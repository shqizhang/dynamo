# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""S2 — Elastic role switching for a Dynamo vLLM worker (RL-Scaling).

This module orchestrates a *role flip* (prefill <-> decode) on a single
worker without restarting the engine. It re-uses the existing
``BaseWorkerHandler.sleep`` / ``wake_up`` machinery so that the GPU is fully
quiesced before any reconfiguration touches device memory.

High-level flow (mirrors design doc S2.4 / S2.7):

    1. Acquire the per-worker switch lock (idempotency guard).
    2. ``handler.sleep(level=2)`` — drain in-flight, free KV memory.
    3. Reconfigure NIXL agent for the new role (stub: see ``_reconfig_nixl``).
    4. Reconfigure KV pool for the new role (stub: see ``_reconfig_kv_pool``).
    5. ``handler.set_disaggregation_mode(target_role)``.
    6. ``handler.wake_up()`` — re-register endpoint to the routing pool.
    7. Emit ``WorkerRoleChanged`` to the KV router (stub: see ``_emit_role_changed``).

The Rust-side reconfig APIs (NIXL, KV pool) and the new ``WorkerRoleChanged``
event variant are tracked as TODO. Until they land, those steps are no-ops
that log a warning so an integration smoke test can still exercise the
sleep/wake path end-to-end. See ``RUST_CHANGES.md`` for the planned patches.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_VALID_ROLES = ("prefill", "decode")


class DualModeWorker:
    """Coordinates a role flip on a single ``BaseWorkerHandler``.

    Parameters
    ----------
    handler:
        A ``BaseWorkerHandler`` (DecodeWorkerHandler / PrefillWorkerHandler).
    initial_role:
        The role the worker started in. Used as the source of truth for
        ``current_role`` until the first successful switch.
    publisher:
        Optional KV-event publisher. If provided, ``WorkerRoleChanged`` is
        emitted after a successful flip. Stubbed today.
    """

    def __init__(
        self,
        handler: Any,
        initial_role: str,
        publisher: Any | None = None,
    ) -> None:
        if initial_role not in _VALID_ROLES:
            raise ValueError(f"initial_role must be one of {_VALID_ROLES}, got {initial_role!r}")
        self._handler = handler
        self._publisher = publisher
        self._lock = asyncio.Lock()
        self._current_role = initial_role
        # Reflect into the handler so other code paths can read it.
        try:
            handler.set_disaggregation_mode(initial_role)
        except Exception:  # noqa: BLE001 - tolerated for handler stubs in tests
            logger.debug("handler does not implement set_disaggregation_mode (test stub?)")

    @property
    def current_role(self) -> str:
        return self._current_role

    # ------------------------------------------------------------------ public
    async def switch_role(self, target_role: str) -> dict:
        """Flip the worker's role.

        Returns a dict suitable for an HTTP response: ``status``,
        ``new_role``, ``switch_time_ms`` (and ``message`` on errors).
        """
        if target_role not in _VALID_ROLES:
            return {
                "status": "error",
                "message": f"target_role must be one of {_VALID_ROLES}, got {target_role!r}",
                "new_role": self._current_role,
                "switch_time_ms": 0.0,
            }
        if self._lock.locked():
            return {
                "status": "busy",
                "message": "another switch is in progress",
                "new_role": self._current_role,
                "switch_time_ms": 0.0,
            }

        async with self._lock:
            if target_role == self._current_role:
                return {
                    "status": "ok",
                    "message": "already in target role",
                    "new_role": self._current_role,
                    "switch_time_ms": 0.0,
                }

            t0 = time.monotonic()
            previous_role = self._current_role
            try:
                # 1. Drain & sleep so the GPU is quiesced.
                sleep_resp = await self._handler.sleep({"level": 2})
                if sleep_resp.get("status") not in {"ok", None}:
                    return {
                        "status": "error",
                        "message": f"sleep failed: {sleep_resp}",
                        "new_role": previous_role,
                        "switch_time_ms": (time.monotonic() - t0) * 1000.0,
                    }

                # 2-3. Reconfigure NIXL + KV pool (stubs — see RUST_CHANGES.md).
                await self._reconfig_nixl(target_role)
                await self._reconfig_kv_pool(target_role)

                # 4. Persist the new role on the handler.
                self._handler.set_disaggregation_mode(target_role)

                # 5. Wake the engine and re-register endpoint.
                wake_resp = await self._handler.wake_up({})
                if wake_resp.get("status") not in {"ok", None}:
                    raise RuntimeError(f"wake_up failed: {wake_resp}")

                # 6. Tell the router the role changed.
                await self._emit_role_changed(previous_role, target_role)

                self._current_role = target_role
                return {
                    "status": "ok",
                    "new_role": target_role,
                    "switch_time_ms": (time.monotonic() - t0) * 1000.0,
                }
            except Exception as exc:  # noqa: BLE001
                logger.exception("role switch %s -> %s failed; attempting recovery", previous_role, target_role)
                # Best-effort: try to wake back up so the worker isn't stuck asleep.
                try:
                    await self._handler.wake_up({})
                    self._handler.set_disaggregation_mode(previous_role)
                except Exception:  # noqa: BLE001
                    logger.exception("recovery wake_up also failed")
                return {
                    "status": "error",
                    "message": str(exc),
                    "new_role": self._current_role,
                    "switch_time_ms": (time.monotonic() - t0) * 1000.0,
                }

    # -------------------------------------------------------- stubbed Rust ops
    async def _reconfig_nixl(self, target_role: str) -> None:
        """TODO(RL-Scaling-S2): call ``handler.engine_client.reconfig_nixl(...)``.

        The Rust-side NIXL agent currently has no reconfig API. Until that
        lands the stub is a no-op so existing single-mode workers continue to
        function unchanged.
        """
        logger.info("[DualMode] _reconfig_nixl(%s) — stubbed; no Rust reconfig API yet", target_role)

    async def _reconfig_kv_pool(self, target_role: str) -> None:
        """TODO(RL-Scaling-S2): call ``handler.engine_client.reconfig_kv_pool(...)``.

        Decode mode wants a large KV pool; prefill mode wants a tiny one.
        Implementing this requires a vLLM patch that exposes the pool
        allocator at runtime. Stubbed for now.
        """
        logger.info("[DualMode] _reconfig_kv_pool(%s) — stubbed; no allocator API yet", target_role)

    async def _emit_role_changed(self, from_role: str, to_role: str) -> None:
        """TODO(RL-Scaling-S2): emit ``WorkerRoleChanged`` to the KV router."""
        if self._publisher is None:
            return
        try:
            publish = getattr(self._publisher, "publish_role_changed", None)
            if publish is None:
                logger.debug("publisher has no publish_role_changed; skipping")
                return
            res = publish(from_role=from_role, to_role=to_role)
            if asyncio.iscoroutine(res):
                await res
        except Exception:  # noqa: BLE001
            logger.exception("failed to publish WorkerRoleChanged")

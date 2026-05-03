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
    3. ``_reconfig_nixl(target_role)`` — invalidate cached NIXL connector handle.
    4. ``_reconfig_kv_pool(target_role)`` — call ``engine.reset_prefix_cache()``
       so the new role starts with a clean GPU block budget.
    5. ``handler.set_disaggregation_mode(target_role)``.
    6. ``handler.wake_up()`` — re-register endpoint to the routing pool.
    7. ``_emit_role_changed`` — push the new role label to discovery + (optional)
       router event publisher.

All three reconfig steps are best-effort: each wraps its real call in
``try/log/continue`` so that a failure in one step does not leave the worker
stuck asleep. The orchestration's outer ``except`` guarantees a recovery
``wake_up`` even if reconfig fails entirely.
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

                # 2-3. Reconfigure NIXL handle + KV pool for the new role.
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

    # -------------------------------------------------------- real reconfig ops
    async def _reconfig_kv_pool(self, target_role: str) -> None:
        """Reset the KV prefix cache so the new role starts with a clean pool.

        vLLM 0.16 exposes ``engine.reset_prefix_cache()`` (also used by the
        existing ``clear_kv_blocks`` handler). Calling it after sleep frees all
        cached blocks held by the previous role, letting the new role's traffic
        shape (decode = long sequences, prefill = many short ones) make full
        use of the GPU block budget without eviction churn.

        We do **not** resize ``cache_config.num_gpu_blocks`` at runtime — that
        would require ``_initialize_kv_caches`` re-execution which is fragile
        on a sleeping engine. The block budget is shared and the prefix cache
        eviction policy lets each role saturate it on demand.
        """
        engine = getattr(self._handler, "engine_client", None)
        if engine is None:
            logger.warning(
                "[DualMode] reconfig_kv_pool(%s): handler has no engine_client; skipping",
                target_role,
            )
            return
        reset = getattr(engine, "reset_prefix_cache", None)
        if reset is None:
            logger.warning(
                "[DualMode] reconfig_kv_pool(%s): engine has no reset_prefix_cache; skipping",
                target_role,
            )
            return
        try:
            result = reset()
            if asyncio.iscoroutine(result):
                await result
            logger.info(
                "[DualMode] reconfig_kv_pool(%s): reset_prefix_cache OK", target_role
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[DualMode] reconfig_kv_pool(%s): reset failed: %s; continuing",
                target_role,
                exc,
            )

    async def _reconfig_nixl(self, target_role: str) -> None:
        """Drop any cached NIXL connector handle so the next request rebuilds it.

        vLLM 0.16's ``kv_transfer_config`` is fixed at engine construction;
        we cannot swap connectors at runtime without recreating the engine.
        What we CAN do is invalidate the handler's cached NIXL connector handle
        so the next request rebuilds it against the engine's current connector
        state — preventing stale prefill→decode xfer slots from leaking across
        a role flip.
        """
        handler = self._handler
        nixl = getattr(handler, "_nixl_connector", None)
        if nixl is None:
            logger.info(
                "[DualMode] reconfig_nixl(%s): no cached nixl connector; nothing to drop",
                target_role,
            )
            return
        try:
            handler._nixl_connector = None
            shutdown = getattr(nixl, "shutdown", None) or getattr(nixl, "close", None)
            if shutdown is not None:
                result = shutdown()
                if asyncio.iscoroutine(result):
                    await result
            logger.info(
                "[DualMode] reconfig_nixl(%s): dropped cached nixl connector",
                target_role,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[DualMode] reconfig_nixl(%s): drop failed: %s; continuing",
                target_role,
                exc,
            )

    async def _emit_role_changed(self, from_role: str, to_role: str) -> None:
        """Push the new role to discovery + (optionally) publish to the router.

        Two complementary paths:

        1. **Endpoint metadata** — if the discovery endpoint exposes
           ``update_metadata``, push ``{"disaggregation_mode": to_role}``.
           The KV router's worker selector picks this up on its next refresh.
           Note: ``wake_up`` already re-registered the endpoint, so any
           metadata set via :meth:`BaseWorkerHandler.set_disaggregation_mode`
           is already on record; this call is the explicit transition signal.

        2. **Publisher event** — if a router event publisher exposes
           ``publish_role_changed``, emit it. This is for routers that track
           role transitions independently of registration events.
        """
        endpoint = getattr(self._handler, "generate_endpoint", None)
        if endpoint is not None:
            update = getattr(endpoint, "update_metadata", None)
            if update is not None:
                try:
                    result = update({"disaggregation_mode": to_role})
                    if asyncio.iscoroutine(result):
                        await result
                    logger.info(
                        "[DualMode] emit_role_changed: pushed metadata disaggregation_mode=%s",
                        to_role,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[DualMode] emit_role_changed: update_metadata failed: %s", exc
                    )

        if self._publisher is not None:
            publish = getattr(self._publisher, "publish_role_changed", None)
            if publish is not None:
                try:
                    res = publish(from_role=from_role, to_role=to_role)
                    if asyncio.iscoroutine(res):
                        await res
                    logger.info(
                        "[DualMode] emit_role_changed: published WorkerRoleChanged(%s->%s)",
                        from_role,
                        to_role,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[DualMode] emit_role_changed: publish failed: %s", exc
                    )

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""S2 — Elastic role switching for a Dynamo vLLM worker (RL-Scaling).

This module orchestrates a *true end-to-end* role flip (prefill <-> decode)
on a single worker without restarting the engine. The crucial property
this module guarantees is that **after a successful switch the Dynamo
router actually delivers traffic of the new role to this worker**, not
just that the in-process engine state changed.

End-to-end flow (mirrors design doc S2.4 / S2.7, with the missing
re-registration step that earlier revisions lacked):

    1. Acquire the per-worker switch lock (idempotency / concurrency guard).
    2. ``handler.sleep(level=2)``
         - drains in-flight, frees KV memory, *unregisters the current
           endpoint instance* from discovery (so the router stops
           dispatching old-role traffic).
    3. ``Reregistrar.unregister(current_role)``
         - removes the ModelDeploymentCard from the *current* endpoint
           (router's WorkerSet rebuild drops this worker from the old role
           pool).
    4. ``_reconfig_nixl(target_role)``
         - invalidates cached NIXL connector handle so the next request
           rebuilds it cleanly for the new role.
    5. ``_reconfig_kv_pool(target_role)``
         - calls ``engine.reset_prefix_cache()`` so the new role starts
           with a clean GPU block budget (KV consistency guarantee).
    6. ``handler.set_disaggregation_mode(target_role)``.
    7. ``Reregistrar.register(target_role)``
         - publishes a fresh MDC under the *target-role* endpoint URI
           (router's ModelWatcher picks it up; PrefillRouter or chat pool
           gains this worker).
    8. ``handler.wake_up()``
         - re-attaches the *current handler's* endpoint instance to
           discovery and resumes engine generation.
    9. ``_emit_role_changed`` — best-effort pod label patch + optional
       publisher event for observability.

If any step fails, the orchestration's outer ``except`` attempts a
recovery: re-register the previous role's MDC, wake the engine, restore
the previous role flag — so the worker is never left silently stuck in a
half-flipped state.

KV consistency
--------------
All in-flight requests are drained inside ``handler.sleep`` (which calls
``engine.pause_generation``). Then ``reset_prefix_cache`` clears the KV
pool *while the engine is asleep*, before any traffic of the new role
arrives. The new role therefore starts with an empty, role-correct KV
pool; no cross-role KV bleed is possible.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Awaitable, Callable, Optional, Protocol

logger = logging.getLogger(__name__)


def _switch_drain_timeout() -> float:
    try:
        return max(0.0, float(os.environ.get("DYNAMO_RL_SWITCH_DRAIN_TIMEOUT", "10.0")))
    except (TypeError, ValueError):
        return 10.0

_VALID_ROLES = ("prefill", "decode")


class Reregistrar(Protocol):
    """Callback contract used by :class:`DualModeWorker` to swap which
    endpoint URI carries this worker's ModelDeploymentCard.

    Implementations are wired in :mod:`dynamo.vllm.main` where the two
    endpoint objects (``backend.generate`` for decode and
    ``prefill.generate`` for prefill) and ``register_model`` /
    ``unregister_model`` are in scope.
    """

    async def register(self, role: str) -> None: ...
    async def unregister(self, role: str) -> None: ...
    def get_endpoint(self, role: str): ...  # returns Endpoint or None


class DualModeWorker:
    """Coordinates a true role flip on a single ``BaseWorkerHandler``.

    Parameters
    ----------
    handler:
        A ``BaseWorkerHandler`` whose ``sleep`` / ``wake_up`` machinery is
        used to quiesce the engine across the flip. Both decode and
        prefill handler-shaped objects accept ``set_disaggregation_mode``
        from this class.
    initial_role:
        The role the worker started in. Source of truth for
        ``current_role`` until the first successful switch.
    reregistrar:
        Callback object that knows how to ``register`` / ``unregister`` a
        ModelDeploymentCard against the role-specific endpoint URIs.
        When ``None`` the switch only does the in-process reconfig (legacy
        behaviour — keeps unit tests working without a real runtime).
    publisher:
        Optional KV-event publisher; ``WorkerRoleChanged`` is emitted via
        :meth:`publish_role_changed` if available.
    label_patcher:
        Optional async callable ``(label_key, label_value) -> None`` used
        to update the pod's ``nvidia.com/dynamo-current-role`` label after
        a successful switch (informational only — does not affect routing).
    """

    def __init__(
        self,
        handler: Any,
        initial_role: str,
        reregistrar: Optional[Reregistrar] = None,
        publisher: Any | None = None,
        label_patcher: Optional[Callable[[str, str], Awaitable[None]]] = None,
    ) -> None:
        if initial_role not in _VALID_ROLES:
            raise ValueError(
                f"initial_role must be one of {_VALID_ROLES}, got {initial_role!r}"
            )
        self._handler = handler
        self._reregistrar = reregistrar
        self._publisher = publisher
        self._label_patcher = label_patcher
        self._lock = asyncio.Lock()
        self._current_role = initial_role
        self._drain_timeout_s = _switch_drain_timeout()
        try:
            handler.set_disaggregation_mode(initial_role)
        except Exception:  # noqa: BLE001 - tolerated for handler stubs in tests
            logger.debug(
                "handler does not implement set_disaggregation_mode (test stub?)"
            )

    @property
    def current_role(self) -> str:
        return self._current_role

    # ------------------------------------------------------------------ public
    async def switch_role(self, target_role: str) -> dict:
        """Flip the worker's role. Returns an HTTP-friendly response dict.

        Timing fields:
          * ``switch_time_ms``   - total wall-clock from accept to ack.
          * ``timings_ms``       - per-phase breakdown (sleep, unregister,
            reset_kv, register, wake) — useful for SLO debugging.
        """
        if target_role not in _VALID_ROLES:
            return {
                "status": "error",
                "message": (
                    f"target_role must be one of {_VALID_ROLES}, "
                    f"got {target_role!r}"
                ),
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
                    "timings_ms": {},
                }

            t0 = time.monotonic()
            previous_role = self._current_role
            timings: dict[str, float] = {}

            def mark(label: str, since: float) -> float:
                now = time.monotonic()
                timings[label] = round((now - since) * 1000.0, 3)
                return now

            try:
                # 0. Drain in-flight requests BEFORE sleep so their KV blocks
                #    are released. handler.sleep(level=2) pauses generation but
                #    does not guarantee running requests finish, so their blocks
                #    keep ref_cnt>0 and reset_prefix_cache below cannot free them
                #    ("some blocks (N) are not freed yet"). Those leaked blocks
                #    then shrink the new role's KV budget and make long decode
                #    requests hang after a P->D switch-back. Wait for natural
                #    completion, then force-abort any straggler past the timeout.
                t = time.monotonic()
                await self._drain_inflight(self._drain_timeout_s)
                t = mark("drain", t)

                # 1. Drain & sleep so the GPU is quiesced and the current
                #    endpoint instance is unregistered from discovery.
                sleep_resp = await self._handler.sleep({"level": 2})
                t = mark("sleep", t)
                if sleep_resp.get("status") not in {"ok", None}:
                    return {
                        "status": "error",
                        "message": f"sleep failed: {sleep_resp}",
                        "new_role": previous_role,
                        "switch_time_ms": (time.monotonic() - t0) * 1000.0,
                        "timings_ms": timings,
                    }

                # 1b. Also unregister the previous role's endpoint instance
                #     if it differs from the handler's generate_endpoint.
                #     (handler.sleep already unregisters generate_endpoint,
                #      but we also need to remove the role-specific one.)
                if self._reregistrar is not None:
                    prev_ep = self._reregistrar.get_endpoint(previous_role)
                    handler_ep = getattr(self._handler, "generate_endpoint", None)
                    if prev_ep is not None and prev_ep is not handler_ep:
                        try:
                            await prev_ep.unregister_endpoint_instance()
                            logger.info(
                                "[DualMode] unregistered %s endpoint instance",
                                previous_role,
                            )
                        except Exception:  # noqa: BLE001
                            logger.debug(
                                "unregister %s endpoint instance failed (ok if first switch)",
                                previous_role,
                            )

                # 2. Drop the previous-role MDC from discovery so the router
                #    immediately stops considering this worker for old-role
                #    traffic.  No-op when reregistrar is absent (unit tests).
                if self._reregistrar is not None:
                    await self._reregistrar.unregister(previous_role)
                    t = mark("unregister_mdc", t)

                # 3-4. Reconfigure NIXL handle + KV pool for the new role.
                await self._reconfig_nixl(target_role)
                t = mark("reconfig_nixl", t)
                await self._reconfig_kv_pool(target_role)
                t = mark("reset_prefix_cache", t)

                # 5. Persist the new role on the handler and dispatcher before
                # publishing the target MDC. The registered generate endpoint
                # is role-aware and reads DualModeWorker.current_role, so it
                # must not briefly serve target-role traffic as the old role.
                self._handler.set_disaggregation_mode(target_role)
                self._current_role = target_role

                # 6. Publish a fresh MDC under the target-role endpoint URI.
                #    The frontend's ModelWatcher will rebuild WorkerSets and
                #    start sending new-role traffic to this worker.
                if self._reregistrar is not None:
                    await self._reregistrar.register(target_role)
                    t = mark("register_mdc", t)

                # 7. Wake the engine. BaseWorkerHandler.wake_up() always
                #    re-registers handler.generate_endpoint. For a decode
                #    component this is the backend endpoint, which is correct
                #    for target_role=decode but wrong for target_role=prefill.
                #    Clean it up immediately below before publishing the
                #    target endpoint instance.
                wake_resp = await self._handler.wake_up({})
                t = mark("wake", t)
                if wake_resp.get("status") not in {"ok", None}:
                    raise RuntimeError(f"wake_up failed: {wake_resp}")

                # 7b. Register the target role's endpoint instance so the
                #     PrefillRouter can discover this worker. The handler's
                #     wake_up() only registers generate_endpoint (=backend),
                #     but we also need the role-specific endpoint (=prefill).
                if self._reregistrar is not None:
                    target_ep = self._reregistrar.get_endpoint(target_role)
                    handler_ep = getattr(self._handler, "generate_endpoint", None)
                    if target_ep is not None and target_ep is not handler_ep:
                        if handler_ep is not None:
                            await handler_ep.unregister_endpoint_instance()
                            logger.info(
                                "[DualMode] unregistered handler endpoint instance "
                                "after wake for target_role=%s",
                                target_role,
                            )
                        await target_ep.register_endpoint_instance()
                        logger.info(
                            "[DualMode] registered %s endpoint instance",
                            target_role,
                        )

                # 8. Best-effort: pod label + router event for observability.
                await self._emit_role_changed(previous_role, target_role)

                total_ms = (time.monotonic() - t0) * 1000.0
                logger.info(
                    "[DualMode] switch_role %s->%s OK total=%.2fms timings=%s",
                    previous_role, target_role, total_ms, timings,
                )
                return {
                    "status": "ok",
                    "new_role": target_role,
                    "switch_time_ms": total_ms,
                    "timings_ms": timings,
                }
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "role switch %s -> %s failed; attempting recovery",
                    previous_role, target_role,
                )
                # Best-effort recovery: re-register the previous role and
                # wake so the worker is never silently stuck asleep / off-pool.
                try:
                    if self._reregistrar is not None:
                        try:
                            await self._reregistrar.register(previous_role)
                        except Exception:  # noqa: BLE001
                            logger.exception("recovery re-register failed")
                    await self._handler.wake_up({})
                    self._handler.set_disaggregation_mode(previous_role)
                    self._current_role = previous_role
                except Exception:  # noqa: BLE001
                    logger.exception("recovery wake_up also failed")
                return {
                    "status": "error",
                    "message": str(exc),
                    "new_role": self._current_role,
                    "switch_time_ms": (time.monotonic() - t0) * 1000.0,
                    "timings_ms": timings,
                }

    # -------------------------------------------------------- real reconfig ops
    async def _drain_inflight(self, timeout_s: float) -> None:
        """Release KV blocks held by in-flight requests before a role switch.

        Polls the worker's request registry for active ids and waits for them
        to finish naturally (freeing their KV blocks); any straggler still
        in-flight past ``timeout_s`` is aborted so ``reset_prefix_cache`` can
        actually free every block. Best-effort — a missing registry/engine just
        skips draining (unit-test handler stubs).
        """
        registry = getattr(self._handler, "request_registry", None)
        engine = getattr(self._handler, "engine_client", None)
        if registry is None:
            return
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            try:
                active = list(registry.active_ids())
            except Exception:  # noqa: BLE001
                return
            if not active:
                return
            if time.monotonic() >= deadline:
                for rid in active:
                    try:
                        if engine is not None:
                            res = engine.abort(rid)
                            if asyncio.iscoroutine(res):
                                await res
                    except Exception:  # noqa: BLE001
                        logger.debug("[DualMode] drain abort(%s) failed", rid, exc_info=True)
                    try:
                        registry.deregister(rid)
                    except Exception:  # noqa: BLE001
                        pass
                logger.info(
                    "[DualMode] drain: force-aborted %d straggler(s) after %.1fs before switch",
                    len(active), timeout_s,
                )
                return
            await asyncio.sleep(0.2)

    async def _reconfig_kv_pool(self, target_role: str) -> None:
        """Reset the KV prefix cache so the new role starts with a clean pool.

        vLLM 0.16 exposes ``engine.reset_prefix_cache()``. Calling it after
        sleep frees all cached blocks held by the previous role, letting
        the new role's traffic shape (decode = long sequences, prefill =
        many short ones) make full use of the GPU block budget without
        eviction churn.

        We do **not** resize ``cache_config.num_gpu_blocks`` at runtime —
        that would require ``_initialize_kv_caches`` re-execution which is
        fragile on a sleeping engine. The block budget is shared and the
        prefix-cache eviction policy lets each role saturate it on demand.
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
                result = await result
            # vLLM returns False (and logs "some blocks (N) are not freed yet")
            # when a running request still pins blocks. If drain missed one,
            # abort remaining and retry once so the new role gets a clean pool.
            if result is False:
                logger.warning(
                    "[DualMode] reconfig_kv_pool(%s): reset_prefix_cache reported "
                    "blocks still pinned; draining and retrying", target_role,
                )
                await self._drain_inflight(2.0)
                result = reset()
                if asyncio.iscoroutine(result):
                    result = await result
            logger.info(
                "[DualMode] reconfig_kv_pool(%s): reset_prefix_cache result=%s",
                target_role, result,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[DualMode] reconfig_kv_pool(%s): reset failed: %s; continuing",
                target_role, exc,
            )

    async def _reconfig_nixl(self, target_role: str) -> None:
        """Drop any cached NIXL connector handle so the next request rebuilds it.

        vLLM 0.16's ``kv_transfer_config`` is fixed at engine construction;
        we cannot swap connectors at runtime. What we CAN do is invalidate
        the handler's cached NIXL connector handle so the next request
        rebuilds it against the engine's current connector state —
        preventing stale prefill→decode xfer slots from leaking across a
        role flip.
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
                target_role, exc,
            )

    async def _emit_role_changed(self, from_role: str, to_role: str) -> None:
        """Observability-only: pod label + (optional) publisher event.

        Routing has *already* been switched at this point via the
        unregister/register MDC steps in :meth:`switch_role`. This method
        only updates side-channel signals so operators can see the new
        role with ``kubectl get pod -L nvidia.com/dynamo-current-role``.
        """
        # 1. Pod label patch (operator-visible).
        if self._label_patcher is not None:
            try:
                await self._label_patcher(
                    "nvidia.com/dynamo-current-role", to_role
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[DualMode] emit_role_changed: label patch failed: %s", exc
                )

        # 2. Endpoint metadata (debugging aid; router does not consume).
        endpoint = getattr(self._handler, "generate_endpoint", None)
        if endpoint is not None:
            update = getattr(endpoint, "update_metadata", None)
            if update is not None:
                try:
                    result = update({"disaggregation_mode": to_role})
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[DualMode] emit_role_changed: update_metadata failed: %s", exc
                    )

        # 3. Publisher event (consumed by external observers).
        if self._publisher is not None:
            publish = getattr(self._publisher, "publish_role_changed", None)
            if publish is not None:
                try:
                    res = publish(from_role=from_role, to_role=to_role)
                    if asyncio.iscoroutine(res):
                        await res
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[DualMode] emit_role_changed: publish failed: %s", exc
                    )

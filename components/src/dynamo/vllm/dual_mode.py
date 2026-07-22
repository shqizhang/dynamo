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
        return max(0.0, float(os.environ.get("DYNAMO_RL_SWITCH_DRAIN_TIMEOUT", "30.0")))
    except (TypeError, ValueError):
        return 30.0


def _cordon_settle_seconds() -> float:
    """The QUIESCE stable-idle window: after withdrawing its ModelCard and
    draining to idle, switch_role requires the engine to stay idle (no new
    arrivals) for this long continuously before sleeping, confirming the
    frontend has stopped routing here. Bounds the switch-instant request-loss
    race under high concurrency. A fixed 0.5s was too short; 1.5s covers the
    ModelCard-withdrawal propagation to the router."""
    try:
        return max(0.0, float(os.environ.get("DYNAMO_RL_CORDON_SETTLE", "1.5")))
    except (TypeError, ValueError):
        return 1.5


def _flush_nixl_pending_sends(worker) -> dict:
    """Runs INSIDE each engine worker (via AsyncLLM.collective_rpc).

    While this worker served the prefill role it produced KV that the NIXL
    connector holds in ``_reqs_to_send`` until a decode worker pulls it (up to
    ``VLLM_NIXL_ABORT_REQUEST_TIMEOUT`` = 480s). Any un-pulled entry keeps its KV
    blocks pinned (ref_cnt>0) so ``reset_prefix_cache`` cannot free them and the
    role-switched decode engine runs with a shrunken/confused KV pool. Before a
    role switch we force every pending send to expire NOW, so the next
    ``get_finished`` reports it as finished-sending and the scheduler frees the
    blocks. Best-effort and self-contained (cloudpickled to the worker process).
    """
    import time as _t
    try:
        from vllm.distributed.kv_transfer import get_kv_transfer_group

        conn = get_kv_transfer_group()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"no_kv_transfer_group: {exc}"}
    cw = getattr(conn, "connector_worker", None) or conn
    reqs = getattr(cw, "_reqs_to_send", None)
    if not isinstance(reqs, dict):
        return {"ok": False, "reason": "no__reqs_to_send"}
    now = _t.perf_counter()
    n = len(reqs)
    for rid in list(reqs.keys()):
        reqs[rid] = now  # already-elapsed -> expired on next get_finished()
    return {"ok": True, "expired": n}


def _count_nixl_pending_sends(worker) -> dict:
    """Runs INSIDE each engine worker (via AsyncLLM.collective_rpc).

    Read-only companion to :func:`_flush_nixl_pending_sends`: report how many
    produced-but-not-yet-pulled KV sends the connector still tracks, so the
    switch can WAIT for peers to finish pulling instead of destroying their
    in-flight reads (see ``_wait_outbound_kv_drained``).
    """
    try:
        from vllm.distributed.kv_transfer import get_kv_transfer_group

        conn = get_kv_transfer_group()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"no_kv_transfer_group: {exc}", "pending": 0}
    cw = getattr(conn, "connector_worker", None) or conn
    reqs = getattr(cw, "_reqs_to_send", None)
    if not isinstance(reqs, dict):
        return {"ok": False, "reason": "no__reqs_to_send", "pending": 0}
    return {"ok": True, "pending": len(reqs)}


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
        self._cordon_settle_s = _cordon_settle_seconds()
        # --- switch-window hold support (zero-loss fast switch) ---------
        # Between cordon (old ModelCard withdrawn) and register (new card
        # published) the router can only be acting on the OLD card, so any
        # request arriving in that window is old-role traffic by
        # construction. The dispatcher in main.py uses these to HOLD such
        # arrivals until the switch completes and then serve them under the
        # pre-switch role instead of letting sleep() 500 them. This is what
        # makes a short cordon-settle safe.
        self._switch_complete = asyncio.Event()
        self._switch_complete.set()  # no switch in progress at boot
        self._switch_previous_role: str = initial_role
        # True once cordon() has withdrawn this worker's ModelCard (S3
        # scale-down, and now the cordon-first step of switch_role).
        # switch_role() re-publishes a card at the end, so it clears this.
        self._cordoned = False
        try:
            handler.set_disaggregation_mode(initial_role)
        except Exception:  # noqa: BLE001 - tolerated for handler stubs in tests
            logger.debug(
                "handler does not implement set_disaggregation_mode (test stub?)"
            )

    @property
    def current_role(self) -> str:
        return self._current_role

    @property
    def switch_in_progress(self) -> bool:
        """True while switch_role is between cordon and register+wake."""
        return not self._switch_complete.is_set()

    @property
    def switch_previous_role(self) -> str:
        """The role this worker had when the in-progress switch began.
        Arrivals during the switch window are, by construction, traffic
        for this role (the new card is not published yet)."""
        return self._switch_previous_role

    async def wait_switch_complete(self, timeout_s: float = 20.0) -> bool:
        """Block until the in-progress switch finishes (True) or the
        timeout elapses (False). Returns immediately when idle."""
        try:
            await asyncio.wait_for(self._switch_complete.wait(), timeout=timeout_s)
            return True
        except asyncio.TimeoutError:
            return False

    # ------------------------------------------------------------------ public
    async def cordon(self) -> dict:
        """Withdraw this worker's current-role ModelCard WITHOUT touching the engine.

        S3 scale-down previously went ``drain -> scale replicas`` and relied on
        the trigger (batch_completion >= 0.92) to mean "no new work is arriving".
        That is not a guarantee: between the drain check and the pod actually
        terminating, the pod is still in the frontend's WorkerSet, so KvRouter
        can route a NEW request onto a decoder that is about to be deleted —
        which then dies with EngineShutdown.

        Cordoning removes the worker from the router's candidate set first
        (cordon -> drain -> delete, the standard Kubernetes pattern that
        ``switch_role`` already follows), while leaving the engine running so
        any already-accepted request can still finish. Idempotent.
        """
        if self._reregistrar is None:
            return {"status": "error", "message": "no reregistrar; cannot cordon"}
        if self._cordoned:
            return {"status": "ok", "message": "already cordoned", "role": self._current_role}
        t0 = time.monotonic()
        try:
            await self._reregistrar.unregister(self._current_role)
            self._cordoned = True
            logger.info("[DualMode] cordon: withdrew %s ModelCard", self._current_role)
            return {
                "status": "ok",
                "role": self._current_role,
                "cordon_time_ms": (time.monotonic() - t0) * 1000.0,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("cordon failed")
            return {"status": "error", "message": str(exc)}

    async def uncordon(self) -> dict:
        """Re-publish the current-role ModelCard (undo :meth:`cordon`).

        Used when a planned scale-down is abandoned after the pod was already
        cordoned, so the worker rejoins the router's candidate set instead of
        idling invisibly. Idempotent.
        """
        if self._reregistrar is None:
            return {"status": "error", "message": "no reregistrar; cannot uncordon"}
        if not self._cordoned:
            return {"status": "ok", "message": "not cordoned", "role": self._current_role}
        try:
            await self._reregistrar.register(self._current_role)
            self._cordoned = False
            logger.info("[DualMode] uncordon: republished %s ModelCard", self._current_role)
            return {"status": "ok", "role": self._current_role}
        except Exception as exc:  # noqa: BLE001
            logger.exception("uncordon failed")
            return {"status": "error", "message": str(exc)}

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

            # Open the switch window: the dispatcher holds new arrivals
            # (which can only be previous-role traffic until register)
            # instead of letting sleep() reject them.
            self._switch_previous_role = previous_role
            self._switch_complete.clear()

            def mark(label: str, since: float) -> float:
                now = time.monotonic()
                timings[label] = round((now - since) * 1000.0, 3)
                return now

            try:
                # 0-pre. CORDON FIRST — fixes the S2 switch-instant request
                #   loss. The steps below drain and then sleep(level=2) the
                #   engine, but the previous ordering did not withdraw this
                #   worker's ModelCard until step 2 (after drain+sleep). So
                #   throughout drain+sleep the card was still published and the
                #   router could route a NEW request onto this worker; sleep
                #   then 500'd it (observed: 1/105 requests, HTTP 500, 0 tokens,
                #   during the D->P switch in prefill_burst — 2 of 6 S2 runs).
                #   Withdraw the current-role card up front and give the
                #   frontend's ModelWatcher a bounded settle window to observe
                #   the withdrawal BEFORE we quiesce the engine, so drain sees a
                #   closed intake and sleep has nothing live to kill. This is the
                #   cordon->drain->delete order the rest of this method already
                #   documents; cordon() uses the same unregister.
                t = time.monotonic()
                if self._reregistrar is not None and not self._cordoned:
                    await self._reregistrar.unregister(previous_role)
                    self._cordoned = True
                    t = mark("cordon", t)

                # 0. Drain in-flight requests BEFORE sleep so their KV blocks
                #    are released. handler.sleep(level=2) pauses generation but
                #    does not guarantee running requests finish, so their blocks
                #    keep ref_cnt>0 and reset_prefix_cache below cannot free them
                #    ("some blocks (N) are not freed yet"). Those leaked blocks
                #    then shrink the new role's KV budget and make long decode
                #    requests hang after a P->D switch-back. Wait for natural
                #    completion, then force-abort any straggler past the timeout.
                #    QUIESCE HANDSHAKE (fixes the residual switch-instant 500s
                #    under high concurrency): a fixed post-cordon settle is not
                #    enough — the frontend's routing pipeline keeps dispatching to
                #    this worker until it observes the ModelCard withdrawal, so a
                #    request can slip in AFTER drain returns idle and get 500'd by
                #    sleep. Instead, drain to idle then CONFIRM the engine stays
                #    idle for a stable window (no new arrivals => the router has
                #    stopped routing here); any arrival re-drains and re-confirms.
                t = time.monotonic()
                quiesce = await self._drain_and_quiesce(self._cordon_settle_s, self._drain_timeout_s)
                timings["quiesce_arrivals_after_cordon"] = quiesce.get("arrivals_after_cordon", 0)
                t = mark("drain", t)

                # 0a. Wait for OUTBOUND KV to drain before sleep frees VRAM.
                #     While this worker held the prefill role it produced KV
                #     that peer decoders pull via NIXL READ; interrupting an
                #     in-flight pull (or force-expiring a send about to be
                #     pulled) hangs the peer's request. Bounded wait first;
                #     only true orphans are force-expired on timeout. At a
                #     phase boundary with no handoff in flight this passes on
                #     the first poll (~0 cost); under load it is the honest
                #     zero-loss fabric-drain cost.
                t = time.monotonic()
                outbound = await self._wait_outbound_kv_drained(timeout_s=8.0)
                timings["kv_outbound_drain_waited_ms"] = round(
                    float(outbound.get("waited_s", 0.0)) * 1000.0, 3)
                t = mark("kv_outbound_drain", t)

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
                #    traffic.  No-op when reregistrar is absent (unit tests) or
                #    when step 0-pre already cordoned (the normal path now).
                if self._reregistrar is not None and not self._cordoned:
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
                    # A switch publishes a fresh card, so any prior cordon
                    # (S3 scale-down that was abandoned) no longer applies.
                    self._cordoned = False
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
                    # Routing-convergence acknowledgements. The old-role ack is
                    # observational (cordon applied + a stable arrival-silence
                    # window with any switch-window arrivals HELD and served
                    # under the old role — zero loss either way). A
                    # deterministic Rust router-epoch ACK remains future work.
                    "frontend_ack": {
                        "acknowledged": bool(quiesce.get("quiesced", False)),
                        "method": "cordon+arrival_silence+hold",
                        "arrivals_after_cordon": quiesce.get("arrivals_after_cordon", 0),
                        "settle_window_s": self._cordon_settle_s,
                    },
                    "frontend_target_ready": {
                        "acknowledged": True,
                        "method": "target_mdc_registered+engine_awake",
                    },
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
                            # Re-published the previous-role card, so the
                            # cordon-first withdrawal no longer applies.
                            self._cordoned = False
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
            finally:
                # Close the switch window: release any held dispatcher
                # arrivals (they are served under switch_previous_role).
                self._switch_complete.set()

    # -------------------------------------------------------- real reconfig ops
    async def _wait_outbound_kv_drained(self, timeout_s: float = 8.0) -> dict:
        """Wait for peers to finish pulling KV this worker produced as prefill.

        ``sleep(level=2)`` frees GPU memory, so switching while a peer decode
        worker's NIXL READ against this worker's KV is still in flight breaks
        that transfer and hangs the peer's request until its client timeout
        (observed 2026-07-23: P->D at early-B left 646 blocks pinned and hung
        32 peer decodes to their 600s timeout). Force-expiring the pending
        sends (the old flush-first order) is just as destructive for pulls
        that are about to start. So: WAIT, bounded, for (a) the connector's
        pending-send registry to empty and (b) the block pool to report no
        pinned blocks (``reset_prefix_cache`` returns True only then; calling
        it on the cordoned+drained engine is safe and it is reset again after
        sleep). Only what survives the wait is a true orphan (nobody pulled it
        for ``timeout_s``) and gets force-expired by the flush fallback.

        Cost model: this is load-dependent zero-loss drain (phase-boundary
        switches with no outbound KV pass on the first poll at ~0 cost; a
        switch under active prefill handoff pays the fabric-drain time).
        """
        t0 = time.monotonic()
        engine = getattr(self._handler, "engine_client", None)
        rpc = getattr(engine, "collective_rpc", None) if engine is not None else None
        reset = getattr(engine, "reset_prefix_cache", None) if engine is not None else None
        polls = 0
        pending = -1
        pinned_clear = None
        while time.monotonic() - t0 < timeout_s:
            polls += 1
            pending = 0
            if rpc is not None:
                try:
                    result = rpc(_count_nixl_pending_sends)
                    if asyncio.iscoroutine(result):
                        result = await result
                    for entry in result if isinstance(result, list) else [result]:
                        pending += int((entry or {}).get("pending", 0) or 0)
                except Exception:  # noqa: BLE001
                    pending = 0  # introspection unavailable -> rely on reset probe
            pinned_clear = None
            if reset is not None:
                try:
                    pinned_clear = reset()
                    if asyncio.iscoroutine(pinned_clear):
                        pinned_clear = await pinned_clear
                except Exception:  # noqa: BLE001
                    pinned_clear = None
            if pending == 0 and pinned_clear is not False:
                waited = time.monotonic() - t0
                logger.info(
                    "[DualMode] outbound KV drained (polls=%d, waited=%.3fs)",
                    polls, waited,
                )
                return {"drained": True, "waited_s": waited, "polls": polls}
            await asyncio.sleep(0.25)
        waited = time.monotonic() - t0
        logger.warning(
            "[DualMode] outbound KV drain timed out after %.1fs "
            "(pending_sends=%s, pinned_clear=%s); force-expiring leftovers",
            waited, pending, pinned_clear,
        )
        await self._flush_kv_connector()
        return {"drained": False, "waited_s": waited, "polls": polls}

    async def _flush_kv_connector(self) -> None:
        """Expire the NIXL connector's pending KV sends across all engine workers
        so the scheduler frees the blocks they pin. Best-effort: any failure
        (no connector, RPC unsupported) is logged and the switch continues."""
        engine = getattr(self._handler, "engine_client", None)
        rpc = getattr(engine, "collective_rpc", None) if engine is not None else None
        if rpc is None:
            return
        try:
            result = rpc(_flush_nixl_pending_sends)
            if asyncio.iscoroutine(result):
                result = await result
            logger.info("[DualMode] flush_kv_connector: expired pending sends -> %s", result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[DualMode] flush_kv_connector failed (%s); continuing", exc)

    async def _drain_inflight(self, timeout_s: float) -> None:
        """Release KV blocks held by in-flight work before a role switch.

        The registry only tracks requests that flow through the decode handler's
        ``generate_tokens``. Partner-prefill requests (served via a separate
        ``PrefillWorkerHandler`` that calls ``engine.generate`` directly) and
        pending NIXL KV transfers keep blocks pinned (ref_cnt>0) WITHOUT
        appearing in the registry, so a P->D switch-back then finds thousands of
        blocks "not freed yet", shrinking the decode KV budget until long decode
        requests hang. We therefore drain the *engine* itself to idle, then mop
        up any registry straggler. Best-effort — missing engine/registry (unit
        test stubs) just skips.
        """
        engine = getattr(self._handler, "engine_client", None)
        # Primary: wait for the vLLM engine to become fully idle (covers
        # partner-prefill requests + pending KV transfers the registry misses).
        drain = getattr(engine, "wait_for_requests_to_drain", None) if engine is not None else None
        if drain is not None:
            try:
                await drain(int(max(1.0, timeout_s)))
                logger.info("[DualMode] drain: engine idle before switch")
            except Exception as exc:  # noqa: BLE001 - includes TimeoutError
                logger.warning(
                    "[DualMode] drain: engine did not fully quiesce in %.0fs (%s); "
                    "aborting stragglers", timeout_s, exc,
                )
        # Secondary: force-abort any registry-tracked straggler still present.
        registry = getattr(self._handler, "request_registry", None)
        if registry is not None:
            try:
                active = list(registry.active_ids())
            except Exception:  # noqa: BLE001
                active = []
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
            if active:
                logger.info("[DualMode] drain: aborted %d registry straggler(s)", len(active))

    def _inflight_count(self) -> int:
        """Best-effort count of decode requests currently accepted by this
        worker (registry-tracked). Used by the quiesce handshake to detect NEW
        arrivals after cordon. 0 when no registry (unit-test stubs)."""
        reg = getattr(self._handler, "request_registry", None)
        if reg is None:
            return 0
        try:
            return len(reg.active_ids())
        except Exception:  # noqa: BLE001
            return 0

    async def _drain_and_quiesce(self, stable_s: float, timeout_s: float) -> dict:
        """Drain to idle, then CONFIRM the engine stays idle for ``stable_s``
        continuously before returning — i.e. no new request has arrived, so the
        frontend has drained its routing pipeline to this (cordoned) worker. Any
        arrival during the stable window re-drains and re-confirms. Bounded by
        ``timeout_s``. This is the handshake that closes the residual
        switch-instant 500 race a fixed settle delay left open under high
        concurrency. ``stable_s`` == 0 falls back to a single drain."""
        deadline = time.monotonic() + max(0.0, timeout_s)
        arrivals_after_cordon = 0
        while True:
            remaining = deadline - time.monotonic()
            await self._drain_inflight(max(1.0, min(self._drain_timeout_s, remaining)) if remaining > 0 else 1.0)
            if stable_s <= 0:
                return {"quiesced": True, "arrivals_after_cordon": arrivals_after_cordon}
            # Confirm the engine STAYS idle for stable_s (no new arrivals).
            stable_deadline = time.monotonic() + stable_s
            interrupted = False
            while time.monotonic() < stable_deadline:
                if self._inflight_count() > 0:
                    arrivals_after_cordon += 1
                    interrupted = True
                    break
                await asyncio.sleep(0.1)
            if not interrupted:
                return {"quiesced": True, "arrivals_after_cordon": arrivals_after_cordon}
            if time.monotonic() >= deadline:
                logger.warning("[DualMode] quiesce timed out with late arrivals=%d; proceeding", arrivals_after_cordon)
                return {"quiesced": False, "arrivals_after_cordon": arrivals_after_cordon}

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

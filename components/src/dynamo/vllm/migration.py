# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""S3 — Request consolidation HTTP handlers (RL-Scaling).

Phase 2 (v4 · coordinated block-hold migration):

* **Phase-2.A** — smart recompute-prefill migration with
  :class:`MigrationPolicy` cost-benefit gate + prefix-cache awareness.

* **Phase-2.B (coordinated)** — when ``connector_enabled=True``
  (env ``DYNAMO_RL_CONNECTOR_ENABLED=1``):

  - ``migrate_out`` **always** defers abort (block-hold), regardless of
    KVBM/NIXL availability.  Enables safe rollback if destination declines.
  - KV transfer (NIXL pull) used only when both KVBM block IDs and NIXL
    coordinates are available; otherwise falls back to recompute-prefill.
  - ``migration_rollback`` releases hold without aborting — source
    request continues as if nothing happened.
  - ``/migrate`` coordinated sidecar endpoint orchestrates the full
    migrate_out → remote migrate_in → complete/rollback flow.

  Block-hold protocol::

    1. migrate_out: hold request (defer abort, blocks pinned)
    2. Orchestrator calls migrate_in on destination
    3a. Success: migration_complete aborts source, frees blocks
    3b. Failure: migration_rollback releases hold, source continues

  Sweeper force-aborts stale holds after DYNAMO_RL_MIGRATION_HOLD_TIMEOUT
  seconds (default 10).

Public surface::

    await handler.migrate_out(body)        -> {status, ..., kv_transfer_params?}
    await handler.migrate_in(body)         -> {status, path, replay_tokens}
    await handler.migration_complete(body) -> {status}
    await handler.migration_rollback(body) -> {status}  # release hold, keep running
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Protocol

logger = logging.getLogger(__name__)


class RequestTracker(Protocol):
    """What :class:`MigrationHandler` needs from the engine."""

    async def get_request_state(self, request_id: str) -> Optional[dict]:
        """Return ``{prompt_tokens, generated_tokens, sampling_params, stop_conditions}`` or ``None``."""

    async def abort_request(self, request_id: str) -> None:
        """Cancel the in-engine request without finalising the client stream."""

    async def submit_request(self, request_id: str, payload: dict) -> None:
        """Resubmit a request to the engine using ``prompt + generated`` as the new prompt."""

    async def list_active_request_ids(self) -> Iterable[str]:
        """Used to resolve the ``"*"`` placeholder."""


class _BlockIdSource(Protocol):
    """Subset of :class:`kvbm.vllm_integration.kv_cache_manager.KvbmCacheManager`
    we depend on. Kept as a Protocol so tests can mock it without importing
    torch / kvbm Rust extensions.
    """

    def get_block_ids(self, request_id: str) -> list[list[int]]: ...


class RequestBlockIndex:
    """Thin adapter over ``KvbmCacheManager.get_block_ids``.

    The Rust KVBM cache manager already maintains a per-request block list
    keyed by ``request_id`` and exposes it via PyO3 at
    ``lib/bindings/kvbm/python/kvbm/vllm_integration/kv_cache_manager.py:295``.
    Phase-2.B's true-D2D migration path needs to know which physical block
    IDs hold a request's KV; this class is the read-only Python view of that.

    Returns ``None`` when the request is unknown, has been freed, or the
    underlying cache manager is unavailable — callers must treat it as a
    best-effort hint, not a guarantee.
    """

    def __init__(self, kvbm_cache_manager: Optional[_BlockIdSource]) -> None:
        self._kvbm = kvbm_cache_manager

    def lookup(self, request_id: str) -> Optional[list[int]]:
        """Return a flat list of block IDs for ``request_id`` or ``None``.

        ``KvbmCacheManager.get_block_ids`` returns ``list[list[int]]`` where
        the outer list is per kv-cache group (always length 1 today). We
        flatten it for callers that just want "the blocks for this request".
        """
        if self._kvbm is None:
            return None
        try:
            grouped = self._kvbm.get_block_ids(request_id)
        except Exception:  # noqa: BLE001
            logger.debug("get_block_ids(%s) failed", request_id, exc_info=True)
            return None
        if not grouped:
            return None
        flat: list[int] = []
        for group in grouped:
            flat.extend(int(b) for b in group)
        return flat or None


#: Callable that returns this worker's NIXL listener coordinates so
#: ``migrate_out`` can advertise them to the destination. Plumbed by
#: ``main.py`` from the engine's ``KvTransferConfig`` once NixlConnector has
#: registered its KV caches. Expected dict shape (matches vLLM 0.16's
#: ``kv_transfer_params`` schema)::
#:
#:     {"engine_id": str, "host": str, "port": int}
#:
#: Returning ``None`` (or the callable being ``None``) means "no NIXL coords
#: available right now" — ``migrate_out`` will simply omit
#: ``kv_transfer_params`` from its response, which makes ``migrate_in`` fall
#: back to the Phase-2.A recompute path automatically.
NixlMetaProvider = Callable[[], Optional[dict]]


# ----------------------------------------------------------------- policy
@dataclass
class MigrationPolicy:
    """Cost-benefit thresholds for ``migrate_in``.

    The cost of a migration is one extra prefill on the destination worker.
    Migrate only when the in-progress decode has enough remaining work that
    the recompute is recouped.

    Defaults are tuned for a ~7B model on a single RTX 3090:
      - ``max_replay_tokens=8192``: above this, recompute prefill exceeds 200 ms
        even with prefix-cache hits — usually not worth it.
      - ``min_generated_tokens=16``: too-young requests have nothing to save.
      - ``min_remaining_tokens=32``: too-old requests will finish faster than a
        migration round-trip.
    """

    max_replay_tokens: int = 8192
    min_generated_tokens: int = 16
    min_remaining_tokens: int = 32


# ---------------------------------------------------------------- handler
class MigrationHandler:
    def __init__(
        self,
        tracker: RequestTracker,
        policy: Optional[MigrationPolicy] = None,
        engine: Any = None,
        block_index: Optional[RequestBlockIndex] = None,
        nixl_meta_provider: Optional[NixlMetaProvider] = None,
        connector_enabled: bool = False,
    ) -> None:
        self._tracker = tracker
        self._policy = policy or MigrationPolicy()
        self._engine = engine
        self._block_index = block_index or RequestBlockIndex(None)
        self._nixl_meta_provider = nixl_meta_provider
        # Phase-2.B block-hold: when connector_enabled=True, migrate_out
        # ALWAYS defers abort (regardless of KVBM/NIXL availability).
        # The orchestrator calls /migration_complete to abort+free, or
        # /migration_rollback to release the hold and let the request
        # continue.  Sweeper force-aborts stale entries.
        self._connector_enabled = connector_enabled
        self._pending_migrations: dict[str, float] = {}  # request_id -> monotonic ts
        self._migration_hold_timeout_s = float(
            os.environ.get("DYNAMO_RL_MIGRATION_HOLD_TIMEOUT", "10.0")
        )
        self._warn_if_prefix_cache_disabled()

    # ------------------------------------------------------------------ API
    async def migrate_out(self, body: dict) -> dict:
        request_id = self._validate_id(body)
        if isinstance(request_id, dict):
            return request_id  # error
        # Resolve "*" -> most-progressed active id (excluding already-held).
        if request_id == "*":
            ids = [
                rid
                for rid in await self._tracker.list_active_request_ids()
                if rid not in self._pending_migrations
            ]
            if not ids:
                return {"status": "error", "message": "no active requests"}
            request_id = await self._pick_most_progressed(ids)

        state = await self._tracker.get_request_state(request_id)
        if state is None:
            return {"status": "error", "message": f"unknown request_id {request_id!r}"}

        # Look up source-side block IDs **before** any abort: KVBM frees
        # blocks synchronously on abort, so the lookup must happen first.
        src_block_ids = self._block_index.lookup(request_id)
        nixl_coords = self._read_nixl_meta()

        # Phase-2.B block-hold: when connector_enabled=True, ALWAYS defer
        # the abort so the orchestrator can rollback if dst declines.
        # KV transfer path (NIXL) additionally requires KVBM + NIXL coords.
        use_block_hold = self._connector_enabled
        use_kv_transfer = (
            self._connector_enabled
            and src_block_ids is not None
            and nixl_coords is not None
        )
        if use_block_hold:
            self._pending_migrations[request_id] = time.monotonic()
            logger.info(
                "[Migration] migrate_out: holding request %s "
                "(block_hold=True, kv_transfer=%s)",
                request_id, use_kv_transfer,
            )
        else:
            await self._tracker.abort_request(request_id)

        response: dict = {
            "status": "ok",
            "request_id": request_id,
            "prompt_tokens": state["prompt_tokens"],
            "generated_tokens": state["generated_tokens"],
            "sampling_params": state["sampling_params"],
            "stop_conditions": state.get("stop_conditions", {}),
        }
        if src_block_ids is not None:
            response["src_block_ids"] = src_block_ids
        if use_kv_transfer:
            # Build the same shape vLLM 0.16's NixlConnector expects on the
            # decode side (see handlers.py:1577 for the disagg-PD use of the
            # same dict). vLLM’s NixlConnectorScheduler at
            # nixl_connector.py:add_new_req_to_recv reads exactly these
            # fields.
            response["kv_transfer_params"] = {
                "do_remote_prefill": True,
                "do_remote_decode": False,
                "remote_engine_id": nixl_coords["engine_id"],
                "remote_block_ids": list(src_block_ids),
                "remote_host": nixl_coords["host"],
                "remote_port": int(nixl_coords["port"]),
                "remote_request_id": request_id,
            }
        return response

    async def migrate_in(self, body: dict) -> dict:
        request_id = self._validate_id(body)
        if isinstance(request_id, dict):
            return request_id
        for required in ("prompt_tokens", "generated_tokens", "sampling_params"):
            if required not in body:
                return {"status": "error", "message": f"missing {required!r}"}

        # Phase-2: cost-benefit gate.
        ok, reason = self._should_migrate(body)
        if not ok:
            logger.info("[Migration] migrate_in declined for %s: %s", request_id, reason)
            return {
                "status": "declined",
                "reason": reason,
                "request_id": request_id,
            }

        replay_prompt = list(body["prompt_tokens"]) + list(body["generated_tokens"])
        kv_transfer_params = body.get("kv_transfer_params")

        # Phase-2.B: when the source provided NIXL coordinates AND the
        # connector path is locally enabled, route via vLLM's NixlConnector.
        # The connector chain (DynamoConnector + NixlConnector via
        # MultiConnector) reads sampling_params.extra_args["kv_transfer_params"]
        # and issues an async NIXL READ pull during the next scheduler step
        # (see vLLM 0.16 nixl_connector.py:start_load_kv -> _read_blocks).
        if self._connector_enabled and kv_transfer_params:
            try:
                payload = {
                    "prompt_tokens": replay_prompt,
                    "sampling_params": body["sampling_params"],
                    "stop_conditions": body.get("stop_conditions", {}),
                    "previously_emitted_tokens": list(body["generated_tokens"]),
                    "kv_transfer_params": dict(kv_transfer_params),
                    "migration_meta": {
                        "path": "connector",
                        "src_block_ids": list(
                            kv_transfer_params.get("remote_block_ids") or []
                        ),
                    },
                }
                await self._tracker.submit_request(request_id, payload)
                return {
                    "status": "ok",
                    "request_id": request_id,
                    "path": "connector",
                    "replay_tokens": len(replay_prompt),
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[Migration] connector path failed for %s (%s); "
                    "falling back to recompute-prefill",
                    request_id,
                    exc,
                )

        # Phase-2.A: recompute-prefill (the new "prompt" is prompt + generated).
        payload = {
            "prompt_tokens": replay_prompt,
            "sampling_params": body["sampling_params"],
            "stop_conditions": body.get("stop_conditions", {}),
            "previously_emitted_tokens": list(body["generated_tokens"]),
        }
        await self._tracker.submit_request(request_id, payload)
        return {
            "status": "ok",
            "request_id": request_id,
            "path": "recompute",
            "replay_tokens": len(replay_prompt),
        }

    async def migration_complete(self, body: dict) -> dict:
        """Phase-2.B ack: destination confirms KV pull is done.

        Called by the orchestrator (test script / RL controller) AFTER
        ``migrate_in`` has successfully submitted the request with
        ``kv_transfer_params``.  The source now aborts the original
        request, freeing the held KV blocks.
        """
        request_id = self._validate_id(body)
        if isinstance(request_id, dict):
            return request_id  # error
        ts = self._pending_migrations.pop(request_id, None)
        if ts is None:
            # Not tracked — might have been swept by timeout already.
            # Still try to abort in case it is live.
            logger.info(
                "[Migration] migration_complete: %s not in pending "
                "(timeout-swept or Phase-2.A); best-effort abort",
                request_id,
            )
        else:
            hold_ms = (time.monotonic() - ts) * 1000.0
            logger.info(
                "[Migration] migration_complete: releasing %s after %.1fms hold",
                request_id,
                hold_ms,
            )
        try:
            await self._tracker.abort_request(request_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[Migration] abort_request(%s) in migration_complete failed: %s",
                request_id,
                exc,
            )
        return {"status": "ok", "request_id": request_id}

    async def sweep_stale_migrations(self) -> int:
        """Force-abort migrations that have been held longer than timeout.

        Returns the number of force-aborted entries.  Designed to be called
        periodically from a background task (see sidecar sweeper).
        """
        if not self._pending_migrations:
            return 0
        now = time.monotonic()
        stale = [
            (rid, ts)
            for rid, ts in self._pending_migrations.items()
            if now - ts > self._migration_hold_timeout_s
        ]
        for rid, ts in stale:
            hold_s = now - ts
            logger.warning(
                "[Migration] timeout: force-aborting held migration %s "
                "(held %.1fs > %.1fs limit)",
                rid,
                hold_s,
                self._migration_hold_timeout_s,
            )
            self._pending_migrations.pop(rid, None)
            try:
                await self._tracker.abort_request(rid)
            except Exception:  # noqa: BLE001
                logger.debug("force-abort %s failed", rid, exc_info=True)
        return len(stale)

    async def migration_rollback(self, body: dict) -> dict:
        """Release a pending migration hold without aborting the request.

        Called when the destination declines or the coordinated migration
        fails.  The source request continues running as if ``migrate_out``
        never happened.  Only meaningful when ``connector_enabled=True``
        (block-hold mode); otherwise returns a harmless no-op.
        """
        request_id = self._validate_id(body)
        if isinstance(request_id, dict):
            return request_id  # error
        ts = self._pending_migrations.pop(request_id, None)
        if ts is None:
            return {
                "status": "ok",
                "request_id": request_id,
                "message": "not in pending (no-op)",
            }
        hold_ms = (time.monotonic() - ts) * 1000.0
        logger.info(
            "[Migration] migration_rollback: releasing hold on %s after "
            "%.1fms (request continues on source)",
            request_id,
            hold_ms,
        )
        return {"status": "ok", "request_id": request_id, "hold_ms": hold_ms}

    # ------------------------------------------------------------------ utils
    def _read_nixl_meta(self) -> Optional[dict]:
        """Best-effort fetch of the source's NIXL listener coordinates."""
        if self._nixl_meta_provider is None:
            return None
        try:
            meta = self._nixl_meta_provider()
        except Exception:  # noqa: BLE001
            logger.debug("nixl_meta_provider() raised", exc_info=True)
            return None
        if not meta:
            return None
        # Validate the shape our wire protocol promises.
        for key in ("engine_id", "host", "port"):
            if key not in meta:
                logger.warning(
                    "[Migration] nixl_meta_provider returned dict missing %r; "
                    "omitting kv_transfer_params from migrate_out response",
                    key,
                )
                return None
        return meta

    # ------------------------------------------------------------------ utils
    def _should_migrate(self, body: dict) -> tuple[bool, str]:
        """Apply the cost-benefit policy. Returns ``(ok, reason)``."""
        prompt_len = len(body.get("prompt_tokens") or [])
        gen_len = len(body.get("generated_tokens") or [])
        replay_total = prompt_len + gen_len
        policy = self._policy

        if replay_total > policy.max_replay_tokens:
            return False, (
                f"replay_total={replay_total} exceeds max_replay_tokens="
                f"{policy.max_replay_tokens} (recompute prefill too expensive)"
            )
        if gen_len < policy.min_generated_tokens:
            return False, (
                f"generated_tokens={gen_len} below min_generated_tokens="
                f"{policy.min_generated_tokens} (request too young to benefit)"
            )
        # Best-effort remaining-work estimate from sampling_params.
        sampling = body.get("sampling_params") or {}
        max_tokens = sampling.get("max_tokens")
        if isinstance(max_tokens, int) and max_tokens > 0:
            remaining = max_tokens - gen_len
            if remaining < policy.min_remaining_tokens:
                return False, (
                    f"remaining_tokens={remaining} below min_remaining_tokens="
                    f"{policy.min_remaining_tokens} (will finish faster than migrate)"
                )
        return True, "policy ok"

    async def _pick_most_progressed(self, ids: list[str]) -> str:
        """Pick the request with the most generated tokens (saves the most work)."""
        best_id = ids[0]
        best_gen = -1
        for rid in ids:
            state = await self._tracker.get_request_state(rid)
            if state is None:
                continue
            gen = len(state.get("generated_tokens") or [])
            if gen > best_gen:
                best_gen = gen
                best_id = rid
        return best_id

    def _warn_if_prefix_cache_disabled(self) -> None:
        if self._engine is None:
            return
        # vLLM 0.16: engine.vllm_config.cache_config.enable_prefix_caching
        try:
            cache_cfg = getattr(
                getattr(self._engine, "vllm_config", None), "cache_config", None
            )
            if cache_cfg is None:
                return
            enabled = getattr(cache_cfg, "enable_prefix_caching", None)
            if enabled is False:
                logger.warning(
                    "[Migration] enable_prefix_caching=False — recompute-prefill "
                    "migration will pay full prefill cost (~100ms+) instead of "
                    "the cached ~30-40ms. Strongly recommend enabling prefix cache."
                )
        except Exception:  # noqa: BLE001
            logger.debug("could not inspect engine cache config", exc_info=True)

    @staticmethod
    def _validate_id(body: dict) -> Any:
        rid = body.get("request_id") if isinstance(body, dict) else None
        if not rid or not isinstance(rid, str):
            return {"status": "error", "message": "request_id is required"}
        return rid

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""S3 — Request consolidation HTTP handlers (RL-Scaling).

Phase 2 (v3.5 · vLLM-native KV-D2D, two-tier delivery):

* **Phase-2.A (已交付 · 零风险)** — smart recompute-prefill migration with
  :class:`MigrationPolicy` cost-benefit gate + prefix-cache awareness.
  ``migrate_out`` additionally exposes ``src_block_ids`` (looked up via
  :class:`RequestBlockIndex`, which wraps the existing
  ``KvbmCacheManager.get_block_ids`` PyO3-exposed API) so a Phase-2.B
  invocation has everything it needs without changing the wire protocol.

* **Phase-2.B (已交付主路径 · 需 GPU验证上线)** — real KV-D2D via vLLM's
  built-in ``NixlConnector``. Architecture insight from fact-check (v3.5):
  vLLM 0.16's ``NixlConnector`` already implements
  ``kv_transfer_params``-driven NIXL READ pull (see vLLM 0.16
  ``nixl_connector.py:177-220, 2063-2371`` and dynamo
  ``handlers.py:1577-1589`` for the disagg-PD usage). When a worker is
  deployed with that connector in its KV-transfer chain (``DynamoConnector``
  + ``NixlConnector`` via ``MultiConnector``, the PD-disagg layout this
  cluster already uses), migrate_in just needs to inject
  ``sampling_params.extra_args["kv_transfer_params"]`` and vLLM does the
  rest — no custom KVConnector subclass required.

  ``MigrationHandler`` is constructed with ``nixl_meta_provider`` (a
  callable returning the *source* worker's NIXL coordinates, plumbed by
  ``main.py`` from the engine's ``KvTransferConfig``); ``migrate_out``
  packages those plus ``src_block_ids`` into the response’s
  ``kv_transfer_params`` field. ``migrate_in`` forwards the dict verbatim
  onto the resubmitted request. Any failure path falls back to Phase-2.A
  recompute.

  **Known correctness gap (block-hold)**: Phase-2.B currently calls
  ``abort_request`` on the source immediately, which frees the src KV
  blocks before the dst's NIXL READ has a chance to complete. In
  production this needs the same "hold blocks alive until dst confirms
  pull" mechanism vLLM disagg-PD uses (``request_finished -> (True, None)``
  + ``get_finished``). Until the hold mechanism is wired,
  Phase-2.B remains gated behind ``connector_enabled=False`` (the safe
  default), and the connector path falls back to Phase-2.A on every
  migrate_in. See ``RL_SCALING_PYTHON_CHANGES.md`` for the GPU-validation
  TODO list.

Public surface (backward-compatible — existing Phase-1 callers unchanged):

    handler = MigrationHandler(
        tracker,
        policy=MigrationPolicy(),
        engine=engine,
        block_index=RequestBlockIndex(kvbm_cache_manager),     # 可选
        nixl_meta_provider=lambda: {"engine_id": ...,           # 可选 (Phase-2.B)
                                    "host": ..., "port": ...},
        connector_enabled=False,                                # 安全默认
    )
    await handler.migrate_out({"request_id": rid})
        -> {status, request_id, prompt_tokens, generated_tokens,
            sampling_params, stop_conditions,
            src_block_ids?,        # iff KVBM block_index available
            kv_transfer_params?}   # iff connector_enabled and NIXL meta available
    await handler.migrate_in({...})
        -> {status: "ok"|"declined"|"error", path: "recompute"|"connector", ...}
"""
from __future__ import annotations

import logging
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
        # Phase-2.B is feature-flagged. Until the source-side block-hold
        # mechanism (analogous to disagg-PD's request_finished -> True
        # + get_finished pattern) is wired, immediate abort_request on the
        # source races with the dst's NIXL READ. With connector_enabled=False
        # we keep producing src_block_ids in migrate_out responses (zero risk,
        # forward-compatible) but migrate_in still goes through the safe
        # recompute path even if the wire protocol carries kv_transfer_params.
        self._connector_enabled = connector_enabled
        self._warn_if_prefix_cache_disabled()

    # ------------------------------------------------------------------ API
    async def migrate_out(self, body: dict) -> dict:
        request_id = self._validate_id(body)
        if isinstance(request_id, dict):
            return request_id  # error
        # Resolve "*" -> most-progressed active id.
        if request_id == "*":
            ids = list(await self._tracker.list_active_request_ids())
            if not ids:
                return {"status": "error", "message": "no active requests"}
            request_id = await self._pick_most_progressed(ids)

        state = await self._tracker.get_request_state(request_id)
        if state is None:
            return {"status": "error", "message": f"unknown request_id {request_id!r}"}

        # Look up source-side block IDs **before** abort: KVBM frees blocks
        # synchronously on abort, so the lookup must happen first.
        src_block_ids = self._block_index.lookup(request_id)
        nixl_coords = self._read_nixl_meta()

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
        if (
            self._connector_enabled
            and src_block_ids is not None
            and nixl_coords is not None
        ):
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

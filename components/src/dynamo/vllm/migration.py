# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""S3 — Request consolidation HTTP handlers (RL-Scaling).

Phase 2 (Option A — pragmatic): smart recompute-prefill migration.

The KV cache itself is **not** transferred between workers (true KV-D2D is
tracked separately and requires deep vLLM scheduler integration). Instead,
``migrate_in`` re-prefills ``prompt + generated`` on the destination and
continues decoding from there, paying one extra prefill in exchange for the
ability to drain a worker before scaling down.

Phase 2 improvements over the Phase-1 always-recompute path:

1. **Cost-benefit gate** — :class:`MigrationPolicy` declines migration when
   the recompute cost likely exceeds the savings. The controller can then
   leave the request on its source worker and pick a different drain target.
2. **Prefix-cache awareness** — the recompute prefill on the destination is
   expected to hit the local prefix cache (when ``enable_prefix_caching=True``)
   so the recompute cost is amortised to ~30-40 ms instead of ~100 ms.
   :class:`MigrationHandler` warns at construction time if the engine's
   prefix-cache flag is missing or False.
3. **Wildcard ``*``** — picks the *most-progressed* request rather than the
   first one, maximising the work that survives the migration.

Public surface (unchanged contract):

    handler = MigrationHandler(tracker, policy=MigrationPolicy(), engine=...)
    await handler.migrate_out({"request_id": rid}) -> serialized state
    await handler.migrate_in({"request_id": rid, ...}) -> {"status": "ok"|"declined"|"error"}
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Protocol

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
    ) -> None:
        self._tracker = tracker
        self._policy = policy or MigrationPolicy()
        self._engine = engine
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
        await self._tracker.abort_request(request_id)
        return {
            "status": "ok",
            "request_id": request_id,
            "prompt_tokens": state["prompt_tokens"],
            "generated_tokens": state["generated_tokens"],
            "sampling_params": state["sampling_params"],
            "stop_conditions": state.get("stop_conditions", {}),
        }

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

        # Recompute-prefill: the new "prompt" is prompt_tokens + generated_tokens.
        replay_prompt = list(body["prompt_tokens"]) + list(body["generated_tokens"])
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
            "replay_tokens": len(replay_prompt),
        }

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

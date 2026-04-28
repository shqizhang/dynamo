# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""S3 — Request Consolidation HTTP handlers (RL-Scaling).

Provides a thin, *recompute-prefill* fallback for moving an in-flight
request from one decode worker to another. The KV cache is **not**
transferred — the controller pays for one extra prefill on the destination
worker. Per the design doc (S3 feasibility alert) this is the only viable
path until a Rust NIXL D2D-migration primitive lands.

Public surface:

    handler = MigrationHandler(engine_client=..., tracker=...)
    await handler.migrate_out({"request_id": rid}) -> serialized state
    await handler.migrate_in({"request_id": rid, "prompt_tokens": [...],
                              "generated_tokens": [...],
                              "sampling_params": {...},
                              "stop_conditions": {...}}) -> {"status": "ok"}

The handler is intentionally engine-agnostic. The ``RequestTracker``
protocol below is what an engine integration must implement; a vLLM-specific
``VllmRequestTracker`` lives close to ``handlers.py`` and is wired in
``main.py`` when the worker is launched with ``--enable-migration``.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Iterable, Optional, Protocol

logger = logging.getLogger(__name__)


class RequestTracker(Protocol):
    """What ``MigrationHandler`` needs from the engine.

    Implementations expose live request state so we can serialise + abort
    a request, or replay one onto the engine.
    """

    async def get_request_state(self, request_id: str) -> Optional[dict]:
        """Return ``{prompt_tokens, generated_tokens, sampling_params, stop_conditions}`` or ``None``."""

    async def abort_request(self, request_id: str) -> None:
        """Cancel the in-engine request without finalising the client stream."""

    async def submit_request(self, request_id: str, payload: dict) -> None:
        """Resubmit a request to the engine using ``prompt + generated`` as the new prompt."""

    async def list_active_request_ids(self) -> Iterable[str]:
        """Used to resolve the ``"*"`` placeholder (next ready request)."""


# ---------------------------------------------------------------- handler
class MigrationHandler:
    def __init__(self, tracker: RequestTracker) -> None:
        self._tracker = tracker

    async def migrate_out(self, body: dict) -> dict:
        request_id = self._validate_id(body)
        if isinstance(request_id, dict):
            return request_id  # error
        # Resolve "*" -> first active id.
        if request_id == "*":
            ids = list(await self._tracker.list_active_request_ids())
            if not ids:
                return {"status": "error", "message": "no active requests"}
            request_id = ids[0]

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

        # Recompute-prefill: the new "prompt" is prompt_tokens + generated_tokens.
        # The engine will redo prefill on the concatenation and continue
        # generation from there.
        replay_prompt = list(body["prompt_tokens"]) + list(body["generated_tokens"])
        payload = {
            "prompt_tokens": replay_prompt,
            "sampling_params": body["sampling_params"],
            "stop_conditions": body.get("stop_conditions", {}),
            # Keep the raw generated tokens too in case the engine wants to
            # short-circuit emit them to the client without re-yielding.
            "previously_emitted_tokens": list(body["generated_tokens"]),
        }
        await self._tracker.submit_request(request_id, payload)
        return {"status": "ok", "request_id": request_id}

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _validate_id(body: dict) -> Any:
        rid = body.get("request_id") if isinstance(body, dict) else None
        if not rid or not isinstance(rid, str):
            return {"status": "error", "message": "request_id is required"}
        return rid

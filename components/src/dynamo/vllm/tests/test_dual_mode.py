# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for S2 (RL-Scaling) DualModeWorker orchestration."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from dynamo.vllm.dual_mode import DualModeWorker
from dynamo.vllm.handlers import BaseWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _StubHandler(BaseWorkerHandler):
    async def generate(self, request, context):  # pragma: no cover - unused
        yield {}


def _make_handler():
    h = _StubHandler.__new__(_StubHandler)
    h.engine_client = SimpleNamespace(
        pause_generation=AsyncMock(),
        sleep=AsyncMock(),
        wake_up=AsyncMock(),
        resume_generation=AsyncMock(),
    )
    h.generate_endpoint = SimpleNamespace(
        unregister_endpoint_instance=AsyncMock(),
        register_endpoint_instance=AsyncMock(),
    )
    h._sleep_wake_lock = asyncio.Lock()
    h._engine_is_sleeping = False
    h._disaggregation_mode = None
    return h


# ---------------------------------------------------------------- handler API
class TestSetDisaggregationMode:
    def test_accepts_known_modes(self):
        h = _make_handler()
        for mode in ("prefill", "decode", "agg"):
            h.set_disaggregation_mode(mode)
            assert h.get_disaggregation_mode() == mode

    def test_rejects_unknown_mode(self):
        h = _make_handler()
        with pytest.raises(ValueError):
            h.set_disaggregation_mode("garbage")


# ------------------------------------------------------------ DualModeWorker
class TestDualModeWorker:
    def test_rejects_invalid_initial_role(self):
        with pytest.raises(ValueError):
            DualModeWorker(_make_handler(), initial_role="weird")

    def test_initial_role_is_persisted_on_handler(self):
        h = _make_handler()
        w = DualModeWorker(h, initial_role="decode")
        assert h.get_disaggregation_mode() == "decode"
        assert w.current_role == "decode"

    @pytest.mark.asyncio
    async def test_invalid_target_role_returns_error(self):
        w = DualModeWorker(_make_handler(), initial_role="decode")
        result = await w.switch_role("garbage")
        assert result["status"] == "error"
        assert w.current_role == "decode"

    @pytest.mark.asyncio
    async def test_already_in_role_is_noop(self):
        h = _make_handler()
        w = DualModeWorker(h, initial_role="decode")
        result = await w.switch_role("decode")
        assert result["status"] == "ok"
        assert "already" in result["message"]
        h.engine_client.sleep.assert_not_awaited()
        h.engine_client.wake_up.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_full_flip_orchestration_order(self):
        h = _make_handler()
        order: list[str] = []
        # Patch sleep/wake on the handler itself so we observe the orchestration.
        original_sleep, original_wake = h.sleep, h.wake_up

        async def tracked_sleep(body):
            order.append("sleep")
            return await original_sleep(body)

        async def tracked_wake(body):
            order.append("wake")
            return await original_wake(body)

        h.sleep = tracked_sleep  # type: ignore
        h.wake_up = tracked_wake  # type: ignore

        publisher = MagicMock()
        publisher.publish_role_changed = AsyncMock()
        w = DualModeWorker(h, initial_role="decode", publisher=publisher)

        result = await w.switch_role("prefill")

        assert result["status"] == "ok"
        assert result["new_role"] == "prefill"
        assert result["switch_time_ms"] >= 0
        assert order == ["sleep", "wake"]
        assert h.get_disaggregation_mode() == "prefill"
        assert w.current_role == "prefill"
        publisher.publish_role_changed.assert_awaited_once_with(
            from_role="decode", to_role="prefill"
        )

    @pytest.mark.asyncio
    async def test_sleep_failure_rolls_back(self):
        h = _make_handler()

        async def bad_sleep(body):
            return {"status": "error", "message": "boom"}

        h.sleep = bad_sleep  # type: ignore
        w = DualModeWorker(h, initial_role="decode")
        result = await w.switch_role("prefill")
        assert result["status"] == "error"
        assert w.current_role == "decode"  # unchanged
        assert h.get_disaggregation_mode() == "decode"

    @pytest.mark.asyncio
    async def test_wake_failure_recovers_to_previous_role(self):
        h = _make_handler()

        async def bad_wake(body):
            return {"status": "error", "message": "wake fail"}

        # Override only the FIRST wake_up call (the orchestrated one) — the
        # recovery path also calls wake_up; let it succeed via the original.
        original_wake = h.wake_up
        call_count = {"n": 0}

        async def wake_router(body):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return await bad_wake(body)
            return await original_wake(body)

        h.wake_up = wake_router  # type: ignore

        w = DualModeWorker(h, initial_role="decode")
        result = await w.switch_role("prefill")

        assert result["status"] == "error"
        # current_role NOT updated because wake failed
        assert w.current_role == "decode"
        # handler role rolled back during recovery
        assert h.get_disaggregation_mode() == "decode"
        assert call_count["n"] == 2  # one orchestrated, one recovery

    @pytest.mark.asyncio
    async def test_no_publisher_does_not_crash(self):
        h = _make_handler()
        w = DualModeWorker(h, initial_role="decode", publisher=None)
        result = await w.switch_role("prefill")
        assert result["status"] == "ok"

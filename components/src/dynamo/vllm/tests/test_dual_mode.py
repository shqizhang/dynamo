# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for S2 (RL-Scaling) DualModeWorker orchestration."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from dynamo.vllm.dual_mode import DualModeWorker

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

# Avoid importing dynamo.vllm.handlers (which pulls in torch) at unit-test time.
# Reproduce the small surface DualModeWorker uses from BaseWorkerHandler.
_VALID_MODES = {"prefill", "decode", "agg"}


class _StubHandler:
    async def generate(self, request, context):  # pragma: no cover - unused
        yield {}

    async def sleep(self, body):
        async with self._sleep_wake_lock:
            if self._engine_is_sleeping:
                return {"status": "ok", "message": "Engine already sleeping"}
            await self.engine_client.sleep((body or {}).get("level", 1))
            self._engine_is_sleeping = True
            return {"status": "ok", "message": "Engine slept"}

    async def wake_up(self, body):
        async with self._sleep_wake_lock:
            if not self._engine_is_sleeping:
                return {"status": "ok", "message": "Engine already awake"}
            await self.engine_client.wake_up()
            self._engine_is_sleeping = False
            return {"status": "ok", "message": "Engine woke"}

    def set_disaggregation_mode(self, mode):
        if mode not in _VALID_MODES:
            raise ValueError(f"unknown disaggregation mode: {mode!r}")
        self._disaggregation_mode = mode

    def get_disaggregation_mode(self):
        return self._disaggregation_mode


def _make_handler():
    h = _StubHandler.__new__(_StubHandler)
    h.engine_client = SimpleNamespace(
        pause_generation=AsyncMock(),
        sleep=AsyncMock(),
        wake_up=AsyncMock(),
        resume_generation=AsyncMock(),
        reset_prefix_cache=AsyncMock(),
    )
    h.generate_endpoint = SimpleNamespace(
        unregister_endpoint_instance=AsyncMock(),
        register_endpoint_instance=AsyncMock(),
        update_metadata=AsyncMock(),
    )
    h._sleep_wake_lock = asyncio.Lock()
    h._engine_is_sleeping = False
    h._disaggregation_mode = None
    h._nixl_connector = None
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


class _StubRegistry:
    """active_ids() returns non-empty for the first `busy_calls` calls, then []."""
    def __init__(self, busy_calls=0):
        self._busy = busy_calls
    def active_ids(self):
        if self._busy > 0:
            self._busy -= 1
            return ["r1"]
        return []
    def deregister(self, rid):
        pass


class TestQuiesce:
    @pytest.mark.asyncio
    async def test_quiesce_returns_when_idle(self):
        h = _make_handler()
        h.request_registry = _StubRegistry(busy_calls=0)
        w = DualModeWorker(h, initial_role="decode")
        res = await w._drain_and_quiesce(stable_s=0.05, timeout_s=5)
        assert res["quiesced"] is True
        assert res["arrivals_after_cordon"] == 0

    @pytest.mark.asyncio
    async def test_quiesce_detects_late_arrival_then_settles(self):
        h = _make_handler()
        # active_ids reports a late arrival on an early stable-window poll, then
        # goes idle -> the loop re-drains and eventually quiesces.
        h.request_registry = _StubRegistry(busy_calls=2)
        w = DualModeWorker(h, initial_role="decode")
        res = await w._drain_and_quiesce(stable_s=0.05, timeout_s=5)
        assert res["quiesced"] is True
        assert res["arrivals_after_cordon"] >= 1

    @pytest.mark.asyncio
    async def test_quiesce_stable_zero_single_drain(self):
        h = _make_handler()
        h.request_registry = _StubRegistry(busy_calls=0)
        w = DualModeWorker(h, initial_role="decode")
        res = await w._drain_and_quiesce(stable_s=0.0, timeout_s=5)
        assert res["quiesced"] is True


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

    @pytest.mark.asyncio
    async def test_reconfig_kv_pool_calls_reset_prefix_cache(self):
        h = _make_handler()
        w = DualModeWorker(h, initial_role="decode")
        await w.switch_role("prefill")
        # Awaited at least twice: once as the outbound-KV-drain pinned-block
        # probe (pre-sleep) and once by reconfig_kv_pool (post-sleep).
        assert h.engine_client.reset_prefix_cache.await_count >= 2

    @pytest.mark.asyncio
    async def test_reconfig_kv_pool_tolerates_missing_api(self):
        h = _make_handler()
        # Drop the reset method to simulate an older engine.
        h.engine_client = SimpleNamespace(
            pause_generation=AsyncMock(),
            sleep=AsyncMock(),
            wake_up=AsyncMock(),
            resume_generation=AsyncMock(),
        )
        w = DualModeWorker(h, initial_role="decode")
        result = await w.switch_role("prefill")
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_reconfig_nixl_drops_cached_connector(self):
        h = _make_handler()
        cached = MagicMock()
        cached.shutdown = AsyncMock()
        h._nixl_connector = cached
        w = DualModeWorker(h, initial_role="decode")
        await w.switch_role("prefill")
        assert h._nixl_connector is None
        cached.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_emit_role_changed_pushes_metadata(self):
        h = _make_handler()
        w = DualModeWorker(h, initial_role="decode")
        await w.switch_role("prefill")
        h.generate_endpoint.update_metadata.assert_awaited_once_with(
            {"disaggregation_mode": "prefill"}
        )

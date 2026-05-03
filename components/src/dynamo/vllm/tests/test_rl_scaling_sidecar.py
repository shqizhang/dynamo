# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the in-process HTTP sidecar.

Covers:
  * InProcessRequestRegistry register/record_tokens/deregister/active_ids.
  * EngineRequestTracker get_request_state / abort_request / submit_request
    / list_active_request_ids using a fake AsyncLLM-like engine.
  * aiohttp app routes return correct shapes (uses aiohttp's TestClient).
  * make_nixl_meta_provider extracts coords and gracefully degrades.
"""
from __future__ import annotations

import asyncio
import importlib
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from dynamo.vllm.rl_scaling_sidecar import (
    EngineRequestTracker,
    InProcessRequestRegistry,
    make_nixl_meta_provider,
)


# --------------------------------------------------------------- registry
class TestRegistry:
    def test_register_and_get(self):
        reg = InProcessRequestRegistry()
        reg.register("r1", [1, 2, 3], {"temperature": 0.7}, {"max_tokens": 100})
        snap = reg.get("r1")
        assert snap.prompt_tokens == [1, 2, 3]
        assert snap.sampling_params_dict == {"temperature": 0.7}
        assert snap.stop_conditions == {"max_tokens": 100}
        assert snap.generated_tokens == []

    def test_record_tokens_appends(self):
        reg = InProcessRequestRegistry()
        reg.register("r1", [1], {})
        reg.record_tokens("r1", [4, 5])
        reg.record_tokens("r1", [6])
        assert reg.get("r1").generated_tokens == [4, 5, 6]

    def test_record_tokens_unknown_id_is_noop(self):
        reg = InProcessRequestRegistry()
        reg.record_tokens("ghost", [1, 2])  # must not raise

    def test_deregister(self):
        reg = InProcessRequestRegistry()
        reg.register("r1", [1], {})
        reg.deregister("r1")
        assert reg.get("r1") is None

    def test_active_ids(self):
        reg = InProcessRequestRegistry()
        for rid in ("a", "b", "c"):
            reg.register(rid, [1], {})
        assert sorted(reg.active_ids()) == ["a", "b", "c"]


# ---------------------------------------------------------------- tracker
class _FakeEngine:
    def __init__(self) -> None:
        self.aborted: list[str] = []

    async def abort(self, rid: str) -> None:
        self.aborted.append(rid)


@pytest.mark.asyncio
class TestTracker:
    async def test_get_request_state_returns_snapshot(self):
        reg = InProcessRequestRegistry()
        reg.register("r1", [1, 2], {"temperature": 0.5}, {})
        reg.record_tokens("r1", [9])
        engine = _FakeEngine()

        async def submit(rid, payload):
            pass

        t = EngineRequestTracker(engine, reg, submit)
        state = await t.get_request_state("r1")
        assert state["prompt_tokens"] == [1, 2]
        assert state["generated_tokens"] == [9]
        assert state["sampling_params"] == {"temperature": 0.5}

    async def test_get_request_state_unknown_returns_none(self):
        t = EngineRequestTracker(_FakeEngine(), InProcessRequestRegistry(), lambda r, p: None)
        assert await t.get_request_state("ghost") is None

    async def test_abort_calls_engine_and_deregisters(self):
        reg = InProcessRequestRegistry()
        reg.register("r1", [1], {})
        engine = _FakeEngine()
        t = EngineRequestTracker(engine, reg, lambda r, p: None)
        await t.abort_request("r1")
        assert engine.aborted == ["r1"]
        assert reg.get("r1") is None

    async def test_abort_swallows_engine_exception(self):
        reg = InProcessRequestRegistry()
        reg.register("r1", [1], {})

        class _Boom:
            async def abort(self, rid):
                raise RuntimeError("simulated")

        t = EngineRequestTracker(_Boom(), reg, lambda r, p: None)
        await t.abort_request("r1")  # must not raise
        assert reg.get("r1") is None  # still cleaned up

    async def test_submit_delegates_to_callback(self):
        seen: list = []

        async def submit(rid, payload):
            seen.append((rid, payload))

        t = EngineRequestTracker(_FakeEngine(), InProcessRequestRegistry(), submit)
        await t.submit_request("r1", {"foo": "bar"})
        assert seen == [("r1", {"foo": "bar"})]

    async def test_list_active(self):
        reg = InProcessRequestRegistry()
        reg.register("a", [1], {})
        reg.register("b", [2], {})
        t = EngineRequestTracker(_FakeEngine(), reg, lambda r, p: None)
        assert sorted(await t.list_active_request_ids()) == ["a", "b"]


# ----------------------------------------------------------- nixl provider
class TestNixlMetaProvider:
    def test_full_coords_returned(self):
        cfg = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                engine_id="eid",
                nixl_side_channel_host="10.0.0.1",
                nixl_side_channel_port=5557,
            )
        )
        provider = make_nixl_meta_provider(cfg)
        assert provider() == {"engine_id": "eid", "host": "10.0.0.1", "port": 5557}

    def test_returns_none_when_kv_transfer_config_missing(self):
        provider = make_nixl_meta_provider(SimpleNamespace(kv_transfer_config=None))
        assert provider() is None

    def test_returns_none_when_partial(self):
        cfg = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                engine_id="eid",
                nixl_side_channel_host=None,
                nixl_side_channel_port=5557,
            )
        )
        assert make_nixl_meta_provider(cfg)() is None

    def test_returns_none_when_attribute_missing(self):
        # vllm_config that lacks the attr entirely
        provider = make_nixl_meta_provider(SimpleNamespace())
        assert provider() is None


# --------------------------------------------------------------- HTTP app
@pytest.fixture
def aiohttp_available():
    try:
        importlib.import_module("aiohttp")
    except ImportError:
        pytest.skip("aiohttp not installed in test env")


class _FakeDualMode:
    def __init__(self):
        self.current_role = "decode"
        self.calls: list[str] = []

    async def switch_role(self, target):
        self.calls.append(target)
        if target not in ("decode", "prefill"):
            return {"status": "error", "message": "bad role"}
        self.current_role = target
        return {"status": "ok", "new_role": target, "switch_time_ms": 1.0}


class _FakeMigration:
    def __init__(self):
        self.out_calls = []
        self.in_calls = []

    async def migrate_out(self, body):
        self.out_calls.append(body)
        return {"status": "ok", "request_id": body.get("request_id"),
                "prompt_tokens": [1], "generated_tokens": [], "sampling_params": {}}

    async def migrate_in(self, body):
        self.in_calls.append(body)
        return {"status": "ok", "request_id": body.get("request_id"), "path": "recompute"}


@pytest.mark.asyncio
class TestSidecarApp:
    async def test_healthz(self, aiohttp_available):
        from aiohttp.test_utils import TestClient, TestServer
        from dynamo.vllm.rl_scaling_sidecar import build_app

        app = build_app()
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/healthz")
            assert resp.status == 200
            assert (await resp.json())["status"] == "ok"

    async def test_get_role_with_dual_mode(self, aiohttp_available):
        from aiohttp.test_utils import TestClient, TestServer
        from dynamo.vllm.rl_scaling_sidecar import build_app

        dm = _FakeDualMode()
        async with TestClient(TestServer(build_app(dual_mode_worker=dm))) as cli:
            resp = await cli.get("/v1/role")
            assert (await resp.json())["current_role"] == "decode"

    async def test_switch_role(self, aiohttp_available):
        from aiohttp.test_utils import TestClient, TestServer
        from dynamo.vllm.rl_scaling_sidecar import build_app

        dm = _FakeDualMode()
        async with TestClient(TestServer(build_app(dual_mode_worker=dm))) as cli:
            resp = await cli.post("/switch_role", json={"target_role": "prefill"})
            assert resp.status == 200
            body = await resp.json()
            assert body["status"] == "ok"
            assert body["new_role"] == "prefill"
            assert dm.calls == ["prefill"]

    async def test_switch_role_503_when_disabled(self, aiohttp_available):
        from aiohttp.test_utils import TestClient, TestServer
        from dynamo.vllm.rl_scaling_sidecar import build_app

        async with TestClient(TestServer(build_app())) as cli:
            resp = await cli.post("/switch_role", json={"target_role": "prefill"})
            assert resp.status == 503

    async def test_migrate_out(self, aiohttp_available):
        from aiohttp.test_utils import TestClient, TestServer
        from dynamo.vllm.rl_scaling_sidecar import build_app

        m = _FakeMigration()
        async with TestClient(TestServer(build_app(migration_handler=m))) as cli:
            resp = await cli.post("/migrate_out", json={"request_id": "r1"})
            body = await resp.json()
            assert body["status"] == "ok"
            assert body["request_id"] == "r1"
            assert m.out_calls == [{"request_id": "r1"}]

    async def test_migrate_in(self, aiohttp_available):
        from aiohttp.test_utils import TestClient, TestServer
        from dynamo.vllm.rl_scaling_sidecar import build_app

        m = _FakeMigration()
        async with TestClient(TestServer(build_app(migration_handler=m))) as cli:
            resp = await cli.post("/migrate_in", json={
                "request_id": "r1",
                "prompt_tokens": [1, 2],
                "generated_tokens": [3],
                "sampling_params": {},
            })
            body = await resp.json()
            assert body["status"] == "ok"
            assert body["path"] == "recompute"
            assert m.in_calls and m.in_calls[0]["request_id"] == "r1"

    async def test_migrate_503_when_disabled(self, aiohttp_available):
        from aiohttp.test_utils import TestClient, TestServer
        from dynamo.vllm.rl_scaling_sidecar import build_app

        async with TestClient(TestServer(build_app())) as cli:
            assert (await cli.post("/migrate_out", json={"request_id": "x"})).status == 503
            assert (await cli.post("/migrate_in", json={})).status == 503

    async def test_active_requests(self, aiohttp_available):
        from aiohttp.test_utils import TestClient, TestServer
        from dynamo.vllm.rl_scaling_sidecar import build_app

        reg = InProcessRequestRegistry()
        reg.register("a", [1], {})
        reg.register("b", [2], {})
        async with TestClient(TestServer(build_app(registry=reg))) as cli:
            assert sorted(await (await cli.get("/v1/active_requests")).json()) == ["a", "b"]

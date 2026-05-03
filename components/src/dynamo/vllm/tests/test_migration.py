# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for S3 (RL-Scaling) MigrationHandler with recompute-prefill fallback."""
from __future__ import annotations

import pytest

from dynamo.vllm.migration import MigrationHandler, MigrationPolicy

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class FakeTracker:
    def __init__(self, store: dict | None = None) -> None:
        self.store: dict = store or {}
        self.aborted: list = []
        self.submitted: list = []

    async def get_request_state(self, request_id: str):
        return self.store.get(request_id)

    async def abort_request(self, request_id: str) -> None:
        self.aborted.append(request_id)

    async def submit_request(self, request_id: str, payload: dict) -> None:
        self.submitted.append((request_id, payload))

    async def list_active_request_ids(self):
        return list(self.store.keys())


@pytest.fixture
def tracker():
    return FakeTracker(
        store={
            "r1": {
                "prompt_tokens": [1, 2, 3],
                "generated_tokens": [10, 11, 12],
                "sampling_params": {"temperature": 0.7, "max_tokens": 100},
                "stop_conditions": {"stop": ["\n\n"]},
            }
        }
    )


# ---------------------------------------------------------------- migrate_out
class TestMigrateOut:
    @pytest.mark.asyncio
    async def test_returns_serialised_state(self, tracker):
        h = MigrationHandler(tracker)
        out = await h.migrate_out({"request_id": "r1"})
        assert out["status"] == "ok"
        assert out["prompt_tokens"] == [1, 2, 3]
        assert out["generated_tokens"] == [10, 11, 12]
        assert out["sampling_params"]["temperature"] == 0.7
        assert tracker.aborted == ["r1"]

    @pytest.mark.asyncio
    async def test_unknown_id_returns_error(self, tracker):
        h = MigrationHandler(tracker)
        out = await h.migrate_out({"request_id": "missing"})
        assert out["status"] == "error"
        assert tracker.aborted == []

    @pytest.mark.asyncio
    async def test_missing_id_returns_error(self, tracker):
        h = MigrationHandler(tracker)
        out = await h.migrate_out({})
        assert out["status"] == "error"

    @pytest.mark.asyncio
    async def test_wildcard_picks_first_active(self, tracker):
        h = MigrationHandler(tracker)
        out = await h.migrate_out({"request_id": "*"})
        assert out["status"] == "ok"
        assert out["request_id"] == "r1"
        assert tracker.aborted == ["r1"]

    @pytest.mark.asyncio
    async def test_wildcard_no_active(self):
        tracker = FakeTracker(store={})
        h = MigrationHandler(tracker)
        out = await h.migrate_out({"request_id": "*"})
        assert out["status"] == "error"


# ---------------------------------------------------------------- migrate_in
class TestMigrateIn:
    @pytest.mark.asyncio
    async def test_recompute_prefill_concatenates_tokens(self):
        tr = FakeTracker()
        h = MigrationHandler(tr, policy=MigrationPolicy(min_generated_tokens=0, min_remaining_tokens=0))
        body = {
            "request_id": "r1",
            "prompt_tokens": [1, 2, 3],
            "generated_tokens": [10, 11, 12],
            "sampling_params": {"temperature": 0.7},
        }
        out = await h.migrate_in(body)
        assert out["status"] == "ok"
        assert len(tr.submitted) == 1
        rid, payload = tr.submitted[0]
        assert rid == "r1"
        assert payload["prompt_tokens"] == [1, 2, 3, 10, 11, 12]
        assert payload["sampling_params"]["temperature"] == 0.7
        assert payload["previously_emitted_tokens"] == [10, 11, 12]

    @pytest.mark.asyncio
    async def test_missing_required_field(self):
        tr = FakeTracker()
        h = MigrationHandler(tr)
        out = await h.migrate_in({"request_id": "r1", "prompt_tokens": [1]})
        assert out["status"] == "error"
        assert tr.submitted == []

    @pytest.mark.asyncio
    async def test_missing_request_id(self):
        tr = FakeTracker()
        h = MigrationHandler(tr)
        out = await h.migrate_in({
            "prompt_tokens": [1], "generated_tokens": [], "sampling_params": {},
        })
        assert out["status"] == "error"
        assert tr.submitted == []


# ---------------------------------------------------------------- end-to-end
class TestRoundtrip:
    @pytest.mark.asyncio
    async def test_out_then_in_preserves_logical_state(self, tracker):
        src = MigrationHandler(tracker)
        permissive = MigrationPolicy(min_generated_tokens=0, min_remaining_tokens=0)
        dst_tracker = FakeTracker()
        dst = MigrationHandler(dst_tracker, policy=permissive)

        out = await src.migrate_out({"request_id": "r1"})
        assert out["status"] == "ok"
        in_resp = await dst.migrate_in(out)
        assert in_resp["status"] == "ok"
        assert dst_tracker.submitted[0][1]["prompt_tokens"] == [1, 2, 3, 10, 11, 12]


# ---------------------------------------------------------------- policy
class TestMigrationPolicy:
    @pytest.mark.asyncio
    async def test_declines_when_replay_too_large(self):
        tr = FakeTracker()
        h = MigrationHandler(tr, policy=MigrationPolicy(max_replay_tokens=10))
        out = await h.migrate_in({
            "request_id": "r1",
            "prompt_tokens": list(range(100)),
            "generated_tokens": list(range(50)),
            "sampling_params": {"max_tokens": 200},
        })
        assert out["status"] == "declined"
        assert "max_replay_tokens" in out["reason"]
        assert tr.submitted == []

    @pytest.mark.asyncio
    async def test_declines_when_too_young(self):
        tr = FakeTracker()
        h = MigrationHandler(tr, policy=MigrationPolicy(min_generated_tokens=20))
        out = await h.migrate_in({
            "request_id": "r1",
            "prompt_tokens": [1, 2, 3],
            "generated_tokens": [10, 11],
            "sampling_params": {"max_tokens": 200},
        })
        assert out["status"] == "declined"
        assert "min_generated_tokens" in out["reason"]

    @pytest.mark.asyncio
    async def test_declines_when_almost_done(self):
        tr = FakeTracker()
        h = MigrationHandler(tr, policy=MigrationPolicy(min_remaining_tokens=50))
        out = await h.migrate_in({
            "request_id": "r1",
            "prompt_tokens": [1, 2, 3],
            "generated_tokens": list(range(180)),
            "sampling_params": {"max_tokens": 200},
        })
        assert out["status"] == "declined"
        assert "min_remaining_tokens" in out["reason"]

    @pytest.mark.asyncio
    async def test_accepts_when_within_policy(self):
        tr = FakeTracker()
        h = MigrationHandler(tr)  # default policy
        out = await h.migrate_in({
            "request_id": "r1",
            "prompt_tokens": list(range(100)),
            "generated_tokens": list(range(50)),
            "sampling_params": {"max_tokens": 500},
        })
        assert out["status"] == "ok"
        assert out["replay_tokens"] == 150


class TestWildcardMostProgressed:
    @pytest.mark.asyncio
    async def test_wildcard_picks_most_generated(self):
        store = {
            "young": {
                "prompt_tokens": [1],
                "generated_tokens": [10],
                "sampling_params": {},
                "stop_conditions": {},
            },
            "old": {
                "prompt_tokens": [1],
                "generated_tokens": list(range(50)),
                "sampling_params": {},
                "stop_conditions": {},
            },
        }
        tracker = FakeTracker(store=store)
        h = MigrationHandler(tracker)
        out = await h.migrate_out({"request_id": "*"})
        assert out["status"] == "ok"
        assert out["request_id"] == "old"
        assert tracker.aborted == ["old"]


class TestPrefixCacheWarning:
    def test_warns_when_prefix_cache_disabled(self, caplog):
        from types import SimpleNamespace
        engine = SimpleNamespace(
            vllm_config=SimpleNamespace(
                cache_config=SimpleNamespace(enable_prefix_caching=False)
            )
        )
        with caplog.at_level("WARNING"):
            MigrationHandler(FakeTracker(), engine=engine)
        assert any(
            "enable_prefix_caching=False" in rec.message for rec in caplog.records
        )

    def test_no_warning_when_prefix_cache_enabled(self, caplog):
        from types import SimpleNamespace
        engine = SimpleNamespace(
            vllm_config=SimpleNamespace(
                cache_config=SimpleNamespace(enable_prefix_caching=True)
            )
        )
        with caplog.at_level("WARNING"):
            MigrationHandler(FakeTracker(), engine=engine)
        assert not any(
            "enable_prefix_caching" in rec.message for rec in caplog.records
        )


# ============================================================================
# Phase 2.B (v3) — RequestBlockIndex + connector path coverage
# ============================================================================
from dynamo.vllm.migration import RequestBlockIndex


class _FakeKvbm:
    """Mimics ``KvbmCacheManager.get_block_ids`` shape ``list[list[int]]``."""

    def __init__(self, mapping: dict[str, list[int]]) -> None:
        self.mapping = mapping
        self.calls: list[str] = []

    def get_block_ids(self, request_id: str) -> list[list[int]]:
        self.calls.append(request_id)
        if request_id not in self.mapping:
            raise KeyError(request_id)
        return [self.mapping[request_id]]


class TestRequestBlockIndex:
    def test_returns_none_when_kvbm_missing(self):
        idx = RequestBlockIndex(None)
        assert idx.lookup("r1") is None

    def test_flattens_grouped_block_ids(self):
        idx = RequestBlockIndex(_FakeKvbm({"r1": [7, 8, 9, 10]}))
        assert idx.lookup("r1") == [7, 8, 9, 10]

    def test_returns_none_when_kvbm_raises(self):
        idx = RequestBlockIndex(_FakeKvbm({}))  # any lookup raises KeyError
        assert idx.lookup("r1") is None

    def test_returns_none_for_empty_groups(self):
        class _Empty:
            def get_block_ids(self, rid):
                return []
        assert RequestBlockIndex(_Empty()).lookup("r1") is None


class TestMigrateOutWithBlockIndex:
    @pytest.mark.asyncio
    async def test_migrate_out_includes_src_block_ids(self, tracker):
        idx = RequestBlockIndex(_FakeKvbm({"r1": [7, 8, 9, 10]}))
        h = MigrationHandler(tracker, block_index=idx)
        out = await h.migrate_out({"request_id": "r1"})
        assert out["status"] == "ok"
        assert out["src_block_ids"] == [7, 8, 9, 10]

    @pytest.mark.asyncio
    async def test_migrate_out_omits_field_when_unknown(self, tracker):
        idx = RequestBlockIndex(_FakeKvbm({}))
        h = MigrationHandler(tracker, block_index=idx)
        out = await h.migrate_out({"request_id": "r1"})
        assert out["status"] == "ok"
        assert "src_block_ids" not in out  # graceful absence

    @pytest.mark.asyncio
    async def test_lookup_happens_before_abort(self, tracker):
        """Abort frees blocks in KVBM; lookup must run first."""
        kvbm = _FakeKvbm({"r1": [7, 8]})
        h = MigrationHandler(tracker, block_index=RequestBlockIndex(kvbm))
        out = await h.migrate_out({"request_id": "r1"})
        assert kvbm.calls == ["r1"]
        assert tracker.aborted == ["r1"]
        assert out["src_block_ids"] == [7, 8]


class _FakeNixlMeta:
    def __init__(self, meta: dict | None) -> None:
        self.meta = meta
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.meta


class TestMigrateOutWithNixlMeta:
    @pytest.mark.asyncio
    async def test_migrate_out_includes_kv_transfer_params(self, tracker):
        idx = RequestBlockIndex(_FakeKvbm({"r1": [7, 8, 9, 10]}))
        provider = _FakeNixlMeta({"engine_id": "src-eid", "host": "10.0.0.1", "port": 5557})
        h = MigrationHandler(
            tracker,
            block_index=idx,
            nixl_meta_provider=provider,
            connector_enabled=True,
        )
        out = await h.migrate_out({"request_id": "r1"})
        assert out["status"] == "ok"
        params = out["kv_transfer_params"]
        assert params["do_remote_prefill"] is True
        assert params["do_remote_decode"] is False
        assert params["remote_engine_id"] == "src-eid"
        assert params["remote_block_ids"] == [7, 8, 9, 10]
        assert params["remote_host"] == "10.0.0.1"
        assert params["remote_port"] == 5557
        assert params["remote_request_id"] == "r1"

    @pytest.mark.asyncio
    async def test_kv_params_omitted_when_connector_disabled(self, tracker):
        idx = RequestBlockIndex(_FakeKvbm({"r1": [7, 8]}))
        provider = _FakeNixlMeta({"engine_id": "src", "host": "h", "port": 1})
        h = MigrationHandler(
            tracker,
            block_index=idx,
            nixl_meta_provider=provider,
            connector_enabled=False,  # <<< default safe path
        )
        out = await h.migrate_out({"request_id": "r1"})
        assert "kv_transfer_params" not in out
        # src_block_ids still surfaced for forward-compatibility
        assert out["src_block_ids"] == [7, 8]

    @pytest.mark.asyncio
    async def test_kv_params_omitted_when_meta_missing_field(self, tracker):
        idx = RequestBlockIndex(_FakeKvbm({"r1": [7]}))
        provider = _FakeNixlMeta({"engine_id": "src"})  # missing host/port
        h = MigrationHandler(
            tracker, block_index=idx, nixl_meta_provider=provider, connector_enabled=True
        )
        out = await h.migrate_out({"request_id": "r1"})
        assert "kv_transfer_params" not in out

    @pytest.mark.asyncio
    async def test_kv_params_omitted_when_provider_raises(self, tracker):
        def boom():
            raise RuntimeError("nixl coords not ready")
        h = MigrationHandler(
            tracker,
            block_index=RequestBlockIndex(_FakeKvbm({"r1": [7]})),
            nixl_meta_provider=boom,
            connector_enabled=True,
        )
        out = await h.migrate_out({"request_id": "r1"})
        assert "kv_transfer_params" not in out


class TestMigrateInConnectorPath:
    @pytest.mark.asyncio
    async def test_migrate_in_uses_connector_when_kv_transfer_params_present(self):
        tr = FakeTracker()
        permissive = MigrationPolicy(min_generated_tokens=0, min_remaining_tokens=0)
        h = MigrationHandler(tr, policy=permissive, connector_enabled=True)
        kv_params = {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "remote_engine_id": "src-eid",
            "remote_block_ids": [7, 8],
            "remote_host": "10.0.0.1",
            "remote_port": 5557,
            "remote_request_id": "r1",
        }
        out = await h.migrate_in({
            "request_id": "r1",
            "prompt_tokens": [1, 2, 3],
            "generated_tokens": [10, 11, 12],
            "sampling_params": {"temperature": 0.7},
            "kv_transfer_params": kv_params,
        })
        assert out["status"] == "ok"
        assert out["path"] == "connector"
        rid, payload = tr.submitted[0]
        assert payload["kv_transfer_params"] == kv_params
        assert payload["migration_meta"]["path"] == "connector"

    @pytest.mark.asyncio
    async def test_migrate_in_uses_recompute_when_connector_disabled_at_dst(self):
        tr = FakeTracker()
        permissive = MigrationPolicy(min_generated_tokens=0, min_remaining_tokens=0)
        # dst worker has connector disabled (e.g. it's not a NIXL-enabled deploy)
        h = MigrationHandler(tr, policy=permissive, connector_enabled=False)
        out = await h.migrate_in({
            "request_id": "r1",
            "prompt_tokens": [1, 2, 3],
            "generated_tokens": [10, 11, 12],
            "sampling_params": {},
            "kv_transfer_params": {"remote_engine_id": "x"},  # ignored
        })
        assert out["path"] == "recompute"
        rid, payload = tr.submitted[0]
        assert "kv_transfer_params" not in payload

    @pytest.mark.asyncio
    async def test_migrate_in_recompute_when_no_kv_transfer_params(self):
        tr = FakeTracker()
        permissive = MigrationPolicy(min_generated_tokens=0, min_remaining_tokens=0)
        h = MigrationHandler(tr, policy=permissive, connector_enabled=True)
        out = await h.migrate_in({
            "request_id": "r1",
            "prompt_tokens": [1, 2, 3],
            "generated_tokens": [10, 11, 12],
            "sampling_params": {},
        })
        assert out["path"] == "recompute"

    @pytest.mark.asyncio
    async def test_migrate_in_falls_back_when_submit_raises_in_connector_path(self, caplog):
        class _Boom(FakeTracker):
            def __init__(self):
                super().__init__()
                self._first = True
            async def submit_request(self, rid, payload):
                if self._first and "kv_transfer_params" in payload:
                    self._first = False
                    raise RuntimeError("injected failure")
                self.submitted.append((rid, payload))
        tr = _Boom()
        permissive = MigrationPolicy(min_generated_tokens=0, min_remaining_tokens=0)
        h = MigrationHandler(tr, policy=permissive, connector_enabled=True)
        with caplog.at_level("WARNING"):
            out = await h.migrate_in({
                "request_id": "r1",
                "prompt_tokens": [1, 2, 3],
                "generated_tokens": [10, 11, 12],
                "sampling_params": {},
                "kv_transfer_params": {"remote_engine_id": "x", "remote_block_ids": [1]},
            })
        assert out["path"] == "recompute"
        # Fallback submit landed and did NOT carry kv_transfer_params.
        assert tr.submitted and "kv_transfer_params" not in tr.submitted[0][1]
        assert any("connector path failed" in r.message for r in caplog.records)

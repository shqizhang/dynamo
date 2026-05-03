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

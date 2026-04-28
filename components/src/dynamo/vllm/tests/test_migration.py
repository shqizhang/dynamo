# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for S3 (RL-Scaling) MigrationHandler with recompute-prefill fallback."""
from __future__ import annotations

import pytest

from dynamo.vllm.migration import MigrationHandler

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
        h = MigrationHandler(tr)
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
        dst_tracker = FakeTracker()
        dst = MigrationHandler(dst_tracker)

        out = await src.migrate_out({"request_id": "r1"})
        assert out["status"] == "ok"
        in_resp = await dst.migrate_in(out)
        assert in_resp["status"] == "ok"
        assert dst_tracker.submitted[0][1]["prompt_tokens"] == [1, 2, 3, 10, 11, 12]

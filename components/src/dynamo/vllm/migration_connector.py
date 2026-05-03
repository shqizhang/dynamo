# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""S3 Phase-2.B — RL-Scaling KV-D2D migration KVConnector (skeleton).

This connector is the destination-side hook that lets a migrated request skip
prefill on the destination engine and instead pull its KV cache directly from
the source worker over NIXL.

The vLLM 0.16 ``KVConnectorBase_V1`` contract used here:

* **Scheduler/leader side**:
    - :meth:`get_num_new_matched_tokens(request, n_computed) -> (k, async)`
      Returning ``(k, True)`` tells the vLLM scheduler "the first ``k`` tokens
      will be loaded externally; treat them as computed and dispatch as
      decode-only at position ``k``."
    - :meth:`update_state_after_alloc(request, blocks, n_external)`
      Receives the freshly-allocated destination block IDs we will NIXL-pull
      into.
    - :meth:`build_connector_meta(scheduler_output) -> KVConnectorMetadata`
      Packs ``{request_id, src_block_ids, dst_block_ids, src_handshake}`` so
      the worker side can act on it.

* **Worker side**:
    - :meth:`bind_connector_metadata` — receives the meta from the leader.
    - :meth:`start_load_kv(forward_context, **kwargs)` — issues the actual
      NIXL pull. Called *before* the forward pass; loading is async and the
      attention layer waits via :meth:`wait_for_layer_load`.
    - :meth:`wait_for_layer_load`, :meth:`save_kv_layer`, :meth:`wait_for_save`
      — required no-ops for migration (we never save during decode-only
      replay; saving is the source side's responsibility).

NIXL invocation in :meth:`start_load_kv` is intentionally **left as a TODO**
guarded by a feature flag — it requires GPU validation against a live
NixlConnector pair and cannot be exercised in unit tests. Until validated,
the connector returns ``(0, False)`` from :meth:`get_num_new_matched_tokens`,
which makes :class:`~dynamo.vllm.migration.MigrationHandler` fall back to the
Phase-2.A recompute path automatically (no behavioural change).

References:
* vLLM 0.16 connector contract:
  ``vllm.distributed.kv_transfer.kv_connector.v1.base.KVConnectorBase_V1``
* Reference implementation:
  ``dynamo/lib/bindings/kvbm/python/kvbm/vllm_integration/connector/dynamo_connector.py``
* Design doc: ``RL-Scaling/tutorial/scaling/RL_Scaling_Unified_Design.md``
  §P2.3 (v3).
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # pragma: no cover - import-only typing
    import torch
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- meta
@dataclass
class _PendingMigration:
    """Per-request migration state carried from migrate_in to start_load_kv."""

    request_id: str
    src_block_ids: list[int]
    nixl_handshake_meta: dict
    replay_token_count: int
    dst_block_ids: list[int] = field(default_factory=list)


# ---------------------------------------------------------------- connector
class RLScalingMigrationConnector:
    """KVConnectorBase_V1 implementation for RL-Scaling Phase-2.B.

    Not a subclass at import time so ``import dynamo.vllm.migration_connector``
    works in environments without vLLM installed (e.g. lint/unit tests).
    The actual subclass binding happens in :func:`_bind_to_vllm` lazily.
    """

    # ----- handler-facing surface (used by MigrationHandler) -----------------

    def __init__(
        self,
        vllm_config: Optional["VllmConfig"] = None,
        role: Optional[Any] = None,
        kv_cache_config: Optional["KVCacheConfig"] = None,
        *,
        feature_enabled: bool = False,
    ) -> None:
        self._vllm_config = vllm_config
        self._role = role
        self._kv_cache_config = kv_cache_config
        # Phase-2.B is GPU-validation gated. Until proven on a live cluster,
        # we deliberately advertise no external matched tokens, which causes
        # the handler to fall back to recompute. Flip via env var or config
        # to enable on a controlled rollout.
        self._feature_enabled = feature_enabled
        self._pending: dict[str, _PendingMigration] = {}
        self._lock = threading.Lock()
        self._nixl_handshake_meta: Optional[dict] = None

    async def attach_remote_blocks(
        self,
        request_id: str,
        src_block_ids: list[int],
        nixl_handshake_meta: dict,
        replay_token_count: int,
    ) -> None:
        """Register a migrated request for the next scheduler step.

        Called by ``MigrationHandler.migrate_in`` *before* the request is
        re-submitted to the engine. The connector remembers the source-side
        block IDs and will return ``(replay_token_count, True)`` from
        :meth:`get_num_new_matched_tokens` when vLLM next schedules this
        request, telling vLLM to skip prefill and dispatch as decode-only.
        """
        with self._lock:
            self._pending[request_id] = _PendingMigration(
                request_id=request_id,
                src_block_ids=list(src_block_ids),
                nixl_handshake_meta=dict(nixl_handshake_meta),
                replay_token_count=int(replay_token_count),
            )
        logger.info(
            "[MigrationConnector] attached request_id=%s src_blocks=%d replay=%d",
            request_id,
            len(src_block_ids),
            replay_token_count,
        )

    def nixl_handshake_meta(self) -> Optional[dict]:
        """Return this worker's NIXL handshake metadata for migrate_out responses.

        Populated at engine boot from the wrapped NixlConnector. ``None``
        until then; ``MigrationHandler.migrate_out`` will simply omit the
        field, which makes Phase-2.B fall back to Phase-2.A on the dst side.
        """
        return self._nixl_handshake_meta

    def set_nixl_handshake_meta(self, meta: dict) -> None:
        """Wired by ``main.py`` once the NixlConnector reports its address."""
        self._nixl_handshake_meta = dict(meta)

    # ----- scheduler/leader-side KVConnectorBase_V1 hooks --------------------

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        if not self._feature_enabled:
            return 0, False
        with self._lock:
            pending = self._pending.get(getattr(request, "request_id", ""))
        if pending is None:
            return 0, False
        # Tell vLLM "the first replay_token_count tokens are externally
        # cached; load them async between scheduler steps."
        external = max(0, pending.replay_token_count - num_computed_tokens)
        return external, True

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        if not self._feature_enabled or num_external_tokens <= 0:
            return
        rid = getattr(request, "request_id", "")
        with self._lock:
            pending = self._pending.get(rid)
            if pending is None:
                return
            # Flatten KVCacheBlocks into block IDs we will NIXL-pull into.
            try:
                pending.dst_block_ids = [
                    int(b.block_id) for b in (blocks.blocks[0] if blocks.blocks else [])
                ]
            except Exception:  # noqa: BLE001
                logger.warning(
                    "[MigrationConnector] failed to extract dst_block_ids for %s",
                    rid,
                    exc_info=True,
                )

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> Any:
        # KVConnectorMetadata is opaque to vLLM; the worker side decodes it
        # in bind_connector_metadata. We pack everything we need.
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: WPS433
            KVConnectorMetadata,
        )

        with self._lock:
            payload = {
                rid: {
                    "src_block_ids": list(p.src_block_ids),
                    "dst_block_ids": list(p.dst_block_ids),
                    "src_handshake": dict(p.nixl_handshake_meta),
                }
                for rid, p in self._pending.items()
                if p.dst_block_ids  # only ship requests that got allocated this step
            }
        meta = KVConnectorMetadata()
        # Use a side-channel attribute to avoid subclassing KVConnectorMetadata.
        meta._rl_scaling_payload = payload  # type: ignore[attr-defined]
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        rid = getattr(request, "request_id", "")
        with self._lock:
            self._pending.pop(rid, None)
        # (False, None) = finalize synchronously, no extra metadata to ship.
        return False, None

    # ----- worker-side hooks -------------------------------------------------

    def bind_connector_metadata(self, connector_metadata: Any) -> None:
        if not self._feature_enabled:
            return
        payload = getattr(connector_metadata, "_rl_scaling_payload", None)
        if not payload:
            return
        # Stash for start_load_kv to consume.
        self._current_step_payload = payload  # type: ignore[attr-defined]

    def clear_connector_metadata(self) -> None:
        if hasattr(self, "_current_step_payload"):
            delattr(self, "_current_step_payload")

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Issue NIXL pulls for all migrated requests scheduled this step.

        TODO(GPU-validation): wire this to the wrapped ``NixlConnector``'s
        transfer primitive. The shape is::

            for rid, info in payload.items():
                nixl_connector.transfer_blocks(
                    src_handshake=info["src_handshake"],
                    src_block_ids=info["src_block_ids"],
                    dst_block_ids=info["dst_block_ids"],
                    direction="pull",
                )

        Until validated against a live NixlConnector pair, this is a no-op
        (and ``feature_enabled`` is False by default, so this branch is
        unreachable in production).
        """
        payload = getattr(self, "_current_step_payload", None)
        if not payload:
            return
        logger.info(
            "[MigrationConnector] start_load_kv would pull %d migrated requests "
            "(NIXL wiring is TODO; falling through to no-op)",
            len(payload),
        )

    def wait_for_layer_load(self, layer_name: str) -> None:
        # Synchronous no-op: with NIXL not yet wired, there's nothing to wait
        # for. When wired, this should block until the layer's blocks are in.
        return None

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: "torch.Tensor",
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        # Migration is destination-side; saving is the source's job (handled
        # by the existing PdConnector).
        return None

    def wait_for_save(self) -> None:
        return None

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        return None, None

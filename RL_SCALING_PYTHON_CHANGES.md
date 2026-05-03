# RL-Scaling — Phase 2 改动清单 (Pure Python · v3 fact-corrected)

> **本文档 v3 取代了 v1 (RL_SCALING_RUST_CHANGES.md) 中的 8 项 Rust 改动**。
> 经代码贯穿检查，**Phase 2 不需要任何 Rust 改动**。事实根据：
>
> 1. NIXL Agent 无方向 — 方向由 `transfer/strategy.rs::TransferStrategy` 在 src/dst layout
>    比较时自动推断。Python 侧丢弃缓存的 connector handle 即完成 "翻转"。
> 2. `request_id → Vec<BlockId>` 映射 **已存在并已通过 PyO3 暴露** —
>    `lib/bindings/kvbm/python/kvbm/vllm_integration/kv_cache_manager.py:get_block_ids(rid)`。
> 3. vLLM 0.16 KVConnector v1 的官方注入 hook 是 `get_num_new_matched_tokens` (返回
>    `(k, async_load)`) + `start_load_kv` (worker-side NIXL pull)；不是早先以为的
>    `recv_kv_caches_and_hidden_states`。
> 4. P→D NIXL transfer 与 D→D NIXL transfer 共用同一段代码，方向由 peer 地址决定。
>
> 设计依据：兄弟仓库 `RL-Scaling/tutorial/scaling/RL_Scaling_Unified_Design.md` §Phase 2 (v3)。

---

## 总图

| ID | 模块 | 文件 | 状态 |
|----|------|------|------|
| **S2-v2-P1** | NIXL connector handle 重交 | `components/src/dynamo/vllm/dual_mode.py:_reconfig_nixl` | ✅ 已交付 (commit `9eb6642460`) |
| **S2-v2-P2** | KV pool reset (公开 API) | `components/src/dynamo/vllm/dual_mode.py:_reconfig_kv_pool` | ✅ 已交付 (commit `9eb6642460`) |
| **S2-v2-P3** | Router metadata 推送 | `components/src/dynamo/vllm/dual_mode.py:_emit_role_changed` | ✅ 已交付 (commit `9eb6642460`) |
| **S3-v2-P1** | 复用 `KvbmCacheManager.get_block_ids` 的 RequestBlockIndex | `components/src/dynamo/vllm/migration.py` | ✅ 本批交付 |
| **S3-v2-P2** | migrate_out 响应携带 src_block_ids + handshake | `components/src/dynamo/vllm/migration.py` | ✅ 本批交付 |
| **S3-v2-P3** | MigrationPolicy cost-benefit 闸门 | `components/src/dynamo/vllm/migration.py` | ✅ 已交付 (commit `9eb6642460`) |
| **S3-v2-P4** | RLScalingMigrationConnector 骨架 | `components/src/dynamo/vllm/migration_connector.py` (新增) | ✅ 本批交付 (骨架 · NIXL 调用环节标 TODO 待 GPU 验证) |
| **S3-v2-P5** | migrate_in 优先 connector，失败回退 recompute | `components/src/dynamo/vllm/migration.py` | ✅ 本批交付 |

---

## 已删除的 v1 项 (作废理由)

| v1 ID | v1 设想 | 删除理由 |
|---|---|---|
| ~~S2-v2-R1~~ | Rust `NixlAgent::set_direction` + PyO3 | NIXL 方向由 `TransferStrategy` 从 layout metadata 推断，无 agent-level 方向字段。 |
| ~~S2-v2-R2~~ | `KvEvent::WorkerRoleChanged` Rust 事件 | `wake_up()` 已含 `register_endpoint_instance()`；router 下次 metadata 拉取自然得到新角色。HTTP 层 `update_metadata` 同步推送即可。 |
| ~~S3-v2-R1~~ | `MigrateRequestBlocks` ZMQ 消息 | 迁移是点对点 worker→worker；HTTP `/migrate_out` + `/migrate_in` 已能传递所有元数据。 |
| ~~S3-v2-R2~~ | leader `migrate_request` API | 同上。 |
| ~~S3-v2-R3~~ | Rust `request_block_map.rs` + PyO3 | `KvbmCacheManager.get_block_ids(rid)` 已实现并已暴露。 |
| ~~S3-v2-V3~~ | connector 实现 `recv_kv_caches_and_hidden_states` | vLLM 0.16 KVConnector v1 用的是 `get_num_new_matched_tokens` + `start_load_kv` + `wait_for_layer_load` + `save_kv_layer` + `wait_for_save`；旧名是 v0 残留。 |

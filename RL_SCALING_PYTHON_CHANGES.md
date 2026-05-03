# RL-Scaling — Phase 2 改动清单 (Pure Python · v3.5 vLLM-native)

> **本文档 v3.5 取代了 v3 中"我们要写自己的 KVConnector"的设计。**
> 经过对 vLLM 0.16 `nixl_connector.py` 的代码贯穿（参见 `nixl_connector.py:177-220`
> 的 `RemoteMeta` + `add_new_req_to_recv` 与 `2063-2371` 的 `start_load_kv` →
> `_read_blocks`），事实是：
>
> 1. **vLLM 0.16 的 `NixlConnector` 已经实现了"按 `kv_transfer_params` 拉取
>    远端 KV"的全部逻辑**。它从 `request.kv_transfer_params` 读出
>    `{do_remote_prefill, remote_engine_id, remote_block_ids, remote_host,
>    remote_port, remote_request_id}`，自动完成：handshake (ZMQ 取
>    `NixlAgentMetadata`) → 注册远端 agent → prepare descriptor lists →
>    issue NIXL READ → 等待完成 → 通知远端释放。
> 2. **dynamo 的 `handlers.py:1577-1589` 已经在用同一套 `kv_transfer_params`**
>    做 disagg-PD（prefill 把 kv_transfer_params 回传给 decode；decode 在
>    `handlers.py:1380` 把它装到 `sampling_params.extra_args` 上提交）。
> 3. **`MultiConnector`** (vLLM 0.16 `multi_connector.py:116-118`) 让
>    `[DynamoConnector, NixlConnector]` 串行运行。当前集群部署的
>    `pd_connector.py` 已经是这个形态。
>
> 因此 **migrate 不需要任何自定义 KVConnector**，只要 migrate_in 把
> `kv_transfer_params` 注入到 `sampling_params.extra_args` 即可。

---

## 总图

| ID | 模块 | 文件 | 状态 |
|----|------|------|------|
| **S2-v2-P1** | NIXL connector handle 重交 | `components/src/dynamo/vllm/dual_mode.py:_reconfig_nixl` | ✅ 已交付 (commit `9eb6642460`) |
| **S2-v2-P2** | KV pool reset (公开 API) | `components/src/dynamo/vllm/dual_mode.py:_reconfig_kv_pool` | ✅ 已交付 (commit `9eb6642460`) |
| **S2-v2-P3** | Router metadata 推送 | `components/src/dynamo/vllm/dual_mode.py:_emit_role_changed` | ✅ 已交付 (commit `9eb6642460`) |
| **S3-v2-P1** | RequestBlockIndex (复用 `KvbmCacheManager.get_block_ids`) | `components/src/dynamo/vllm/migration.py` | ✅ 已交付 (commit `fd1f3ab6d6`) |
| **S3-v2-P2** | MigrationPolicy cost-benefit 闸门 | `components/src/dynamo/vllm/migration.py` | ✅ 已交付 (commit `9eb6642460`) |
| **S3-v2-P3** | migrate_out 携带 `src_block_ids` + `kv_transfer_params` | `components/src/dynamo/vllm/migration.py` | ✅ 本批交付 (v3.5) |
| **S3-v2-P4** | migrate_in 注入 `sampling_params.extra_args["kv_transfer_params"]` | `components/src/dynamo/vllm/migration.py` | ✅ 本批交付 (v3.5) |
| **S3-v2-P5** | NixlMetaProvider 由 `main.py` 从 KvTransferConfig 装填 | `components/src/dynamo/vllm/main.py` | ⚠️ 集群侧装配步骤 (NIXL coords plumb-through) |
| **S3-v2-P6** | submit_request 路径透传 `kv_transfer_params` 到 sampling_params | `components/src/dynamo/vllm/handlers.py` (复用既有 `decode` 路径 1380 行) | ⚠️ 需在 generate 路径上确认 path 复用 |

`✅ = 代码 + 单测交付`；`⚠️ = 集群部署装配步骤，已在文档中明示，需 GPU 在线验证`。

---

## 重要架构说明

### 为什么不需要自定义 KVConnector

migrate_in 的流程：
```
controller HTTP /migrate_in {kv_transfer_params: {...}}
   ↓
MigrationHandler.migrate_in
   ↓ (cost-benefit 通过)
tracker.submit_request(rid, payload={..., kv_transfer_params: {...}})
   ↓
generate handler → sampling_params.extra_args["kv_transfer_params"] = ...
   ↓
engine.add_request(prompt, sampling_params, rid, ...)
   ↓
vLLM scheduler step → MultiConnector.get_num_new_matched_tokens(req, ...)
   ↓
NixlConnectorScheduler 看到 do_remote_prefill=True → 返回 (prompt_len, async=True)
   ↓
NixlConnectorScheduler.add_new_req_to_recv → reqs_to_recv
   ↓
NixlConnectorWorker.start_load_kv → _read_blocks_for_req → make_prepped_xfer("READ", ...)
   ↓
NIXL pulls src_block_ids from remote_engine into freshly-allocated dst_block_ids
   ↓
forward pass uses populated KV — first new token generated ~2-15 ms later
```

vLLM 自带 NixlConnector 已经做完全部 7 个步骤。我们 RL-Scaling 唯一要做的：
- 在 migrate_out 收集源端 NIXL 坐标（`KvTransferConfig.engine_id` /
  `nixl_side_channel_host` / `nixl_side_channel_port`）
- 把它装进 HTTP 响应
- 在 migrate_in 把它装到 sampling_params 上

### 已知正确性 gap：源端 block-hold

vLLM 0.16 的 disagg-PD 用 `KVConnectorBase_V1.request_finished()` 返回
`(True, None)` 让 vLLM **延迟释放 blocks** 直到 `get_finished()` 报告安全
（参见 dynamo 的 `connector_leader.py:226-244` 实现）。

migrate_out 当前直接 `abort_request`，这会让源端 KV blocks 立即释放，
**dst 的 NIXL READ 可能拉到垃圾**。

正确做法（待 GPU 验证后实现）：
1. migrate_out **不**立即 `abort_request`；改为通知源端 connector "this request 即将外迁"
2. 源端 KvbmCacheManager 把该 rid 的 block 标记为"延迟释放"
3. 源端 NixlConnector 监听 NIXL 的 send-completion notification
4. 收到 notification 后才真正释放 block

实现位置：`migration.py:migrate_out` 替换 `await self._tracker.abort_request(rid)`
为 `await self._tracker.mark_for_migration(rid)`，并在 `lib/bindings/kvbm/python/kvbm/vllm_integration/connector_leader.py`
中加 "migration-pinned" 状态。

**当前 mitigation**：`MigrationHandler` 默认 `connector_enabled=False`，
即使 migrate_out/migrate_in 协议层支持 NIXL 字段，dst 也走安全的 recompute
路径。此 flag 需要在 src/dst 都加上 block-hold 后由 cluster operator 显式打开。

---

## 已删除的 v1/v3 项 (作废理由)

| ID | 设想 | 删除理由 |
|---|---|---|
| ~~v1 S2-v2-R1~~ | Rust `NixlAgent::set_direction` | NIXL 方向由 `TransferStrategy` 从 layout metadata 推断 |
| ~~v1 S2-v2-R2~~ | `KvEvent::WorkerRoleChanged` | `wake_up()` 已含 `register_endpoint_instance()` |
| ~~v1 S3-v2-R1/R2~~ | `MigrateRequestBlocks` ZMQ + leader API | HTTP `/migrate_*` 已能传递所有元数据 |
| ~~v1 S3-v2-R3~~ | Rust `request_block_map.rs` | `KvbmCacheManager.get_block_ids` 已存在 |
| ~~v3 S3-v2-V3~~ | 自写 `RLScalingMigrationConnector` | vLLM 0.16 NixlConnector + kv_transfer_params 已经做完全部 |

---

## GPU 验证 TODO

部署到双 GPU 节点后必须依次验证：

1. ✅ **冒烟**: 启用 disagg-PD KvTransferConfig 的 worker 能正常处理普通推理
2. ⚠️ **migrate_out coords**: `curl /migrate_out` 返回的 `kv_transfer_params` 字段非空且 host/port 可达
3. ⚠️ **migrate_in connector path**: `curl /migrate_in` 时 worker 日志出现 `[NixlConnector] _read_blocks_for_req`
4. ⚠️ **D2D 字节计数**: `kvbm_offload_blocks_d2d` 增量等于 `len(remote_block_ids)`
5. ⚠️ **block-hold**: 不要在没有 block-hold 的情况下打开 `connector_enabled=True`，否则会读到错误数据

(2)–(5) 必须 GPU 在线一次过；任何一步失败保持 `connector_enabled=False` 走安全 recompute。

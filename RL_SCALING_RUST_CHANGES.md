# RL-Scaling — Phase 2 必做清单 (Rust + vLLM wrapper)

> 本文档替代了原先的 "future work / 暂时 stub" 表述。
> Phase 1 (S2 stub + S3 recompute-prefill) 已完成并跑通了控制面正确性，
> 但不满足 RL Scaling 的最终目标 ——
> **端到端最小化 batch 处理时间**。
> Phase 2 把所有 stub / fallback 都升级为真实实现。
>
> 设计依据：兄弟仓库 `RL-Scaling/tutorial/scaling/RL_Scaling_Unified_Design.md`
> 中新增的 **Phase 2 — 端到端最优实现** 章节 (P2.0–P2.5)。

---

## 总图

| ID | 模块 | 文件 | 工作量 | 阻塞测试 |
|----|------|------|--------|----------|
| **S2-v2-R1** | NIXL agent 方向切换 | `lib/llm/src/block_manager/v2/physical/transfer/nixl_agent/mod.rs` + `lib/bindings/python/` | M | S2 真实翻转 |
| **S2-v2-R2** | `WorkerRoleChanged` 事件 | `lib/llm/src/kv_router/{protocols.rs, scheduler.rs}` | S | Router 重路由 |
| **S2-v2-V1** | KV 池运行时缩放 | `components/src/dynamo/vllm/dual_mode.py` + 调 vLLM 内部 `reset_prefix_cache` / `_initialize_kv_caches` | M | Decode 模式吞吐 |
| **S3-v2-R1** | `MigrateRequestBlocks` ZMQ 消息 | `lib/llm/src/block_manager/distributed/utils.rs` | S | S3 端到端 |
| **S3-v2-R2** | leader 端 `migrate_request` API | `lib/llm/src/block_manager/distributed/leader.rs` | M | S3 端到端 |
| **S3-v2-R3** | `request_id → Vec<BlockId>` 映射 | 新增 `lib/llm/src/block_manager/v2/physical/request_block_map.rs` + PyO3 | M | S3 端到端 |
| **S3-v2-V1** | `RLScalingKVConnector` | `components/src/dynamo/vllm/migration_connector.py` (新增) | M | S3 端到端 |
| **S3-v2-V2** | `migrate_in` 改用 connector，不再 recompute | `components/src/dynamo/vllm/migration.py` | L | S3 端到端 |

`L=large(>200 LoC), M=medium(80–200), S=small(<80)`.

---

## S2-v2 — 真实角色翻转

### S2-v2-R1：NIXL agent 方向切换

**已有基础**：
`lib/llm/src/block_manager/v2/physical/transfer/mod.rs` 中
`transfer_blocks(src, dst, src_block_ids, dst_block_ids, ctx)` 已经在做
worker 间 NIXL D2D；但 NIXL agent 内部的 send/recv loop **没有暴露切换接口**。

**新增 API**：
```rust
// lib/llm/src/block_manager/v2/physical/transfer/nixl_agent/mod.rs
pub enum Direction { Sender, Receiver }

impl NixlAgent {
    pub async fn set_direction(&self, dir: Direction) -> Result<()>;
    pub async fn rebind_buffers(&self, layout: &PhysicalLayout) -> Result<()>;
    pub fn stats(&self) -> NixlAgentStats; // 必须暴露 active_transfers，给测试用
}
```

**PyO3 binding**：在 `lib/bindings/python/rust/llm/block_manager.rs` 中
暴露为 `nixl_agent.set_direction("sender"|"receiver")` 与 `.rebind(layout)`。

**Python 调用点**：`components/src/dynamo/vllm/dual_mode.py:_reconfig_nixl`
把 `logger.info(...)` 替换为：
```python
agent = self._handler.engine_client.nixl_agent
new_dir = "sender" if target_role == "prefill" else "receiver"
await agent.set_direction(new_dir)
await agent.rebind(self._handler.engine_client.kv_layout)
```

### S2-v2-R2：`WorkerRoleChanged` 事件

```rust
// lib/llm/src/kv_router/protocols.rs
pub enum KvEvent {
    BlockStored  { worker_id: u64, block_id: u64, tokens: Vec<u32> },
    BlockRemoved { worker_id: u64, block_id: u64 },
    WorkerRoleChanged { worker_id: u64, old_role: String, new_role: String }, // 新增
}
```

```rust
// lib/llm/src/kv_router/scheduler.rs::handle_event 新增分支
KvEvent::WorkerRoleChanged { worker_id, new_role, .. } => {
    self.kv_indexer.remove_all_entries_for_worker(worker_id);
    if let Some(s) = self.worker_states.get_mut(&worker_id) {
        s.active_blocks = 0;
        s.in_flight_requests = 0;
        s.role = new_role.clone();
    }
    self.repartition_role_pools(worker_id, &new_role);
}
```

### S2-v2-V1：KV 池运行时缩放

vLLM 1.0.1 内部 **已有** `reset_prefix_cache()` 和 `_initialize_kv_caches()`，
不需要 fork vLLM。在 worker wrapper 层调用：

```python
# dual_mode.py:_reconfig_kv_pool
async def _reconfig_kv_pool(self, target_role: str) -> None:
    engine = self._handler.engine_client.engine_core   # vLLM AsyncLLMEngine 内核
    # 1) 清掉 prefix cache 引用
    engine.reset_prefix_cache()
    # 2) 修改 num_gpu_blocks_override
    new_blocks = self._target_blocks_for_role(target_role)
    engine.cache_config.num_gpu_blocks_override = new_blocks
    # 3) 重建 KV cache（engine 已 sleep，显存安全）
    engine._initialize_kv_caches(engine.vllm_config)
    # 4) 让 NIXL 重新登记 buffer (否则 D2D 还指向旧地址)
    await self._handler.engine_client.nixl_agent.rebind(
        engine.kv_cache_layout()
    )
```

`_target_blocks_for_role` 由 `cache_config.num_gpu_blocks` 基线 × 角色比例
得到（decode=0.85, prefill=0.20，可配）。

---

## S3-v2 — 真实 KV-D2D 请求迁移

### S3-v2-R1：ZMQ 消息

```rust
// lib/llm/src/block_manager/distributed/utils.rs
pub const ZMQ_MIGRATE_REQUEST_BLOCKS_MESSAGE: &str = "migrate_request_blocks";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrateRequestBlocks {
    pub request_id: String,
    pub src_worker: u64,
    pub dst_worker: u64,
    pub src_block_ids: Vec<u64>,
    pub dst_block_ids: Vec<u64>,
}
```

### S3-v2-R2：leader 端 API

```rust
// lib/llm/src/block_manager/distributed/leader.rs
impl BlockManagerLeader {
    pub async fn migrate_request(
        &self,
        req: MigrateRequestBlocks,
    ) -> anyhow::Result<oneshot::Receiver<()>> {
        let zmq = self.zmq_leader.get()
            .ok_or_else(|| anyhow::anyhow!("ZMQ leader not ready"))?;
        let data = vec![serde_json::to_vec(&req)?];
        zmq.broadcast(ZMQ_MIGRATE_REQUEST_BLOCKS_MESSAGE, data).await
    }
}
```

每个 worker 收到 `MigrateRequestBlocks` 后：
- 若 `worker_id == src_worker`：调用既有 `transfer_blocks(src=local, dst=remote, src_ids, dst_ids, ctx)`，源不动 dst 由对端预分配。
- 若 `worker_id == dst_worker`：等待 transfer 完成 notification，触发 connector 注入。

### S3-v2-R3：`request_id → Vec<BlockId>` 映射

KVPublisher 已经按 block 发出 `BlockStored {worker_id, block_id, tokens}` 与
`BlockRemoved {worker_id, block_id}` 事件
（参见 `dynamo/docs/design_docs/router-design.md`）。新增一个
**worker 本地** 的订阅者：

```rust
// lib/llm/src/block_manager/v2/physical/request_block_map.rs (新文件)
pub struct RequestBlockMap {
    map: DashMap<RequestId, Vec<BlockId>>,
}
impl RequestBlockMap {
    pub fn on_block_stored(&self, rid: RequestId, bid: BlockId);
    pub fn on_block_removed(&self, bid: BlockId); // O(1) reverse index
    pub fn lookup(&self, rid: &RequestId) -> Option<Vec<BlockId>>;
}
```

通过 PyO3 暴露 `request_block_map.lookup(rid) -> list[int]`。

### S3-v2-V1 + V2：connector 路径替换 recompute

```python
# components/src/dynamo/vllm/migration_connector.py (新增)
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase

class RLScalingKVConnector(KVConnectorBase):
    """把已经在显存里的 dst_block_ids 接到一个新 sequence 上，
    让 scheduler 把它当成 'decode-only, position=K' 加入 batch。"""

    def recv_kv_caches_and_hidden_states(self, model_input, kv_caches, hidden_states):
        # no-op: 我们的 block 已经被 leader 通过 NIXL D2D 灌进显存，
        # 这里只是告诉 vLLM "别 prefill 了，从 position=K 接着 decode"。
        return None, True, model_input
```

```python
# migration.py:MigrationHandler.migrate_out 改造
async def migrate_out(self, body):
    rid = body["request_id"]
    src_blocks = self._block_map.lookup(rid)             # 来自 R3
    dst_blocks = await self._dst_client.allocate(len(src_blocks))
    await self._leader.migrate_request(MigrateRequestBlocks(
        request_id=rid, src_worker=self.wid, dst_worker=body["dst"],
        src_block_ids=src_blocks, dst_block_ids=dst_blocks,
    ))
    await self._tracker.abort_request(rid)
    last_token = self._tracker.last_token(rid)
    return {"status":"ok","request_id":rid,
            "dst_block_ids":dst_blocks,
            "last_token":last_token,
            "sampling_params":self._tracker.sampling_params(rid)}

# migrate_in: 用 connector 把 block 接进去，不再 submit_request(prompt+gen)
async def migrate_in(self, body):
    self._connector.attach_blocks(body["request_id"],
                                  body["dst_block_ids"],
                                  position=len(body["dst_block_ids"]) * BLOCK_SIZE)
    await self._tracker.add_decode_only_request(
        body["request_id"], body["last_token"], body["sampling_params"])
    return {"status":"ok"}
```

> 注：`add_decode_only_request` 是 wrapper 层的便捷方法，底层调用
> vLLM 的 `engine.add_request(...)` 但绕过 prefill 阶段，依赖 connector
> 在 `recv_kv_caches_and_hidden_states` 中告诉 scheduler "skip prefill"。

---

## 验收（与 TEST_PLAN.zh-CN.md Phase 2 节联动）

| 项目 | 通过标准 |
|---|---|
| S2-v2 真实翻转 | 翻转后 5s 内 `dynamo_frontend_requests_total{worker_id=W}` 在新角色路由池中开始增长；旧角色池 ≈ 0 |
| KV 池缩放 | 翻转前后 `engine.cache_config.num_gpu_blocks` 与目标角色比例相符 (±5%) |
| S3-v2 D2D 路径 | `kvbm_offload_blocks_d2d` 出现非零计数；`recompute_prefill_count` 不增加 |
| 端到端时间 | 同一 batch、同一 seed，开启迁移比关闭迁移的 wall-clock 短 ≥ 20% |
| 输出确定性 | greedy + seed 固定下，迁移后续写与基线 sha256 一致 |

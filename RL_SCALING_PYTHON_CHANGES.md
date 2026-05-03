# RL-Scaling — Python Changes (v3.6 · in-process HTTP sidecar shipped)

> **v3.6 update**: discovered (and fixed) a fundamental gap from earlier
> phases: the dynamo-vllm worker pod has **no HTTP server**. dynamo
> runtime's `serve_endpoint()` is NATS RPC, not HTTP, so neither
> `/switch_role` (S2) nor `/migrate_out` `/migrate_in` (S3) were ever
> actually reachable from the rl-scaling-controller. The controller's
> `dual_mode_client.py` and `migration_client.py` calls would always 404
> on a real cluster. Test scripts caught this with `fail "/migrate_out
> not registered"` but the failure was treated as "redeploy needed"
> rather than "wire missing".
>
> **What v3.6 ships**: `components/src/dynamo/vllm/rl_scaling_sidecar.py`
> — an in-process aiohttp server that exposes the missing routes
> directly from the worker process, sharing the same AsyncLLM engine
> in-memory. No new container, no NATS hop, no marshalling overhead.

---

## 总图

| ID | 模块 | 文件 | 状态 |
|----|------|------|------|
| **S1** | RL signal SDK + controller HTTP | `RL-Scaling/rl-scaling-controller/...` | ✅ 已交付 |
| **S2-P1** | NIXL connector handle 重交 | `dual_mode.py:_reconfig_nixl` | ✅ 已交付 |
| **S2-P2** | KV pool reset (公开 API) | `dual_mode.py:_reconfig_kv_pool` | ✅ 已交付 |
| **S2-P3** | Router metadata 推送 | `dual_mode.py:_emit_role_changed` | ✅ 已交付 |
| **S3.A-P1** | RequestBlockIndex (复用 `KvbmCacheManager.get_block_ids`) | `migration.py` | ✅ 已交付 |
| **S3.A-P2** | MigrationPolicy cost-benefit 闸门 | `migration.py` | ✅ 已交付 |
| **S3.B-P1** | migrate_out 携 `src_block_ids` + `kv_transfer_params` | `migration.py` | ✅ 已交付 |
| **S3.B-P2** | migrate_in 注入 `kv_transfer_params` 到 sampling_params | `migration.py` | ✅ 已交付 |
| **S3.B-P3** | NixlMetaProvider 由 `main.py` 装填 | `rl_scaling_sidecar.py:make_nixl_meta_provider` | ✅ 本批 (v3.6) |
| **S3.B-P4** | submit_request 透传 kv_transfer_params | `rl_scaling_sidecar.py:make_submit_request_callback` | ✅ 本批 (v3.6) |
| **SC-P1** | **In-process HTTP sidecar** (`/switch_role`, `/v1/role`, `/migrate_out`, `/migrate_in`, `/healthz`, `/v1/active_requests`) | `rl_scaling_sidecar.py` | ✅ 本批 (v3.6) |
| **SC-P2** | **In-process request registry** (handler.generate_tokens 钩入) | `handlers.py:1228+` `rl_scaling_sidecar.py:InProcessRequestRegistry` | ✅ 本批 (v3.6) |
| **SC-P3** | **EngineRequestTracker** (RequestTracker Protocol 实现) | `rl_scaling_sidecar.py:EngineRequestTracker` | ✅ 本批 (v3.6) |
| **SC-P4** | main.py 启动 sidecar，wire DualMode + Migration | `main.py:840+` | ✅ 本批 (v3.6) |
| **S3.B-P5** | 源端块持留 (block-hold) | `migration.py:migrate_out` 替 `abort_request` 为 `mark_for_migration` + `connector_leader.py` migration-pinned slot | ⏳ 待 GPU 验证 |
| **S3.C** | 前端连接重路由 (frontend-side) | `lib/llm/.../frontend` | ⏳ 后续 |

---

## 架构简图

```
┌────────────────────────── worker pod (single process) ──────────────────────────┐
│                                                                                  │
│  ┌──────────────────┐        ┌──────────────────────────────────────┐           │
│  │ dynamo runtime   │        │  in-process aiohttp sidecar (NEW)    │           │
│  │ NATS endpoints   │        │  port 9090 (DYNAMO_RL_SIDECAR_PORT)  │           │
│  │ - generate       │        │  - GET  /healthz                     │           │
│  │ - clear_kv_blocks│        │  - GET  /v1/role                     │           │
│  │ - sleep/wake_up  │        │  - POST /switch_role     ──────┐    │           │
│  │ ...              │        │  - POST /migrate_out  ────┐    │    │           │
│  └────────┬─────────┘        │  - POST /migrate_in   ─┐  │    │    │           │
│           │                  │  - GET  /v1/active...  │  │    │    │           │
│           ▼                  └────────────────────────┼──┼────┼────┘           │
│  ┌──────────────────┐                                 │  │    │                 │
│  │ DecodeWorkerHandler                                │  │    │                 │
│  │ .generate_tokens ──── records ─→ InProcessRequestRegistry  │                 │
│  │ .engine_client                                  ▲  │  │    │                 │
│  └────────┬─────────┘                              │  │  │    │                 │
│           │                                        │  │  │    │                 │
│           ▼                                        │  ▼  ▼    ▼                 │
│  ┌──────────────────┐         ┌────────────────────┴────────────────┐           │
│  │ vLLM AsyncLLM    │◀────────│ EngineRequestTracker  MigrationHandler│         │
│  │ + NixlConnector  │         │ + MigrationPolicy + RequestBlockIndex │         │
│  └──────────────────┘         │ + nixl_meta_provider(vllm_config)     │         │
│                               └────────────────────────────┬──────────┘         │
│                                                            │                    │
│  ┌──────────────────┐                                      │                    │
│  │ DualModeWorker   │◀─────────────────────────────────────┘                    │
│  └──────────────────┘                                                           │
└──────────────────────────────────────────────────────────────────────────────────┘
                                      ▲
                                      │ HTTP (curl from controller pod)
                                      │
                         ┌────────────┴────────────┐
                         │ rl-scaling-controller   │
                         │ - dual_mode_client.py   │
                         │ - migration_client.py   │
                         └─────────────────────────┘
```

---

## 数据流

**请求注册** (handlers.py:1228 钩子)
```
client → frontend → NATS → DecodeWorkerHandler.generate_tokens(rid, prompt, sp)
   │
   ├─→ registry.register(rid, prompt_tokens, sp_dict)         # 起点
   │
   ├─→ async for tok in engine.generate(...):
   │      registry.record_tokens(rid, tok.new_token_ids)      # 流式记录
   │      yield tok                                            # 同时给客户端
   │
   └─→ finally: registry.deregister(rid)                       # 清理
```

**migrate_out** (Phase-2.A 默认 / Phase-2.B 当 connector_enabled=True)
```
controller → POST :9090/migrate_out {request_id: rid}
   │
   ├─→ MigrationHandler.migrate_out(body):
   │     state = registry.get(rid)                             # snapshot
   │     src_block_ids = block_index.lookup(rid)               # KVBM
   │     nixl_coords  = nixl_meta_provider()                   # KvTransferConfig
   │     engine.abort(rid)                                     # ⚠ 立即 (S3.B-P5 未做)
   │     return {
   │       status: ok, prompt_tokens, generated_tokens, sampling_params,
   │       src_block_ids,
   │       kv_transfer_params (如果 connector_enabled+coords 都有)
   │     }
   │
   └─→ controller forwards to dst worker
```

**migrate_in (recompute, 默认)**
```
controller → POST dst:9090/migrate_in {payload}
   │
   ├─→ MigrationHandler.migrate_in(body):
   │     policy.gate(prompt+gen, generated, max)               # cost-benefit
   │     submit_request_callback(rid, {prompt+gen, sp})        # re-inject
   │     return {status: ok, path: recompute, replay_tokens: N}
   │
   └─→ engine.generate(prompt+gen, sp, rid) → drain in background task
       (the original client connection is gone; tokens discarded server-side)
```

**migrate_in (connector path, Phase-2.B)**
```
controller → POST dst:9090/migrate_in {payload + kv_transfer_params}
   │
   ├─→ MigrationHandler.migrate_in:
   │     submit_request_callback(rid, payload)
   │       sp.extra_args["kv_transfer_params"] = payload["kv_transfer_params"]
   │       engine.generate(prompt+gen, sp, rid)
   │
   └─→ vLLM scheduler step:
       MultiConnector.get_num_new_matched_tokens(req)
         → NixlConnectorScheduler 看 do_remote_prefill=True
         → reqs_to_recv ∋ rid
       NixlConnectorWorker.start_load_kv:
         _read_blocks_for_req → make_prepped_xfer("READ", ...)
         → NIXL pulls src_block_ids → dst_block_ids (~0.1ms NVLink / ~5ms PCIe)
       第一个新 token 在 ~3 ms 内生成
```

---

## 已知 gap (按严重程度)

### Gap 1 (可能造成数据错误 · feature flag 默认关) — 源端 block-hold

migrate_out 立即 `engine.abort(rid)` → 源端 KV 块同步释放。如果同时 dst 的
NIXL READ 还没完成（异步），dst 拉到的将是已被覆写的垃圾。

**当前 mitigation**：`MigrationHandler(connector_enabled=False)` 默认关。
在 `DYNAMO_RL_CONNECTOR_ENABLED=1` 之前，migrate_in 始终走 recompute 路径，
即使 body 里带了 kv_transfer_params 也忽略。

**修复方案** (S3.B-P5)：参考 vLLM 0.16 disagg-PD 的
`request_finished -> (True, None) + get_finished` 模式
(dynamo `connector_leader.py:226-244`)：
1. `migrate_out` 改为调 `tracker.mark_for_migration(rid)` 而不是 `abort`；
2. 在源端 connector_leader 加一个 "migration-pinned" slot；
3. NIXL send-completion notification 到达后才真正 free + abort；
4. 超时兜底 (e.g. 5 s) 防止悬挂。

**为何没做**：~150 行代码 + 必须 GPU 在线一次过验证；先把基础设施铺好。

### Gap 2 (功能性 · 服务侧已尽职) — 客户端连接重路由

migrate_out 调用 `engine.abort` 后，原来由 frontend 持有的 streaming HTTP
连接也会断；客户端会看到 5xx 或截断。当前 sidecar 把 resubmitted request
的输出 tokens 丢掉（drain task 内部消费），因为没有办法把新 worker 的
token 流回灌到旧的 frontend 连接上。

**当前 mitigation**：migration 在工程上视为"GPU 内部优化"。从 RL trainer
角度看，trainer 收到 abort/retry，重新发起请求即可——已经是 RL 的常规行为
（rollout 失败重发）。完全无需 frontend 改动也能跑通 RL 场景。

**完整解决方案** (S3.C, 后续 Phase)：dynamo frontend 加 in-flight request
table，把 client conn ID 与 dst worker ID 挂钩，dst worker token
stream 经 NATS 回传给 frontend，frontend 续写到原 conn。属于 frontend
侧改动，与本仓 sidecar 解耦。

### Gap 3 (运维) — 8 卡 RTX3090 NVLink 拓扑限制

`nvidia-smi topo -m` 显示：
- (GPU0,GPU2) (GPU1,GPU3) (GPU4,GPU6) (GPU5,GPU7) 是 NV4 配对，互拉走 ~30 GB/s
- 跨配对走 PIX (PCIe Switch) ~12 GB/s
- 跨 NUMA (GPU0-3 vs GPU4-7) 走 NODE 链接 ~5 GB/s

**对 NIXL pull 的影响**：同一 NUMA 域内拉 KV 是最优；跨 NUMA 应避免。
deploy/manifests 应通过 nodeSelector + topologySpreadConstraints 把
prefill/decode worker 配对到同一 NV4 群里。

---

## 测试

| 测试范围 | 文件 | 通过数 |
|---|---|---|
| Phase-1 dual-mode unit | `tests/test_dual_mode.py` | 23/23 |
| Phase-2 migration (recompute + connector) unit | `tests/test_migration.py` | 22/22 |
| Sidecar (registry + tracker + HTTP routes + nixl provider) unit | `tests/test_rl_scaling_sidecar.py` | 23/23 |
| **总计** | | **68/68** |

集群 e2e (待镜像 rebuild + redeploy)：
- `RL-Scaling/test-scripts/test-s2.sh` — /switch_role + 实际 reset_prefix_cache
- `RL-Scaling/test-scripts/test-s3.sh` §1-3 — /migrate_out, /migrate_in recompute 路径
- `RL-Scaling/test-scripts/test-s3.sh` §4 — Phase-2.B kv_transfer_params shape
  验证 (设 `DYNAMO_RL_CONNECTOR_ENABLED=1` 后才有效)

---

## 部署变更

### worker 启动环境变量

| env | 默认 | 说明 |
|---|---|---|
| `DYNAMO_RL_SIDECAR_PORT` | `9090` | sidecar 监听端口 |
| `DYNAMO_RL_SIDECAR_DISABLED` | unset | 设 `1` 完全关掉 sidecar (回到 v3.5 之前的 NATS-only 行为) |
| `DYNAMO_RL_CONNECTOR_ENABLED` | unset | 设 `1` 在 migrate_out 中输出 kv_transfer_params 并在 migrate_in 走 connector 路径。**警告：未实现块持留前会读到错误数据**，仅 GPU 在线测试时打开 |

### K8s manifest

worker container 需要：
- `containerPort: 9090` exposed
- HTTP probe `/healthz` 取代或并存现有的 NATS health
- 如开 Phase-2.B：`env: DYNAMO_RL_CONNECTOR_ENABLED=1`

### 镜像

`fern.config.json` / `rl-scaling-build.yml` 已配置自动 push 到 GHCR；
本批改动会在 push 后约 5-10 分钟出新镜像 `dynamo-vllm-runtime:rl-scaling-<sha>`。

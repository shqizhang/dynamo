# RL-Scaling — Pending Rust Changes

This file tracks Rust patches required to fully realise the RL-Scaling S2/S3
features. Until they land, the corresponding Python code paths are stubbed
no-ops with logging so existing single-mode workers continue to function
unchanged. See the design doc `tutorial/scaling/RL_Scaling_Unified_Design.md`
in the sibling RL-Scaling repo for full context.

---

## S2 — Elastic Role Switch

### 1. NIXL agent reconfig API

* **Where**: `lib/llm/src/block_manager/distributed/transfer/nixl_agent.rs`
* **What**: Add `pub async fn reconfig(&self, role: WorkerRole) -> Result<()>`
  that re-binds the agent to the buffers needed by the new role.
* **Why**: Switching prefill ↔ decode requires reattaching NIXL endpoints to
  a different KV layout.

### 2. KV pool reallocation API

* **Where**: vLLM block-manager bridge in `dynamo-llm`.
* **What**: Add a runtime API to grow / shrink the KV-cache pool without an
  engine restart.
* **Why**: Decode mode wants ≥80% of GPU memory for KV; prefill mode wants
  ≤20%. Today this is fixed at engine startup.

### 3. `WorkerRoleChanged` discovery event

* **Where**: `lib/llm/src/kv_router/protocols.rs`
* **What**: New variant on the existing worker-event enum.
* **Why**: The KV router needs to repartition its routing table when a
  worker flips role.

---

## S3 — Request Consolidation (full NIXL D2D path)

* **Where**: `lib/llm/src/block_manager/distributed/transfer/`
* **What**: Add a new `MigrateBlocks { request_id, src, dst }` transfer
  primitive that streams a single request's KV blocks between workers.
* **Today**: The Python layer falls back to **recompute-prefill** —
  serialising the generated tokens back to the source request and resubmitting
  it on the target worker. Costs one extra prefill pass per migrated request
  but requires zero new transport code.

---

## Why stubbed?

The design doc explicitly flags both reconfig and KV-D2D-migration as
*low-feasibility* until upstream vLLM exposes the relevant runtime APIs. The
Python orchestration (`DualModeWorker`, migration HTTP endpoints) is correct
and tested today; flipping the stubs to real implementations is a localised
change inside each Rust module.

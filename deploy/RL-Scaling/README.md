# RL-Scaling Deployment

Entry point for deploying the **RL-Scaling-flavoured** Dynamo (the worker
built from this repo's [`RL-Scaling` branch](https://github.com/shqizhang/dynamo/tree/RL-Scaling))
together with the [`rl-scaling-controller`](https://github.com/shqizhang/RL-Scaling)
that drives RL-aware scaling decisions.

> **Repo layout assumption**
> ```
> <workspace>/
> ├── dynamo/          ← this repo (RL-Scaling branch)
> │   ├── deploy/RL-Scaling/   ← you are here
> │   └── .github/workflows/rl-scaling-build.yml
> └── RL-Scaling/      ← controller + tutorial assets
>     ├── deploy/deploy-controller.sh
>     └── tutorial/dynamo-auto-deploy/1.0.1/
> ```
> Sibling layout is auto-detected. Override with `RL_SCALING_REPO=/abs/path`
> if your checkout is elsewhere.

The intent is **maximum reuse, minimum surprise**: every base script comes
from the existing `tutorial/dynamo-auto-deploy/1.0.1/` flow in the RL-Scaling
repo. We only override the image reference.

## Files in this folder

| File | Purpose |
| ---- | ------- |
| [build-dynamo-image.sh](build-dynamo-image.sh) | Local/dev: render Dockerfile via `container/render.py` and `docker build && docker push` to `${REGISTRY}/${IMAGE_REPO}:rl-scaling-<sha>`. CI does the equivalent in `.github/workflows/rl-scaling-build.yml`. |
| [deploy-dynamo.sh](deploy-dynamo.sh) | Thin wrapper: copies upstream `manifests/` to a temp dir, `sed`-rewrites the image registry, then `exec`s the upstream `01-deploy-dynamo-1.0.1.sh` with `MANIFEST_DIR=<tmp>`. |
| [deploy-all.sh](deploy-all.sh) | Calls `${RL_SCALING_REPO}/deploy/deploy-controller.sh` then `deploy-dynamo.sh`. |

The only persistent edit to the tutorial flow is one line in
`01-deploy-dynamo-1.0.1.sh` honouring `MANIFEST_DIR`. Everything else is reused.

---

## End-to-end deployment

### 0. Pre-reqs (cluster level, one time)

```bash
# Prometheus / Grafana  (run from RL-Scaling repo)
GRAFANA_ADMIN_PASSWORD=... \
  bash ${RL_SCALING_REPO}/tutorial/dynamo-auto-deploy/k8s/deploy-Prometheus-Grafana.sh
```

### 1. Build the worker image

**Option A — CI (recommended, one-click).**
Push to the `RL-Scaling` branch (or fire `workflow_dispatch` /
`repository_dispatch:rl-scaling-build`). The
[`rl-scaling-build`](../../.github/workflows/rl-scaling-build.yml) workflow
builds and pushes:

```
ghcr.io/<owner>/dynamo-vllm-runtime:rl-scaling-<short-sha>
ghcr.io/<owner>/dynamo-vllm-runtime:rl-scaling-latest
```

It also `repository_dispatch`-notifies the controller repo with a
`dynamo-image-ready` event (payload contains `image` + `sha`) so a CD
workflow on that side can pick it up.

**Option B — Local build.**

```bash
REGISTRY=ghcr.io/<you> \
IMAGE_REPO=dynamo-vllm-runtime \
PUSH=true \
./deploy/RL-Scaling/build-dynamo-image.sh
```

### 2. Deploy controller + dynamo (server-side, one shot)

```bash
export CONTROLLER_IMAGE=ghcr.io/<you>/rl-scaling-controller:rl-scaling-abc1234
export DYNAMO_IMAGE_REGISTRY=ghcr.io/<you>
export DYNAMO_IMAGE_REPO=dynamo-vllm-runtime
export RELEASE_VERSION=rl-scaling-abc1234
export HF_TOKEN=...
export NGC_API_KEY=...

# If the GHCR package is private:
# export GHCR_USERNAME=<you>
# export GHCR_PAT=ghp_xxx          # `read:packages` scope

./deploy/RL-Scaling/deploy-all.sh --router
```

The script will:
1. Build/push (skipped if `PUSH=false`) and apply the controller's RBAC +
   ConfigMap + Deployment in the `dynamo` namespace.
2. Install/upgrade the `dynamo-platform` Helm release in `dynamo-system`.
3. Apply DGD/DGDSA with the custom RL-Scaling worker image.
4. Apply ingress.
5. Smoke-test the frontend.

### 3. Verify

```bash
kubectl -n dynamo-system get pods
kubectl -n dynamo       get deploy/rl-scaling-controller
kubectl -n dynamo-system get dgd vllm-v1-disagg-router \
  -o jsonpath='{.spec.services.VllmDecodeWorker.extraPodSpec.mainContainer.image}'
# → ghcr.io/<you>/dynamo-vllm-runtime:rl-scaling-abc1234
```

Then run the scenario E2E suite from the RL-Scaling repo root:

```bash
CONTROLLER_URL=http://<svc-ip>:8080 \
  ./test-scripts/run-all.sh e2e
```

---

## Image registry & pull secrets

| Image | Registry | Built by | Pull-secret on cluster |
| ----- | -------- | -------- | ---------------------- |
| `dynamo-vllm-runtime:rl-scaling-<sha>` | `ghcr.io/<owner>` | this repo (`rl-scaling-build.yml`) | none if package is **public**; `ghcr-imagepullsecret` (auto-created when you set `GHCR_USERNAME`/`GHCR_PAT`) if private |
| `rl-scaling-controller:<tag>` | `ghcr.io/<owner>` (default) | RL-Scaling repo (`deploy/deploy-controller.sh`) | same as above |
| `nats`, operator images, etc. | upstream | Helm chart | none (public) |

The 1.0.1 deployer always creates `nvcr-imagepullsecret` (for `nvcr.io`).
That is **harmless** even when the worker image now comes from GHCR — it just
goes unused. `deploy-dynamo.sh` will additionally create `ghcr-imagepullsecret`
**only** when `GHCR_USERNAME` + `GHCR_PAT` are exported (and patches the DGD
manifests to reference it).

> **Recommendation**: make the GHCR package public
> (`Package settings → Change visibility → Public`). Then no pull-secret
> management is required and the deploy script stays minimal.

---

## CI/CD: how close to "one-click"?

Today's pipeline:

```
push to RL-Scaling branch ──▶ rl-scaling-build.yml
                              ├── render + buildx + push to ghcr.io
                              └── repository_dispatch → shqizhang/RL-Scaling
                                                        (event-type: dynamo-image-ready)
```

To close the loop into a real one-click CD you have **two choices**:

1. **Self-hosted runner on `gpu14`** — register a self-hosted GitHub runner
   on the K8s server. Add a CD workflow to the RL-Scaling repo that listens
   on `repository_dispatch:dynamo-image-ready` and runs:

   ```yaml
   - run: |
       export CONTROLLER_IMAGE=ghcr.io/${{ github.repository_owner }}/rl-scaling-controller:${{ github.event.client_payload.sha }}
       export DYNAMO_IMAGE_REGISTRY=ghcr.io/${{ github.repository_owner }}
       export DYNAMO_IMAGE_REPO=dynamo-vllm-runtime
       export RELEASE_VERSION=rl-scaling-${{ github.event.client_payload.sha }}
       ./dynamo/deploy/RL-Scaling/deploy-all.sh --router
   ```

   Result: every push → image built → cluster updated, fully automatic.

2. **Manual one-liner** (current state) — the workflow's job summary already
   prints the exact deploy command (with the freshly-built tag). Copy-paste
   that command on the server.

Both paths use the same `deploy-all.sh` entry point, so switching is a
matter of where you run it.

### Required GitHub repo configuration

| Secret / setting | Where | Why |
| ---------------- | ----- | --- |
| `GITHUB_TOKEN` (auto) with `packages: write` | dynamo repo | push to GHCR |
| `CONTROLLER_DISPATCH_TOKEN` (PAT, `repo` + `workflow` scope on `shqizhang/RL-Scaling`) | dynamo repo → Settings → Secrets | fire `repository_dispatch` to the controller repo |
| GHCR package visibility | `ghcr.io/<owner>/dynamo-vllm-runtime` | public ⇒ no pull-secret needed on cluster |

---

## Upgrade / rollback

Pure tag swap; nothing else changes:

```bash
# upgrade
RELEASE_VERSION=rl-scaling-NEWSHA ./deploy/RL-Scaling/deploy-dynamo.sh --router

# rollback
RELEASE_VERSION=rl-scaling-OLDSHA ./deploy/RL-Scaling/deploy-dynamo.sh --router
```

`helm upgrade --install` makes both directions idempotent.

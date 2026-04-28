#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy-all.sh — Orchestrate a full RL-Scaling deployment:
#   1) Deploy/Upgrade the rl-scaling-controller (lives in the RL-Scaling repo)
#   2) Deploy/Upgrade the Dynamo platform + DGD with the custom RL-Scaling
#      vLLM-runtime image (this folder's deploy-dynamo.sh)
#
# Both steps are idempotent. Any flags passed to this script are forwarded to
# deploy-dynamo.sh (e.g. --router | --planner | --mocker | --skip-monitoring).
#
# Required env vars:
#   CONTROLLER_IMAGE        e.g. ghcr.io/shqizhang/rl-scaling-controller:v0.1.0
#   DYNAMO_IMAGE_REGISTRY   e.g. ghcr.io/shqizhang
#   DYNAMO_IMAGE_REPO       e.g. dynamo-vllm-runtime
#   RELEASE_VERSION         e.g. rl-scaling-<short-sha>
#   HF_TOKEN, NGC_API_KEY
# Optional:
#   RL_SCALING_REPO         path to RL-Scaling checkout (auto-detected sibling)
#   GHCR_USERNAME / GHCR_PAT  if the RL-Scaling image lives in a private package
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── Locate the RL-Scaling repo (holds deploy-controller.sh) ──────────────────
if [[ -z "${RL_SCALING_REPO:-}" ]]; then
  for candidate in \
      "${DYNAMO_DIR}/../RL-Scaling" \
      "${HOME}/projects/RL-Scaling" \
      "${HOME}/RL-Scaling"; do
    if [[ -f "${candidate}/deploy/deploy-controller.sh" ]]; then
      RL_SCALING_REPO="$(cd "${candidate}" && pwd)"
      break
    fi
  done
fi
: "${RL_SCALING_REPO:?cannot locate RL-Scaling repo; set RL_SCALING_REPO=/path/to/RL-Scaling}"
export RL_SCALING_REPO

CONTROLLER_DEPLOYER="${RL_SCALING_REPO}/deploy/deploy-controller.sh"
[[ -f "${CONTROLLER_DEPLOYER}" ]] || { echo "controller deployer not found: ${CONTROLLER_DEPLOYER}"; exit 1; }

: "${CONTROLLER_IMAGE:?CONTROLLER_IMAGE not set}"
: "${RELEASE_VERSION:?RELEASE_VERSION not set}"

echo "==> [1/2] Deploy rl-scaling-controller (${CONTROLLER_IMAGE})"
IMAGE="${CONTROLLER_IMAGE}" PUSH="${PUSH:-false}" bash "${CONTROLLER_DEPLOYER}"

echo "==> [2/2] Deploy Dynamo with RL-Scaling image (${RELEASE_VERSION})"
bash "${SCRIPT_DIR}/deploy-dynamo.sh" "$@"

echo
echo "Done. Sanity check:"
echo "  kubectl -n dynamo get deploy,svc,dgd,dgdsa"
echo "  kubectl -n dynamo-system get pods"

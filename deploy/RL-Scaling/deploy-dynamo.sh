#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy-dynamo.sh — Reuse the existing 1.0.1 platform deploy script with a
# *custom RL-Scaling image* by:
#   1) copying the original manifests/ to a temp dir,
#   2) sed-rewriting the image registry/repo references in that copy,
#   3) invoking the existing 01-deploy-dynamo-1.0.1.sh with MANIFEST_DIR
#      and RELEASE_VERSION pointing at the new image.
#
# The upstream deployer + manifests live in the **RL-Scaling repo** (sibling
# to this dynamo checkout). They are located via RL_SCALING_REPO:
#   1. ${RL_SCALING_REPO}                    if set
#   2. ${DYNAMO_DIR}/../RL-Scaling           sibling auto-detect
#   3. ${HOME}/projects/RL-Scaling           last-resort default
#
# Usage:
#   DYNAMO_IMAGE_REGISTRY=ghcr.io/your-name \
#   DYNAMO_IMAGE_REPO=dynamo-vllm-runtime \
#   RELEASE_VERSION=rl-scaling-abc1234 \
#   HF_TOKEN=... NGC_API_KEY=... \
#   ./deploy/RL-Scaling/deploy-dynamo.sh --router
#
# Pulls from a private GHCR? See README §"Image registry & pull secrets".
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── Locate the RL-Scaling repo (holds the upstream deployer + manifests) ─────
if [[ -z "${RL_SCALING_REPO:-}" ]]; then
  for candidate in \
      "${DYNAMO_DIR}/../RL-Scaling" \
      "${HOME}/projects/RL-Scaling" \
      "${HOME}/RL-Scaling"; do
    if [[ -d "${candidate}/tutorial/dynamo-auto-deploy/1.0.1" ]]; then
      RL_SCALING_REPO="$(cd "${candidate}" && pwd)"
      break
    fi
  done
fi
: "${RL_SCALING_REPO:?cannot locate RL-Scaling repo; set RL_SCALING_REPO=/path/to/RL-Scaling}"

UPSTREAM_DEPLOYER="${RL_SCALING_REPO}/tutorial/dynamo-auto-deploy/1.0.1/01-deploy-dynamo-1.0.1.sh"
UPSTREAM_MANIFESTS="${RL_SCALING_REPO}/tutorial/dynamo-auto-deploy/1.0.1/manifests"

[[ -f "${UPSTREAM_DEPLOYER}" ]]  || { echo "deployer not found: ${UPSTREAM_DEPLOYER}"; exit 1; }
[[ -d "${UPSTREAM_MANIFESTS}" ]] || { echo "manifests not found: ${UPSTREAM_MANIFESTS}"; exit 1; }
[[ -x "${UPSTREAM_DEPLOYER}" ]]  || chmod +x "${UPSTREAM_DEPLOYER}" || true

# ── overrides ────────────────────────────────────────────────────────────────
DYNAMO_IMAGE_REGISTRY="${DYNAMO_IMAGE_REGISTRY:-ghcr.io/shqizhang}"
DYNAMO_IMAGE_REPO="${DYNAMO_IMAGE_REPO:-dynamo-vllm-runtime}"
# RELEASE_VERSION is consumed by both the wrapper and the upstream deployer
RELEASE_VERSION="${RELEASE_VERSION:?must set RELEASE_VERSION (e.g. rl-scaling-<sha>)}"

echo "==> dynamo repo:    ${DYNAMO_DIR}"
echo "==> RL-Scaling repo: ${RL_SCALING_REPO}"
echo "==> Image override:  ${DYNAMO_IMAGE_REGISTRY}/${DYNAMO_IMAGE_REPO}:${RELEASE_VERSION}"

# Build a sed-modified manifests copy
TMP_MANIFESTS="$(mktemp -d /tmp/rl-scaling-manifests-XXXXXX)"
trap 'rm -rf "${TMP_MANIFESTS}"' EXIT
cp -r "${UPSTREAM_MANIFESTS}/." "${TMP_MANIFESTS}/"

# Rewrite the image base. We keep the ${RELEASE_VERSION} placeholder intact so
# the upstream renderer (envsubst) still substitutes the tag.
NEW_BASE="${DYNAMO_IMAGE_REGISTRY}/${DYNAMO_IMAGE_REPO}"
find "${TMP_MANIFESTS}" -type f -name '*.yaml' -print0 | xargs -0 \
    sed -i "s|nvcr.io/nvidia/ai-dynamo/vllm-runtime|${NEW_BASE}|g; \
            s|nvcr.io/nvidia/ai-dynamo/mocker-runtime|${NEW_BASE}|g"

echo "==> Patched manifests at ${TMP_MANIFESTS}"
grep -RH "image: " "${TMP_MANIFESTS}" | head -n 5 || true

# Optional: ghcr image pull secret if the package is private. The upstream
# deployer always creates `nvcr-imagepullsecret` (for nvcr.io). If the
# RL-Scaling image lives in a *private* GHCR package, also create a secret
# named `ghcr-imagepullsecret` and reference it in the DGD template.
if [[ -n "${GHCR_USERNAME:-}" && -n "${GHCR_PAT:-}" ]]; then
  : "${NAMESPACE:=dynamo-system}"
  kubectl create namespace "${NAMESPACE}" 2>/dev/null || true
  kubectl -n "${NAMESPACE}" delete secret ghcr-imagepullsecret 2>/dev/null || true
  kubectl -n "${NAMESPACE}" create secret docker-registry ghcr-imagepullsecret \
      --docker-server=ghcr.io \
      --docker-username="${GHCR_USERNAME}" \
      --docker-password="${GHCR_PAT}"
  # Inject the ghcr secret next to nvcr-imagepullsecret in every manifest.
  find "${TMP_MANIFESTS}" -type f -name '*.yaml' -print0 | xargs -0 \
      sed -i "s|- name: nvcr-imagepullsecret|- name: nvcr-imagepullsecret\n          - name: ghcr-imagepullsecret|g"
fi

# Invoke upstream deployer
export MANIFEST_DIR="${TMP_MANIFESTS}"
export RELEASE_VERSION
echo "==> Invoking upstream deployer with MANIFEST_DIR=${MANIFEST_DIR} RELEASE_VERSION=${RELEASE_VERSION}"
exec bash "${UPSTREAM_DEPLOYER}" "$@"

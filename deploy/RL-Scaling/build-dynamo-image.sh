#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# build-dynamo-image.sh — Build & push the RL-Scaling-flavoured Dynamo
# vLLM-runtime image directly from this dynamo checkout (RL-Scaling branch).
#
# Reuses the upstream container build flow (container/render.py + docker build)
# and tags the resulting image so it can be plugged into the existing
# 1.0.1 deploy manifests by overriding only the registry+tag.
#
# This script lives inside the dynamo repo at deploy/RL-Scaling/.
# Therefore DYNAMO_DIR is auto-detected as the repo root — no env var needed.
#
# Output image name pattern:
#   ${REGISTRY}/${IMAGE_REPO}:${IMAGE_TAG}
# Default:
#   ghcr.io/<gh-user>/dynamo-vllm-runtime:rl-scaling-<short-sha>
#
# Usage (local):
#   REGISTRY=ghcr.io/<you> IMAGE_REPO=dynamo-vllm-runtime PUSH=true \
#     ./deploy/RL-Scaling/build-dynamo-image.sh
#
# In CI (.github/workflows/rl-scaling-build.yml) the equivalent steps are
# inlined for cache-friendly buildx; this script is for manual / dev use.
#
# Prereqs: docker, python3 (with pyyaml + jinja2 for render.py), git.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repo root = dynamo/  (this script lives at dynamo/deploy/RL-Scaling/)
DYNAMO_DIR="${DYNAMO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

REGISTRY="${REGISTRY:-ghcr.io/shqizhang}"
IMAGE_REPO="${IMAGE_REPO:-dynamo-vllm-runtime}"
FRAMEWORK="${FRAMEWORK:-vllm}"
TARGET="${TARGET:-runtime}"
CUDA_VERSION="${CUDA_VERSION:-12.9}"
PLATFORM="${PLATFORM:-amd64}"
PUSH="${PUSH:-false}"

[[ -f "${DYNAMO_DIR}/container/render.py" ]] || {
  echo "ERROR: container/render.py not found under DYNAMO_DIR=${DYNAMO_DIR}"
  echo "       This script must run from inside a dynamo checkout."
  exit 1
}

cd "${DYNAMO_DIR}"
SHORT_SHA="$(git rev-parse --short HEAD)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
IMAGE_TAG="${IMAGE_TAG:-rl-scaling-${SHORT_SHA}}"
FULL_IMAGE="${REGISTRY}/${IMAGE_REPO}:${IMAGE_TAG}"

echo "==> Branch: ${BRANCH}"
echo "==> Image:  ${FULL_IMAGE}"
[[ "${BRANCH}" == "RL-Scaling" ]] || echo "    WARN: not on RL-Scaling branch"

echo "==> Render Dockerfile (${FRAMEWORK}/${TARGET}, cuda=${CUDA_VERSION})"
python3 container/render.py \
    --framework "${FRAMEWORK}" \
    --target "${TARGET}" \
    --cuda-version "${CUDA_VERSION}" \
    --platform "${PLATFORM}" \
    --output-short-filename

DOCKERFILE="container/rendered.Dockerfile"
[[ -f "${DOCKERFILE}" ]] || { echo "Dockerfile not produced at ${DOCKERFILE}"; exit 1; }

echo "==> docker build"
docker build \
    --platform "linux/${PLATFORM}" \
    -f "${DOCKERFILE}" \
    -t "${FULL_IMAGE}" \
    --label "rl-scaling.branch=${BRANCH}" \
    --label "rl-scaling.sha=$(git rev-parse HEAD)" \
    .

if [[ "${PUSH}" == "true" ]]; then
  echo "==> docker push ${FULL_IMAGE}"
  docker push "${FULL_IMAGE}"
fi

echo
echo "Built: ${FULL_IMAGE}"
echo "Use it with deploy-dynamo.sh by setting:"
echo "  export DYNAMO_IMAGE_REGISTRY=${REGISTRY}"
echo "  export DYNAMO_IMAGE_REPO=${IMAGE_REPO}"
echo "  export RELEASE_VERSION=${IMAGE_TAG}"

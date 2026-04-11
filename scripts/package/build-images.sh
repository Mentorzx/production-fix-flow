#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CPU_IMAGE="${PFF_DOCKER_IMAGE_CPU:-pff:cpu}"
CUDA_IMAGE="${PFF_DOCKER_IMAGE_CUDA:-pff:cuda}"
TARGET="${1:-all}"

build_image() {
  local accelerator="$1"
  local tag="$2"
  local stage="runtime-${accelerator}"

  echo "Construindo imagem ${tag} (acelerador=${accelerator})"
  docker build \
    --build-arg "PFF_ACCELERATOR=${accelerator}" \
    --target "${stage}" \
    -t "${tag}" \
    "${ROOT_DIR}"
}

case "${TARGET}" in
  cpu)
    build_image "cpu" "${CPU_IMAGE}"
    ;;
  cuda|gpu)
    build_image "cuda" "${CUDA_IMAGE}"
    ;;
  all)
    build_image "cpu" "${CPU_IMAGE}"
    build_image "cuda" "${CUDA_IMAGE}"
    ;;
  *)
    echo "Uso: $0 [cpu|cuda|all]" >&2
    exit 1
    ;;
esac

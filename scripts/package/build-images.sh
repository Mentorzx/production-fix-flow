#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CPU_IMAGE="${PFF_DOCKER_IMAGE_CPU:-pff:cpu}"
CUDA_IMAGE="${PFF_DOCKER_IMAGE_CUDA:-pff:cuda}"
TOOLS_IMAGE="${PFF_DOCKER_IMAGE_TOOLS:-pff:tools}"
TEST_IMAGE="${PFF_DOCKER_IMAGE_TEST:-pff:test}"
TARGET="${1:-cpu}"

if [[ -z "${DOCKER_BUILDKIT:-}" ]]; then
  if docker buildx version >/dev/null 2>&1; then
    export DOCKER_BUILDKIT=1
  else
    export DOCKER_BUILDKIT=0
  fi
fi

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

build_tools_image() {
  echo "Construindo imagem ${TOOLS_IMAGE} (target=tools)"
  docker build \
    --build-arg "PFF_ACCELERATOR=cpu" \
    --target tools \
    -t "${TOOLS_IMAGE}" \
    "${ROOT_DIR}"
}

build_test_image() {
  echo "Construindo imagem ${TEST_IMAGE} (target=test)"
  docker build \
    --build-arg "PFF_ACCELERATOR=cpu" \
    --target test \
    -t "${TEST_IMAGE}" \
    "${ROOT_DIR}"
}

show_sizes() {
  local -a images=("$@")

  echo
  echo "Tamanhos das imagens geradas:"
  for image in "${images[@]}"; do
    local bytes
    bytes="$(docker image inspect "${image}" --format '{{.Size}}' 2>/dev/null || true)"
    if [[ -n "${bytes}" ]]; then
      awk -v image="${image}" -v bytes="${bytes}" \
        'BEGIN { printf "%s %.2f GB\n", image, bytes / 1073741824 }'
    fi
  done
}

case "${TARGET}" in
  cpu)
    build_image "cpu" "${CPU_IMAGE}"
    show_sizes "${CPU_IMAGE}"
    ;;
  cuda|gpu)
    build_image "cuda" "${CUDA_IMAGE}"
    show_sizes "${CUDA_IMAGE}"
    ;;
  tools)
    build_tools_image
    show_sizes "${TOOLS_IMAGE}"
    ;;
  test)
    build_test_image
    show_sizes "${TEST_IMAGE}"
    ;;
  runtime)
    build_image "cpu" "${CPU_IMAGE}"
    build_image "cuda" "${CUDA_IMAGE}"
    show_sizes "${CPU_IMAGE}" "${CUDA_IMAGE}"
    ;;
  all)
    build_image "cpu" "${CPU_IMAGE}"
    build_image "cuda" "${CUDA_IMAGE}"
    build_tools_image
    build_test_image
    show_sizes "${CPU_IMAGE}" "${CUDA_IMAGE}" "${TOOLS_IMAGE}" "${TEST_IMAGE}"
    ;;
  *)
    echo "Uso: $0 [cpu|cuda|runtime|tools|test|all]" >&2
    exit 1
    ;;
esac

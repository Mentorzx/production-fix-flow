#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CPU_IMAGE="${PFF_DOCKER_IMAGE_CPU:-pff:cpu}"
CUDA_IMAGE="${PFF_DOCKER_IMAGE_CUDA:-pff:cuda}"
BUILD_TARGET="${PFF_SMOKE_BUILD_TARGET:-cpu}"
RUN_GPU="${PFF_SMOKE_RUN_GPU:-auto}"
REQUIRE_GPU="${PFF_SMOKE_REQUIRE_GPU:-0}"
SMOKE_WORK_DIR="${PFF_SMOKE_WORK_DIR:-}"
SMOKE_KEEP_WORK_DIR="${PFF_SMOKE_KEEP_WORK_DIR:-0}"
SMOKE_DATA_DIR="${PFF_SMOKE_DATA_DIR:-}"
SMOKE_LOGS_DIR="${PFF_SMOKE_LOGS_DIR:-}"
SMOKE_OUTPUTS_DIR="${PFF_SMOKE_OUTPUTS_DIR:-}"

docker_supports_gpu() {
  docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'
}

image_exists() {
  docker image inspect "$1" >/dev/null 2>&1
}

should_run_gpu_smoke() {
  if [[ "${REQUIRE_GPU}" == "1" || "${RUN_GPU}" == "1" ]]; then
    return 0
  fi
  if [[ "${RUN_GPU}" == "0" ]]; then
    return 1
  fi
  case "${BUILD_TARGET}" in
    cuda|gpu|runtime|all)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

build_required_images() {
  case "${BUILD_TARGET}" in
    none|skip)
      echo "Build de smoke ignorado: PFF_SMOKE_BUILD_TARGET=${BUILD_TARGET}"
      ;;
    cpu|cuda|gpu|runtime|all)
      "${ROOT_DIR}/scripts/package/build-images.sh" "${BUILD_TARGET}"
      ;;
    *)
      echo "PFF_SMOKE_BUILD_TARGET invalido: ${BUILD_TARGET}" >&2
      echo "Use: none|cpu|cuda|runtime|all" >&2
      exit 2
      ;;
  esac
}

setup_smoke_workspace() {
  if [[ -z "${SMOKE_WORK_DIR}" ]]; then
    SMOKE_WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pff-package-smoke.XXXXXXXX")"
  else
    mkdir -p "${SMOKE_WORK_DIR}"
  fi

  SMOKE_DATA_DIR="${SMOKE_DATA_DIR:-${SMOKE_WORK_DIR}/data}"
  SMOKE_LOGS_DIR="${SMOKE_LOGS_DIR:-${SMOKE_WORK_DIR}/logs}"
  SMOKE_OUTPUTS_DIR="${SMOKE_OUTPUTS_DIR:-${SMOKE_WORK_DIR}/outputs}"
  mkdir -p "${SMOKE_DATA_DIR}" "${SMOKE_LOGS_DIR}" "${SMOKE_OUTPUTS_DIR}"
}

cleanup_smoke_workspace() {
  if [[ "${SMOKE_KEEP_WORK_DIR}" == "1" ]]; then
    echo "Workspace do smoke preservado: ${SMOKE_WORK_DIR}" >&2
    return 0
  fi
  if [[ -n "${SMOKE_WORK_DIR}" && "${SMOKE_WORK_DIR}" == "${TMPDIR:-/tmp}"/pff-package-smoke.* ]]; then
    rm -rf "${SMOKE_WORK_DIR}"
  fi
}

run_cli() {
  local image="$1"
  local accelerator="$2"
  shift 2

  local -a args=(
    run
    --rm
    --user "$(id -u):$(id -g)"
    -e "PFF_ACCELERATOR=${accelerator}"
    -e "SECRET_KEY=package-smoke-secret-key-0123456789"
    -e "API_KEY=package-smoke-api-key"
    -e "POSTGRES_PASSWORD=package-smoke-postgres"
    -e "PFF_HPO_STORAGE_BACKEND=journal"
    -e "PFF_HPO_SMOKE_MODE=1"
    -e "PFF_HPO_DISABLE_DASHBOARD=1"
    -e "PFF_HPO_USE_SYNTHETIC=1"
    -v "${ROOT_DIR}/config:/app/config:ro"
    -v "${SMOKE_DATA_DIR}:/app/data"
    -v "${SMOKE_LOGS_DIR}:/app/logs"
    -v "${SMOKE_OUTPUTS_DIR}:/app/outputs"
  )

  if [[ "${accelerator}" == "cuda" ]]; then
    args+=(--gpus all)
  fi

  docker "${args[@]}" "${image}" "$@"
}

validate_gpu_image() {
  docker run --rm --gpus all --entrypoint python "${CUDA_IMAGE}" -c \
    "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)"
}

main() {
  setup_smoke_workspace
  trap cleanup_smoke_workspace EXIT

  build_required_images

  echo "Smoke CPU: clean deep"
  run_cli "${CPU_IMAGE}" cpu clean deep -y

  echo "Smoke CPU: hpo"
  run_cli "${CPU_IMAGE}" cpu hpo --trials 1 --synthetic-data --no-dashboard --no-update-config --no-bert

  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && docker_supports_gpu; then
    if ! should_run_gpu_smoke; then
      echo "Smoke GPU ignorado. Use PFF_SMOKE_RUN_GPU=1 ou PFF_SMOKE_BUILD_TARGET=runtime para validar CUDA." >&2
      return 0
    fi
    if ! image_exists "${CUDA_IMAGE}"; then
      echo "Imagem CUDA ausente para smoke GPU: ${CUDA_IMAGE}" >&2
      echo "Use PFF_SMOKE_BUILD_TARGET=runtime para gerar CPU+CUDA." >&2
      exit 1
    fi

    echo "Smoke GPU: validacao de CUDA"
    validate_gpu_image

    echo "Smoke GPU: clean deep"
    run_cli "${CUDA_IMAGE}" cuda clean deep -y

    echo "Smoke GPU: hpo"
    run_cli "${CUDA_IMAGE}" cuda hpo --trials 1 --synthetic-data --no-dashboard --no-update-config --no-bert
  else
    if [[ "${REQUIRE_GPU}" == "1" ]]; then
      echo "GPU NVIDIA ausente ou runtime Docker GPU indisponivel." >&2
      exit 1
    fi
    echo "GPU NVIDIA ausente ou runtime Docker GPU indisponivel; smoke GPU ignorado." >&2
  fi
}

main "$@"

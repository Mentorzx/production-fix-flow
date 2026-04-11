#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CPU_IMAGE="${PFF_DOCKER_IMAGE_CPU:-pff:cpu}"
CUDA_IMAGE="${PFF_DOCKER_IMAGE_CUDA:-pff:cuda}"
docker_supports_gpu() {
  docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'
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
    -v "${ROOT_DIR}/config:/app/config"
    -v "${ROOT_DIR}/data:/app/data"
    -v "${ROOT_DIR}/logs:/app/logs"
    -v "${ROOT_DIR}/outputs:/app/outputs"
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
  "${ROOT_DIR}/scripts/package/build-images.sh" all

  echo "Smoke CPU: clean deep"
  run_cli "${CPU_IMAGE}" cpu clean deep -y

  echo "Smoke CPU: hpo"
  run_cli "${CPU_IMAGE}" cpu hpo --trials 1 --synthetic-data --no-dashboard --no-update-config --no-bert

  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && docker_supports_gpu; then
    echo "Smoke GPU: validacao de CUDA"
    validate_gpu_image

    echo "Smoke GPU: clean deep"
    run_cli "${CUDA_IMAGE}" cuda clean deep -y

    echo "Smoke GPU: hpo"
    run_cli "${CUDA_IMAGE}" cuda hpo --trials 1 --synthetic-data --no-dashboard --no-update-config --no-bert
  else
    echo "GPU NVIDIA ausente ou runtime Docker GPU indisponivel; smoke GPU ignorado." >&2
  fi
}

main "$@"

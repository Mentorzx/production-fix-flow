#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

CPU_IMAGE="${PFF_DOCKER_IMAGE_CPU:-pff:cpu}"
CUDA_IMAGE="${PFF_DOCKER_IMAGE_CUDA:-pff:cuda}"
TOOLS_IMAGE="${PFF_DOCKER_IMAGE_TOOLS:-pff:tools}"
TEST_IMAGE="${PFF_DOCKER_IMAGE_TEST:-pff:test}"

CPU_BUDGET_GB="${PFF_IMAGE_BUDGET_CPU_GB:-3}"
CUDA_BUDGET_GB="${PFF_IMAGE_BUDGET_CUDA_GB:-10}"
TOOLS_BUDGET_GB="${PFF_IMAGE_BUDGET_TOOLS_GB:-15}"
TEST_BUDGET_GB="${PFF_IMAGE_BUDGET_TEST_GB:-16.5}"
DEFAULT_BUDGET_GB="${PFF_IMAGE_BUDGET_DEFAULT_GB:-}"

BASELINE_FILE=""
OUTPUT_FILE=""
FAIL_ON_BUDGET=0

usage() {
  cat <<'EOF'
Uso: measure-image-sizes.sh [--baseline arquivo.tsv] [--output arquivo.tsv] [--fail-on-budget] [imagem...]

Mede imagens Docker existentes e, opcionalmente, compara com um baseline TSV.
Formato do baseline: image<TAB>bytes, com ou sem linha de cabecalho.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline)
      BASELINE_FILE="${2:?missing baseline path}"
      shift 2
      ;;
    --output)
      OUTPUT_FILE="${2:?missing output path}"
      shift 2
      ;;
    --fail-on-budget)
      FAIL_ON_BUDGET=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Opcao desconhecida: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ -n "${OUTPUT_FILE}" ]]; then
  exec >"${OUTPUT_FILE}"
fi

if [[ $# -eq 0 ]]; then
  set -- "${CPU_IMAGE}" "${CUDA_IMAGE}" "${TOOLS_IMAGE}" "${TEST_IMAGE}"
fi

baseline_bytes_for() {
  local image="$1"

  if [[ -z "${BASELINE_FILE}" || ! -f "${BASELINE_FILE}" ]]; then
    return 0
  fi

  awk -F '\t' -v image="${image}" '
    $1 == image && $2 ~ /^[0-9]+$/ { print $2; found = 1; exit }
    END { if (!found) exit 0 }
  ' "${BASELINE_FILE}"
}

budget_gb_for() {
  local image="$1"

  case "${image}" in
    "${CPU_IMAGE}") echo "${CPU_BUDGET_GB}" ;;
    "${CUDA_IMAGE}") echo "${CUDA_BUDGET_GB}" ;;
    "${TOOLS_IMAGE}") echo "${TOOLS_BUDGET_GB}" ;;
    "${TEST_IMAGE}") echo "${TEST_BUDGET_GB}" ;;
    *) echo "${DEFAULT_BUDGET_GB}" ;;
  esac
}

within_budget() {
  local bytes="$1"
  local budget_gb="$2"

  awk -v bytes="${bytes}" -v budget="${budget_gb}" 'BEGIN { exit !((bytes / 1073741824) <= budget) }'
}

budget_failed=0

printf "image\tstatus\tbytes\tgib\tbaseline_bytes\tdelta_gib\tdelta_pct\tbudget_gib\tbudget_status\n"

for image in "$@"; do
  bytes="$(docker image inspect "${image}" --format '{{.Size}}' 2>/dev/null || true)"
  baseline_bytes="$(baseline_bytes_for "${image}")"
  budget_gb="$(budget_gb_for "${image}")"

  if [[ -z "${bytes}" ]]; then
    printf "%s\tmissing\t\t\t%s\t\t\t%s\tunknown\n" "${image}" "${baseline_bytes}" "${budget_gb}"
    continue
  fi

  awk \
    -v image="${image}" \
    -v bytes="${bytes}" \
    -v baseline="${baseline_bytes}" \
    -v budget="${budget_gb}" '
    BEGIN {
      gib = bytes / 1073741824
      delta_gib = ""
      delta_pct = ""
      if (baseline ~ /^[0-9]+$/ && baseline > 0) {
        delta_gib = (bytes - baseline) / 1073741824
        delta_pct = ((bytes - baseline) / baseline) * 100
      }
      budget_status = "unknown"
      if (budget != "") {
        budget_status = (gib <= budget) ? "pass" : "fail"
      }
      printf "%s\tpresent\t%s\t%.2f\t%s\t%s\t%s\t%s\t%s\n",
        image,
        bytes,
        gib,
        baseline,
        (delta_gib == "" ? "" : sprintf("%.2f", delta_gib)),
        (delta_pct == "" ? "" : sprintf("%.1f", delta_pct)),
        budget,
        budget_status
    }
  '
  if [[ "${FAIL_ON_BUDGET}" == "1" && -n "${budget_gb}" ]] && ! within_budget "${bytes}" "${budget_gb}"; then
    echo "Image budget exceeded: image=${image} bytes=${bytes} budget_gib=${budget_gb}" >&2
    budget_failed=1
  fi
done

if [[ "${FAIL_ON_BUDGET}" == "1" && "${budget_failed}" != "0" ]]; then
  exit 1
fi

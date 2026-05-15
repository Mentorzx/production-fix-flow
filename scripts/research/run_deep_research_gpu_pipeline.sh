#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="deep_research_advisor_20260506"
STUDY_NAME="deep_research_advisor_real50_gpu_20260506"
OUTPUT_DIR="${ROOT_DIR}/outputs/research/${RUN_ID}"
AUDIT_OUTPUT="${ROOT_DIR}/outputs/benches/search_space_advisor/deep_research_audit_20260506.json"
PAIRED_BENCHMARK_OUTPUT="${ROOT_DIR}/outputs/benches/search_space_advisor/paired_benchmark_50.json"
REPORT_MD="${OUTPUT_DIR}/deep-research-report-abnt.md"
REPORT_HTML="${OUTPUT_DIR}/deep-research-report-abnt.html"
REPORT_PDF="${OUTPUT_DIR}/deep-research-report-abnt.pdf"
DOCS_REPORT_MD="${ROOT_DIR}/docs/deep-research-report-abnt.md"
DOCS_REPORT_HTML="${ROOT_DIR}/docs/deep-research-report-abnt.html"
DOCS_REPORT_PDF="${ROOT_DIR}/docs/deep-research-report-abnt.pdf"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "$(dirname "${AUDIT_OUTPUT}")"

cd "${ROOT_DIR}"

./pff hpo \
  --trials 50 \
  --no-update-config \
  --no-bert \
  --no-dashboard \
  --study-name "${STUDY_NAME}"

if ! docker image inspect pff:tools >/dev/null 2>&1; then
  ./scripts/package/build-images.sh tools
fi

docker run --rm \
  -v "${ROOT_DIR}:/workspace" \
  -w /workspace \
  --entrypoint /app/.venv/bin/python \
  pff:tools \
  scripts/benchmarks/search_space_advisor_audit.py \
  --input outputs/.cache/hpo/dashboard_data.json \
  --output "${AUDIT_OUTPUT#${ROOT_DIR}/}" \
  --min-prefix 8

docker run --rm \
  -v "${ROOT_DIR}:/workspace" \
  -w /workspace \
  --entrypoint /app/.venv/bin/python \
  pff:tools \
  scripts/benchmarks/search_space_advisor_paired_benchmark.py \
  --trials 50 \
  --output "${PAIRED_BENCHMARK_OUTPUT#${ROOT_DIR}/}"

docker run --rm \
  -v "${ROOT_DIR}:/workspace" \
  -w /workspace \
  --entrypoint /app/.venv/bin/python \
  pff:tools \
  scripts/research/deep_research_advisor_artifacts.py \
  --dashboard outputs/.cache/hpo/dashboard_data.json \
  --audit "${AUDIT_OUTPUT#${ROOT_DIR}/}" \
  --hpo-summary outputs/optimization/kg_dslfm/hpo_summary.json \
  --paired-benchmark "${PAIRED_BENCHMARK_OUTPUT#${ROOT_DIR}/}" \
  --output-dir "${OUTPUT_DIR#${ROOT_DIR}/}"

python3 scripts/render_academic_pdf.py \
  "${REPORT_MD#${ROOT_DIR}/}" \
  --html-output "${REPORT_HTML#${ROOT_DIR}/}" \
  --pdf-output "${REPORT_PDF#${ROOT_DIR}/}" \
  --author "Alex de Lira Neto" \
  --institution "Universidade Federal da Bahia (UFBA)"

cp "${REPORT_MD}" "${DOCS_REPORT_MD}"
cp "${REPORT_HTML}" "${DOCS_REPORT_HTML}"
cp "${REPORT_PDF}" "${DOCS_REPORT_PDF}"

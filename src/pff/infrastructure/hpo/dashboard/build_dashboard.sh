#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# PFF Dashboard Pure ESM Build Script
# v18.3.0 - SOTA ESM + Import Maps + Content-Hash Cache
# guard v4 - lint + typecheck + size guard + source-hash skip

DASHBOARD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="$DASHBOARD_DIR/dist"
BUILD_DIR="$DASHBOARD_DIR/build"
REPO_ROOT="$(cd "$DASHBOARD_DIR/../../../../.." && pwd)"
WASM_CRATE_DIR="$REPO_ROOT/src/pff_rust/search_core_wasm"
WASM_TARGET_DIR="$REPO_ROOT/outputs/.cache/hpo/dashboard_wasm_target"
ENTRY_FILE="$DASHBOARD_DIR/static/js/main.jsx"
CSS_IN_FILE="$DASHBOARD_DIR/static/css/input.css"
CSS_OUT_FILE="$DIST_DIR/dashboard.css"
BUILD_ID_FILE="$DIST_DIR/build_id.txt"
BIN_DIR="$DASHBOARD_DIR/node_modules/.bin"
LOCK_FILE="$DASHBOARD_DIR/package-lock.json"
LOCK_HASH_FILE="$BUILD_DIR/npm_lock.sha256"
SOURCE_HASH_FILE="$BUILD_DIR/source.sha256"
MAX_BUNDLE_BYTES="${MAX_BUNDLE_BYTES:-950000}"
FORCE_BUILD="${PFF_DASHBOARD_FORCE_BUILD:-0}"

mkdir -p "$DIST_DIR" "$BUILD_DIR"
mkdir -p "$WASM_TARGET_DIR"

DASHBOARD_VERSION_CURRENT="$(grep -o '"version": "[^"]*"' "$DASHBOARD_DIR/package.json" | cut -d'"' -f4 || echo "desconhecida")"
LAST_BUILT_VERSION="$(grep -o '"version":"[^"]*"' "$DIST_DIR/version.json" 2>/dev/null | head -n1 | cut -d'"' -f4 || true)"
echo "======================================================================"
echo "[REMINDER] Verifique se a versao do dashboard em package.json foi incrementada corretamente antes do build."
echo "[REMINDER] Versao atual detectada: ${DASHBOARD_VERSION_CURRENT}"
if [ -n "${LAST_BUILT_VERSION}" ]; then
    echo "[REMINDER] Ultima versao buildada: ${LAST_BUILT_VERSION}"
    if [ "${DASHBOARD_VERSION_CURRENT}" = "${LAST_BUILT_VERSION}" ]; then
        echo "[WARNING] A versao atual e igual a ultima buildada. Considere incrementar antes de publicar."
    fi
fi
echo "======================================================================"

# --- Content-hash cache: skip entire build if sources unchanged ---
_compute_source_hash() {
    {
        find "$DASHBOARD_DIR/static" -type f \( -name '*.jsx' -o -name '*.js' -o -name '*.css' -o -name '*.html' \) \
            -exec sha256sum {} + 2>/dev/null
        find "$WASM_CRATE_DIR/src" -type f -name '*.rs' -exec sha256sum {} + 2>/dev/null
        sha256sum "$WASM_CRATE_DIR/Cargo.toml" 2>/dev/null || true
        sha256sum "$WASM_CRATE_DIR/Cargo.lock" 2>/dev/null || true
        sha256sum "$DASHBOARD_DIR/package.json" 2>/dev/null || true
        sha256sum "$DASHBOARD_DIR/build_dashboard.sh" 2>/dev/null || true
    } | sort | sha256sum | awk '{print $1}'
}

_build_wasm_core() {
    if [ ! -f "$WASM_CRATE_DIR/Cargo.toml" ]; then
        echo "[ERROR] WASM crate not found at $WASM_CRATE_DIR"
        exit 1
    fi

    if ! command -v cargo >/dev/null 2>&1; then
        echo "[ERROR] cargo not found; install Rust toolchain to build search_core wasm."
        exit 1
    fi

    if ! command -v rustup >/dev/null 2>&1; then
        echo "[ERROR] rustup not found; cannot ensure wasm32 target."
        exit 1
    fi

    if ! command -v wasm-bindgen >/dev/null 2>&1; then
        echo "[ERROR] wasm-bindgen CLI not found."
        echo "[ERROR] Install with: cargo install wasm-bindgen-cli"
        exit 1
    fi

    if ! rustup target list --installed | grep -q '^wasm32-unknown-unknown$'; then
        echo "[INFO] Installing Rust target wasm32-unknown-unknown..."
        rustup target add wasm32-unknown-unknown
    fi

    echo "[INFO] Building Rust WASM search core..."
    cargo build \
        --manifest-path "$WASM_CRATE_DIR/Cargo.toml" \
        --target wasm32-unknown-unknown \
        --release \
        --target-dir "$WASM_TARGET_DIR"

    local wasm_binary="$WASM_TARGET_DIR/wasm32-unknown-unknown/release/search_core_wasm.wasm"
    if [ ! -f "$wasm_binary" ]; then
        echo "[ERROR] Expected wasm binary not found: $wasm_binary"
        exit 1
    fi

    wasm-bindgen \
        --target web \
        --out-dir "$DIST_DIR" \
        --out-name search_core \
        "$wasm_binary"
}

if [ "$FORCE_BUILD" != "1" ] && [ -f "$DIST_DIR/dashboard.js" ] && [ -f "$SOURCE_HASH_FILE" ]; then
    current_source_hash="$(_compute_source_hash)"
    cached_source_hash="$(cat "$SOURCE_HASH_FILE")"
    if [ "$current_source_hash" = "$cached_source_hash" ]; then
        echo "[SKIP] Dashboard sources/package unchanged — using cached build"
        echo "[SKIP] Versao mantida: ${DASHBOARD_VERSION_CURRENT}"
        exit 0
    fi
fi

if [ ! -f "$LOCK_FILE" ]; then
    echo "[INFO] package-lock.json ausente, instalando com npm install..."
    cd "$DASHBOARD_DIR" && npm install --silent --no-audit --no-fund
else
    current_hash="$(sha256sum "$LOCK_FILE" | awk '{print $1}')"
    cached_hash=""
    if [ -f "$LOCK_HASH_FILE" ]; then
        cached_hash="$(cat "$LOCK_HASH_FILE")"
    fi
    if [ ! -d "$DASHBOARD_DIR/node_modules" ] || [ "$current_hash" != "$cached_hash" ]; then
        echo "[INFO] Installing dashboard dependencies (npm ci)..."
        cd "$DASHBOARD_DIR" && npm ci --silent --no-audit --no-fund
    fi
    echo "$current_hash" > "$LOCK_HASH_FILE"
fi

export NODE_ENV=production
export CI=true

_build_wasm_core

echo "[INFO] Running TypeScript checks (guard v4)..."
cd "$DASHBOARD_DIR"
npm run typecheck

echo "[INFO] Running ESLint (guard v4)..."
npm run lint

echo "[INFO] Building Tailwind CSS..."
BROWSERSLIST_IGNORE_OLD_DATA=1 "$BIN_DIR/tailwindcss" \
    -i "$CSS_IN_FILE" \
    -o "$CSS_OUT_FILE" \
    --minify

echo "[INFO] Building dashboard bundle (Standalone Bundle)..."
"$BIN_DIR/esbuild" "$ENTRY_FILE" \
    --bundle \
    --minify \
    --sourcemap \
    --format=esm \
    --target=es2020 \
    --loader:.jsx=jsx \
    --loader:.js=jsx \
    --outfile="$DIST_DIR/dashboard.js" \
    --define:process.env.NODE_ENV='"production"'

BUILD_ID="$(date +%s)"
echo "$BUILD_ID" > "$BUILD_ID_FILE"
echo "[INFO] Build id: $BUILD_ID"

# Generate version info for frontend
VERSION_FILE="$DIST_DIR/version.json"
DASHBOARD_VERSION="$(grep -o '"version": "[^"]*"' "$DASHBOARD_DIR/package.json" | cut -d'"' -f4)"
echo "{\"version\":\"$DASHBOARD_VERSION\",\"buildId\":\"$BUILD_ID\",\"buildDate\":\"$(date -Iseconds)\"}" > "$VERSION_FILE"

echo "[SUCCESS] Dashboard build complete: dist/dashboard.js"
ls -lh "$DIST_DIR/dashboard.js"
bundle_bytes="$(wc -c < "$DIST_DIR/dashboard.js" | tr -d ' ')"
if [ "$bundle_bytes" -gt "$MAX_BUNDLE_BYTES" ]; then
    echo "[ERROR] Bundle size guard failed: ${bundle_bytes} bytes (limit ${MAX_BUNDLE_BYTES})"
    exit 1
fi

# Persist source hash so subsequent runs skip the build when sources are unchanged
_compute_source_hash > "$SOURCE_HASH_FILE"

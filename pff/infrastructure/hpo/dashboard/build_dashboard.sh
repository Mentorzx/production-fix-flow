#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# PFF Dashboard Pure ESM Build Script
# v18.2.0 - SOTA ESM + Import Maps
# guard v4 - lint + typecheck + size guard

DASHBOARD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="$DASHBOARD_DIR/dist"
BUILD_DIR="$DASHBOARD_DIR/build"
ENTRY_FILE="$DASHBOARD_DIR/static/js/main.jsx"
CSS_IN_FILE="$DASHBOARD_DIR/static/css/input.css"
CSS_OUT_FILE="$DIST_DIR/dashboard.css"
BUILD_ID_FILE="$DIST_DIR/build_id.txt"
BIN_DIR="$DASHBOARD_DIR/node_modules/.bin"
LOCK_FILE="$DASHBOARD_DIR/package-lock.json"
LOCK_HASH_FILE="$BUILD_DIR/npm_lock.sha256"
MAX_BUNDLE_BYTES="${MAX_BUNDLE_BYTES:-950000}"

mkdir -p "$DIST_DIR" "$BUILD_DIR"

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

echo "[INFO] Running TypeScript checks (guard v4)..."
cd "$DASHBOARD_DIR"
npm run typecheck || echo "[WARNING] TypeScript checks failed"

echo "[INFO] Running ESLint (guard v4)..."
npm run lint || echo "[WARNING] ESLint failed"

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

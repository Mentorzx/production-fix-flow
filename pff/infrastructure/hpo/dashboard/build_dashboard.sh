#!/bin/bash
set -e

# PFF Dashboard Pure ESM Build Script
# v18.2.0 - SOTA ESM + Import Maps
# guard v3 - lint obrigatório

DASHBOARD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="$DASHBOARD_DIR/dist"
BUILD_DIR="$DASHBOARD_DIR/build"
ENTRY_FILE="$DASHBOARD_DIR/static/js/main.jsx"
CSS_IN_FILE="$DASHBOARD_DIR/static/css/input.css"
CSS_OUT_FILE="$DIST_DIR/dashboard.css"
BUILD_ID_FILE="$DIST_DIR/build_id.txt"
BIN_DIR="$DASHBOARD_DIR/node_modules/.bin"

mkdir -p "$DIST_DIR" "$BUILD_DIR"

if [ ! -f "$BIN_DIR/esbuild" ] || [ ! -f "$BIN_DIR/tailwindcss" ]; then
    echo "[INFO] Installing dashboard dependencies (npm)..."
    cd "$DASHBOARD_DIR" && npm install --silent --no-audit --no-fund
fi

echo "[INFO] Building Tailwind CSS..."
cd "$DASHBOARD_DIR"
BROWSERSLIST_IGNORE_OLD_DATA=1 "$BIN_DIR/tailwindcss" \
    -i "$CSS_IN_FILE" \
    -o "$CSS_OUT_FILE" \
    --minify

echo "[INFO] Running ESLint (guard v3)..."
npm run lint

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

echo "[SUCCESS] Dashboard build complete: dist/dashboard.js"
ls -lh "$DIST_DIR/dashboard.js"

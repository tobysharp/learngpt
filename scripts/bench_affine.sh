#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR="$ROOT_DIR/infergpt/build"
BENCH="$BUILD_DIR/bench/affine_bench"

cmake --build "$BUILD_DIR" --target affine_bench --parallel
exec "$BENCH" "$@"
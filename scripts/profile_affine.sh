#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR="$ROOT_DIR/infergpt/build"
BENCH="$BUILD_DIR/bench/affine_bench"
PROFILE_DIR=${PROFILE_DIR:-/tmp/affine_bench_perf}
DATA_FILE="$PROFILE_DIR/perf.data"
REPORT_FILE="$PROFILE_DIR/perf-report.txt"
EVENT=${PERF_EVENT:-cycles:u}

cmake --build "$BUILD_DIR" --target affine_bench --parallel
rm -rf "$PROFILE_DIR"
mkdir -p "$PROFILE_DIR"

perf record -g -e "$EVENT" -o "$DATA_FILE" -- "$BENCH" "$@" | tee "$PROFILE_DIR/benchmark.txt"

perf report --stdio --no-children --percent-limit 0.5 --sort symbol -i "$DATA_FILE" > "$REPORT_FILE"

echo
echo "Profile summary (top perf symbols)"
awk '
	BEGIN { shown = 0; in_table = 0 }
	/^# Overhead/ { in_table = 1; print; next }
	in_table && /^# / { print; next }
	in_table && /^$/ {
		if (shown > 0) exit
		next
	}
	in_table && $1 ~ /^[0-9.]+%$/ {
		print
		++shown
		if (shown >= 12) exit
	}
' "$REPORT_FILE"

echo
echo "Saved full profile to $REPORT_FILE"
echo "Saved benchmark output to $PROFILE_DIR/benchmark.txt"
echo "Saved raw perf data to $DATA_FILE"
echo "Useful next commands:"
echo "  perf annotate --stdio -i $DATA_FILE 'Affine<float>::operator()(Matrix<float> const&) const'"
echo "  perf report -i $DATA_FILE"
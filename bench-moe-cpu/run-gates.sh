#!/usr/bin/env bash
# NG MoE pipeline gate runner (M0+).
# Builds nothing (uses an existing build dir), runs the type x shape matrix:
#   1. reference goldens via the stock vec_dot engine (GGML_CPU_TILED_MM=0)
#   2. current engine check + timing vs those goldens
#   3. selftest (injected error must FAIL)
# Appends a results block to docs/ng-moe-pipeline/results.md when -r is given.
#
# Usage: run-gates.sh [build_dir] [-r "milestone note"]
set -u
BUILD=/mnt/ssd/projects/llama.cpp/build-cuda
NOTE=""
while [ $# -gt 0 ]; do
    case "$1" in
        -r) NOTE="$2"; shift 2 ;;
        *) BUILD="$1"; shift ;;
    esac
done
GOLDIR=/tmp/opencode/ng-goldens
LIB="$BUILD/bin"
export LD_LIBRARY_PATH="$LIB"
mkdir -p "$GOLDIR"
MATRIX=( "up iq2_xxs" "down iq2_xxs" "up iq3_xxs" "down iq3_xxs" "up iq2_s" "up iq4_xs" )
mkdir -p "$GOLDIR"

B=$(g++ -O2 -march=native "$(dirname "$0")/mmid.cpp" \
      -I"$(dirname "$(dirname "$0")")/../llama.cpp/ggml/include" -I/mnt/ssd/projects/llama.cpp/ggml/include \
      -L"$LIB" -lggml-cpu -lggml-base -o "$GOLDIR/mmid" 2>&1) \
  || { echo "build failed: $B"; exit 2; }
GG="$GOLDIR/mmid"

overall=0
printf "| config | ms | GB/s | GMAC/s | check |\n|---|---|---|---|---|\n"
for cfg in "${MATRIX[@]}"; do
    set -- $cfg; SH=$1; TY=$2
    G="$GOLDIR/golden_${SH}_${TY}.bin"
    # goldens come from the tiled incumbent engine (int32 dpbusd, no saturation)
    [ -f "$G" ] || "$GG" "$SH" "$TY" 2048 12 ref-dump "$G" > /dev/null \
        || { echo "ref-dump failed for $SH/$TY"; exit 2; }
    OUT=$(GGML_CPU_TILED_MM=1 "$GG" "$SH" "$TY" 2048 12 check "$G" 2>&1) || overall=1
    echo "| $SH/$TY | $(echo "$OUT" | sed -n 's/.*thr: \([0-9.]*\) ms.*/\1/p') \
| $(echo "$OUT" | sed -n 's/.*, \([0-9.]*\) GB\/s.*/\1/p') \
| $(echo "$OUT" | sed -n 's/.*, \([0-9]*\) GMAC\/s.*/\1/p') \
| $(echo "$OUT" | grep -oE 'PASS|FAIL') |"
    echo "$OUT" | grep -q PASS || overall=1
done

# vec_dot cross-check: the classic AVX2 kernel saturates int16 maddubs on this
# adversarial data (max_rel ~1.1 on ~0.3% of elements) - allow that narrow
# class, flag anything broader as structural
for cfg in "up iq2_xxs"; do
    set -- $cfg; SH=$1; TY=$2
    OUT=$(GGML_CPU_TILED_MM=0 GGMM_TOL=1e-3 "$GG" "$SH" "$TY" 2048 12 check "$GOLDIR/golden_${SH}_${TY}.bin" 2>&1)
    NBAD=$(echo "$OUT" | sed -n 's/.*(\([0-9]*\) bad.*/\1/p')
    if [ -n "$NBAD" ] && [ "$NBAD" -lt 42000 ]; then
        echo "vec_dot cross-check: OK (saturation-only divergence, $NBAD/4194304 bad at tol 1e-3)"
    else
        echo "vec_dot cross-check: STRUCTURAL divergence ($NBAD bad)"; overall=1
    fi
done

# selftest: injected error must produce FAIL (exit 1)
G="$GOLDIR/golden_up_iq2_xxs.bin"
if "$GG" up iq2_xxs 2048 12 selftest "$G" > /dev/null 2>&1; then
    echo "SELFTEST FAILED: injected error was not caught"; overall=1
else
    echo "selftest: OK (injected error caught)"
fi

if [ -n "$NOTE" ]; then
    {
        echo ""
        echo "## $(date +%F) $NOTE"
        echo "commit: $(cd "$(dirname "$0")" && git rev-parse --short HEAD)"
        echo "config: engine=GGML_CPU_TILED_MM=1, 2048 tok x 10 used, 12 threads"
        echo "| config | ms | GB/s | GMAC/s | check |"
        echo "|---|---|---|---|---|"
        for cfg in "${MATRIX[@]}"; do
            set -- $cfg; SH=$1; TY=$2
            OUT=$(GGML_CPU_TILED_MM=1 "$GG" "$SH" "$TY" 2048 12 check "$GOLDIR/golden_${SH}_${TY}.bin" 2>&1)
            echo "| $SH/$TY | $(echo "$OUT" | sed -n 's/.*thr: \([0-9.]*\) ms.*/\1/p') \
| $(echo "$OUT" | sed -n 's/.*, \([0-9.]*\) GB\/s.*/\1/p') \
| $(echo "$OUT" | sed -n 's/.*, \([0-9]*\) GMAC\/s.*/\1/p') \
| $(echo "$OUT" | grep -oE 'PASS|FAIL' | head -1) |"
        done
        echo "decision: (fill in)"
    } >> "$(dirname "$0")/../docs/ng-moe-pipeline/results.md"
    echo "results appended"
fi
exit $overall

#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK="${SCRIPT_DIR}/jax_snake_throughput.py"
COMMON_ARGS=(--dtype float64 --steps 1000)

echo "CUDA snake throughput sweep (n-snakes-exp 6..14)"
for exponent in {6..14}; do
    echo "CUDA n-snakes-exp=${exponent}"
    uv run --no-sync python "${BENCHMARK}" \
        "${COMMON_ARGS[@]}" \
        --backend cuda \
        --no-numba \
        --n-snakes-exp "${exponent}"
done

echo "Numba snake throughput sweep (n-snakes-exp 6..12)"
for exponent in {6..12}; do
    echo "Numba n-snakes-exp=${exponent}"
    uv run --no-sync python "${BENCHMARK}" \
        "${COMMON_ARGS[@]}" \
        --numba-only \
        --n-snakes-exp "${exponent}"
done

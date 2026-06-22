"""Benchmark multi-snake rollout throughput: Numba vs JAX."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import time

import numpy as np

from _jax_snake_common import (
    DEFAULT_DT,
    DEFAULT_N_ELEM,
    DEFAULT_N_SNAKES_EXP,
    DEFAULT_STEPS,
    benchmark_config,
    build_cpu_sim,
    build_jax_sim,
    collect_cpu_state,
    collect_jax_state,
    emit_report,
    max_abs_diff,
    select_device,
    snake_count_from_exponent,
    validate_dtype_for_device,
)

import elastica as ea

try:
    import jax
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "This benchmark requires JAX. Install the optional GPU extra first."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--n-snakes-exp", type=int, default=DEFAULT_N_SNAKES_EXP)
    parser.add_argument("--n-elem", type=int, default=DEFAULT_N_ELEM)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument(
        "--transfer-guard",
        choices=("allow", "log", "disallow", "log_explicit", "disallow_explicit"),
        default="allow",
    )
    parser.add_argument("--log", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert args.n_elem > 1, "n-elem must be greater than 1."
    assert args.steps > 0, "steps must be positive."
    assert args.dt > 0.0, "dt must be positive."
    assert args.warmup_runs >= 0, "warmup-runs must be nonnegative."

    n_snakes = snake_count_from_exponent(args.n_snakes_exp)
    device = select_device(args.backend)
    dtype = np.dtype(np.float32 if args.dtype == "float32" else np.float64)
    validate_dtype_for_device(dtype, device)
    backend_label = "jax-cpu" if device.platform == "cpu" else f"jax-{device.platform}"
    config = benchmark_config(n_snakes=n_snakes, n_elem=args.n_elem, dt=args.dt)
    final_time = np.float64(args.steps * args.dt)

    numba_instantiate_start = time.perf_counter()
    cpu_sim, cpu_rods = build_cpu_sim(**config)
    numba_instantiate_elapsed = time.perf_counter() - numba_instantiate_start
    cpu_stepper = ea.PositionVerlet()
    if args.warmup_runs > 0:
        cpu_warmup_sim, _ = build_cpu_sim(**config)
        cpu_warmup_stepper = ea.PositionVerlet()
        warmup_time_value = np.float64(0.0)
        for _ in range(args.warmup_runs):
            warmup_time_value = np.float64(0.0)
            for _ in range(args.steps):
                warmup_time_value = cpu_warmup_stepper.step(
                    cpu_warmup_sim,
                    warmup_time_value,
                    np.float64(args.dt),
                )

    time_value = np.float64(0.0)
    start = time.perf_counter()
    for _ in range(args.steps):
        time_value = cpu_stepper.step(cpu_sim, time_value, np.float64(args.dt))
    cpu_elapsed = time.perf_counter() - start
    assert np.isclose(time_value, final_time), "CPU rollout did not end at final_time."
    cpu_state = collect_cpu_state(cpu_rods)

    with jax.default_device(device):
        jax_instantiate_start = time.perf_counter()
        jax_sim, jax_block = build_jax_sim(
            device=device,
            device_dtype=dtype,
            **config,
        )
        jax.block_until_ready(jax_block.position_collection_device)
        jax_instantiate_elapsed = time.perf_counter() - jax_instantiate_start
        jax_stepper = ea.PositionVerletGPU()
        guard_context = (
            jax.transfer_guard(args.transfer_guard)
            if args.transfer_guard != "allow"
            else nullcontext()
        )
        with guard_context:
            for _ in range(args.warmup_runs):
                initial_state = dict(jax_block.jax_get_state())
                jax_stepper.integrate(
                    jax_sim,
                    time=np.float64(0.0),
                    final_time=final_time,
                    dt=np.float64(args.dt),
                )
                jax.block_until_ready(jax_block.position_collection_device)
                jax_block.jax_set_state(initial_state)

            start = time.perf_counter()
            jax_stepper.integrate(
                jax_sim,
                time=np.float64(0.0),
                final_time=final_time,
                dt=np.float64(args.dt),
            )
            jax.block_until_ready(jax_block.position_collection_device)
            jax_elapsed = time.perf_counter() - start
        jax_state = collect_jax_state(jax_block, n_snakes)

    report_lines = [
        f"device: {device}",
        f"dtype: {dtype}",
        f"n_snakes: {n_snakes}",
        f"n_elem: {args.n_elem}",
        f"steps: {args.steps}",
        f"dt: {args.dt}",
        f"warmup_runs: {args.warmup_runs}",
        f"transfer_guard: {args.transfer_guard}",
        f"numba_instantiate_seconds: {numba_instantiate_elapsed:.6f}",
        f"{backend_label}_instantiate_seconds: {jax_instantiate_elapsed:.6f}",
        f"numba_seconds: {cpu_elapsed:.6f}",
        f"{backend_label}_seconds: {jax_elapsed:.6f}",
        f"speedup: {cpu_elapsed / jax_elapsed:.3f}x",
        "Max absolute differences vs numba:",
    ]
    for key in (
        "position_collection",
        "director_collection",
        "velocity_collection",
        "omega_collection",
        "sigma",
        "kappa",
    ):
        report_lines.append(
            f"  {key}: {max_abs_diff(jax_state[key], cpu_state[key]):.6e}"
        )
    emit_report(report_lines, args.log)


if __name__ == "__main__":
    main()

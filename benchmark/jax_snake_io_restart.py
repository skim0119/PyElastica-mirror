"""Benchmark multi-snake save/load and restart I/O: Numba vs JAX."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import tempfile
from pathlib import Path

import numpy as np

from _jax_snake_common import (
    DEFAULT_DT,
    DEFAULT_N_ELEM,
    benchmark_config,
    build_cpu_sim,
    build_jax_sim,
    emit_report,
    load_jax_state_npz,
    restore_jax_state_from_host,
    save_jax_state_npz,
    select_device,
    snake_count_from_exponent,
    snapshot_jax_state_to_host,
    time_average,
    validate_dtype_for_device,
)

import elastica as ea

try:
    import jax
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "This benchmark requires JAX. Install the optional GPU extra first."
    ) from exc


DEFAULT_MIN_N_SNAKES_EXP = 8
DEFAULT_MAX_N_SNAKES_EXP = 10


@dataclass(frozen=True)
class IOBenchmarkResult:
    n_snakes_exp: int
    n_snakes: int
    jax_instantiate_seconds: float
    jax_save_seconds: float
    jax_load_seconds: float
    numba_instantiate_seconds: float | None = None
    numba_save_seconds: float | None = None
    numba_load_seconds: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("auto", "cpu", "cuda", "mps"), default="cuda"
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--min-n-snakes-exp", type=int, default=DEFAULT_MIN_N_SNAKES_EXP
    )
    parser.add_argument(
        "--max-n-snakes-exp", type=int, default=DEFAULT_MAX_N_SNAKES_EXP
    )
    parser.add_argument("--n-elem", type=int, default=DEFAULT_N_ELEM)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--no-numba",
        action="store_true",
        help="Skip the Numba save/load and instantiation benchmarks.",
    )
    parser.add_argument("--no-external-loads", action="store_true")
    parser.add_argument("--log", type=Path, default=None)
    return parser.parse_args()


def benchmark_size(
    *,
    n_snakes_exp: int,
    n_elem: int,
    dt: float,
    iterations: int,
    include_numba: bool,
    include_external_loads: bool,
    device: jax.Device,
    dtype: np.dtype,
    tmp_path: Path,
) -> IOBenchmarkResult:
    n_snakes = snake_count_from_exponent(n_snakes_exp)
    config = benchmark_config(
        n_snakes=n_snakes,
        n_elem=n_elem,
        dt=dt,
        include_external_loads=include_external_loads,
    )

    jax_file = tmp_path / f"jax_block_state_{n_snakes}.npz"
    numba_instantiate_avg = None
    numba_save_avg = None
    numba_load_avg = None
    if include_numba:
        numba_dir = tmp_path / f"numba_restart_{n_snakes}"
        cpu_sim, _ = build_cpu_sim(**config)
        ea.save_state(cpu_sim, directory=str(numba_dir), time=np.float64(0.0))
        cpu_load_sim, _ = build_cpu_sim(**config)

        def _instantiate_numba_sim() -> None:
            build_cpu_sim(**config)

        numba_instantiate_avg = time_average(iterations, _instantiate_numba_sim)
        numba_save_avg = time_average(
            iterations,
            lambda: ea.save_state(
                cpu_sim, directory=str(numba_dir), time=np.float64(0.0)
            ),
        )
        numba_load_avg = time_average(
            iterations,
            lambda: ea.load_state(cpu_load_sim, directory=str(numba_dir)),
        )

    with jax.default_device(device):
        _, jax_block = build_jax_sim(
            device=device,
            device_dtype=dtype,
            **config,
        )
        jax.block_until_ready(jax_block.position_collection_device)
        initial_host_state = snapshot_jax_state_to_host(jax_block.jax_get_state())
        save_jax_state_npz(jax_file, initial_host_state)

        def _instantiate_jax_sim() -> None:
            _, block = build_jax_sim(
                device=device,
                device_dtype=dtype,
                **config,
            )
            jax.block_until_ready(block.position_collection_device)

        jax_instantiate_avg = time_average(iterations, _instantiate_jax_sim)
        jax_save_avg = time_average(
            iterations,
            lambda: save_jax_state_npz(
                jax_file,
                snapshot_jax_state_to_host(jax_block.jax_get_state()),
            ),
        )

        def _load_jax_state() -> None:
            restored_host_state = load_jax_state_npz(jax_file)
            restored_state = restore_jax_state_from_host(
                restored_host_state,
                device,
            )
            jax.block_until_ready(restored_state["position_collection"])
            jax_block.jax_set_state(restored_state)

        jax_load_avg = time_average(iterations, _load_jax_state)

    return IOBenchmarkResult(
        n_snakes_exp=n_snakes_exp,
        n_snakes=n_snakes,
        jax_instantiate_seconds=jax_instantiate_avg,
        jax_save_seconds=jax_save_avg,
        jax_load_seconds=jax_load_avg,
        numba_instantiate_seconds=numba_instantiate_avg,
        numba_save_seconds=numba_save_avg,
        numba_load_seconds=numba_load_avg,
    )


def main() -> None:
    args = parse_args()
    assert args.min_n_snakes_exp >= 0, "min-n-snakes-exp must be nonnegative."
    assert args.max_n_snakes_exp >= args.min_n_snakes_exp, (
        "max-n-snakes-exp must be greater than or equal to min-n-snakes-exp."
    )
    assert args.n_elem > 1, "n-elem must be greater than 1."
    assert args.dt > 0.0, "dt must be positive."
    assert args.iterations > 0, "iterations must be positive."

    device = select_device(args.backend)
    dtype = np.dtype(np.float32 if args.dtype == "float32" else np.float64)
    validate_dtype_for_device(dtype, device)
    backend_label = "jax-cpu" if device.platform == "cpu" else f"jax-{device.platform}"

    with tempfile.TemporaryDirectory(prefix="snake_restart_io_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        results = [
            benchmark_size(
                n_snakes_exp=exponent,
                n_elem=args.n_elem,
                dt=args.dt,
                iterations=args.iterations,
                include_numba=not args.no_numba,
                include_external_loads=not args.no_external_loads,
                device=device,
                dtype=dtype,
                tmp_path=tmp_path,
            )
            for exponent in range(args.min_n_snakes_exp, args.max_n_snakes_exp + 1)
        ]

    report_lines = [
        f"device: {device}",
        f"dtype: {dtype}",
        f"n_elem: {args.n_elem}",
        f"iterations: {args.iterations}",
        f"no_numba: {args.no_numba}",
        f"no_external_loads: {args.no_external_loads}",
    ]
    for result in results:
        report_lines.extend(
            (
                "",
                f"n_snakes: 2^{result.n_snakes_exp} ({result.n_snakes})",
                f"{backend_label}_instantiate_avg_seconds: "
                f"{result.jax_instantiate_seconds:.6f}",
                f"{backend_label}_save_avg_seconds: {result.jax_save_seconds:.6f}",
                f"{backend_label}_load_avg_seconds: {result.jax_load_seconds:.6f}",
            )
        )
        if not args.no_numba:
            assert result.numba_instantiate_seconds is not None, (
                "Numba instantiation timing must be available when Numba is enabled."
            )
            assert result.numba_save_seconds is not None, (
                "Numba save timing must be available when Numba is enabled."
            )
            assert result.numba_load_seconds is not None, (
                "Numba load timing must be available when Numba is enabled."
            )
            report_lines.extend(
                (
                    "numba_instantiate_avg_seconds: "
                    f"{result.numba_instantiate_seconds:.6f}",
                    f"numba_save_avg_seconds: {result.numba_save_seconds:.6f}",
                    f"numba_load_avg_seconds: {result.numba_load_seconds:.6f}",
                )
            )
    emit_report(report_lines, args.log)


if __name__ == "__main__":
    main()

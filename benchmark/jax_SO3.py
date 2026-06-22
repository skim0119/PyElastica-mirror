"""Benchmark SO(3) forward and inverse rotation kernels for Numba vs JAX.

Notes
-----
This file lives under ``benchmark/``, which also contains ``jax.py``. When a
script in this directory imports ``jax``, Python can accidentally resolve the
local file instead of the third-party package. The path fixup below removes the
script directory from ``sys.path`` before importing JAX.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if sys.path and Path(sys.path[0]).resolve() == SCRIPT_DIR:
    sys.path.pop(0)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "This benchmark requires JAX. Install the optional GPU extra first."
    ) from exc

jax.config.update("jax_enable_x64", True)

from elastica._jax_rotations import _jax_get_rotation_matrix, _jax_inv_rotate
from elastica._rotations import _get_rotation_matrix, _inv_rotate


@dataclass
class BenchResult:
    name: str
    numba_seconds_total: float
    jax_seconds_total: float
    iterations: int
    max_abs_diff: float

    @property
    def speedup(self) -> float:
        return self.numba_seconds_total / self.jax_seconds_total

    @property
    def numba_seconds_per_iteration(self) -> float:
        return self.numba_seconds_total / self.iterations

    @property
    def jax_seconds_per_iteration(self) -> float:
        return self.jax_seconds_total / self.iterations


def _select_device(platform: str) -> jax.Device:
    assert platform in ("auto", "cpu", "mps", "cuda"), (
        "platform must be one of auto, cpu, mps, or cuda."
    )
    if platform == "auto":
        return jax.devices()[0]
    devices = jax.devices(platform)
    assert devices, f"No JAX device found for platform {platform!r}."
    return devices[0]


def _resolve_dtype(dtype_name: str, device: jax.Device) -> np.dtype:
    assert dtype_name in ("float32", "float64"), "dtype must be float32 or float64."
    dtype = np.dtype(np.float32 if dtype_name == "float32" else np.float64)
    if dtype == np.dtype(np.float64) and device.platform == "mps":
        raise SystemExit(
            "MPS/MLX does not support float64. Use --dtype float32 or run on CPU/CUDA."
        )
    return dtype


def _make_forward_inputs(
    blocksize: int, dtype: np.dtype, seed: int, iterations: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    axis_collection = rng.standard_normal((iterations, 3, blocksize)).astype(
        dtype, copy=False
    )
    scales = np.linspace(0.05, 0.2, iterations, dtype=dtype)
    return axis_collection, scales


def _make_inverse_inputs(
    blocksize: int, dtype: np.dtype, seed: int, iterations: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    director_collection = np.zeros((iterations, 3, 3, blocksize + 1), dtype=dtype)
    for i in range(iterations):
        axis_collection = (0.2 * rng.standard_normal((3, blocksize))).astype(
            dtype, copy=False
        )
        director_collection[i, :, :, 0] = np.eye(3, dtype=dtype)
        increments = _get_rotation_matrix(dtype.type(1.0), axis_collection)
        for k in range(blocksize):
            director_collection[i, :, :, k + 1] = (
                increments[:, :, k] @ director_collection[i, :, :, k]
            )
    return director_collection


def _benchmark_forward(
    *,
    blocksize: int,
    device: jax.Device,
    dtype: np.dtype,
    warmup_runs: int,
    timed_runs: int,
    seed: int,
) -> BenchResult:
    axis_collection, scales = _make_forward_inputs(
        blocksize, dtype, seed, warmup_runs + timed_runs
    )
    warmup_axes = axis_collection[:warmup_runs]
    warmup_scales = scales[:warmup_runs]
    timed_axes = axis_collection[warmup_runs:]
    timed_scales = scales[warmup_runs:]

    for idx in range(warmup_runs):
        _get_rotation_matrix(warmup_scales[idx], warmup_axes[idx])

    start = time.perf_counter()
    numba_output = None
    for idx in range(timed_runs):
        numba_output = _get_rotation_matrix(timed_scales[idx], timed_axes[idx])
    numba_seconds = time.perf_counter() - start
    assert numba_output is not None, "Numba forward output must be populated."

    with jax.default_device(device):
        warmup_axes_jax = jnp.asarray(warmup_axes, dtype=dtype)
        warmup_scales_jax = jnp.asarray(warmup_scales, dtype=dtype)
        timed_axes_jax = jnp.asarray(timed_axes, dtype=dtype)
        timed_scales_jax = jnp.asarray(timed_scales, dtype=dtype)

        def forward_loop(scales_jax: jax.Array, axes_jax: jax.Array) -> jax.Array:
            def body_fn(idx: int, _carry: jax.Array) -> jax.Array:
                return _jax_get_rotation_matrix(scales_jax[idx], axes_jax[idx])

            init = _jax_get_rotation_matrix(scales_jax[0], axes_jax[0])
            return jax.lax.fori_loop(0, scales_jax.shape[0], body_fn, init)

        compiled = jax.jit(forward_loop)
        if warmup_runs > 0:
            warm = compiled(warmup_scales_jax, warmup_axes_jax)
            jax.block_until_ready(warm)

        start = time.perf_counter()
        jax_output = compiled(timed_scales_jax, timed_axes_jax)
        jax.block_until_ready(jax_output)
        jax_seconds = time.perf_counter() - start

    assert jax_output is not None, "JAX forward output must be populated."
    max_abs_diff = float(np.max(np.abs(numba_output - np.asarray(jax_output))))
    return BenchResult(
        name="forward_rotation_matrix",
        numba_seconds_total=numba_seconds,
        jax_seconds_total=jax_seconds,
        iterations=timed_runs,
        max_abs_diff=max_abs_diff,
    )


def _benchmark_inverse(
    *,
    blocksize: int,
    device: jax.Device,
    dtype: np.dtype,
    warmup_runs: int,
    timed_runs: int,
    seed: int,
) -> BenchResult:
    director_collection = _make_inverse_inputs(
        blocksize, dtype, seed, warmup_runs + timed_runs
    )
    warmup_directors = director_collection[:warmup_runs]
    timed_directors = director_collection[warmup_runs:]

    for idx in range(warmup_runs):
        _inv_rotate(warmup_directors[idx])

    start = time.perf_counter()
    numba_output = None
    for idx in range(timed_runs):
        numba_output = _inv_rotate(timed_directors[idx])
    numba_seconds = time.perf_counter() - start
    assert numba_output is not None, "Numba inverse output must be populated."

    with jax.default_device(device):
        warmup_directors_jax = jnp.asarray(warmup_directors, dtype=dtype)
        timed_directors_jax = jnp.asarray(timed_directors, dtype=dtype)

        def inverse_loop(directors_jax: jax.Array) -> jax.Array:
            def body_fn(idx: int, _carry: jax.Array) -> jax.Array:
                return _jax_inv_rotate(directors_jax[idx])

            init = _jax_inv_rotate(directors_jax[0])
            return jax.lax.fori_loop(0, directors_jax.shape[0], body_fn, init)

        compiled = jax.jit(inverse_loop)
        if warmup_runs > 0:
            warm = compiled(warmup_directors_jax)
            jax.block_until_ready(warm)

        start = time.perf_counter()
        jax_output = compiled(timed_directors_jax)
        jax.block_until_ready(jax_output)
        jax_seconds = time.perf_counter() - start

    assert jax_output is not None, "JAX inverse output must be populated."
    max_abs_diff = float(np.max(np.abs(numba_output - np.asarray(jax_output))))
    return BenchResult(
        name="inverse_rotation_extract",
        numba_seconds_total=numba_seconds,
        jax_seconds_total=jax_seconds,
        iterations=timed_runs,
        max_abs_diff=max_abs_diff,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--blocksizes",
        type=str,
        default="128,512,2048,8192,32768,131072,524288",
        help="Comma-separated block sizes to benchmark.",
    )
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--timed-runs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20250621)
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("benchmark/jax_SO3.png"),
        help="Path to save the benchmark plot.",
    )
    return parser.parse_args()


def _plot_results(
    blocksizes: tuple[int, ...],
    forward_results: list[BenchResult],
    inverse_results: list[BenchResult],
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)

    for ax, results, title in (
        (axes[0], forward_results, "Forward Rotation"),
        (axes[1], inverse_results, "Inverse Rotation"),
    ):
        numba_times = [result.numba_seconds_per_iteration for result in results]
        jax_times = [result.jax_seconds_per_iteration for result in results]
        diffs = [result.max_abs_diff for result in results]

        ax.loglog(blocksizes, numba_times, marker="o", label="Numba per iter")
        ax.loglog(blocksizes, jax_times, marker="o", label="JAX per iter")
        ax2 = ax.twinx()
        ax2.loglog(
            blocksizes,
            diffs,
            marker="s",
            linestyle="--",
            color="tab:green",
            label="max abs diff",
        )

        ax.set_title(title)
        ax.set_xlabel("blocksize")
        ax.set_ylabel("seconds / iteration")
        ax2.set_ylabel("max abs diff")
        ax.grid(True, which="both", alpha=0.3)

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")

    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    assert args.warmup_runs >= 0, "warmup-runs must be nonnegative."
    assert args.timed_runs > 0, "timed-runs must be positive."
    blocksizes = tuple(int(item) for item in args.blocksizes.split(",") if item.strip())
    assert blocksizes, "At least one blocksize must be provided."
    for blocksize in blocksizes:
        assert blocksize > 0, "Every blocksize must be positive."

    device = _select_device(args.platform)
    dtype = _resolve_dtype(args.dtype, device)

    print(f"device: {device}")
    print(f"dtype: {dtype}")
    print(f"blocksizes: {blocksizes}")
    print(f"warmup_runs: {args.warmup_runs}")
    print(f"timed_runs: {args.timed_runs}")
    print("Results:")
    forward_results: list[BenchResult] = []
    inverse_results: list[BenchResult] = []
    for blocksize in blocksizes:
        forward = _benchmark_forward(
            blocksize=blocksize,
            device=device,
            dtype=dtype,
            warmup_runs=args.warmup_runs,
            timed_runs=args.timed_runs,
            seed=args.seed,
        )
        inverse = _benchmark_inverse(
            blocksize=blocksize,
            device=device,
            dtype=dtype,
            warmup_runs=args.warmup_runs,
            timed_runs=args.timed_runs,
            seed=args.seed + 1,
        )
        forward_results.append(forward)
        inverse_results.append(inverse)
        print(f"  blocksize={blocksize}")
        for result in (forward, inverse):
            print(f"    {result.name}")
            print(f"      numba_seconds_total: {result.numba_seconds_total:.6e}")
            print(
                f"      numba_seconds_per_iteration: {result.numba_seconds_per_iteration:.6e}"
            )
            print(f"      jax_seconds_total:   {result.jax_seconds_total:.6e}")
            print(
                f"      jax_seconds_per_iteration:   {result.jax_seconds_per_iteration:.6e}"
            )
            print(f"      speedup:       {result.speedup:.3f}x")
            print(f"      max_abs_diff:  {result.max_abs_diff:.6e}")

    _plot_results(blocksizes, forward_results, inverse_results, args.plot_output)
    print(f"plot: {args.plot_output}")


if __name__ == "__main__":
    main()

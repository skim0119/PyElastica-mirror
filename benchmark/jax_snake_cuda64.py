"""Benchmark multi-snake CPU Numba vs JAX CUDA float64 rollout.

Notes
-----
This script is intended for larger accelerator-capable machines. It builds one
simulator containing many independent continuum snakes, each with 50 elements by
default, and compares the original PyElastica CPU path against the JAX rollout.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if sys.path and Path(sys.path[0]).resolve() == SCRIPT_DIR:
    sys.path.pop(0)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

import elastica as ea
from examples.ContinuumSnakeGPUCase.run_continuum_snake_gpu import (
    SnakeMuscleTorquesJax,
    SnakePlaneContactJax,
    build_rod,
    default_b_coeff,
)

try:
    import jax
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "This benchmark requires JAX. Install the optional GPU extra first."
    ) from exc


jax.config.update("jax_enable_x64", True)


class MultiSnakeReferenceSimulator(
    ea.BaseSystemCollection, ea.Forcing, ea.Damping, ea.Contact
):
    pass


class MultiSnakeJAXSimulator(ea.BaseSystemCollection, ea.JAXOps):
    pass


class _ConfiguredSnakeMemoryBlock(ea.MemoryBlockCosseratRodJax):
    device_dtype = np.dtype(np.float64)
    device = None

    def __init__(self, systems, system_idx_list):
        super().__init__(
            systems,
            system_idx_list,
            device_dtype=self.device_dtype,
            device=self.device,
        )


def _select_device(platform: str) -> jax.Device:
    assert platform in ("auto", "cpu", "mps", "cuda"), (
        "platform must be one of auto, cpu, mps, or cuda."
    )
    if platform == "auto":
        for candidate in ("cuda", "mps", "cpu"):
            try:
                devices = jax.devices(candidate)
            except Exception:
                continue
            if devices:
                return devices[0]
        raise RuntimeError("No JAX devices are available.")
    devices = jax.devices(platform)
    assert devices, f"No JAX device found for platform {platform!r}."
    return devices[0]


def _snake_start(index: int, spacing: float) -> np.ndarray:
    return np.array([index * spacing, 0.0, 0.0], dtype=np.float64)


def _build_cpu_sim(
    *,
    n_snakes: int,
    n_elem: int,
    period: float,
    base_length: float,
    density: float,
    youngs_modulus: float,
    poisson_ratio: float,
    gravitational_acc: float,
    time_step: float,
) -> tuple[MultiSnakeReferenceSimulator, list[ea.CosseratRod]]:
    b_coeff = default_b_coeff()
    normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    wave_length = float(b_coeff[-1])
    froude = 0.1
    mu = base_length / (period * period * np.abs(gravitational_acc) * froude)
    kinetic_mu_array = np.array([mu, 1.5 * mu, 2.0 * mu], dtype=np.float64)
    static_mu_array = np.zeros(kinetic_mu_array.shape, dtype=np.float64)
    spacing = 1.5 * base_length

    sim = MultiSnakeReferenceSimulator()
    rods: list[ea.CosseratRod] = []
    ground_plane = ea.Plane(
        plane_origin=np.array([0.0, -base_length * 0.011, 0.0], dtype=np.float64),
        plane_normal=normal,
    )
    sim.append(ground_plane)

    for idx in range(n_snakes):
        rod = build_rod(
            n_elem=n_elem,
            base_length=base_length,
            density=density,
            youngs_modulus=youngs_modulus,
            poisson_ratio=poisson_ratio,
        )
        start = _snake_start(idx, spacing)
        rod.position_collection[...] = rod.position_collection + start[:, None]
        sim.append(rod)
        sim.add_forcing_to(rod).using(
            ea.GravityForces,
            acc_gravity=np.array([0.0, gravitational_acc, 0.0], dtype=np.float64),
        )
        sim.add_forcing_to(rod).using(
            ea.MuscleTorques,
            base_length=base_length,
            b_coeff=b_coeff[:-1],
            period=period,
            wave_number=2.0 * np.pi / wave_length,
            phase_shift=0.0,
            rest_lengths=rod.rest_lengths,
            ramp_up_time=period,
            direction=normal,
            with_spline=True,
        )
        sim.detect_contact_between(rod, ground_plane).using(
            ea.RodPlaneContactWithAnisotropicFriction,
            k=1.0,
            nu=1.0e-6,
            slip_velocity_tol=1.0e-8,
            static_mu_array=static_mu_array,
            kinetic_mu_array=kinetic_mu_array,
        )
        sim.dampen(rod).using(
            ea.AnalyticalLinearDamper,
            damping_constant=2.0e-3,
            time_step=time_step,
        )
        rods.append(rod)

    sim.finalize()
    return sim, rods


def _build_jax_sim(
    *,
    device: jax.Device,
    device_dtype: np.dtype,
    n_snakes: int,
    n_elem: int,
    period: float,
    base_length: float,
    density: float,
    youngs_modulus: float,
    poisson_ratio: float,
    gravitational_acc: float,
    time_step: float,
) -> tuple[MultiSnakeJAXSimulator, ea.MemoryBlockCosseratRodJax]:
    b_coeff = default_b_coeff()
    froude = 0.1
    mu = base_length / (period * period * np.abs(gravitational_acc) * froude)
    kinetic_mu_array = np.array([mu, 1.5 * mu, 2.0 * mu], dtype=np.float64)
    static_mu_array = np.zeros(kinetic_mu_array.shape, dtype=np.float64)
    spacing = 1.5 * base_length

    _ConfiguredSnakeMemoryBlock.device = device
    _ConfiguredSnakeMemoryBlock.device_dtype = np.dtype(device_dtype)

    sim = MultiSnakeJAXSimulator()
    sim.enable_block_supports(ea.CosseratRod, _ConfiguredSnakeMemoryBlock)
    for idx in range(n_snakes):
        rod = build_rod(
            n_elem=n_elem,
            base_length=base_length,
            density=density,
            youngs_modulus=youngs_modulus,
            poisson_ratio=poisson_ratio,
        )
        start = _snake_start(idx, spacing)
        rod.position_collection[...] = rod.position_collection + start[:, None]
        sim.append(rod)
        sim.using(rod).operate(
            SnakeMuscleTorquesJax,
            b_coeff=b_coeff,
            period=period,
            base_length=base_length,
            gravitational_acc=gravitational_acc,
        )
        sim.using(rod).operate(
            SnakePlaneContactJax,
            plane_origin=np.array([0.0, -base_length * 0.011, 0.0], dtype=np.float64),
            plane_normal=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            slip_velocity_tol=1.0e-8,
            k=1.0,
            nu=1.0e-6,
            static_mu_array=static_mu_array,
            kinetic_mu_array=kinetic_mu_array,
        )
        sim.using(rod).operate(
            ea.AnalyticalLinearDamperJax,
            time_step=np.float64(time_step),
            damping_constant=2.0e-3,
        )

    sim.finalize()
    block = tuple(sim.final_systems())[0]
    return sim, block


def _collect_cpu_state(rods: list[ea.CosseratRod]) -> dict[str, np.ndarray]:
    return {
        "position_collection": np.concatenate(
            [rod.position_collection for rod in rods], axis=1
        ),
        "director_collection": np.concatenate(
            [rod.director_collection for rod in rods], axis=2
        ),
        "velocity_collection": np.concatenate(
            [rod.velocity_collection for rod in rods], axis=1
        ),
        "omega_collection": np.concatenate([rod.omega_collection for rod in rods], axis=1),
        "sigma": np.concatenate([rod.sigma for rod in rods], axis=1),
        "kappa": np.concatenate([rod.kappa for rod in rods], axis=1),
    }


def _collect_jax_state(
    block: ea.MemoryBlockCosseratRodJax, n_snakes: int
) -> dict[str, np.ndarray]:
    state = block.jax_get_state()
    position_chunks = []
    director_chunks = []
    velocity_chunks = []
    omega_chunks = []
    sigma_chunks = []
    kappa_chunks = []
    for idx in range(n_snakes):
        node_slice = slice(
            int(block.start_idx_in_rod_nodes[idx]),
            int(block.end_idx_in_rod_nodes[idx]),
        )
        elem_slice = slice(
            int(block.start_idx_in_rod_elems[idx]),
            int(block.end_idx_in_rod_elems[idx]),
        )
        voronoi_slice = slice(
            int(block.start_idx_in_rod_voronoi[idx]),
            int(block.end_idx_in_rod_voronoi[idx]),
        )
        position_chunks.append(np.asarray(state["position_collection"])[:, node_slice])
        director_chunks.append(np.asarray(state["director_collection"])[:, :, elem_slice])
        velocity_chunks.append(np.asarray(state["velocity_collection"])[:, node_slice])
        omega_chunks.append(np.asarray(state["omega_collection"])[:, elem_slice])
        sigma_chunks.append(np.asarray(state["sigma"])[:, elem_slice])
        kappa_chunks.append(np.asarray(state["kappa"])[:, voronoi_slice])
    return {
        "position_collection": np.concatenate(position_chunks, axis=1),
        "director_collection": np.concatenate(director_chunks, axis=2),
        "velocity_collection": np.concatenate(velocity_chunks, axis=1),
        "omega_collection": np.concatenate(omega_chunks, axis=1),
        "sigma": np.concatenate(sigma_chunks, axis=1),
        "kappa": np.concatenate(kappa_chunks, axis=1),
    }


def _max_abs_diff(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.max(np.abs(first - second)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--n-snakes", type=int, default=200)
    parser.add_argument("--n-elem", type=int, default=50)
    parser.add_argument("--final-time", type=float, default=0.1)
    parser.add_argument("--time-step", type=float, default=1.0e-4)
    parser.add_argument("--warmup-runs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert args.n_snakes > 0, "n-snakes must be positive."
    assert args.n_elem > 1, "n-elem must be greater than 1."
    assert args.final_time > 0.0, "final-time must be positive."
    assert args.time_step > 0.0, "time-step must be positive."
    assert args.warmup_runs >= 0, "warmup-runs must be nonnegative."

    device = _select_device(args.backend)
    dtype = np.dtype(np.float32 if args.dtype == "float32" else np.float64)
    if dtype == np.dtype(np.float64) and device.platform == "mps":
        raise SystemExit("MPS/MLX does not support float64. Use CPU/CUDA or float32.")

    total_steps = int(args.final_time / args.time_step)
    assert total_steps > 0, "final-time / time-step must yield at least one step."
    snapped_final_time = total_steps * args.time_step
    backend_label = "jax-cpu" if device.platform == "cpu" else f"jax-{device.platform}"

    cpu_sim, cpu_rods = _build_cpu_sim(
        n_snakes=args.n_snakes,
        n_elem=args.n_elem,
        period=2.0,
        base_length=0.35,
        density=1000.0,
        youngs_modulus=1.0e6,
        poisson_ratio=0.5,
        gravitational_acc=-9.80665,
        time_step=args.time_step,
    )
    cpu_stepper = ea.PositionVerlet()

    time_value = np.float64(0.0)
    start = time.perf_counter()
    for _ in range(total_steps):
        time_value = cpu_stepper.step(cpu_sim, time_value, np.float64(args.time_step))
    cpu_elapsed = time.perf_counter() - start
    assert np.isclose(time_value, snapped_final_time), (
        "CPU rollout did not end on the expected time grid."
    )
    cpu_state = _collect_cpu_state(cpu_rods)

    with jax.default_device(device):
        jax_sim, jax_block = _build_jax_sim(
            device=device,
            device_dtype=dtype,
            n_snakes=args.n_snakes,
            n_elem=args.n_elem,
            period=2.0,
            base_length=0.35,
            density=1000.0,
            youngs_modulus=1.0e6,
            poisson_ratio=0.5,
            gravitational_acc=-9.80665,
            time_step=args.time_step,
        )
        jax_stepper = ea.PositionVerletGPU()

        for _ in range(args.warmup_runs):
            initial_state = dict(jax_block.jax_get_state())
            jax_stepper.integrate(
                jax_sim,
                time=np.float64(0.0),
                final_time=np.float64(snapped_final_time),
                dt=np.float64(args.time_step),
            )
            jax.block_until_ready(jax_block.position_collection_device)
            jax_block.jax_set_state(initial_state)

        start = time.perf_counter()
        jax_stepper.integrate(
            jax_sim,
            time=np.float64(0.0),
            final_time=np.float64(snapped_final_time),
            dt=np.float64(args.time_step),
        )
        jax.block_until_ready(jax_block.position_collection_device)
        jax_elapsed = time.perf_counter() - start
        jax_state = _collect_jax_state(jax_block, args.n_snakes)

    print(f"device: {device}")
    print(f"dtype: {dtype}")
    print(f"n_snakes: {args.n_snakes}")
    print(f"n_elem: {args.n_elem}")
    print(f"steps: {total_steps}")
    print(f"numba_seconds: {cpu_elapsed:.6f}")
    print(f"{backend_label}_seconds: {jax_elapsed:.6f}")
    print(f"speedup: {cpu_elapsed / jax_elapsed:.3f}x")
    print("Max absolute differences vs numba:")
    for key in (
        "position_collection",
        "director_collection",
        "velocity_collection",
        "omega_collection",
        "sigma",
        "kappa",
    ):
        print(f"  {key}: {_max_abs_diff(jax_state[key], cpu_state[key]):.6e}")


if __name__ == "__main__":
    main()

"""Shared helpers for multi-snake Numba vs JAX benchmarks."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

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

DEFAULT_PERIOD = 2.0
DEFAULT_BASE_LENGTH = 0.35
DEFAULT_DENSITY = 1000.0
DEFAULT_YOUNGS_MODULUS = 1.0e6
DEFAULT_POISSON_RATIO = 0.5
DEFAULT_GRAVITY = -9.80665
DEFAULT_DAMPING = 2.0e-3
DEFAULT_FROUDE = 0.1
DEFAULT_N_ELEM = 50
DEFAULT_N_SNAKES_EXP = 8
DEFAULT_STEPS = 1000
DEFAULT_DT = 1.0e-4


class MultiSnakeReferenceSimulator(
    ea.BaseSystemCollection, ea.Forcing, ea.Damping, ea.Contact
):
    pass


class MultiSnakeJAXSimulator(ea.BaseSystemCollection, ea.JAXOps):
    pass


class ConfiguredSnakeMemoryBlock(ea.MemoryBlockCosseratRodJax):
    device_dtype = np.dtype(np.float64)
    device = None

    def __init__(self, systems, system_idx_list):
        super().__init__(
            systems,
            system_idx_list,
            device_dtype=self.device_dtype,
            device=self.device,
        )


def select_device(platform: str) -> jax.Device:
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


def snake_count_from_exponent(exponent: int) -> int:
    assert exponent >= 0, "n-snakes-exp must be nonnegative."
    return 2**exponent


def validate_dtype_for_device(dtype: np.dtype, device: jax.Device) -> None:
    if dtype == np.dtype(np.float64) and device.platform == "mps":
        raise SystemExit("MPS/MLX does not support float64. Use CPU/CUDA or float32.")


def snake_start(index: int, spacing: float) -> np.ndarray:
    return np.array([index * spacing, 0.0, 0.0], dtype=np.float64)


def build_cpu_sim(
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
    mu = base_length / (period * period * np.abs(gravitational_acc) * DEFAULT_FROUDE)
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
        start = snake_start(idx, spacing)
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
            damping_constant=DEFAULT_DAMPING,
            time_step=time_step,
        )
        rods.append(rod)

    sim.finalize()
    return sim, rods


def build_jax_sim(
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
    mu = base_length / (period * period * np.abs(gravitational_acc) * DEFAULT_FROUDE)
    kinetic_mu_array = np.array([mu, 1.5 * mu, 2.0 * mu], dtype=np.float64)
    static_mu_array = np.zeros(kinetic_mu_array.shape, dtype=np.float64)
    spacing = 1.5 * base_length

    ConfiguredSnakeMemoryBlock.device = device
    ConfiguredSnakeMemoryBlock.device_dtype = np.dtype(device_dtype)

    sim = MultiSnakeJAXSimulator()
    sim.enable_block_supports(ea.CosseratRod, ConfiguredSnakeMemoryBlock)
    for idx in range(n_snakes):
        rod = build_rod(
            n_elem=n_elem,
            base_length=base_length,
            density=density,
            youngs_modulus=youngs_modulus,
            poisson_ratio=poisson_ratio,
        )
        start = snake_start(idx, spacing)
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
            damping_constant=DEFAULT_DAMPING,
        )

    sim.finalize()
    block = tuple(sim.final_systems())[0]
    return sim, block


def collect_cpu_state(rods: list[ea.CosseratRod]) -> dict[str, np.ndarray]:
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


def collect_jax_state(
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


def max_abs_diff(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.max(np.abs(first - second)))


def time_average(n_iter: int, fn) -> float:  # type: ignore[no-untyped-def]
    assert n_iter > 0, "n_iter must be positive."
    start = time.perf_counter()
    for _ in range(n_iter):
        fn()
    return (time.perf_counter() - start) / n_iter


def snapshot_jax_state_to_host(
    state: dict[str, jax.Array],
) -> dict[str, np.ndarray]:
    host_state = jax.device_get(state)
    return {key: np.asarray(value).copy() for key, value in host_state.items()}


def restore_jax_state_from_host(
    host_state: dict[str, np.ndarray],
    device: jax.Device,
) -> dict[str, jax.Array]:
    return {
        key: jax.device_put(np.asarray(value), device=device)
        for key, value in host_state.items()
    }


def save_jax_state_npz(path: Path, state: dict[str, np.ndarray]) -> None:
    np.savez(path, **state)


def load_jax_state_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(value).copy() for key, value in data.items()}


def emit_report(lines: list[str], log_path: Path | None) -> None:
    report = "\n".join(lines)
    print(report)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(report + "\n", encoding="utf-8")


def benchmark_config(
    *,
    n_snakes: int,
    n_elem: int,
    dt: float,
) -> dict[str, Any]:
    return {
        "n_snakes": n_snakes,
        "n_elem": n_elem,
        "period": DEFAULT_PERIOD,
        "base_length": DEFAULT_BASE_LENGTH,
        "density": DEFAULT_DENSITY,
        "youngs_modulus": DEFAULT_YOUNGS_MODULUS,
        "poisson_ratio": DEFAULT_POISSON_RATIO,
        "gravitational_acc": DEFAULT_GRAVITY,
        "time_step": dt,
    }

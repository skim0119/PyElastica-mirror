"""
Butterfly GPU Prototype
=======================

This example mirrors the original butterfly case while running the free rod
rollout through the JAX-owned memory block and PositionVerletGPU. It is used
as a Hamiltonian-style sanity check on the free integration path and energy
drift.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import elastica as ea
from elastica.utils import MaxDimension

try:
    import jax
    from jax import config as jax_config
except ModuleNotFoundError as exc:  # pragma: no cover - runtime-only guard
    raise SystemExit(
        "This example requires JAX. Install the optional GPU dependency first, "
        'for example with `uv add --optional gpu "jax[cuda13]"`.'
    ) from exc


jax_config.update("jax_enable_x64", True)


class ButterflyReferenceSimulator(ea.BaseSystemCollection):
    pass


class ButterflyJAXSimulator(ea.BaseSystemCollection, ea.JAXOps):
    pass


class _ConfiguredButterflyMemoryBlock(ea.MemoryBlockCosseratRodJax):
    device_dtype = np.dtype(np.float64)
    device = None

    def __init__(self, systems, system_idx_list):
        super().__init__(
            systems,
            system_idx_list,
            device_dtype=self.device_dtype,
            device=self.device,
        )


def build_positions(
    *,
    n_elem: int,
    total_length: float,
    angle_of_inclination: float,
) -> np.ndarray:
    half_n_elem = n_elem // 2
    origin = np.zeros((3, 1))
    horizontal_direction = np.array([0.0, 0.0, 1.0]).reshape(-1, 1)
    vertical_direction = np.array([1.0, 0.0, 0.0]).reshape(-1, 1)

    positions = np.empty((MaxDimension.value(), n_elem + 1))
    dl = total_length / n_elem
    first_half = np.arange(half_n_elem + 1.0).reshape(1, -1)
    positions[..., : half_n_elem + 1] = origin + dl * first_half * (
        np.cos(angle_of_inclination) * horizontal_direction
        + np.sin(angle_of_inclination) * vertical_direction
    )
    positions[..., half_n_elem:] = positions[
        ..., half_n_elem : half_n_elem + 1
    ] + dl * first_half * (
        np.cos(angle_of_inclination) * horizontal_direction
        - np.sin(angle_of_inclination) * vertical_direction
    )
    return positions


def build_rod(
    *,
    n_elem: int = 4,
    total_length: float = 3.0,
    base_radius: float = 0.25,
    density: float = 5000.0,
    youngs_modulus: float = 1.0e4,
    poisson_ratio: float = 0.5,
    angle_of_inclination: float = np.deg2rad(45.0),
) -> ea.CosseratRod:
    n_elem += n_elem % 2
    positions = build_positions(
        n_elem=n_elem,
        total_length=total_length,
        angle_of_inclination=angle_of_inclination,
    )
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)
    return ea.CosseratRod.straight_rod(
        n_elem,
        start=np.zeros(3),
        direction=np.array([0.0, 0.0, 1.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        base_length=total_length,
        base_radius=base_radius,
        density=density,
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
        position=positions,
    )


def build_cpu_reference_sim(
    *,
    n_elem: int = 4,
    total_length: float = 3.0,
    base_radius: float = 0.25,
    density: float = 5000.0,
    youngs_modulus: float = 1.0e4,
    poisson_ratio: float = 0.5,
    angle_of_inclination: float = np.deg2rad(45.0),
) -> tuple[ButterflyReferenceSimulator, ea.CosseratRod, float]:
    sim = ButterflyReferenceSimulator()
    rod = build_rod(
        n_elem=n_elem,
        total_length=total_length,
        base_radius=base_radius,
        density=density,
        youngs_modulus=youngs_modulus,
        poisson_ratio=poisson_ratio,
        angle_of_inclination=angle_of_inclination,
    )
    sim.append(rod)
    sim.finalize()
    dt = 0.01 * (total_length / rod.n_elems)
    return sim, rod, dt


def build_jax_sim(
    *,
    device: jax.Device,
    device_dtype: np.dtype,
    n_elem: int = 4,
    total_length: float = 3.0,
    base_radius: float = 0.25,
    density: float = 5000.0,
    youngs_modulus: float = 1.0e4,
    poisson_ratio: float = 0.5,
    angle_of_inclination: float = np.deg2rad(45.0),
) -> tuple[ButterflyJAXSimulator, ea.MemoryBlockCosseratRodJax, ea.CosseratRod, float]:
    _ConfiguredButterflyMemoryBlock.device = device
    _ConfiguredButterflyMemoryBlock.device_dtype = np.dtype(device_dtype)

    sim = ButterflyJAXSimulator()
    sim.enable_block_supports(ea.CosseratRod, _ConfiguredButterflyMemoryBlock)
    rod = build_rod(
        n_elem=n_elem,
        total_length=total_length,
        base_radius=base_radius,
        density=density,
        youngs_modulus=youngs_modulus,
        poisson_ratio=poisson_ratio,
        angle_of_inclination=angle_of_inclination,
    )
    sim.append(rod)
    sim.finalize()
    block = tuple(sim.final_systems())[0]
    dt = 0.01 * (total_length / rod.n_elems)
    return sim, block, rod, dt


def available_platforms() -> dict[str, jax.Device]:
    platforms: dict[str, jax.Device] = {}
    for backend_name in ("cpu", "gpu", "cuda", "metal", "mps"):
        try:
            backend_devices = jax.devices(backend_name)
        except Exception:
            continue
        if not backend_devices:
            continue
        device = backend_devices[0]
        platforms.setdefault(backend_name, device)
        platforms.setdefault(device.platform.lower(), device)

    if "metal" in platforms and "mps" not in platforms:
        platforms["mps"] = platforms["metal"]
    if "gpu" in platforms:
        platforms.setdefault("cuda", platforms["gpu"])
    if "cuda" in platforms:
        platforms.setdefault("gpu", platforms["cuda"])
    return platforms


def select_device(requested_backend: str) -> tuple[str, jax.Device]:
    platforms = available_platforms()
    if requested_backend == "auto":
        for candidate in ("cuda", "mps", "gpu", "cpu"):
            if candidate in platforms:
                return candidate, platforms[candidate]
        raise RuntimeError("No JAX devices are available.")

    assert requested_backend in platforms, (
        f"Requested backend {requested_backend!r} is not available. "
        f"Found: {sorted(platforms)}"
    )
    return requested_backend, platforms[requested_backend]


def preferred_dtype(device: jax.Device) -> np.dtype:
    if device.platform.lower() == "cpu":
        return np.float64
    return np.float32


def total_energy(rod: ea.CosseratRod) -> float:
    return float(
        rod.compute_translational_energy()
        + rod.compute_rotational_energy()
        + rod.compute_shear_energy()
        + rod.compute_bending_energy()
    )


def run_cpu_reference(
    *,
    final_time: float,
    n_elem: int = 4,
) -> tuple[dict[str, np.ndarray], dict[str, float], float, float]:
    sim, rod, dt = build_cpu_reference_sim(n_elem=n_elem)
    stepper = ea.PositionVerlet()
    total_steps = int(final_time / dt)
    snapped_final_time = total_steps * dt
    time_value = np.float64(0.0)
    initial_total_energy = total_energy(rod)

    start = time.perf_counter()
    for _ in range(total_steps):
        time_value = stepper.step(sim, time_value, np.float64(dt))
    elapsed = time.perf_counter() - start
    assert np.isclose(
        time_value, snapped_final_time
    ), "CPU butterfly rollout did not end on the expected time grid."
    final_total_energy = total_energy(rod)
    return (
        {
            "position_collection": rod.position_collection.copy(),
            "director_collection": rod.director_collection.copy(),
            "velocity_collection": rod.velocity_collection.copy(),
            "omega_collection": rod.omega_collection.copy(),
            "sigma": rod.sigma.copy(),
            "kappa": rod.kappa.copy(),
        },
        {
            "initial_total_energy": initial_total_energy,
            "final_total_energy": final_total_energy,
            "energy_drift": final_total_energy - initial_total_energy,
        },
        elapsed,
        dt,
    )


def run_jax_rollout(
    *,
    device: jax.Device,
    device_dtype: np.dtype,
    final_time: float,
    n_elem: int = 4,
) -> tuple[dict[str, np.ndarray], dict[str, float], float, float]:
    sim, block, rod, dt = build_jax_sim(
        device=device,
        device_dtype=device_dtype,
        n_elem=n_elem,
    )
    stepper = ea.PositionVerletGPU()
    total_steps = int(final_time / dt)
    snapped_final_time = total_steps * dt
    initial_state = dict(block.jax_get_state())
    initial_total_energy = total_energy(rod)

    stepper.integrate(
        sim,
        time=np.float64(0.0),
        final_time=np.float64(snapped_final_time),
        dt=np.float64(dt),
    )
    jax.block_until_ready(block.position_collection_device)

    block.jax_set_state(initial_state)
    start = time.perf_counter()
    stepper.integrate(
        sim,
        time=np.float64(0.0),
        final_time=np.float64(snapped_final_time),
        dt=np.float64(dt),
    )
    jax.block_until_ready(block.position_collection_device)
    elapsed = time.perf_counter() - start

    block.from_device(update_rods=True)
    final_total_energy = total_energy(rod)
    state = block.jax_get_state()
    return (
        {
            "position_collection": np.asarray(state["position_collection"]),
            "director_collection": np.asarray(state["director_collection"]),
            "velocity_collection": np.asarray(state["velocity_collection"]),
            "omega_collection": np.asarray(state["omega_collection"]),
            "sigma": np.asarray(state["sigma"]),
            "kappa": np.asarray(state["kappa"]),
        },
        {
            "initial_total_energy": initial_total_energy,
            "final_total_energy": final_total_energy,
            "energy_drift": final_total_energy - initial_total_energy,
        },
        elapsed,
        dt,
    )


def max_abs_diff(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.max(np.abs(first - second)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="JAX backend to target for the GPU-style rollout.",
    )
    parser.add_argument("--n-elem", type=int, default=4)
    parser.add_argument(
        "--final-time",
        type=float,
        default=40.0,
        help="Final simulation time for the free butterfly rollout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend_name, device = select_device(args.backend)
    dtype = preferred_dtype(device)

    cpu_state, cpu_energy, cpu_elapsed, dt = run_cpu_reference(
        n_elem=args.n_elem,
        final_time=args.final_time,
    )
    jax_state, jax_energy, jax_elapsed, _ = run_jax_rollout(
        device=device,
        device_dtype=dtype,
        n_elem=args.n_elem,
        final_time=args.final_time,
    )
    total_steps = int(args.final_time / dt)

    print(f"Selected backend alias: {backend_name}")
    print(f"JAX device: {device} (platform={device.platform})")
    print(f"JAX rollout dtype: {dtype}")
    print(f"Butterfly rollout steps: {total_steps}")
    print(f"CPU reference elapsed: {cpu_elapsed:.4f} s")
    print(f"JAX rollout elapsed: {jax_elapsed:.4f} s")
    print("Max absolute differences vs CPU reference:")
    for key in (
        "position_collection",
        "director_collection",
        "velocity_collection",
        "omega_collection",
        "sigma",
        "kappa",
    ):
        print(f"  {key}: {max_abs_diff(jax_state[key], cpu_state[key]):.3e}")
    print("Energy summary:")
    print(f"  CPU drift: {cpu_energy['energy_drift']:.6e}")
    print(f"  JAX drift: {jax_energy['energy_drift']:.6e}")
    print(
        "  Final total energy diff: "
        f"{abs(jax_energy['final_total_energy'] - cpu_energy['final_total_energy']):.6e}"
    )


if __name__ == "__main__":
    main()

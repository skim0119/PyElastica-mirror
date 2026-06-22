"""
Timoshenko Beam GPU Prototype
=============================

This example mirrors the shearable Timoshenko beam setup while replacing the
host-side constraints, forcing, and damping modules with JAX-compatible
operators. It is intended as the first planar-rotation validation case after
the purely axial stretching example.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import elastica as ea

try:
    import jax
    from jax import config as jax_config
except ModuleNotFoundError as exc:  # pragma: no cover - runtime-only guard
    raise SystemExit(
        "This example requires JAX. Install the optional GPU dependency first, "
        'for example with `uv add --optional gpu "jax[cuda13]"`.'
    ) from exc


jax_config.update("jax_enable_x64", True)


class TimoshenkoReferenceSimulator(
    ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.CallBacks, ea.Damping
):
    pass


class TimoshenkoJAXSimulator(ea.BaseSystemCollection, ea.JAXOps):
    pass


class _ConfiguredTimoshenkoMemoryBlock(ea.MemoryBlockCosseratRodJax):
    device_dtype = np.dtype(np.float64)
    device = None

    def __init__(self, systems, system_idx_list):
        super().__init__(
            systems,
            system_idx_list,
            device_dtype=self.device_dtype,
            device=self.device,
        )


def build_rod(
    *,
    start: np.ndarray,
    n_elem: int = 100,
    base_length: float = 3.0,
    base_radius: float = 0.25,
    density: float = 5000.0,
    youngs_modulus: float = 1.0e6,
    shear_modulus: float,
) -> ea.CosseratRod:
    return ea.CosseratRod.straight_rod(
        n_elem,
        start,
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
    )


def build_cpu_reference_sim(
    *,
    n_elem: int = 100,
    simulation_time: float = 10.0,
    base_length: float = 3.0,
    base_radius: float = 0.25,
    density: float = 5000.0,
    youngs_modulus: float = 1.0e6,
) -> tuple[TimoshenkoReferenceSimulator, ea.CosseratRod, float]:
    poisson_ratio = 99.0
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)
    base_area = np.pi * base_radius**2
    damping_constant = 0.1 / 7.0 / density / base_area
    dl = base_length / n_elem
    dt = 0.07 * dl
    end_force = np.array([-15.0, 0.0, 0.0], dtype=np.float64)

    sim = TimoshenkoReferenceSimulator()
    shearable_rod = build_rod(
        start=np.zeros(3),
        n_elem=n_elem,
        base_length=base_length,
        base_radius=base_radius,
        density=density,
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
    )
    sim.append(shearable_rod)
    sim.dampen(shearable_rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=dt,
    )
    sim.constrain(shearable_rod).using(
        ea.OneEndFixedBC,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
    )
    sim.add_forcing_to(shearable_rod).using(
        ea.EndpointForces,
        0.0 * end_force,
        end_force,
        ramp_up_time=simulation_time / 2.0,
    )

    sim.finalize()
    return sim, shearable_rod, dt


def build_jax_sim(
    *,
    device: jax.Device,
    device_dtype: np.dtype,
    n_elem: int = 100,
    simulation_time: float = 10.0,
    base_length: float = 3.0,
    base_radius: float = 0.25,
    density: float = 5000.0,
    youngs_modulus: float = 1.0e6,
) -> tuple[TimoshenkoJAXSimulator, ea.MemoryBlockCosseratRodJax, float]:
    poisson_ratio = 99.0
    shearable_shear_modulus = youngs_modulus / (poisson_ratio + 1.0)
    base_area = np.pi * base_radius**2
    damping_constant = 0.1 / 7.0 / density / base_area
    dl = base_length / n_elem
    dt = 0.07 * dl
    end_force = np.array([-15.0, 0.0, 0.0], dtype=np.float64)

    _ConfiguredTimoshenkoMemoryBlock.device = device
    _ConfiguredTimoshenkoMemoryBlock.device_dtype = np.dtype(device_dtype)

    sim = TimoshenkoJAXSimulator()
    sim.enable_block_supports(ea.CosseratRod, _ConfiguredTimoshenkoMemoryBlock)
    rod = build_rod(
        start=np.zeros(3),
        n_elem=n_elem,
        base_length=base_length,
        base_radius=base_radius,
        density=density,
        youngs_modulus=youngs_modulus,
        shear_modulus=shearable_shear_modulus,
    )
    sim.append(rod)
    sim.using(rod).operate(ea.OneEndFixedJax)
    sim.using(rod).operate(
        ea.EndpointForcesJax,
        0.0 * end_force,
        end_force,
        ramp_up_time=simulation_time / 2.0,
    )
    sim.using(rod).operate(
        ea.AnalyticalLinearDamperJax,
        time_step=np.float64(dt),
        damping_constant=damping_constant,
    )

    sim.finalize()
    block = tuple(sim.final_systems())[0]
    return sim, block, dt


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


def _collect_cpu_state(
    shearable_rod: ea.CosseratRod,
) -> dict[str, np.ndarray]:
    return {
        "shearable_position_collection": shearable_rod.position_collection.copy(),
        "shearable_director_collection": shearable_rod.director_collection.copy(),
        "shearable_velocity_collection": shearable_rod.velocity_collection.copy(),
        "shearable_omega_collection": shearable_rod.omega_collection.copy(),
        "shearable_sigma": shearable_rod.sigma.copy(),
        "shearable_kappa": shearable_rod.kappa.copy(),
    }


def _collect_block_state(block: ea.MemoryBlockCosseratRodJax) -> dict[str, np.ndarray]:
    state = block.jax_get_state()
    node_slice = slice(
        int(block.start_idx_in_rod_nodes[0]),
        int(block.end_idx_in_rod_nodes[0]),
    )
    elem_slice = slice(
        int(block.start_idx_in_rod_elems[0]),
        int(block.end_idx_in_rod_elems[0]),
    )
    voronoi_slice = slice(
        int(block.start_idx_in_rod_voronoi[0]),
        int(block.end_idx_in_rod_voronoi[0]),
    )
    return {
        "shearable_position_collection": np.asarray(state["position_collection"])[
            :, node_slice
        ],
        "shearable_director_collection": np.asarray(state["director_collection"])[
            :, :, elem_slice
        ],
        "shearable_velocity_collection": np.asarray(state["velocity_collection"])[
            :, node_slice
        ],
        "shearable_omega_collection": np.asarray(state["omega_collection"])[
            :, elem_slice
        ],
        "shearable_sigma": np.asarray(state["sigma"])[:, elem_slice],
        "shearable_kappa": np.asarray(state["kappa"])[:, voronoi_slice],
    }


def run_cpu_reference(
    *,
    final_time: float,
    n_elem: int = 100,
) -> tuple[dict[str, np.ndarray], float, float]:
    sim, shearable_rod, dt = build_cpu_reference_sim(
        n_elem=n_elem,
        simulation_time=final_time,
    )
    stepper = ea.PositionVerlet()
    total_steps = int(final_time / dt)
    snapped_final_time = total_steps * dt
    time_value = np.float64(0.0)

    start = time.perf_counter()
    for _ in range(total_steps):
        time_value = stepper.step(sim, time_value, np.float64(dt))
    elapsed = time.perf_counter() - start
    assert np.isclose(time_value, snapped_final_time), (
        "CPU timoshenko rollout did not end on the expected time grid."
    )
    return _collect_cpu_state(shearable_rod), elapsed, dt


def run_jax_rollout(
    *,
    device: jax.Device,
    device_dtype: np.dtype,
    final_time: float,
    n_elem: int = 100,
) -> tuple[dict[str, np.ndarray], float, float]:
    sim, block, dt = build_jax_sim(
        device=device,
        device_dtype=device_dtype,
        n_elem=n_elem,
        simulation_time=final_time,
    )
    stepper = ea.PositionVerletGPU()
    total_steps = int(final_time / dt)
    snapped_final_time = total_steps * dt
    initial_state = dict(block.jax_get_state())

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
    return _collect_block_state(block), elapsed, dt


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
    parser.add_argument("--n-elem", type=int, default=100)
    parser.add_argument(
        "--final-time",
        type=float,
        default=10.0,
        help="Final simulation time for the planar bending validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend_name, device = select_device(args.backend)
    dtype = preferred_dtype(device)

    cpu_state, cpu_elapsed, dt = run_cpu_reference(
        n_elem=args.n_elem,
        final_time=args.final_time,
    )
    jax_state, jax_elapsed, _ = run_jax_rollout(
        device=device,
        device_dtype=dtype,
        n_elem=args.n_elem,
        final_time=args.final_time,
    )
    total_steps = int(args.final_time / dt)

    print(f"Selected backend alias: {backend_name}")
    print(f"JAX device: {device} (platform={device.platform})")
    print(f"JAX rollout dtype: {dtype}")
    print(f"Timoshenko rollout steps: {total_steps}")
    print(f"CPU reference elapsed: {cpu_elapsed:.4f} s")
    print(f"JAX rollout elapsed: {jax_elapsed:.4f} s")
    print("Max absolute differences vs CPU reference:")
    for key in (
        "shearable_position_collection",
        "shearable_director_collection",
        "shearable_velocity_collection",
        "shearable_omega_collection",
        "shearable_sigma",
        "shearable_kappa",
    ):
        print(f"  {key}: {max_abs_diff(jax_state[key], cpu_state[key]):.3e}")


if __name__ == "__main__":
    main()

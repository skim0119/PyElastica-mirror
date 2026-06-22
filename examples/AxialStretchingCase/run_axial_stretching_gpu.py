"""
Axial Stretching GPU Prototype
==============================

This example mirrors the original axial stretching case while swapping the
host-side forcing, damping, and constraint modules for JAX-compatible
operators. It is meant to validate the linear dynamics path before moving to
examples with stronger rotational coupling.
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


class AxialStretchingReferenceSimulator(
    ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Damping
):
    pass


class AxialStretchingJAXSimulator(ea.BaseSystemCollection, ea.JAXOps):
    pass


class _ConfiguredAxialMemoryBlock(ea.MemoryBlockCosseratRodJax):
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
    n_elem: int = 19,
    base_length: float = 1.0,
    base_radius: float = 0.025,
    density: float = 1000.0,
    youngs_modulus: float = 1.0e4,
    poisson_ratio: float = 0.5,
) -> ea.CosseratRod:
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)
    return ea.CosseratRod.straight_rod(
        n_elem,
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
    )


def build_cpu_reference_sim(
    *,
    n_elem: int = 19,
    base_length: float = 1.0,
    base_radius: float = 0.025,
    density: float = 1000.0,
    youngs_modulus: float = 1.0e4,
    poisson_ratio: float = 0.5,
    end_force_x: float = 1.0,
    damping_constant: float = 0.1,
    time_step: float = 0.1 / 19.0,
) -> tuple[AxialStretchingReferenceSimulator, ea.CosseratRod]:
    sim = AxialStretchingReferenceSimulator()
    rod = build_rod(
        n_elem=n_elem,
        base_length=base_length,
        base_radius=base_radius,
        density=density,
        youngs_modulus=youngs_modulus,
        poisson_ratio=poisson_ratio,
    )
    sim.append(rod)
    sim.constrain(rod).using(
        ea.OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )
    end_force = np.array([end_force_x, 0.0, 0.0], dtype=np.float64)
    sim.add_forcing_to(rod).using(
        ea.EndpointForces, 0.0 * end_force, end_force, ramp_up_time=1.0e-2
    )
    sim.dampen(rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=time_step,
    )
    sim.finalize()
    return sim, rod


def build_jax_sim(
    *,
    device: jax.Device,
    device_dtype: np.dtype,
    n_elem: int = 19,
    base_length: float = 1.0,
    base_radius: float = 0.025,
    density: float = 1000.0,
    youngs_modulus: float = 1.0e4,
    poisson_ratio: float = 0.5,
    end_force_x: float = 1.0,
    damping_constant: float = 0.1,
    time_step: float = 0.1 / 19.0,
) -> tuple[AxialStretchingJAXSimulator, ea.MemoryBlockCosseratRodJax]:
    _ConfiguredAxialMemoryBlock.device = device
    _ConfiguredAxialMemoryBlock.device_dtype = np.dtype(device_dtype)

    sim = AxialStretchingJAXSimulator()
    sim.enable_block_supports(ea.CosseratRod, _ConfiguredAxialMemoryBlock)
    rod = build_rod(
        n_elem=n_elem,
        base_length=base_length,
        base_radius=base_radius,
        density=density,
        youngs_modulus=youngs_modulus,
        poisson_ratio=poisson_ratio,
    )
    sim.append(rod)
    sim.using(rod).operate(ea.OneEndFixedJax)
    end_force = np.array([end_force_x, 0.0, 0.0], dtype=np.float64)
    sim.using(rod).operate(
        ea.EndpointForcesJax,
        0.0 * end_force,
        end_force,
        ramp_up_time=1.0e-2,
    )
    sim.using(rod).operate(
        ea.AnalyticalLinearDamperJax,
        time_step=np.float64(time_step),
        damping_constant=damping_constant,
    )
    sim.finalize()
    block = tuple(sim.final_systems())[0]
    return sim, block


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


def run_cpu_reference(
    *,
    final_time: float,
    time_step: float,
    **kwargs: float | int,
) -> tuple[dict[str, np.ndarray], float]:
    sim, rod = build_cpu_reference_sim(time_step=time_step, **kwargs)
    stepper = ea.PositionVerlet()
    total_steps = int(final_time / time_step)
    snapped_final_time = total_steps * time_step
    time_value = np.float64(0.0)
    dt = np.float64(time_step)

    start = time.perf_counter()
    for _ in range(total_steps):
        time_value = stepper.step(sim, time_value, dt)
    elapsed = time.perf_counter() - start
    assert np.isclose(time_value, snapped_final_time), (
        "CPU axial stretching rollout did not end on the expected time grid."
    )

    state = {
        "position_collection": rod.position_collection.copy(),
        "director_collection": rod.director_collection.copy(),
        "velocity_collection": rod.velocity_collection.copy(),
        "omega_collection": rod.omega_collection.copy(),
        "internal_forces": rod.internal_forces.copy(),
        "internal_torques": rod.internal_torques.copy(),
        "sigma": rod.sigma.copy(),
        "kappa": rod.kappa.copy(),
    }
    return state, elapsed


def run_jax_rollout(
    *,
    device: jax.Device,
    device_dtype: np.dtype,
    final_time: float,
    time_step: float,
    **kwargs: float | int,
) -> tuple[dict[str, np.ndarray], float]:
    sim, block = build_jax_sim(
        device=device,
        device_dtype=device_dtype,
        time_step=time_step,
        **kwargs,
    )
    stepper = ea.PositionVerletGPU()
    total_steps = int(final_time / time_step)
    snapped_final_time = total_steps * time_step
    initial_state = dict(block.jax_get_state())
    stepper.integrate(
        sim,
        time=np.float64(0.0),
        final_time=np.float64(snapped_final_time),
        dt=np.float64(time_step),
    )
    jax.block_until_ready(block.position_collection_device)

    block.jax_set_state(initial_state)
    start = time.perf_counter()
    stepper.integrate(
        sim,
        time=np.float64(0.0),
        final_time=np.float64(snapped_final_time),
        dt=np.float64(time_step),
    )
    jax.block_until_ready(block.position_collection_device)
    elapsed = time.perf_counter() - start

    return {key: np.asarray(value) for key, value in block.jax_get_state().items()}, elapsed


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
    parser.add_argument("--n-elem", type=int, default=19)
    parser.add_argument("--final-time", type=float, default=0.2)
    parser.add_argument("--time-step", type=float, default=0.1 / 19.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend_name, device = select_device(args.backend)
    dtype = preferred_dtype(device)
    total_steps = int(args.final_time / args.time_step)
    snapped_final_time = total_steps * args.time_step

    cpu_state, cpu_elapsed = run_cpu_reference(
        n_elem=args.n_elem,
        final_time=snapped_final_time,
        time_step=args.time_step,
    )
    jax_state, jax_elapsed = run_jax_rollout(
        device=device,
        device_dtype=dtype,
        n_elem=args.n_elem,
        final_time=snapped_final_time,
        time_step=args.time_step,
    )

    print(f"Selected backend alias: {backend_name}")
    print(f"JAX device: {device} (platform={device.platform})")
    print(f"JAX rollout dtype: {dtype}")
    print(f"Axial stretching rollout steps: {total_steps}")
    print(f"CPU reference elapsed: {cpu_elapsed:.4f} s")
    print(f"JAX rollout elapsed: {jax_elapsed:.4f} s")
    print("Max absolute differences vs CPU reference:")
    for key in (
        "position_collection",
        "director_collection",
        "velocity_collection",
        "omega_collection",
        "internal_forces",
        "internal_torques",
        "sigma",
        "kappa",
    ):
        print(f"  {key}: {max_abs_diff(jax_state[key], cpu_state[key]):.3e}")


if __name__ == "__main__":
    main()

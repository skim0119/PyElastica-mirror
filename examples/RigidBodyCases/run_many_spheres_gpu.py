"""
Many Spheres GPU Prototype
==========================

Demonstrate a JAX-owned rigid-body rollout using many spheres connected by
nearest-neighbor springs and driven by gravity. The first sphere is pinned in
space so the chain hangs and oscillates under the spring network.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import elastica as ea

try:
    import jax
    import jax.numpy as jnp
    from jax import config as jax_config
except ModuleNotFoundError as exc:  # pragma: no cover - runtime-only guard
    raise SystemExit(
        "This example requires JAX. Install the optional GPU dependency first, "
        'for example with `uv add --optional gpu "jax[cuda13]"`.'
    ) from exc


jax_config.update("jax_enable_x64", True)


@jax.jit
def _apply_gravity_and_springs(
    state: dict[str, jax.Array],
    gravity: jax.Array,
    rest_lengths: jax.Array,
    spring_constant: jax.Array,
    damping_constant: jax.Array,
) -> dict[str, jax.Array]:
    position_collection = state["position_collection"]
    velocity_collection = state["velocity_collection"]
    mass = state["mass"]

    external_forces = mass[jnp.newaxis, :] * gravity[:, jnp.newaxis]
    displacement = position_collection[:, 1:] - position_collection[:, :-1]
    distance = jnp.linalg.norm(displacement, axis=0)
    safe_distance = jnp.maximum(distance, jnp.asarray(1.0e-12, dtype=distance.dtype))
    direction = displacement / safe_distance[jnp.newaxis, :]

    relative_velocity = velocity_collection[:, 1:] - velocity_collection[:, :-1]
    axial_relative_velocity = jnp.sum(relative_velocity * direction, axis=0)

    spring_magnitude = spring_constant * (distance - rest_lengths)
    damping_magnitude = damping_constant * axial_relative_velocity
    pair_force = (spring_magnitude + damping_magnitude)[jnp.newaxis, :] * direction

    external_forces = external_forces.at[:, :-1].add(pair_force)
    external_forces = external_forces.at[:, 1:].add(-pair_force)

    updated = dict(state)
    updated["external_forces"] = external_forces
    updated["external_torques"] = jnp.zeros_like(state["external_torques"])
    return updated


@jax.jit
def _constrain_first_sphere_values(
    state: dict[str, jax.Array],
    fixed_position: jax.Array,
    fixed_director: jax.Array,
) -> dict[str, jax.Array]:
    updated = dict(state)
    updated["position_collection"] = state["position_collection"].at[:, 0].set(
        fixed_position
    )
    updated["director_collection"] = state["director_collection"].at[:, :, 0].set(
        fixed_director
    )
    return updated


@jax.jit
def _constrain_first_sphere_rates(
    state: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    updated = dict(state)
    updated["velocity_collection"] = state["velocity_collection"].at[:, 0].set(0.0)
    updated["omega_collection"] = state["omega_collection"].at[:, 0].set(0.0)
    updated["acceleration_collection"] = state["acceleration_collection"].at[:, 0].set(
        0.0
    )
    updated["alpha_collection"] = state["alpha_collection"].at[:, 0].set(0.0)
    return updated


class ManySphereSpringJAXSimulator(ea.BaseSystemCollection):
    def __init__(
        self,
        *,
        gravity: np.ndarray,
        spring_constant: float,
        damping_constant: float,
    ) -> None:
        self.gravity = np.asarray(gravity, dtype=np.float64)
        self.spring_constant = np.float64(spring_constant)
        self.damping_constant = np.float64(damping_constant)
        self._block: ea.MemoryBlockRigidBodyJax | None = None
        self._fixed_position_device: jax.Array | None = None
        self._fixed_director_device: jax.Array | None = None
        self._rest_lengths_device: jax.Array | None = None
        self._spring_constant_device: jax.Array | None = None
        self._damping_constant_device: jax.Array | None = None
        self._gravity_device: jax.Array | None = None
        super().__init__()
        self._feature_group_finalize.append(self._finalize_jax_state)

    def _finalize_jax_state(self) -> None:
        final_systems = tuple(self.final_systems())
        assert len(final_systems) == 1, (
            "ManySphereSpringJAXSimulator expects exactly one block system after finalize."
        )
        block = final_systems[0]
        assert isinstance(block, ea.MemoryBlockRigidBodyJax), (
            "ManySphereSpringJAXSimulator requires MemoryBlockRigidBodyJax block support."
        )
        assert block.n_elems >= 2, "ManySphereSpringJAXSimulator needs at least two spheres."

        self._block = block

        rest_lengths = np.linalg.norm(
            np.diff(block.position_collection, axis=1),
            axis=0,
        )
        assert np.all(rest_lengths > 0.0), (
            "Adjacent spheres must start at distinct positions to define spring rest lengths."
        )

        reference_device = block.position_collection_device.device
        device_dtype = block.device_dtype
        self._fixed_position_device = jax.device_put(
            np.asarray(block.position_collection[:, 0], dtype=device_dtype),
            device=reference_device,
        )
        self._fixed_director_device = jax.device_put(
            np.asarray(block.director_collection[:, :, 0], dtype=device_dtype),
            device=reference_device,
        )
        self._rest_lengths_device = jax.device_put(
            np.asarray(rest_lengths, dtype=device_dtype),
            device=reference_device,
        )
        self._spring_constant_device = jax.device_put(
            np.asarray(self.spring_constant, dtype=device_dtype),
            device=reference_device,
        )
        self._damping_constant_device = jax.device_put(
            np.asarray(self.damping_constant, dtype=device_dtype),
            device=reference_device,
        )
        self._gravity_device = jax.device_put(
            np.asarray(self.gravity, dtype=device_dtype),
            device=reference_device,
        )

    def jax_constrain_values(
        self,
        states: tuple[dict[str, jax.Array], ...],
        time: np.float64,
    ) -> tuple[dict[str, jax.Array], ...]:
        assert len(states) == 1, "ManySphereSpringJAXSimulator expects one block state."
        assert self._fixed_position_device is not None, (
            "Pinned-position metadata must be initialized during finalize()."
        )
        assert self._fixed_director_device is not None, (
            "Pinned-director metadata must be initialized during finalize()."
        )
        return (
            _constrain_first_sphere_values(
                states[0],
                self._fixed_position_device,
                self._fixed_director_device,
            ),
        )

    def jax_synchronize(
        self,
        states: tuple[dict[str, jax.Array], ...],
        time: np.float64,
    ) -> tuple[dict[str, jax.Array], ...]:
        assert len(states) == 1, "ManySphereSpringJAXSimulator expects one block state."
        assert self._gravity_device is not None, (
            "Gravity metadata must be initialized during finalize()."
        )
        assert self._rest_lengths_device is not None, (
            "Spring metadata must be initialized during finalize()."
        )
        assert self._spring_constant_device is not None, (
            "Spring metadata must be initialized during finalize()."
        )
        assert self._damping_constant_device is not None, (
            "Damping metadata must be initialized during finalize()."
        )
        return (
            _apply_gravity_and_springs(
                states[0],
                self._gravity_device,
                self._rest_lengths_device,
                self._spring_constant_device,
                self._damping_constant_device,
            ),
        )

    def jax_constrain_rates(
        self,
        states: tuple[dict[str, jax.Array], ...],
        time: np.float64,
    ) -> tuple[dict[str, jax.Array], ...]:
        assert len(states) == 1, "ManySphereSpringJAXSimulator expects one block state."
        return (_constrain_first_sphere_rates(states[0]),)


class _ConfiguredRigidBodyMemoryBlock(ea.MemoryBlockRigidBodyJax):
    device_dtype = np.dtype(np.float64)
    device = None

    def __init__(self, systems, system_idx_list):
        super().__init__(
            systems,
            system_idx_list,
            device_dtype=self.device_dtype,
            device=self.device,
        )


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


def build_simulator(
    *,
    device: jax.Device,
    device_dtype: np.dtype,
    n_spheres: int,
    sphere_radius: float,
    density: float,
    spacing: float,
    gravity: np.ndarray,
    spring_constant: float,
    damping_constant: float,
) -> tuple[ManySphereSpringJAXSimulator, ea.MemoryBlockRigidBodyJax, list[ea.Sphere]]:
    assert n_spheres >= 2, "n_spheres must be at least 2."
    assert spacing > 2.0 * sphere_radius, (
        "spacing must exceed the sphere diameter so adjacent centers are distinct."
    )

    _ConfiguredRigidBodyMemoryBlock.device = device
    _ConfiguredRigidBodyMemoryBlock.device_dtype = np.dtype(device_dtype)

    simulator = ManySphereSpringJAXSimulator(
        gravity=gravity,
        spring_constant=spring_constant,
        damping_constant=damping_constant,
    )
    simulator.enable_block_supports(ea.Sphere, _ConfiguredRigidBodyMemoryBlock)

    spheres: list[ea.Sphere] = []
    for idx in range(n_spheres):
        center = np.array([spacing * idx, 0.0, 0.0], dtype=np.float64)
        sphere = ea.Sphere(center, sphere_radius, density)
        spheres.append(sphere)
        simulator.append(sphere)

    simulator.finalize()
    block = tuple(simulator.final_systems())[0]
    assert isinstance(block, ea.MemoryBlockRigidBodyJax), (
        "Rigid-body GPU example expected a JAX rigid-body memory block."
    )
    return simulator, block, spheres


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="JAX backend to target for the GPU-style rollout.",
    )
    parser.add_argument("--n-spheres", type=int, default=16)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--density", type=float, default=1000.0)
    parser.add_argument("--spacing", type=float, default=0.14)
    parser.add_argument("--spring-k", type=float, default=2.5e3)
    parser.add_argument("--spring-nu", type=float, default=20.0)
    parser.add_argument("--gravity", type=float, default=9.80665)
    parser.add_argument("--final-time", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=5.0e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend_name, device = select_device(args.backend)
    dtype = preferred_dtype(device)

    simulator, block, spheres = build_simulator(
        device=device,
        device_dtype=dtype,
        n_spheres=args.n_spheres,
        sphere_radius=args.radius,
        density=args.density,
        spacing=args.spacing,
        gravity=np.array([0.0, -args.gravity, 0.0], dtype=np.float64),
        spring_constant=args.spring_k,
        damping_constant=args.spring_nu,
    )

    stepper = ea.PositionVerletGPU()
    total_steps = int(args.final_time / args.dt)
    snapped_final_time = total_steps * args.dt
    assert total_steps > 0, "final_time must be at least one time step."

    stepper.integrate(
        simulator,
        time=np.float64(0.0),
        final_time=np.float64(snapped_final_time),
        dt=np.float64(args.dt),
    )
    jax.block_until_ready(block.position_collection_device)

    start = time.perf_counter()
    stepper.integrate(
        simulator,
        time=np.float64(0.0),
        final_time=np.float64(snapped_final_time),
        dt=np.float64(args.dt),
    )
    jax.block_until_ready(block.position_collection_device)
    elapsed = time.perf_counter() - start

    block.from_device(update_rods=True)

    tip_position = spheres[-1].position_collection[:, 0]
    center_of_mass = np.mean(block.position_collection, axis=1)
    displacement = block.position_collection - block.position_collection[:, :1]
    segment_lengths = np.linalg.norm(displacement[:, 1:] - displacement[:, :-1], axis=0)

    print(f"Selected backend alias: {backend_name}")
    print(f"JAX device: {device} (platform={device.platform})")
    print(f"JAX rollout dtype: {dtype}")
    print(f"Sphere count: {args.n_spheres}")
    print(f"Spring rollout steps: {total_steps}")
    print(f"Elapsed: {elapsed:.4f} s")
    print(f"Tip position: {tip_position}")
    print(f"Center of mass: {center_of_mass}")
    print(
        "Segment length range: "
        f"[{segment_lengths.min():.6f}, {segment_lengths.max():.6f}]"
    )


if __name__ == "__main__":
    main()

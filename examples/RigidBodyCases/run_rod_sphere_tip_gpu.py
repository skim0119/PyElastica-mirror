"""
Rod-Sphere Tip GPU Prototype
============================

Demonstrate a JAX-owned rollout where a fixed-base Cosserat rod is connected at
its tip to a rigid sphere through a spring-damper attachment. Gravity and the
tip-to-sphere interaction are applied through the mixed JAX rod/rigid-body
operator path.
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


class RodSphereTipJAXSimulator(
    ea.BaseSystemCollection,
    ea.JAXOps,
    ea.JAXOpsRodRigidBody,
):
    pass


class _ConfiguredRodMemoryBlock(ea.MemoryBlockCosseratRodJax):
    device_dtype = np.dtype(np.float64)
    device = None

    def __init__(self, systems, system_idx_list):
        super().__init__(
            systems,
            system_idx_list,
            device_dtype=self.device_dtype,
            device=self.device,
        )


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


class TipToSphereSpringJax(ea.NoRodRigidBodyJax):
    """Spring-damper attachment between the rod tip and a point on the sphere."""

    def __init__(
        self,
        *,
        stiffness: float,
        damping: float,
        gravity: np.ndarray,
        point_rigid_body: np.ndarray,
        _first_system: object,
        _second_system: object,
    ) -> None:
        del _first_system, _second_system
        self.stiffness = np.float64(stiffness)
        self.damping = np.float64(damping)
        self.gravity = np.asarray(gravity, dtype=np.float64)
        self.point_rigid_body = np.asarray(point_rigid_body, dtype=np.float64)

    def jax_operate_synchronize(self, rod_view, rigid_body_view, time):
        del time
        dtype = rod_view.external_forces.dtype
        gravity = jnp.asarray(self.gravity, dtype=dtype)
        stiffness = jnp.asarray(self.stiffness, dtype=dtype)
        damping = jnp.asarray(self.damping, dtype=dtype)
        point_rigid_body = jnp.asarray(self.point_rigid_body, dtype=dtype)

        rod_view.external_forces = (
            rod_view.external_forces
            + rod_view.mass[jnp.newaxis, :] * gravity[:, jnp.newaxis]
        )
        rigid_body_view.external_forces = (
            rigid_body_view.external_forces
            + rigid_body_view.mass[jnp.newaxis, :] * gravity[:, jnp.newaxis]
        )

        rod_tip_position = rod_view.position_collection[:, -1]
        rod_tip_velocity = rod_view.velocity_collection[:, -1]

        rigid_body_position = rigid_body_view.position_collection[:, 0]
        rigid_body_velocity = rigid_body_view.velocity_collection[:, 0]
        rigid_body_director = rigid_body_view.director_collection[:, :, 0]
        rigid_body_omega = rigid_body_view.omega_collection[:, 0]

        rigid_body_offset = rigid_body_director.T @ point_rigid_body
        rigid_body_attachment_position = rigid_body_position + rigid_body_offset
        rigid_body_omega_lab = rigid_body_director.T @ rigid_body_omega
        rigid_body_attachment_velocity = (
            rigid_body_velocity + jnp.cross(rigid_body_omega_lab, rigid_body_offset)
        )

        distance = rigid_body_attachment_position - rod_tip_position
        relative_velocity = rigid_body_attachment_velocity - rod_tip_velocity
        connection_force = stiffness * distance + damping * relative_velocity

        rod_view.external_forces = rod_view.external_forces.at[:, -1].add(
            connection_force
        )
        rigid_body_view.external_forces = rigid_body_view.external_forces.at[:, 0].add(
            -connection_force
        )

        rigid_body_torque_lab = jnp.cross(rigid_body_offset, -connection_force)
        rigid_body_torque_local = rigid_body_director @ rigid_body_torque_lab
        rigid_body_view.external_torques = rigid_body_view.external_torques.at[
            :, 0
        ].add(rigid_body_torque_local)

        return rod_view, rigid_body_view


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


def build_rod(
    *,
    n_elem: int,
    base_length: float,
    base_radius: float,
    density: float,
    youngs_modulus: float,
    poisson_ratio: float,
) -> ea.CosseratRod:
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([0.0, 1.0, 0.0])
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)
    return ea.CosseratRod.straight_rod(
        n_elements=n_elem,
        start=np.zeros(3),
        direction=direction,
        normal=normal,
        base_length=base_length,
        base_radius=base_radius,
        density=density,
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
    )


def build_sphere(
    *,
    rod: ea.CosseratRod,
    sphere_radius: float,
    density: float,
) -> ea.Sphere:
    rod_tip_position = rod.position_collection[:, -1]
    sphere_center = rod_tip_position + np.array([0.0, 0.0, sphere_radius])
    return ea.Sphere(
        center=sphere_center,
        base_radius=sphere_radius,
        density=density,
    )


def build_simulator(
    *,
    device: jax.Device,
    device_dtype: np.dtype,
    n_elem: int,
    rod_length: float,
    rod_radius: float,
    rod_density: float,
    rod_youngs_modulus: float,
    rod_poisson_ratio: float,
    sphere_radius: float,
    sphere_density: float,
    joint_stiffness: float,
    joint_damping: float,
    gravity: np.ndarray,
) -> tuple[
    RodSphereTipJAXSimulator,
    ea.MemoryBlockCosseratRodJax,
    ea.MemoryBlockRigidBodyJax,
    ea.CosseratRod,
    ea.Sphere,
]:
    _ConfiguredRodMemoryBlock.device = device
    _ConfiguredRodMemoryBlock.device_dtype = np.dtype(device_dtype)
    _ConfiguredRigidBodyMemoryBlock.device = device
    _ConfiguredRigidBodyMemoryBlock.device_dtype = np.dtype(device_dtype)

    simulator = RodSphereTipJAXSimulator()
    simulator.enable_block_supports(ea.CosseratRod, _ConfiguredRodMemoryBlock)
    simulator.enable_block_supports(ea.Sphere, _ConfiguredRigidBodyMemoryBlock)

    rod = build_rod(
        n_elem=n_elem,
        base_length=rod_length,
        base_radius=rod_radius,
        density=rod_density,
        youngs_modulus=rod_youngs_modulus,
        poisson_ratio=rod_poisson_ratio,
    )
    sphere = build_sphere(
        rod=rod,
        sphere_radius=sphere_radius,
        density=sphere_density,
    )

    simulator.append(rod)
    simulator.append(sphere)

    simulator.using(rod).operate(ea.OneEndFixedJax)
    simulator.using_on(rod, sphere).operate(
        TipToSphereSpringJax,
        stiffness=joint_stiffness,
        damping=joint_damping,
        gravity=gravity,
        point_rigid_body=np.array([0.0, 0.0, -sphere_radius], dtype=np.float64),
    )

    simulator.finalize()
    final_systems = tuple(simulator.final_systems())
    assert len(final_systems) == 2, (
        "Rod-sphere JAX example expects one rod block and one rigid-body block."
    )
    rod_block = next(
        system for system in final_systems if isinstance(system, ea.MemoryBlockCosseratRodJax)
    )
    rigid_body_block = next(
        system for system in final_systems if isinstance(system, ea.MemoryBlockRigidBodyJax)
    )
    return simulator, rod_block, rigid_body_block, rod, sphere


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="JAX backend to target for the GPU-style rollout.",
    )
    parser.add_argument("--n-elem", type=int, default=20)
    parser.add_argument("--rod-length", type=float, default=0.3)
    parser.add_argument("--rod-radius", type=float, default=0.012)
    parser.add_argument("--rod-density", type=float, default=1000.0)
    parser.add_argument("--rod-youngs-modulus", type=float, default=5.0e5)
    parser.add_argument("--rod-poisson-ratio", type=float, default=0.5)
    parser.add_argument("--sphere-radius", type=float, default=0.04)
    parser.add_argument("--sphere-density", type=float, default=1000.0)
    parser.add_argument("--joint-k", type=float, default=2.5e4)
    parser.add_argument("--joint-nu", type=float, default=25.0)
    parser.add_argument("--gravity", type=float, default=9.80665)
    parser.add_argument("--final-time", type=float, default=0.5)
    parser.add_argument("--dt", type=float, default=1.0e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend_name, device = select_device(args.backend)
    dtype = preferred_dtype(device)

    simulator, rod_block, rigid_body_block, rod, sphere = build_simulator(
        device=device,
        device_dtype=dtype,
        n_elem=args.n_elem,
        rod_length=args.rod_length,
        rod_radius=args.rod_radius,
        rod_density=args.rod_density,
        rod_youngs_modulus=args.rod_youngs_modulus,
        rod_poisson_ratio=args.rod_poisson_ratio,
        sphere_radius=args.sphere_radius,
        sphere_density=args.sphere_density,
        joint_stiffness=args.joint_k,
        joint_damping=args.joint_nu,
        gravity=np.array([0.0, -args.gravity, 0.0], dtype=np.float64),
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
    jax.block_until_ready(rod_block.position_collection_device)
    jax.block_until_ready(rigid_body_block.position_collection_device)

    start = time.perf_counter()
    stepper.integrate(
        simulator,
        time=np.float64(0.0),
        final_time=np.float64(snapped_final_time),
        dt=np.float64(args.dt),
    )
    jax.block_until_ready(rod_block.position_collection_device)
    jax.block_until_ready(rigid_body_block.position_collection_device)
    elapsed = time.perf_counter() - start

    rod_block.from_device(update_rods=True)
    rigid_body_block.from_device(update_rods=True)

    rod_tip_position = rod.position_collection[:, -1]
    sphere_center = sphere.position_collection[:, 0]
    sphere_attachment_point = (
        sphere.position_collection[:, 0]
        + sphere.director_collection[:, :, 0].T @ np.array([0.0, 0.0, -sphere.radius])
    )
    tip_to_attachment_distance = np.linalg.norm(sphere_attachment_point - rod_tip_position)

    print(f"Selected backend alias: {backend_name}")
    print(f"JAX device: {device} (platform={device.platform})")
    print(f"JAX rollout dtype: {dtype}")
    print(f"Rod-sphere rollout steps: {total_steps}")
    print(f"Elapsed: {elapsed:.4f} s")
    print(f"Rod tip position: {rod_tip_position}")
    print(f"Sphere center: {sphere_center}")
    print(f"Sphere attachment point: {sphere_attachment_point}")
    print(f"Tip-to-attachment distance: {tip_to_attachment_distance:.6e}")


if __name__ == "__main__":
    main()

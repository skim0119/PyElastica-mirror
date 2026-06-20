"""
Continuum Snake GPU Prototype
=============================

This example is a reduced, JAX-backed prototype of the continuum snake case.
It keeps the rod initialization and muscle-actuation parameters from the
original example, but removes callbacks, damping, and rod-plane contact so the
rollout can stay fully inside a JAX loop on the selected accelerator.

The script can:

1. Run a pure JAX Position-Verlet rollout on CPU, Metal/MPS, or CUDA.
2. Compare the final state against a CPU PyElastica reference on the same
   reduced problem.

It is intended as a framework-validation example rather than a replacement for
the full continuum snake benchmark.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from functools import partial

import numpy as np

import elastica as ea

try:
    import jax
    from jax import config as jax_config
    import jax.numpy as jnp
except ModuleNotFoundError as exc:  # pragma: no cover - runtime-only guard
    raise SystemExit(
        "This example requires JAX. Install the optional GPU dependency first, "
        'for example with `uv add --optional gpu "jax[cuda13]"`.'
    ) from exc


jax_config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class SnakeConfig:
    n_elem: int = 50
    period: float = 2.0
    final_time: float = 0.002
    time_step: float = 1.0e-4
    base_length: float = 0.35
    density: float = 1000.0
    youngs_modulus: float = 1.0e6
    poisson_ratio: float = 0.5
    gravitational_acc: float = -9.80665

    @property
    def base_radius(self) -> float:
        return self.base_length * 0.011

    @property
    def shear_modulus(self) -> float:
        return self.youngs_modulus / (self.poisson_ratio + 1.0)

    @property
    def total_steps(self) -> int:
        return int(self.final_time / self.time_step)


def default_b_coeff() -> np.ndarray:
    return np.array([3.4, 3.3, 4.2, 2.6, 3.6, 3.5, 1.0], dtype=np.float64)


def build_rod(config: SnakeConfig) -> ea.CosseratRod:
    return ea.CosseratRod.straight_rod(
        config.n_elem,
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        config.base_length,
        config.base_radius,
        config.density,
        youngs_modulus=config.youngs_modulus,
        shear_modulus=config.shear_modulus,
    )


class SnakeForcingReference(ea.BaseSystemCollection, ea.Forcing):
    pass


def build_cpu_reference_sim(
    config: SnakeConfig, b_coeff: np.ndarray
) -> tuple[SnakeForcingReference, ea.CosseratRod]:
    sim = SnakeForcingReference()
    rod = build_rod(config)
    sim.append(rod)

    normal = np.array([0.0, 1.0, 0.0])
    wave_length = float(b_coeff[-1])
    sim.add_forcing_to(rod).using(
        ea.GravityForces,
        acc_gravity=np.array([0.0, config.gravitational_acc, 0.0]),
    )
    sim.add_forcing_to(rod).using(
        ea.MuscleTorques,
        base_length=config.base_length,
        b_coeff=b_coeff[:-1],
        period=config.period,
        wave_number=2.0 * np.pi / wave_length,
        phase_shift=0.0,
        rest_lengths=rod.rest_lengths,
        ramp_up_time=config.period,
        direction=normal,
        with_spline=True,
    )
    sim.finalize()
    return sim, rod


def run_cpu_reference(
    config: SnakeConfig, b_coeff: np.ndarray
) -> tuple[dict[str, np.ndarray], float]:
    sim, rod = build_cpu_reference_sim(config, b_coeff)
    stepper = ea.PositionVerlet()
    time_value = np.float64(0.0)
    dt = np.float64(config.time_step)

    start = time.perf_counter()
    for _ in range(config.total_steps):
        time_value = stepper.step(sim, time_value, dt)
    elapsed = time.perf_counter() - start

    state = {
        "position_collection": rod.position_collection.copy(),
        "director_collection": rod.director_collection.copy(),
        "velocity_collection": rod.velocity_collection.copy(),
        "omega_collection": rod.omega_collection.copy(),
        "acceleration_collection": rod.acceleration_collection.copy(),
        "alpha_collection": rod.alpha_collection.copy(),
        "internal_forces": rod.internal_forces.copy(),
        "internal_torques": rod.internal_torques.copy(),
        "sigma": rod.sigma.copy(),
        "kappa": rod.kappa.copy(),
        "lengths": rod.lengths.copy(),
        "tangents": rod.tangents.copy(),
        "radius": rod.radius.copy(),
        "dilatation": rod.dilatation.copy(),
        "voronoi_dilatation": rod.voronoi_dilatation.copy(),
    }
    return state, elapsed


def build_gpu_problem(
    config: SnakeConfig, b_coeff: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rod = build_rod(config)

    torque_template = ea.MuscleTorques(
        base_length=config.base_length,
        b_coeff=b_coeff[:-1],
        period=config.period,
        wave_number=2.0 * np.pi / float(b_coeff[-1]),
        phase_shift=0.0,
        direction=np.array([0.0, 1.0, 0.0]),
        rest_lengths=rod.rest_lengths,
        ramp_up_time=config.period,
        with_spline=True,
    )

    state = {
        "position_collection": rod.position_collection.copy(),
        "director_collection": rod.director_collection.copy(),
        "velocity_collection": rod.velocity_collection.copy(),
        "omega_collection": rod.omega_collection.copy(),
        "acceleration_collection": rod.acceleration_collection.copy(),
        "alpha_collection": rod.alpha_collection.copy(),
        "internal_forces": rod.internal_forces.copy(),
        "internal_torques": rod.internal_torques.copy(),
        "external_forces": rod.external_forces.copy(),
        "external_torques": rod.external_torques.copy(),
        "internal_stress": rod.internal_stress.copy(),
        "internal_couple": rod.internal_couple.copy(),
        "sigma": rod.sigma.copy(),
        "kappa": rod.kappa.copy(),
        "lengths": rod.lengths.copy(),
        "tangents": rod.tangents.copy(),
        "radius": rod.radius.copy(),
        "dilatation": rod.dilatation.copy(),
        "dilatation_rate": rod.dilatation_rate.copy(),
        "voronoi_dilatation": rod.voronoi_dilatation.copy(),
    }

    constants = {
        "mass": rod.mass.copy(),
        "volume": rod.volume.copy(),
        "rest_lengths": rod.rest_lengths.copy(),
        "rest_voronoi_lengths": rod.rest_voronoi_lengths.copy(),
        "rest_sigma": rod.rest_sigma.copy(),
        "rest_kappa": rod.rest_kappa.copy(),
        "shear_matrix": rod.shear_matrix.copy(),
        "bend_matrix": rod.bend_matrix.copy(),
        "mass_second_moment_of_inertia": rod.mass_second_moment_of_inertia.copy(),
        "inv_mass_second_moment_of_inertia": (
            rod.inv_mass_second_moment_of_inertia.copy()
        ),
        "gravity": np.array([0.0, config.gravitational_acc, 0.0], dtype=np.float64),
        "muscle_direction": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "muscle_s": np.asarray(torque_template.s, dtype=np.float64),
        "muscle_spline": np.asarray(torque_template.my_spline, dtype=np.float64),
        "muscle_angular_frequency": np.float64(2.0 * np.pi / config.period),
        "muscle_wave_number": np.float64(2.0 * np.pi / float(b_coeff[-1])),
        "muscle_phase_shift": np.float64(0.0),
        "muscle_ramp_up_time": np.float64(config.period),
    }
    return state, constants


def _batch_matvec(
    matrix_collection: jax.Array, vector_collection: jax.Array
) -> jax.Array:
    return (
        matrix_collection[:, 0, :] * vector_collection[0][None, :]
        + matrix_collection[:, 1, :] * vector_collection[1][None, :]
        + matrix_collection[:, 2, :] * vector_collection[2][None, :]
    )


def _batch_matmul(
    first_matrix_collection: jax.Array, second_matrix_collection: jax.Array
) -> jax.Array:
    result = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(
                first_matrix_collection[i, 0, :] * second_matrix_collection[0, j, :]
                + first_matrix_collection[i, 1, :] * second_matrix_collection[1, j, :]
                + first_matrix_collection[i, 2, :] * second_matrix_collection[2, j, :]
            )
        result.append(jnp.stack(row, axis=0))
    return jnp.stack(result, axis=0)


def _batch_cross(
    first_vector_collection: jax.Array, second_vector_collection: jax.Array
) -> jax.Array:
    return jnp.stack(
        (
            first_vector_collection[1] * second_vector_collection[2]
            - first_vector_collection[2] * second_vector_collection[1],
            first_vector_collection[2] * second_vector_collection[0]
            - first_vector_collection[0] * second_vector_collection[2],
            first_vector_collection[0] * second_vector_collection[1]
            - first_vector_collection[1] * second_vector_collection[0],
        ),
        axis=0,
    )


def _batch_dot(
    first_vector_collection: jax.Array, second_vector_collection: jax.Array
) -> jax.Array:
    return jnp.sum(first_vector_collection * second_vector_collection, axis=0)


def _position_difference(position_collection: jax.Array) -> jax.Array:
    return position_collection[:, 1:] - position_collection[:, :-1]


def _position_average(vector: jax.Array) -> jax.Array:
    return 0.5 * (vector[1:] + vector[:-1])


def _two_point_difference_for_single_rod(array_collection: jax.Array) -> jax.Array:
    blocksize = array_collection.shape[1]
    temp_collection = jnp.zeros((3, blocksize + 1), dtype=array_collection.dtype)
    temp_collection = temp_collection.at[:, 0].set(array_collection[:, 0])
    temp_collection = temp_collection.at[:, blocksize].set(-array_collection[:, -1])
    temp_collection = temp_collection.at[:, 1:blocksize].set(
        array_collection[:, 1:] - array_collection[:, :-1]
    )
    return temp_collection


def _trapezoidal_for_single_rod(array_collection: jax.Array) -> jax.Array:
    blocksize = array_collection.shape[1]
    temp_collection = jnp.zeros((3, blocksize + 1), dtype=array_collection.dtype)
    temp_collection = temp_collection.at[:, 0].set(0.5 * array_collection[:, 0])
    temp_collection = temp_collection.at[:, blocksize].set(
        0.5 * array_collection[:, -1]
    )
    temp_collection = temp_collection.at[:, 1:blocksize].set(
        0.5 * (array_collection[:, 1:] + array_collection[:, :-1])
    )
    return temp_collection


def _inv_rotate(director_collection: jax.Array) -> jax.Array:
    d0 = director_collection[:, :, :-1]
    d1 = director_collection[:, :, 1:]

    v0 = (
        d1[2, 0] * d0[1, 0]
        + d1[2, 1] * d0[1, 1]
        + d1[2, 2] * d0[1, 2]
        - d1[1, 0] * d0[2, 0]
        - d1[1, 1] * d0[2, 1]
        - d1[1, 2] * d0[2, 2]
    )
    v1 = (
        d1[0, 0] * d0[2, 0]
        + d1[0, 1] * d0[2, 1]
        + d1[0, 2] * d0[2, 2]
        - d1[2, 0] * d0[0, 0]
        - d1[2, 1] * d0[0, 1]
        - d1[2, 2] * d0[0, 2]
    )
    v2 = (
        d1[1, 0] * d0[0, 0]
        + d1[1, 1] * d0[0, 1]
        + d1[1, 2] * d0[0, 2]
        - d1[0, 0] * d0[1, 0]
        - d1[0, 1] * d0[1, 1]
        - d1[0, 2] * d0[1, 2]
    )

    trace = (
        d1[0, 0] * d0[0, 0]
        + d1[0, 1] * d0[0, 1]
        + d1[0, 2] * d0[0, 2]
        + d1[1, 0] * d0[1, 0]
        + d1[1, 1] * d0[1, 1]
        + d1[1, 2] * d0[1, 2]
        + d1[2, 0] * d0[2, 0]
        + d1[2, 1] * d0[2, 1]
        + d1[2, 2] * d0[2, 2]
    )
    trace = jnp.clip(trace, -1.0, 3.0)
    theta = jnp.arccos(0.5 * trace - 0.5) + 1.0e-14
    magnitude = -0.5 * theta / jnp.sin(theta)

    return jnp.stack((v0 * magnitude, v1 * magnitude, v2 * magnitude), axis=0)


def _rotation_matrix(scale: jax.Array, axis_collection: jax.Array) -> jax.Array:
    theta = jnp.linalg.norm(axis_collection, axis=0)
    theta_eps = theta + 1.0e-14
    v0 = axis_collection[0] / theta_eps
    v1 = axis_collection[1] / theta_eps
    v2 = axis_collection[2] / theta_eps

    theta = theta * scale
    sin_theta = jnp.sin(theta)
    one_minus_cos_theta = 1.0 - jnp.cos(theta)

    return jnp.stack(
        (
            1.0 - one_minus_cos_theta * (v1 * v1 + v2 * v2),
            sin_theta * v2 + one_minus_cos_theta * v0 * v1,
            -sin_theta * v1 + one_minus_cos_theta * v0 * v2,
            -sin_theta * v2 + one_minus_cos_theta * v0 * v1,
            1.0 - one_minus_cos_theta * (v0 * v0 + v2 * v2),
            sin_theta * v0 + one_minus_cos_theta * v1 * v2,
            sin_theta * v1 + one_minus_cos_theta * v0 * v2,
            -sin_theta * v0 + one_minus_cos_theta * v1 * v2,
            1.0 - one_minus_cos_theta * (v0 * v0 + v1 * v1),
        ),
        axis=0,
    ).reshape(3, 3, axis_collection.shape[1])


def _apply_gravity_and_muscle_torques(
    *,
    time_value: jax.Array,
    director_collection: jax.Array,
    mass: jax.Array,
    gravity: jax.Array,
    muscle_direction: jax.Array,
    muscle_s: jax.Array,
    muscle_spline: jax.Array,
    muscle_angular_frequency: jax.Array,
    muscle_wave_number: jax.Array,
    muscle_phase_shift: jax.Array,
    muscle_ramp_up_time: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    external_forces = gravity[:, None] * mass[None, :]
    external_torques = jnp.zeros(
        (3, director_collection.shape[2]), dtype=director_collection.dtype
    )

    factor = jnp.minimum(1.0, time_value / muscle_ramp_up_time)
    torque_mag = (
        factor
        * muscle_spline
        * jnp.sin(
            muscle_angular_frequency * time_value
            - muscle_wave_number * muscle_s
            + muscle_phase_shift
        )
    )
    torque = muscle_direction[:, None] * torque_mag[::-1][None, :]
    torque_world = _batch_matvec(director_collection, torque)

    external_torques = external_torques.at[:, 1:].add(torque_world[:, 1:])
    external_torques = external_torques.at[:, :-1].add(
        -_batch_matvec(director_collection[:, :, :-1], torque[:, 1:])
    )
    return external_forces, external_torques


def _compute_internal_forces_and_torques(
    state: dict[str, jax.Array], constants: dict[str, jax.Array]
) -> dict[str, jax.Array]:
    position_diff = _position_difference(state["position_collection"])
    lengths = jnp.linalg.norm(position_diff, axis=0) + 1.0e-14
    tangents = position_diff / lengths[None, :]
    radius = jnp.sqrt(constants["volume"] / lengths / jnp.pi)
    dilatation = lengths / constants["rest_lengths"]
    voronoi_lengths = _position_average(lengths)
    voronoi_dilatation = voronoi_lengths / constants["rest_voronoi_lengths"]

    sigma = dilatation[None, :] * _batch_matvec(
        state["director_collection"], tangents
    ) - jnp.array([[0.0], [0.0], [1.0]], dtype=state["position_collection"].dtype)
    internal_stress = _batch_matvec(
        constants["shear_matrix"], sigma - constants["rest_sigma"]
    )

    cosserat_internal_stress = jnp.transpose(state["director_collection"], (1, 0, 2))
    cosserat_internal_stress = _batch_matvec(cosserat_internal_stress, internal_stress)
    cosserat_internal_stress = cosserat_internal_stress / dilatation[None, :]
    internal_forces = _two_point_difference_for_single_rod(cosserat_internal_stress)

    kappa = (
        _inv_rotate(state["director_collection"])
        / constants["rest_voronoi_lengths"][None, :]
    )
    internal_couple = _batch_matvec(
        constants["bend_matrix"], kappa - constants["rest_kappa"]
    )

    r_dot_v = _batch_dot(state["position_collection"], state["velocity_collection"])
    r_plus_one_dot_v = _batch_dot(
        state["position_collection"][:, 1:],
        state["velocity_collection"][:, :-1],
    )
    r_dot_v_plus_one = _batch_dot(
        state["position_collection"][:, :-1],
        state["velocity_collection"][:, 1:],
    )
    dilatation_rate = (
        (r_dot_v[:-1] + r_dot_v[1:] - r_dot_v_plus_one - r_plus_one_dot_v)
        / lengths
        / constants["rest_lengths"]
    )

    voronoi_dilatation_inv_cube = 1.0 / (voronoi_dilatation**3)
    bend_twist_couple_2d = _two_point_difference_for_single_rod(
        internal_couple * voronoi_dilatation_inv_cube[None, :]
    )
    bend_twist_couple_3d = _trapezoidal_for_single_rod(
        _batch_cross(kappa, internal_couple)
        * constants["rest_voronoi_lengths"][None, :]
        * voronoi_dilatation_inv_cube[None, :]
    )
    shear_stretch_couple = (
        _batch_cross(
            _batch_matvec(state["director_collection"], tangents), internal_stress
        )
        * constants["rest_lengths"][None, :]
    )
    j_omega_upon_e = (
        _batch_matvec(
            constants["mass_second_moment_of_inertia"], state["omega_collection"]
        )
        / dilatation[None, :]
    )
    lagrangian_transport = _batch_cross(j_omega_upon_e, state["omega_collection"])
    unsteady_dilatation = (
        j_omega_upon_e * dilatation_rate[None, :] / dilatation[None, :]
    )
    internal_torques = (
        bend_twist_couple_2d
        + bend_twist_couple_3d
        + shear_stretch_couple
        + lagrangian_transport
        + unsteady_dilatation
    )

    updated = dict(state)
    updated["lengths"] = lengths
    updated["tangents"] = tangents
    updated["radius"] = radius
    updated["dilatation"] = dilatation
    updated["voronoi_dilatation"] = voronoi_dilatation
    updated["sigma"] = sigma
    updated["kappa"] = kappa
    updated["internal_stress"] = internal_stress
    updated["internal_couple"] = internal_couple
    updated["dilatation_rate"] = dilatation_rate
    updated["internal_forces"] = internal_forces
    updated["internal_torques"] = internal_torques
    return updated


def _update_accelerations(
    state: dict[str, jax.Array], constants: dict[str, jax.Array]
) -> dict[str, jax.Array]:
    acceleration_collection = (
        state["internal_forces"] + state["external_forces"]
    ) / constants["mass"][None, :]
    alpha_collection = (
        _batch_matvec(
            constants["inv_mass_second_moment_of_inertia"],
            state["internal_torques"] + state["external_torques"],
        )
        * state["dilatation"][None, :]
    )

    updated = dict(state)
    updated["acceleration_collection"] = acceleration_collection
    updated["alpha_collection"] = alpha_collection
    return updated


def _update_kinematics(
    state: dict[str, jax.Array], prefac: jax.Array
) -> dict[str, jax.Array]:
    position_collection = (
        state["position_collection"] + prefac * state["velocity_collection"]
    )
    rotation_matrix = _rotation_matrix(prefac, state["omega_collection"])
    director_collection = _batch_matmul(rotation_matrix, state["director_collection"])

    updated = dict(state)
    updated["position_collection"] = position_collection
    updated["director_collection"] = director_collection
    return updated


def _update_dynamics(
    state: dict[str, jax.Array], prefac: jax.Array
) -> dict[str, jax.Array]:
    updated = dict(state)
    updated["velocity_collection"] = (
        state["velocity_collection"] + prefac * state["acceleration_collection"]
    )
    updated["omega_collection"] = (
        state["omega_collection"] + prefac * state["alpha_collection"]
    )
    return updated


@partial(jax.jit, static_argnames=("n_steps",))
def rollout_position_verlet(
    initial_state: dict[str, jax.Array],
    constants: dict[str, jax.Array],
    *,
    dt: jax.Array,
    n_steps: int,
) -> dict[str, jax.Array]:
    half_dt = 0.5 * dt

    def body_fn(_, carry):
        time_value, state = carry

        state = _update_kinematics(state, half_dt)
        external_forces, external_torques = _apply_gravity_and_muscle_torques(
            time_value=time_value + half_dt,
            director_collection=state["director_collection"],
            mass=constants["mass"],
            gravity=constants["gravity"],
            muscle_direction=constants["muscle_direction"],
            muscle_s=constants["muscle_s"],
            muscle_spline=constants["muscle_spline"],
            muscle_angular_frequency=constants["muscle_angular_frequency"],
            muscle_wave_number=constants["muscle_wave_number"],
            muscle_phase_shift=constants["muscle_phase_shift"],
            muscle_ramp_up_time=constants["muscle_ramp_up_time"],
        )
        state["external_forces"] = external_forces
        state["external_torques"] = external_torques

        state = _compute_internal_forces_and_torques(state, constants)
        state = _update_accelerations(state, constants)
        state = _update_dynamics(state, dt)
        state = _update_kinematics(state, half_dt)
        state["external_forces"] = jnp.zeros_like(state["external_forces"])
        state["external_torques"] = jnp.zeros_like(state["external_torques"])

        return time_value + dt, state

    _, final_state = jax.lax.fori_loop(
        0, n_steps, body_fn, (jnp.asarray(0.0, dtype=dt.dtype), initial_state)
    )
    return final_state


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

    if requested_backend not in platforms:
        raise RuntimeError(
            f"Requested backend {requested_backend!r} is not available. "
            f"Found: {sorted(platforms)}"
        )
    return requested_backend, platforms[requested_backend]


def preferred_dtype(device: jax.Device) -> np.dtype:
    if device.platform.lower() == "cpu":
        return np.float64
    return np.float32


def to_device_pytree(
    tree: dict[str, np.ndarray], device: jax.Device, dtype: np.dtype
) -> dict[str, jax.Array]:
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(np.asarray(x, dtype=dtype), device=device), tree
    )


def max_abs_diff(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.max(np.abs(first - second)))


def summarize_results(
    cpu_state: dict[str, np.ndarray], gpu_state: dict[str, np.ndarray]
) -> dict[str, float]:
    return {
        "position_collection": max_abs_diff(
            cpu_state["position_collection"], gpu_state["position_collection"]
        ),
        "director_collection": max_abs_diff(
            cpu_state["director_collection"], gpu_state["director_collection"]
        ),
        "velocity_collection": max_abs_diff(
            cpu_state["velocity_collection"], gpu_state["velocity_collection"]
        ),
        "omega_collection": max_abs_diff(
            cpu_state["omega_collection"], gpu_state["omega_collection"]
        ),
        "internal_forces": max_abs_diff(
            cpu_state["internal_forces"], gpu_state["internal_forces"]
        ),
        "internal_torques": max_abs_diff(
            cpu_state["internal_torques"], gpu_state["internal_torques"]
        ),
        "sigma": max_abs_diff(cpu_state["sigma"], gpu_state["sigma"]),
        "kappa": max_abs_diff(cpu_state["kappa"], gpu_state["kappa"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "gpu", "cuda", "mps"),
        default="auto",
        help="Execution backend for the JAX rollout.",
    )
    parser.add_argument(
        "--n-elem", type=int, default=50, help="Number of rod elements."
    )
    parser.add_argument(
        "--final-time",
        type=float,
        default=0.002,
        help="Final simulation time of the reduced snake case.",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=1.0e-4,
        help="Time step used by both the CPU and JAX rollouts.",
    )
    parser.add_argument(
        "--position-tol",
        type=float,
        default=2.0e-4,
        help="Maximum allowed absolute error for final positions.",
    )
    parser.add_argument(
        "--velocity-tol",
        type=float,
        default=2.0e-4,
        help="Maximum allowed absolute error for final velocities.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SnakeConfig(
        n_elem=args.n_elem,
        final_time=args.final_time,
        time_step=args.time_step,
    )
    b_coeff = default_b_coeff()

    backend_name, device = select_device(args.backend)
    dtype = preferred_dtype(device)
    print(f"Selected backend alias: {backend_name}")
    print(f"JAX device: {device} (platform={device.platform})")
    print(f"JAX rollout dtype: {dtype}")
    print(f"Reduced snake rollout steps: {config.total_steps}")

    cpu_state, cpu_elapsed = run_cpu_reference(config, b_coeff)
    print(f"CPU reference elapsed: {cpu_elapsed:.4f} s")

    initial_state, constants = build_gpu_problem(config, b_coeff)
    initial_state_device = to_device_pytree(initial_state, device, dtype)
    constants_device = to_device_pytree(constants, device, dtype)
    dt_device = jax.device_put(np.asarray(config.time_step, dtype=dtype), device=device)

    warm_state = rollout_position_verlet(
        initial_state_device,
        constants_device,
        dt=dt_device,
        n_steps=config.total_steps,
    )
    jax.block_until_ready(warm_state["position_collection"])

    start = time.perf_counter()
    final_state_device = rollout_position_verlet(
        initial_state_device,
        constants_device,
        dt=dt_device,
        n_steps=config.total_steps,
    )
    jax.block_until_ready(final_state_device["position_collection"])
    gpu_elapsed = time.perf_counter() - start
    print(f"JAX rollout elapsed: {gpu_elapsed:.4f} s")

    gpu_state = jax.tree_util.tree_map(np.asarray, final_state_device)
    diffs = summarize_results(cpu_state, gpu_state)

    print("Max absolute differences vs CPU reference:")
    for key, value in diffs.items():
        print(f"  {key}: {value:.3e}")

    failed = False
    if diffs["position_collection"] > args.position_tol:
        failed = True
        print(
            f"Position mismatch {diffs['position_collection']:.3e} exceeds "
            f"tolerance {args.position_tol:.3e}."
        )
    if diffs["velocity_collection"] > args.velocity_tol:
        failed = True
        print(
            f"Velocity mismatch {diffs['velocity_collection']:.3e} exceeds "
            f"tolerance {args.velocity_tol:.3e}."
        )

    if failed:
        print("Verification failed.")
    else:
        print("Verification passed.")


if __name__ == "__main__":
    main()

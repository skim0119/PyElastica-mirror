"""
Benchmark the original PyElastica CPU path against JAX backends.

Internally, the script runs two benchmark components and reports their sum:

1. the currently ported block-kernel stepping path
2. the fully embedded JAX rollout with on-device external loads

That keeps the terminal output to one number per requested backend while still
covering both execution styles.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

import elastica as ea

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This benchmark requires JAX. Install the optional GPU extra first, for example:\n"
        '  uv add --optional gpu "jax[cuda13]"'
    ) from exc


@dataclass
class BenchmarkResult:
    backend: str
    cpu_seconds: float
    accel_seconds: float | None

    @property
    def speedup(self) -> float:
        if self.accel_seconds is None:
            return 1.0
        return self.cpu_seconds / self.accel_seconds


def build_rod(n_elems: int) -> ea.CosseratRod:
    rod = ea.CosseratRod.straight_rod(
        n_elements=n_elems,
        start=np.zeros(3),
        direction=np.array([0.0, 0.0, 1.0]),
        normal=np.array([1.0, 0.0, 0.0]),
        base_length=1.0,
        base_radius=0.01,
        density=1_000.0,
        youngs_modulus=1.0e6,
    )
    rng = np.random.default_rng(20240620)
    rod.velocity_collection[...] = rng.standard_normal(rod.velocity_collection.shape)
    rod.omega_collection[...] = rng.standard_normal(rod.omega_collection.shape)
    rod.acceleration_collection[...] = 0.0
    rod.alpha_collection[...] = 0.0
    rod.external_forces[...] = 0.0
    rod.external_torques[...] = 0.0
    rod.internal_forces[...] = 0.0
    rod.internal_torques[...] = 0.0
    rod.rest_sigma[...] = rng.standard_normal(rod.rest_sigma.shape)
    rod.rest_kappa[...] = rng.standard_normal(rod.rest_kappa.shape)
    return rod


def build_blocks(
    n_elems: int,
    *,
    device_dtype: np.dtype,
) -> tuple[ea.MemoryBlockCosseratRod, ea.MemoryBlockCosseratRodJax]:
    cpu_rod = build_rod(n_elems)
    gpu_rod = build_rod(n_elems)
    cpu_block = ea.MemoryBlockCosseratRod([cpu_rod], [0])
    gpu_block = ea.MemoryBlockCosseratRodJax(
        [gpu_rod], [0], device_dtype=device_dtype
    )
    return cpu_block, gpu_block


def build_cpu_block(n_elems: int) -> ea.MemoryBlockCosseratRod:
    cpu_rod = build_rod(n_elems)
    return ea.MemoryBlockCosseratRod([cpu_rod], [0])


def build_gpu_block(
    n_elems: int, *, device_dtype: np.dtype, device: jax.Device
) -> ea.MemoryBlockCosseratRodJax:
    gpu_rod = build_rod(n_elems)
    with jax.default_device(device):
        return ea.MemoryBlockCosseratRodJax(
            [gpu_rod],
            [0],
            device_dtype=device_dtype,
            device=device,
        )


def get_available_platforms() -> dict[str, jax.Device]:
    available_platforms: dict[str, jax.Device] = {}
    for backend_name in ("cpu", "cuda", "mps", "metal"):
        try:
            devices = jax.devices(backend_name)
        except Exception:
            continue
        if not devices:
            continue
        platform = devices[0].platform.lower()
        if platform not in available_platforms:
            available_platforms[platform] = devices[0]

    # Apple Metal backends may surface as "metal". Accept "mps" as the user-facing alias.
    if "metal" in available_platforms and "mps" not in available_platforms:
        available_platforms["mps"] = available_platforms["metal"]
    return available_platforms


def move_block_to_device(
    block: ea.MemoryBlockCosseratRodJax, device: jax.Device
) -> None:
    for attr in block._normalize_attr_names():
        block._device_state[attr] = jax.device_put(
            jnp.asarray(
                np.asarray(getattr(block, attr)),
                dtype=block.device_dtype,
            ),
            device=device,
        )
    block._device_platform = device.platform
    block._refresh_device_views()


def _device_array(
    array: np.ndarray | jax.Array,
    *,
    dtype: np.dtype,
    device: jax.Device,
) -> jax.Array:
    return jax.device_put(jnp.asarray(array, dtype=dtype), device=device)


def _device_scalar(
    value: float | np.floating,
    *,
    dtype: np.dtype,
    device: jax.Device,
) -> jax.Array:
    return jax.device_put(jnp.asarray(value, dtype=dtype), device=device)


def _batch_matvec_matrix_vector_np(
    matrix_collection: np.ndarray, vector_collection: np.ndarray
) -> np.ndarray:
    out = np.empty_like(vector_collection)
    out[0, :] = (
        matrix_collection[0, 0, :] * vector_collection[0, :]
        + matrix_collection[0, 1, :] * vector_collection[1, :]
        + matrix_collection[0, 2, :] * vector_collection[2, :]
    )
    out[1, :] = (
        matrix_collection[1, 0, :] * vector_collection[0, :]
        + matrix_collection[1, 1, :] * vector_collection[1, :]
        + matrix_collection[1, 2, :] * vector_collection[2, :]
    )
    out[2, :] = (
        matrix_collection[2, 0, :] * vector_collection[0, :]
        + matrix_collection[2, 1, :] * vector_collection[1, :]
        + matrix_collection[2, 2, :] * vector_collection[2, :]
    )
    return out


def _batch_matvec_matrix_vector_jax(
    matrix_collection: jax.Array, vector_collection: jax.Array
) -> jax.Array:
    out = jnp.empty_like(vector_collection)
    out = out.at[0, :].set(
        matrix_collection[0, 0, :] * vector_collection[0, :]
        + matrix_collection[0, 1, :] * vector_collection[1, :]
        + matrix_collection[0, 2, :] * vector_collection[2, :]
    )
    out = out.at[1, :].set(
        matrix_collection[1, 0, :] * vector_collection[0, :]
        + matrix_collection[1, 1, :] * vector_collection[1, :]
        + matrix_collection[1, 2, :] * vector_collection[2, :]
    )
    out = out.at[2, :].set(
        matrix_collection[2, 0, :] * vector_collection[0, :]
        + matrix_collection[2, 1, :] * vector_collection[1, :]
        + matrix_collection[2, 2, :] * vector_collection[2, :]
    )
    return out


def _batch_matmul_3x3_jax(
    first_matrix_collection: jax.Array, second_matrix_collection: jax.Array
) -> jax.Array:
    out = jnp.empty_like(second_matrix_collection)
    out = out.at[0, 0, :].set(
        first_matrix_collection[0, 0, :] * second_matrix_collection[0, 0, :]
        + first_matrix_collection[0, 1, :] * second_matrix_collection[1, 0, :]
        + first_matrix_collection[0, 2, :] * second_matrix_collection[2, 0, :]
    )
    out = out.at[0, 1, :].set(
        first_matrix_collection[0, 0, :] * second_matrix_collection[0, 1, :]
        + first_matrix_collection[0, 1, :] * second_matrix_collection[1, 1, :]
        + first_matrix_collection[0, 2, :] * second_matrix_collection[2, 1, :]
    )
    out = out.at[0, 2, :].set(
        first_matrix_collection[0, 0, :] * second_matrix_collection[0, 2, :]
        + first_matrix_collection[0, 1, :] * second_matrix_collection[1, 2, :]
        + first_matrix_collection[0, 2, :] * second_matrix_collection[2, 2, :]
    )
    out = out.at[1, 0, :].set(
        first_matrix_collection[1, 0, :] * second_matrix_collection[0, 0, :]
        + first_matrix_collection[1, 1, :] * second_matrix_collection[1, 0, :]
        + first_matrix_collection[1, 2, :] * second_matrix_collection[2, 0, :]
    )
    out = out.at[1, 1, :].set(
        first_matrix_collection[1, 0, :] * second_matrix_collection[0, 1, :]
        + first_matrix_collection[1, 1, :] * second_matrix_collection[1, 1, :]
        + first_matrix_collection[1, 2, :] * second_matrix_collection[2, 1, :]
    )
    out = out.at[1, 2, :].set(
        first_matrix_collection[1, 0, :] * second_matrix_collection[0, 2, :]
        + first_matrix_collection[1, 1, :] * second_matrix_collection[1, 2, :]
        + first_matrix_collection[1, 2, :] * second_matrix_collection[2, 2, :]
    )
    out = out.at[2, 0, :].set(
        first_matrix_collection[2, 0, :] * second_matrix_collection[0, 0, :]
        + first_matrix_collection[2, 1, :] * second_matrix_collection[1, 0, :]
        + first_matrix_collection[2, 2, :] * second_matrix_collection[2, 0, :]
    )
    out = out.at[2, 1, :].set(
        first_matrix_collection[2, 0, :] * second_matrix_collection[0, 1, :]
        + first_matrix_collection[2, 1, :] * second_matrix_collection[1, 1, :]
        + first_matrix_collection[2, 2, :] * second_matrix_collection[2, 1, :]
    )
    out = out.at[2, 2, :].set(
        first_matrix_collection[2, 0, :] * second_matrix_collection[0, 2, :]
        + first_matrix_collection[2, 1, :] * second_matrix_collection[1, 2, :]
        + first_matrix_collection[2, 2, :] * second_matrix_collection[2, 2, :]
    )
    return out


def _cpu_apply_external_loads(
    block: ea.MemoryBlockCosseratRod,
    *,
    gravity: np.ndarray,
    spring_anchor: np.ndarray,
    spring_constant: float,
    spring_damping: float,
    torque_vector: np.ndarray,
) -> None:
    block.external_forces[...] = gravity[:, None] * block.mass[None, :]

    tip_displacement = block.position_collection[:, -1] - spring_anchor
    tip_velocity = block.velocity_collection[:, -1]
    tip_force = -spring_constant * tip_displacement - spring_damping * tip_velocity
    block.external_forces[:, -1] += tip_force

    torque_per_element = np.repeat(
        (torque_vector / block.n_elems).reshape(3, 1), block.n_elems, axis=1
    )
    block.external_torques[...] = _batch_matvec_matrix_vector_np(
        block.director_collection, torque_per_element
    )


def cpu_loop(
    block: ea.MemoryBlockCosseratRod,
    n_steps: int,
    dt: np.float64,
    *,
    include_internal_forces: bool,
    warmup_runs: int,
) -> float:
    for _ in range(warmup_runs):
        time_value = np.float64(0.0)
        for _ in range(n_steps):
            if include_internal_forces:
                block.compute_internal_forces_and_torques(time_value)
            block.update_accelerations(time_value, dt)
            block.update_dynamics(time_value, dt)
            block.update_kinematics(time_value, dt)
            block.zeroed_out_external_forces_and_torques(time_value)
            time_value += dt

    start = time.perf_counter()
    time_value = np.float64(0.0)
    for _ in range(n_steps):
        if include_internal_forces:
            block.compute_internal_forces_and_torques(time_value)
        block.update_accelerations(time_value, dt)
        block.update_dynamics(time_value, dt)
        block.update_kinematics(time_value, dt)
        block.zeroed_out_external_forces_and_torques(time_value)
        time_value += dt
    return time.perf_counter() - start


def gpu_loop(
    block: ea.MemoryBlockCosseratRodJax,
    n_steps: int,
    dt: np.float64,
    *,
    include_internal_forces: bool,
    device: jax.Device,
    warmup_runs: int,
) -> float:
    with jax.default_device(device):
        move_block_to_device(block, device)
        block.allow_cpu_fallback = include_internal_forces

        for _ in range(warmup_runs):
            warmup_time = np.float64(0.0)
            for _ in range(n_steps):
                if include_internal_forces:
                    block.compute_internal_forces_and_torques(warmup_time)
                block.update_accelerations(warmup_time, dt)
                block.update_dynamics(warmup_time, dt)
                block.update_kinematics(warmup_time, dt)
                block.zeroed_out_external_forces_and_torques(warmup_time)
                warmup_time += dt
            jax.block_until_ready(block.position_collection_device)

        start = time.perf_counter()
        time_value = np.float64(0.0)
        for _ in range(n_steps):
            if include_internal_forces:
                block.compute_internal_forces_and_torques(time_value)
            block.update_accelerations(time_value, dt)
            block.update_dynamics(time_value, dt)
            block.update_kinematics(time_value, dt)
            block.zeroed_out_external_forces_and_torques(time_value)
            time_value += dt
        jax.block_until_ready(block.position_collection_device)
        elapsed = time.perf_counter() - start

        block.from_device(attrs=("position_collection", "velocity_collection"))
        return elapsed


@jax.jit
def _jax_apply_external_loads(
    position_collection: jax.Array,
    velocity_collection: jax.Array,
    director_collection: jax.Array,
    mass: jax.Array,
    gravity: jax.Array,
    spring_anchor: jax.Array,
    spring_constant: float,
    spring_damping: float,
    torque_vector: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    external_forces = gravity[:, None] * mass[None, :]
    tip_displacement = position_collection[:, -1] - spring_anchor
    tip_velocity = velocity_collection[:, -1]
    tip_force = -spring_constant * tip_displacement - spring_damping * tip_velocity
    external_forces = external_forces.at[:, -1].add(tip_force)

    n_elems = director_collection.shape[2]
    torque_per_element = jnp.broadcast_to(
        (torque_vector / n_elems).reshape(3, 1), (3, n_elems)
    )
    external_torques = _batch_matvec_matrix_vector_jax(
        director_collection, torque_per_element
    )
    return external_forces, external_torques


@jax.jit
def _jax_update_kinematics(
    position_collection: jax.Array,
    director_collection: jax.Array,
    velocity_collection: jax.Array,
    omega_collection: jax.Array,
    prefac: float,
) -> tuple[jax.Array, jax.Array]:
    axis_collection = prefac * omega_collection
    theta = jnp.sqrt(
        axis_collection[0] * axis_collection[0]
        + axis_collection[1] * axis_collection[1]
        + axis_collection[2] * axis_collection[2]
    )
    theta_eps = theta + jnp.asarray(1.0e-14, dtype=axis_collection.dtype)
    v0 = axis_collection[0] / theta_eps
    v1 = axis_collection[1] / theta_eps
    v2 = axis_collection[2] / theta_eps
    sin_theta = jnp.sin(theta)
    one_minus_cos_theta = 1.0 - jnp.cos(theta)

    rot = jnp.stack(
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
    ).reshape(3, 3, omega_collection.shape[1])

    position_collection = position_collection + prefac * velocity_collection
    director_collection = _batch_matmul_3x3_jax(rot, director_collection)
    return position_collection, director_collection


@jax.jit
def _jax_update_accelerations(
    external_forces: jax.Array,
    mass: jax.Array,
    external_torques: jax.Array,
    inv_mass_second_moment_of_inertia: jax.Array,
    dilatation: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    acceleration_collection = external_forces / mass[jnp.newaxis, :]
    alpha_collection = _batch_matvec_matrix_vector_jax(
        inv_mass_second_moment_of_inertia, external_torques
    ) * dilatation[jnp.newaxis, :]
    return acceleration_collection, alpha_collection


@jax.jit
def _jax_update_dynamics(
    velocity_collection: jax.Array,
    omega_collection: jax.Array,
    acceleration_collection: jax.Array,
    alpha_collection: jax.Array,
    prefac: float,
) -> tuple[jax.Array, jax.Array]:
    return (
        velocity_collection + prefac * acceleration_collection,
        omega_collection + prefac * alpha_collection,
    )


def cpu_full_rollout_loop(
    block: ea.MemoryBlockCosseratRod,
    n_steps: int,
    dt: np.float64,
    *,
    gravity: np.ndarray,
    spring_anchor: np.ndarray,
    spring_constant: float,
    spring_damping: float,
    torque_vector: np.ndarray,
    warmup_runs: int,
) -> float:
    half_dt = np.float64(0.5 * dt)
    for _ in range(warmup_runs):
        time_value = np.float64(0.0)
        for _ in range(n_steps):
            block.update_kinematics(time_value, half_dt)
            _cpu_apply_external_loads(
                block,
                gravity=gravity,
                spring_anchor=spring_anchor,
                spring_constant=spring_constant,
                spring_damping=spring_damping,
                torque_vector=torque_vector,
            )
            block.update_accelerations(time_value, dt)
            block.update_dynamics(time_value, dt)
            block.update_kinematics(time_value, half_dt)
            block.zeroed_out_external_forces_and_torques(time_value)
            time_value += dt

    start = time.perf_counter()
    time_value = np.float64(0.0)
    for _ in range(n_steps):
        block.update_kinematics(time_value, half_dt)
        _cpu_apply_external_loads(
            block,
            gravity=gravity,
            spring_anchor=spring_anchor,
            spring_constant=spring_constant,
            spring_damping=spring_damping,
            torque_vector=torque_vector,
        )
        block.update_accelerations(time_value, dt)
        block.update_dynamics(time_value, dt)
        block.update_kinematics(time_value, half_dt)
        block.zeroed_out_external_forces_and_torques(time_value)
        time_value += dt
    return time.perf_counter() - start


def gpu_full_rollout_loop(
    block: ea.MemoryBlockCosseratRodJax,
    n_steps: int,
    dt: np.float64,
    *,
    gravity: np.ndarray,
    spring_anchor: np.ndarray,
    spring_constant: float,
    spring_damping: float,
    torque_vector: np.ndarray,
    device: jax.Device,
    warmup_runs: int,
) -> float:
    with jax.default_device(device):
        move_block_to_device(block, device)
        initial_state = {
            "position_collection": block.position_collection_device,
            "director_collection": block.director_collection_device,
            "velocity_collection": block.velocity_collection_device,
            "omega_collection": block.omega_collection_device,
        }
        constants = {
            "mass": block._device_state["mass"],
            "inv_mass_second_moment_of_inertia": block._device_state[
                "inv_mass_second_moment_of_inertia"
            ],
            "dilatation": block._device_state["dilatation"],
            "gravity": _device_array(
                gravity,
                dtype=block.device_dtype,
                device=device,
            ),
            "spring_anchor": _device_array(
                spring_anchor,
                dtype=block.device_dtype,
                device=device,
            ),
            "torque_vector": _device_array(
                torque_vector,
                dtype=block.device_dtype,
                device=device,
            ),
        }
        half_dt = _device_scalar(0.5 * dt, dtype=block.device_dtype, device=device)
        full_dt = _device_scalar(dt, dtype=block.device_dtype, device=device)
        spring_constant = _device_scalar(
            spring_constant, dtype=block.device_dtype, device=device
        )
        spring_damping = _device_scalar(
            spring_damping, dtype=block.device_dtype, device=device
        )

    def step_fn(state):
        position_collection, director_collection = _jax_update_kinematics(
            state["position_collection"],
            state["director_collection"],
            state["velocity_collection"],
            state["omega_collection"],
            half_dt,
        )
        external_forces, external_torques = _jax_apply_external_loads(
            position_collection,
            state["velocity_collection"],
            director_collection,
            constants["mass"],
            constants["gravity"],
            constants["spring_anchor"],
            spring_constant,
            spring_damping,
            constants["torque_vector"],
        )
        acceleration_collection, alpha_collection = _jax_update_accelerations(
            external_forces,
            constants["mass"],
            external_torques,
            constants["inv_mass_second_moment_of_inertia"],
            constants["dilatation"],
        )
        velocity_collection, omega_collection = _jax_update_dynamics(
            state["velocity_collection"],
            state["omega_collection"],
            acceleration_collection,
            alpha_collection,
            full_dt,
        )
        position_collection, director_collection = _jax_update_kinematics(
            position_collection,
            director_collection,
            velocity_collection,
            omega_collection,
            half_dt,
        )
        return {
            "position_collection": position_collection,
            "director_collection": director_collection,
            "velocity_collection": velocity_collection,
            "omega_collection": omega_collection,
        }

    with jax.default_device(device):
        @jax.jit
        def rollout(state):
            def body_fn(_, body_state):
                return step_fn(body_state)

            return jax.lax.fori_loop(0, n_steps, body_fn, state)

        for _ in range(warmup_runs):
            warm_state = rollout(initial_state)
            jax.block_until_ready(warm_state["position_collection"])

        start = time.perf_counter()
        final_state = rollout(initial_state)
        jax.block_until_ready(final_state["position_collection"])
        elapsed = time.perf_counter() - start

        block._device_state["position_collection"] = final_state["position_collection"]
        block._device_state["director_collection"] = final_state["director_collection"]
        block._device_state["velocity_collection"] = final_state["velocity_collection"]
        block._device_state["omega_collection"] = final_state["omega_collection"]
        block._refresh_device_views()
        block.from_device(attrs=("position_collection", "velocity_collection"))
        return elapsed


def benchmark_mode(
    backend: str,
    device: jax.Device | None,
    n_elems: int,
    n_steps: int,
    dt: np.float64,
    *,
    include_internal_forces: bool,
    warmup_runs: int,
    device_dtype: np.dtype,
) -> BenchmarkResult:
    cpu_block = build_cpu_block(n_elems)
    cpu_seconds = cpu_loop(
        cpu_block,
        n_steps,
        dt,
        include_internal_forces=include_internal_forces,
        warmup_runs=warmup_runs,
    )
    if backend == "numba":
        return BenchmarkResult(backend, cpu_seconds, None)

    assert device is not None
    gpu_block = build_gpu_block(n_elems, device_dtype=device_dtype, device=device)
    accel_seconds = gpu_loop(
        gpu_block,
        n_steps,
        dt,
        include_internal_forces=include_internal_forces,
        device=device,
        warmup_runs=warmup_runs,
    )
    return BenchmarkResult(backend, cpu_seconds, accel_seconds)


def benchmark_full_jax_rollout(
    backend: str,
    device: jax.Device | None,
    n_elems: int,
    n_steps: int,
    dt: np.float64,
    *,
    gravity: np.ndarray,
    spring_anchor: np.ndarray,
    spring_constant: float,
    spring_damping: float,
    torque_vector: np.ndarray,
    warmup_runs: int,
    device_dtype: np.dtype,
) -> BenchmarkResult:
    cpu_block = build_cpu_block(n_elems)
    cpu_seconds = cpu_full_rollout_loop(
        cpu_block,
        n_steps,
        dt,
        gravity=gravity,
        spring_anchor=spring_anchor,
        spring_constant=spring_constant,
        spring_damping=spring_damping,
        torque_vector=torque_vector,
        warmup_runs=warmup_runs,
    )
    if backend == "numba":
        return BenchmarkResult(backend, cpu_seconds, None)

    assert device is not None
    gpu_block = build_gpu_block(n_elems, device_dtype=device_dtype, device=device)
    accel_seconds = gpu_full_rollout_loop(
        gpu_block,
        n_steps,
        dt,
        gravity=gravity,
        spring_anchor=spring_anchor,
        spring_constant=spring_constant,
        spring_damping=spring_damping,
        torque_vector=torque_vector,
        device=device,
        warmup_runs=warmup_runs,
    )
    return BenchmarkResult(backend, cpu_seconds, accel_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CPU and JAX/GPU Cosserat-rod block stepping."
    )
    parser.add_argument(
        "--numba",
        action="store_true",
        help="Run the original PyElastica/Numba benchmark cases.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run the JAX CPU benchmark cases if a CPU device is available.",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Run the JAX CUDA benchmark cases if a CUDA device is available.",
    )
    parser.add_argument(
        "--mps",
        action="store_true",
        help="Run the JAX Metal/MPS benchmark cases if available.",
    )
    parser.add_argument(
        "--n-elems",
        type=int,
        default=20_000,
        help="Number of elements in the straight rod.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=200,
        help="Number of time steps to benchmark.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0e-5,
        help="Time-step size.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of untimed warm-up runs to execute for every benchmark case.",
    )
    parser.add_argument(
        "--device-dtype",
        choices=("float32", "float64"),
        default="float64",
        help="Device-side dtype used for the JAX memory block and benchmark constants.",
    )
    parser.add_argument(
        "--skip-full-jax-rollout",
        action="store_true",
        help="Skip the fully device-side JAX rollout benchmark.",
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=-9.80665,
        help="Gravity acceleration applied in the y direction for the full JAX rollout.",
    )
    parser.add_argument(
        "--spring-k",
        type=float,
        default=5.0e3,
        help="Linear spring constant applied at the rod tip in the full JAX rollout.",
    )
    parser.add_argument(
        "--spring-nu",
        type=float,
        default=5.0,
        help="Linear spring damping coefficient applied at the rod tip.",
    )
    parser.add_argument(
        "--torque",
        type=float,
        default=1.0e-2,
        help="Distributed torque magnitude applied about +z in the full JAX rollout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jax.config.update("jax_enable_x64", args.device_dtype == "float64")
    available_platforms = get_available_platforms()
    device_dtype = np.dtype(args.device_dtype)
    scalar_dtype = np.float64 if args.device_dtype == "float64" else np.float32
    gravity = np.array([0.0, args.gravity, 0.0], dtype=scalar_dtype)
    spring_anchor = np.array([0.0, 0.0, 1.0], dtype=scalar_dtype)
    torque_vector = np.array([0.0, 0.0, args.torque], dtype=scalar_dtype)
    requested_backends = [
        platform
        for platform, enabled in (
            ("numba", args.numba),
            ("cpu", args.cpu),
            ("cuda", args.cuda),
            ("mps", args.mps),
        )
        if enabled
    ]

    print("Benchmark configuration")
    print(f"  JAX backend : {jax.default_backend()}")
    print(f"  devices     : {', '.join(sorted(available_platforms)) or 'none'}")
    print(f"  n_elems     : {args.n_elems}")
    print(f"  n_steps     : {args.n_steps}")
    print(f"  dt          : {args.dt}")
    print(f"  warmups     : {args.warmup_runs}")
    print(f"  dtype       : {args.device_dtype}")
    print()

    if not requested_backends:
        print("Dry run only. No benchmark backends selected.")
        print("Pass one or more of `--numba`, `--cpu`, `--cuda`, or `--mps` to run cases.")
        return

    results: dict[str, BenchmarkResult] = {}
    for backend in requested_backends:
        device = None if backend == "numba" else available_platforms.get(backend)
        if backend != "numba" and device is None:
            print(f"Skipping `{backend}`: backend not available in this JAX environment.")
            continue

        try:
            total_cpu_seconds = 0.0
            total_accel_seconds = 0.0 if backend != "numba" else None

            kernel_result = benchmark_mode(
                backend,
                device,
                args.n_elems,
                args.n_steps,
                scalar_dtype(args.dt),
                include_internal_forces=False,
                warmup_runs=args.warmup_runs,
                device_dtype=device_dtype,
            )
            total_cpu_seconds += kernel_result.cpu_seconds
            if total_accel_seconds is not None:
                total_accel_seconds += kernel_result.accel_seconds

            if not args.skip_full_jax_rollout:
                rollout_result = benchmark_full_jax_rollout(
                    backend,
                    device,
                    args.n_elems,
                    args.n_steps,
                    scalar_dtype(args.dt),
                    gravity=gravity,
                    spring_anchor=spring_anchor,
                    spring_constant=args.spring_k,
                    spring_damping=args.spring_nu,
                    torque_vector=torque_vector,
                    warmup_runs=args.warmup_runs,
                    device_dtype=device_dtype,
                )
                total_cpu_seconds += rollout_result.cpu_seconds
                if total_accel_seconds is not None:
                    total_accel_seconds += rollout_result.accel_seconds

            results[backend] = BenchmarkResult(
                backend=backend,
                cpu_seconds=total_cpu_seconds,
                accel_seconds=total_accel_seconds,
            )
        except Exception as exc:
            print(f"Skipping `{backend}` after runtime failure: {exc}")

    if not results:
        print("No benchmark cases were executed.")
        return

    print("Results")
    for backend in requested_backends:
        result = results.get(backend)
        if result is None:
            continue
        print(f"  backend     : {result.backend}")
        if result.backend == "numba":
            print(f"    CPU       : {result.cpu_seconds:.6f} s")
        else:
            print(f"    CPU       : {result.cpu_seconds:.6f} s")
            print(f"    JAX       : {result.accel_seconds:.6f} s")
            print(f"    speedup   : {result.speedup:.2f}x")

if __name__ == "__main__":
    main()

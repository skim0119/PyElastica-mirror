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
from elastica._jax_linalg import (
    _jax_batch_matmul as _batch_matmul_3x3_jax,
    _jax_batch_matvec as _batch_matvec_matrix_vector_jax,
)
from elastica._jax_rotations import _jax_get_rotation_matrix

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


class BenchmarkCPUFullSimulator(
    ea.BaseSystemCollection, ea.Forcing, ea.Damping
):
    pass


class BenchmarkJAXSimulator(
    ea.BaseSystemCollection, ea.JAXOps
):
    pass


class _ConfiguredBenchmarkMemoryBlock(ea.MemoryBlockCosseratRodJax):
    device_dtype = np.dtype(np.float64)
    device = None

    def __init__(self, systems, system_idx_list):
        super().__init__(
            systems,
            system_idx_list,
            device_dtype=self.device_dtype,
            device=self.device,
        )


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
        dt_device = _device_scalar(dt, dtype=block.device_dtype, device=device)

        for _ in range(warmup_runs):
            warmup_time = np.float64(0.0)
            warmup_state = block.jax_get_state()
            for _ in range(n_steps):
                if include_internal_forces:
                    warmup_state = block.jax_compute_internal_forces_and_torques(
                        warmup_state, warmup_time
                    )
                warmup_state = block.jax_dynamic_step(
                    warmup_state, warmup_time, dt_device
                )
                warmup_state = block.jax_kinematic_step(
                    warmup_state, warmup_time, dt_device
                )
                warmup_state = block.jax_zero_external_loads(warmup_state, warmup_time)
                warmup_time += dt
            block.jax_set_state(warmup_state)
            jax.block_until_ready(block.position_collection_device)

        start = time.perf_counter()
        time_value = np.float64(0.0)
        state = block.jax_get_state()
        for _ in range(n_steps):
            if include_internal_forces:
                state = block.jax_compute_internal_forces_and_torques(state, time_value)
            state = block.jax_dynamic_step(state, time_value, dt_device)
            state = block.jax_kinematic_step(state, time_value, dt_device)
            state = block.jax_zero_external_loads(state, time_value)
            time_value += dt
        block.jax_set_state(state)
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
    rot = _jax_get_rotation_matrix(prefac, omega_collection)
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


def build_cpu_full_sim(
    n_elems: int,
    dt: np.float64,
    *,
    gravity: np.ndarray,
    uniform_damping_constant: float,
) -> tuple[BenchmarkCPUFullSimulator, ea.CosseratRod]:
    sim = BenchmarkCPUFullSimulator()
    rod = build_rod(n_elems)
    sim.append(rod)
    sim.add_forcing_to(rod).using(ea.GravityForces, acc_gravity=gravity)
    sim.dampen(rod).using(
        ea.AnalyticalLinearDamper,
        uniform_damping_constant=uniform_damping_constant,
        time_step=dt,
    )
    sim.finalize()
    return sim, rod


def cpu_full_rollout_loop(
    n_elems: int,
    n_steps: int,
    dt: np.float64,
    *,
    gravity: np.ndarray,
    uniform_damping_constant: float,
    warmup_runs: int,
) -> float:
    stepper = ea.PositionVerlet()

    for _ in range(warmup_runs):
        sim, _ = build_cpu_full_sim(
            n_elems,
            dt,
            gravity=gravity,
            uniform_damping_constant=uniform_damping_constant,
        )
        warmup_time = np.float64(0.0)
        for _ in range(n_steps):
            warmup_time = stepper.step(sim, warmup_time, dt)

    sim, _ = build_cpu_full_sim(
        n_elems,
        dt,
        gravity=gravity,
        uniform_damping_constant=uniform_damping_constant,
    )
    start = time.perf_counter()
    time_value = np.float64(0.0)
    for _ in range(n_steps):
        time_value = stepper.step(sim, time_value, dt)
    return time.perf_counter() - start


def gpu_full_rollout_loop(
    n_elems: int,
    n_steps: int,
    dt: np.float64,
    *,
    gravity: np.ndarray,
    uniform_damping_constant: float,
    device: jax.Device,
    warmup_runs: int,
    device_dtype: np.dtype,
) -> float:
    _ConfiguredBenchmarkMemoryBlock.device = device
    _ConfiguredBenchmarkMemoryBlock.device_dtype = np.dtype(device_dtype)
    stepper = ea.PositionVerletGPU()

    def build_sim():
        sim = BenchmarkJAXSimulator()
        sim.enable_block_supports(ea.CosseratRod, _ConfiguredBenchmarkMemoryBlock)
        rod = build_rod(n_elems)
        sim.append(rod)
        sim.using(rod).operate(
            ea.GravityAnalyticalDamperJax,
            acc_gravity=gravity,
            uniform_damping_constant=uniform_damping_constant,
            time_step=dt,
        )
        sim.finalize()
        block = tuple(sim.final_systems())[0]
        return sim, block

    for _ in range(warmup_runs):
        warm_sim, warm_block = build_sim()
        stepper.integrate(
            warm_sim,
            time=np.float64(0.0),
            final_time=np.float64(n_steps * dt),
            dt=dt,
        )
        jax.block_until_ready(warm_block.position_collection_device)

    timed_sim, timed_block = build_sim()
    start = time.perf_counter()
    stepper.integrate(
        timed_sim,
        time=np.float64(0.0),
        final_time=np.float64(n_steps * dt),
        dt=dt,
    )
    jax.block_until_ready(timed_block.position_collection_device)
    return time.perf_counter() - start


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
    uniform_damping_constant: float,
    warmup_runs: int,
    device_dtype: np.dtype,
) -> BenchmarkResult:
    cpu_seconds = cpu_full_rollout_loop(
        n_elems,
        n_steps,
        dt,
        gravity=gravity,
        uniform_damping_constant=uniform_damping_constant,
        warmup_runs=warmup_runs,
    )
    if backend == "numba":
        return BenchmarkResult(backend, cpu_seconds, None)

    assert device is not None
    accel_seconds = gpu_full_rollout_loop(
        n_elems,
        n_steps,
        dt,
        gravity=gravity,
        uniform_damping_constant=uniform_damping_constant,
        device=device,
        warmup_runs=warmup_runs,
        device_dtype=device_dtype,
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
        "--uniform-damping-constant",
        type=float,
        default=5.0,
        help="Uniform analytical damping constant used in the full-rollout case.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jax.config.update("jax_enable_x64", args.device_dtype == "float64")
    available_platforms = get_available_platforms()
    device_dtype = np.dtype(args.device_dtype)
    scalar_dtype = np.float64 if args.device_dtype == "float64" else np.float32
    gravity = np.array([0.0, args.gravity, 0.0], dtype=scalar_dtype)
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
                    uniform_damping_constant=args.uniform_damping_constant,
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

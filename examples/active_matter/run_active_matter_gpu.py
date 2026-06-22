"""GPU JAX reproduction of the active-matter snake cases.

The simulation keeps all rod dynamics, actuation, broad/fine contact detection,
and contact-force scattering inside one compiled JAX rollout.  Every rod element
is represented by a capsule.  Use ``--case snake-on-plane`` for the four snakes
falling onto a floor or ``--case snake-pit`` for the enclosed random packing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import time

import numpy as np

import elastica as ea

try:
    import jax
    import jax.numpy as jnp
    from jax import config as jax_config
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "This example requires JAX. Install the optional GPU dependency first, "
        'for example with `uv add --optional gpu "jax[cuda13]"`.'
    ) from exc


jax_config.update("jax_enable_x64", True)

CONTACT_THRESHOLD = 1.0e-8
SPLINE_X = np.array(
    [0.05, 0.13, 0.21, 0.29, 0.37, 0.45, 0.53, 0.61, 0.69, 0.77, 0.85, 0.93]
)
SPLINE_Y = np.array(
    [
        0.003394,
        0.003474,
        0.003604,
        0.003697,
        0.003673,
        0.003525,
        0.003330,
        0.003170,
        0.003126,
        0.003219,
        0.003371,
        0.003489,
    ]
)


@dataclass(frozen=True)
class CaseParameters:
    name: str
    n_elements: int
    n_snakes: int = 4
    length: float = 0.35
    radius_ratio: float = 0.011
    density: float = 1000.0
    youngs_modulus: float = 1.0e6
    time_period: float = 2.0
    wave_length: float = 1.0
    contact_stiffness: float = 1.0e4
    contact_damping: float = 1.0e-3
    gravitational_acc: float = 0.0
    damping_rate: float = 1.0e-4
    time_step: float = 5.0e-5
    final_time: float = 20.0
    activation_start_time_nd: float = 5.0
    packing_initial_vertical_span_ratio: float = 1.0
    packing_initial_radial_span_ratio: float = 1.0
    wall_distance_ratio: float = 2.0
    initialization_protocol: str = "random-cylinder"
    steps_between_detection: int = 0

    @property
    def radius(self) -> float:
        return self.radius_ratio * self.length


def case_parameters(name: str) -> CaseParameters:
    if name == "snake-on-plane":
        return CaseParameters(
            name=name,
            n_elements=50,
            gravitational_acc=-9.80665,
            damping_rate=0.01,
            final_time=2.0,
        )
    assert name == "snake-pit", f"Unknown active-matter case {name!r}."
    return CaseParameters(name=name, n_elements=20)


class ActiveMatterJAXSimulator(ea.BaseSystemCollection, ea.JAXOpsBlock):
    pass


class _ConfiguredRodBlock(ea.MemoryBlockCosseratRodJax):
    device = None
    device_dtype = np.dtype(np.float64)

    def __init__(self, systems, system_idx_list):
        super().__init__(
            systems,
            system_idx_list,
            device=self.device,
            device_dtype=self.device_dtype,
        )


def _closest_points_on_segments(p0, p1, q0, q1):
    """Vectorized closest points for equally shaped ``(..., 3)`` segments."""
    u = p1 - p0
    v = q1 - q0
    w = p0 - q0
    a = jnp.sum(u * u, axis=-1)
    b = jnp.sum(u * v, axis=-1)
    c = jnp.sum(v * v, axis=-1)
    d = jnp.sum(u * w, axis=-1)
    e = jnp.sum(v * w, axis=-1)
    denominator = a * c - b * b
    safe = jnp.maximum(denominator, CONTACT_THRESHOLD)
    s = jnp.where(denominator > CONTACT_THRESHOLD, (b * e - c * d) / safe, 0.0)
    s = jnp.clip(s, 0.0, 1.0)
    t = jnp.clip((b * s + e) / jnp.maximum(c, CONTACT_THRESHOLD), 0.0, 1.0)
    s = jnp.clip((b * t - d) / jnp.maximum(a, CONTACT_THRESHOLD), 0.0, 1.0)
    return p0 + s[..., None] * u, q0 + t[..., None] * v


def _scatter_element_loads(
    external_forces,
    external_torques,
    element_indices,
    force,
    torque_world,
    directors,
):
    nodal_force = 0.5 * force
    external_forces = external_forces.at[:, element_indices].add(nodal_force.T)
    external_forces = external_forces.at[:, element_indices + 1].add(nodal_force.T)
    torque_material = jnp.einsum("nij,nj->ni", directors, torque_world)
    external_torques = external_torques.at[:, element_indices].add(torque_material.T)
    return external_forces, external_torques


class ActiveMatterForcingAndContactJax(ea.NoBlockOpJax):
    """Block-wide forcing and capsule contact operator."""

    def __init__(self, *, parameters: CaseParameters, _system) -> None:
        block = _system
        widths = block.end_idx_in_rod_elems - block.start_idx_in_rod_elems
        assert np.all(
            widths == widths[0]
        ), "Active-matter JAX contact requires equal element counts for all rods."
        assert (
            int(widths[0]) == parameters.n_elements
        ), "Packed rod width must match the active-matter element count."
        self.parameters = parameters
        offsets = np.arange(parameters.n_elements, dtype=np.int32)
        self.element_indices = (
            block.start_idx_in_rod_elems[:, None].astype(np.int32) + offsets[None, :]
        )
        self.node_indices = (
            block.start_idx_in_rod_nodes[:, None].astype(np.int32)
            + np.arange(parameters.n_elements + 1, dtype=np.int32)[None, :]
        )

        rod_id = np.repeat(np.arange(parameters.n_snakes), parameters.n_elements)
        local_id = np.tile(np.arange(parameters.n_elements), parameters.n_snakes)
        first, second = np.triu_indices(rod_id.size, k=1)
        keep = rod_id[first] != rod_id[second]
        self.pair_first = first[keep].astype(np.int32)
        self.pair_second = second[keep].astype(np.int32)
        self.flat_rod_id = rod_id.astype(np.int32)
        self.flat_local_id = local_id.astype(np.int32)

        centers = (np.arange(parameters.n_elements) + 0.5) / parameters.n_elements
        # Evaluate the specified natural cubic spline at element centers.
        try:
            from scipy.interpolate import CubicSpline

            self.spline_amplitude = CubicSpline(SPLINE_X, SPLINE_Y)(centers)
        except ModuleNotFoundError:  # pragma: no cover - project depends on SciPy
            self.spline_amplitude = np.interp(centers, SPLINE_X, SPLINE_Y)
        self.wall_origins, self.wall_normals = make_walls(parameters)

    def jax_block_operate_synchronize(self, state, time):
        dtype = state["position_collection"].dtype
        elem = jnp.asarray(self.element_indices).reshape(-1)
        nodes = jnp.asarray(self.node_indices)
        positions = state["position_collection"][:, nodes]
        velocities = state["velocity_collection"][:, nodes]
        masses = state["mass"][nodes]
        centers = 0.5 * (positions[:, :, :-1] + positions[:, :, 1:])
        numerator = masses[:, :-1][None, :, :] * velocities[:, :, :-1]
        numerator += masses[:, 1:][None, :, :] * velocities[:, :, 1:]
        element_velocity = numerator / (masses[:, :-1] + masses[:, 1:])[None, :, :]

        centers = jnp.moveaxis(centers, 0, -1).reshape(-1, 3)
        element_velocity = jnp.moveaxis(element_velocity, 0, -1).reshape(-1, 3)
        axes = state["tangents"][:, elem].T
        lengths = state["lengths"][elem]
        radii = state["radius"][elem]
        directors = jnp.moveaxis(state["director_collection"][:, :, elem], 2, 0)
        omega_material = state["omega_collection"][:, elem].T
        omega_world = jnp.einsum("nji,nj->ni", directors, omega_material)

        gravity_axis = jnp.array(
            (
                [0.0, 1.0, 0.0]
                if self.parameters.name == "snake-on-plane"
                else [0.0, 0.0, 1.0]
            ),
            dtype=dtype,
        )
        gravity = self.parameters.gravitational_acc * gravity_axis
        external_forces = state["mass"][None, :] * gravity[:, None]
        external_torques = jnp.zeros_like(state["external_torques"])

        s = jnp.arange(self.parameters.n_elements, dtype=dtype) + 0.5
        s /= self.parameters.n_elements
        wave = jnp.sin(
            2.0 * jnp.pi * time / self.parameters.time_period
            - 2.0 * jnp.pi * s / self.parameters.wave_length
        )
        if self.parameters.name == "snake-on-plane":
            ramp = jnp.minimum(1.0, time / self.parameters.time_period)
            actuation_scale = 1.0
        else:
            start = (
                self.parameters.activation_start_time_nd * self.parameters.time_period
            )
            phase = jnp.clip((time - start) / self.parameters.time_period, 0.0, 1.0)
            ramp = 0.5 * (1.0 - jnp.cos(jnp.pi * phase))
            actuation_scale = 0.5
        torque_magnitude = (
            actuation_scale
            * ramp
            * jnp.asarray(self.spline_amplitude, dtype=dtype)
            * wave
        )
        torque_world = torque_magnitude[None, None, :] * gravity_axis[None, :, None]
        torque_world = jnp.broadcast_to(
            torque_world, (self.parameters.n_snakes, 3, self.parameters.n_elements)
        )
        torque_field = jnp.einsum(
            "neij,nje->nei",
            directors.reshape(
                self.parameters.n_snakes, self.parameters.n_elements, 3, 3
            ),
            torque_world,
        )
        torque_couple = jnp.zeros_like(torque_field)
        torque_couple = torque_couple.at[:, 1:, :].add(torque_field[:, 1:, :])
        torque_couple = torque_couple.at[:, :-1, :].add(-torque_field[:, 1:, :])
        external_torques = external_torques.at[:, self.element_indices].add(
            jnp.moveaxis(torque_couple, -1, 0)
        )

        (
            external_forces,
            external_torques,
            candidate_mask,
            last_detection_time,
        ) = self._rod_contacts(
            centers,
            element_velocity,
            axes,
            lengths,
            radii,
            omega_world,
            directors,
            elem,
            external_forces,
            external_torques,
            state["active_matter_candidate_mask"],
            state["active_matter_last_detection_time"],
            time,
        )
        external_forces, external_torques = self._wall_contacts(
            centers,
            element_velocity,
            axes,
            lengths,
            radii,
            omega_world,
            directors,
            elem,
            external_forces,
            external_torques,
        )
        updated = dict(state)
        updated["external_forces"] = external_forces
        updated["external_torques"] = external_torques
        updated["active_matter_candidate_mask"] = candidate_mask
        updated["active_matter_last_detection_time"] = last_detection_time
        return updated

    def _contact_force(self, distance, normal, relative_velocity):
        normal_speed = -jnp.sum(relative_velocity * normal, axis=-1)
        stiffness = 0.5 * self.parameters.contact_stiffness
        damping = 0.5 * self.parameters.contact_damping
        magnitude = jnp.maximum(0.0, stiffness * (-distance) + damping * normal_speed)
        tangent_velocity = -relative_velocity - normal_speed[..., None] * normal
        tangent_speed = jnp.linalg.norm(tangent_velocity, axis=-1)
        tangent_magnitude = jnp.minimum(damping * tangent_speed, 1.0e-10 * magnitude)
        tangent = tangent_velocity / jnp.maximum(
            tangent_speed[..., None], CONTACT_THRESHOLD
        )
        active = distance < -CONTACT_THRESHOLD
        return jnp.where(
            active[..., None],
            magnitude[..., None] * normal + tangent_magnitude[..., None] * tangent,
            0.0,
        )

    def _rod_contacts(
        self,
        centers,
        velocities,
        axes,
        lengths,
        radii,
        omega,
        directors,
        elem,
        external_forces,
        external_torques,
        cached_candidates,
        last_detection_time,
        time,
    ):
        first = jnp.asarray(self.pair_first)
        second = jnp.asarray(self.pair_second)
        c1, c2 = centers[first], centers[second]
        a1, a2 = axes[first], axes[second]
        half1, half2 = 0.5 * lengths[first], 0.5 * lengths[second]
        p1, p2 = _closest_points_on_segments(
            c1 - half1[:, None] * a1,
            c1 + half1[:, None] * a1,
            c2 - half2[:, None] * a2,
            c2 + half2[:, None] * a2,
        )
        delta = p1 - p2
        axis_distance = jnp.linalg.norm(delta, axis=-1)
        normal = delta / jnp.maximum(axis_distance[:, None], CONTACT_THRESHOLD)
        distance = axis_distance - radii[first] - radii[second]

        # Uniform hash-grid neighborhoods generate candidates; AABB overlap is
        # the final broad-phase acceptance rule. The dense pair arrays keep the
        # JAX shapes static while masks exclude non-neighboring grid cells.
        cell_size = 2.0 * jnp.max(radii) + jnp.max(lengths)
        cells1 = jnp.floor(c1 / cell_size).astype(jnp.int32)
        cells2 = jnp.floor(c2 / cell_size).astype(jnp.int32)
        grid_neighbors = jnp.all(jnp.abs(cells1 - cells2) <= 1, axis=-1)
        extent1 = half1[:, None] * jnp.abs(a1) + radii[first, None]
        extent2 = half2[:, None] * jnp.abs(a2) + radii[second, None]
        detected_candidates = grid_neighbors & jnp.all(
            jnp.abs(c1 - c2) <= extent1 + extent2, axis=-1
        )
        detection_interval = (
            self.parameters.steps_between_detection * self.parameters.time_step
        )
        detection_due = (detection_interval == 0.0) | (
            time - last_detection_time >= detection_interval
        )
        broad_phase = jnp.where(detection_due, detected_candidates, cached_candidates)
        last_detection_time = jnp.where(detection_due, time, last_detection_time)
        distance = jnp.where(broad_phase, distance, 1.0)
        contact = p2 + (radii[second] + 0.5 * distance)[:, None] * normal
        arm1, arm2 = contact - c1, contact - c2
        v1 = velocities[first] + jnp.cross(omega[first], arm1)
        v2 = velocities[second] + jnp.cross(omega[second], arm2)
        force = self._contact_force(distance, normal, v1 - v2)
        force = jnp.where((axis_distance > CONTACT_THRESHOLD)[:, None], force, 0.0)
        parallel_overlap = (
            jnp.abs(jnp.sum(a1 * a2, axis=-1)) > 1.0 - CONTACT_THRESHOLD
        ) & (jnp.abs(jnp.sum((c2 - c1) * a1, axis=-1)) < lengths[first])
        # The fine detector emits two equivalent contacts over a parallel axial
        # overlap. Their force/torque accumulations can be combined exactly here.
        force *= jnp.where(parallel_overlap, 2.0, 1.0)[:, None]

        f1 = jnp.zeros_like(centers).at[first].add(force)
        f2 = jnp.zeros_like(centers).at[second].add(-force)
        t1 = jnp.zeros_like(centers).at[first].add(jnp.cross(arm1, force))
        t2 = jnp.zeros_like(centers).at[second].add(jnp.cross(arm2, -force))
        total_force, total_torque = f1 + f2, t1 + t2
        external_forces, external_torques = _scatter_element_loads(
            external_forces,
            external_torques,
            elem,
            total_force,
            total_torque,
            directors,
        )
        return external_forces, external_torques, broad_phase, last_detection_time

    def _wall_contacts(
        self,
        centers,
        velocities,
        axes,
        lengths,
        radii,
        omega,
        directors,
        elem,
        external_forces,
        external_torques,
    ):
        origins = jnp.asarray(self.wall_origins, dtype=centers.dtype)
        normals = jnp.asarray(self.wall_normals, dtype=centers.dtype)
        cosine = jnp.einsum("ni,wi->nw", axes, normals)
        sign = jnp.where(cosine > 0.0, -1.0, 1.0)
        closest = centers[:, None, :] + sign[..., None] * (
            0.5 * lengths[:, None, None] * axes[:, None, :]
        )
        parallel = jnp.abs(cosine) < CONTACT_THRESHOLD
        closest = jnp.where(parallel[..., None], centers[:, None, :], closest)
        distance = jnp.einsum("nwi,wi->nw", closest - origins[None, :, :], normals)
        distance -= radii[:, None]
        normal = jnp.broadcast_to(normals[None, :, :], closest.shape)
        contact = closest - (radii[:, None] + 0.5 * distance)[..., None] * normal
        arm = contact - centers[:, None, :]
        contact_velocity = velocities[:, None, :] + jnp.cross(omega[:, None, :], arm)
        force = self._contact_force(distance, normal, contact_velocity)
        total_force = jnp.sum(force, axis=1)
        total_torque = jnp.sum(jnp.cross(arm, force), axis=1)
        return _scatter_element_loads(
            external_forces,
            external_torques,
            elem,
            total_force,
            total_torque,
            directors,
        )


def make_walls(parameters: CaseParameters) -> tuple[np.ndarray, np.ndarray]:
    if parameters.name == "snake-on-plane":
        return np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]])
    origins = [[0.0, 0.0, 0.0]]
    normals = [[0.0, 0.0, 1.0]]
    if parameters.wall_distance_ratio > 0.0:
        half = 0.5 * parameters.wall_distance_ratio * parameters.length
        origins += [
            [-half, 0.0, 0.0],
            [half, 0.0, 0.0],
            [0.0, -half, 0.0],
            [0.0, half, 0.0],
        ]
        normals += [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    return np.asarray(origins), np.asarray(normals)


def _random_cylinder_rods(
    parameters: CaseParameters, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    rods = []
    radius = parameters.packing_initial_radial_span_ratio * parameters.length
    height = parameters.packing_initial_vertical_span_ratio * parameters.length
    for _ in range(parameters.n_snakes):
        for _attempt in range(10000):
            rho = radius * np.sqrt(rng.random())
            phi = 2.0 * np.pi * rng.random()
            start = np.array(
                [
                    rho * np.cos(phi),
                    rho * np.sin(phi),
                    parameters.radius + rng.random() * (height - parameters.radius),
                ]
            )
            alpha = 2.0 * np.pi * rng.random()
            beta = 0.2 * np.pi * rng.random()
            direction = np.array(
                [
                    np.cos(alpha) * np.cos(beta),
                    np.sin(alpha) * np.cos(beta),
                    np.sin(beta),
                ]
            )
            end = start + parameters.length * direction
            if (
                np.hypot(end[0], end[1]) <= radius
                and parameters.radius <= end[2] <= height
            ):
                rods.append((start, direction))
                break
        else:
            raise RuntimeError("Could not place a snake inside the random cylinder.")
    return rods


def initial_rods(
    parameters: CaseParameters, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    if parameters.name == "snake-pit":
        assert parameters.initialization_protocol == "random-cylinder", (
            "This GPU example currently implements the snake-pit default "
            "initialization protocol, 'random-cylinder'."
        )
        return _random_cylinder_rods(parameters, seed)
    return [
        (
            np.array([3.0 * parameters.radius * i, parameters.radius, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
        for i in range(parameters.n_snakes)
    ]


def build_simulator(parameters: CaseParameters, *, device, dtype, seed: int):
    assert parameters.n_elements >= 2, "Each snake must contain at least two elements."
    assert parameters.n_snakes >= 1, "The active-matter case needs at least one snake."
    assert parameters.time_step > 0.0, "The simulation time step must be positive."
    _ConfiguredRodBlock.device = device
    _ConfiguredRodBlock.device_dtype = np.dtype(dtype)
    simulator = ActiveMatterJAXSimulator()
    simulator.enable_block_supports(ea.CosseratRod, _ConfiguredRodBlock)
    shear_modulus = parameters.youngs_modulus / 1.5
    rods = []
    for start, direction in initial_rods(parameters, seed):
        normal_seed = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(normal_seed, direction)) > 0.9:
            normal_seed = np.array([0.0, 1.0, 0.0])
        normal = normal_seed - np.dot(normal_seed, direction) * direction
        normal /= np.linalg.norm(normal)
        rod = ea.CosseratRod.straight_rod(
            parameters.n_elements,
            start,
            direction,
            normal,
            parameters.length,
            parameters.radius,
            parameters.density,
            youngs_modulus=parameters.youngs_modulus,
            shear_modulus=shear_modulus,
        )
        simulator.append(rod)
        rods.append(rod)
    simulator.operate_block(ea.CosseratRod).using(
        ActiveMatterForcingAndContactJax, parameters=parameters
    )
    simulator.operate_block(ea.CosseratRod).using(
        ea.AnalyticalLinearDamperJax,
        damping_constant=parameters.damping_rate,
        time_step=parameters.time_step,
    )
    simulator.finalize()
    block = tuple(simulator.final_systems())[0]
    assert isinstance(
        block, ea.MemoryBlockCosseratRodJax
    ), "Active-matter simulation requires a JAX Cosserat-rod memory block."
    state = block.jax_get_state()
    pair_count = sum(
        parameters.n_elements * parameters.n_elements
        for _ in range(parameters.n_snakes * (parameters.n_snakes - 1) // 2)
    )
    state["active_matter_candidate_mask"] = jax.device_put(
        np.zeros(pair_count, dtype=bool), device=device
    )
    state["active_matter_last_detection_time"] = jax.device_put(
        np.asarray(-np.inf, dtype=dtype), device=device
    )
    block.jax_set_state(state)
    return simulator, block, rods


def available_platforms() -> dict[str, jax.Device]:
    result = {}
    for name in ("cpu", "gpu", "cuda", "metal", "mps"):
        try:
            devices = jax.devices(name)
        except Exception:
            continue
        if devices:
            result.setdefault(name, devices[0])
            result.setdefault(devices[0].platform.lower(), devices[0])
    if "gpu" in result:
        result.setdefault("cuda", result["gpu"])
    if "metal" in result:
        result.setdefault("mps", result["metal"])
    return result


def select_device(backend: str) -> tuple[str, jax.Device]:
    platforms = available_platforms()
    if backend == "auto":
        for candidate in ("cuda", "mps", "gpu", "cpu"):
            if candidate in platforms:
                return candidate, platforms[candidate]
        raise RuntimeError("No JAX devices are available.")
    assert (
        backend in platforms
    ), f"Requested backend {backend!r} is unavailable; found {sorted(platforms)}."
    return backend, platforms[backend]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("snake-on-plane", "snake-pit"), default="snake-pit"
    )
    parser.add_argument(
        "--backend", choices=("auto", "cpu", "gpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--n-elements", type=int)
    parser.add_argument("--n-snakes", type=int)
    parser.add_argument("--final-time", type=float)
    parser.add_argument("--time-step", type=float)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parameters = case_parameters(args.case)
    replacements = {}
    for field, argument in (
        ("n_elements", args.n_elements),
        ("n_snakes", args.n_snakes),
        ("final_time", args.final_time),
        ("time_step", args.time_step),
    ):
        if argument is not None:
            replacements[field] = argument
    if replacements:
        parameters = CaseParameters(**{**parameters.__dict__, **replacements})
    assert parameters.final_time >= 0.0, "Final simulation time must be nonnegative."
    steps = int(round(parameters.final_time / parameters.time_step))
    assert np.isclose(
        steps * parameters.time_step, parameters.final_time
    ), "Final time must be an integer multiple of the time step."
    backend_name, device = select_device(args.backend)
    dtype = np.float64 if device.platform.lower() == "cpu" else np.float32
    simulator, block, _ = build_simulator(
        parameters, device=device, dtype=dtype, seed=args.seed
    )
    stepper = ea.PositionVerletGPU()
    start = time.perf_counter()
    stepper.integrate(
        simulator, time=0.0, final_time=parameters.final_time, dt=parameters.time_step
    )
    jax.block_until_ready(block.position_collection_device)
    elapsed = time.perf_counter() - start
    positions = np.asarray(block.position_collection_device)
    print(f"case={parameters.name} backend={backend_name} device={device}")
    print(
        f"snakes={parameters.n_snakes} elements={parameters.n_elements} steps={steps} dtype={dtype}"
    )
    print(f"elapsed={elapsed:.6f}s finite={np.isfinite(positions).all()}")


if __name__ == "__main__":
    main()

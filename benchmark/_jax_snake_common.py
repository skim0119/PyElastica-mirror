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
    build_rod,
    default_b_coeff,
)

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "This benchmark requires JAX. Install the optional GPU extra first."
    ) from exc

from elastica._jax_linalg import _jax_batch_matvec


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


class MultiSnakeJAXBlockSimulator(ea.BaseSystemCollection, ea.JAXOpsBlock):
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


def _uniform_index_matrix(
    start_idx: np.ndarray,
    end_idx: np.ndarray,
) -> np.ndarray:
    widths = end_idx - start_idx
    assert np.all(widths == widths[0]), "All rods must share the same discretization."
    offsets = np.arange(int(widths[0]), dtype=np.int32)
    return start_idx[:, None].astype(np.int32) + offsets[None, :]


def _gather_vector_batch(array: jax.Array, indices: jax.Array) -> jax.Array:
    return jnp.moveaxis(jnp.take(array, indices, axis=-1), 1, 0)


def _gather_tensor_batch(array: jax.Array, indices: jax.Array) -> jax.Array:
    return jnp.moveaxis(jnp.take(array, indices, axis=-1), 2, 0)


def _gather_scalar_batch(array: jax.Array, indices: jax.Array) -> jax.Array:
    return jnp.take(array, indices, axis=-1)


def _scatter_set_vector_batch(
    array: jax.Array, indices: jax.Array, values: jax.Array
) -> jax.Array:
    return array.at[:, indices].set(jnp.moveaxis(values, 0, 1))


def _scatter_add_vector_batch(
    array: jax.Array, indices: jax.Array, values: jax.Array
) -> jax.Array:
    return array.at[:, indices].add(jnp.moveaxis(values, 0, 1))


def _node_to_element_position_batch(position_collection: jax.Array) -> jax.Array:
    return 0.5 * (position_collection[:, :, 1:] + position_collection[:, :, :-1])


def _node_to_element_velocity_batch(
    mass: jax.Array, velocity_collection: jax.Array
) -> jax.Array:
    numerator = (
        mass[:, None, 1:] * velocity_collection[:, :, 1:]
        + mass[:, None, :-1] * velocity_collection[:, :, :-1]
    )
    denominator = mass[:, None, 1:] + mass[:, None, :-1]
    return numerator / denominator


def _node_to_element_mass_or_force_batch(nodal_collection: jax.Array) -> jax.Array:
    elemental_collection = 0.5 * (
        nodal_collection[:, :, :-1] + nodal_collection[:, :, 1:]
    )
    elemental_collection = elemental_collection.at[:, :, 0].add(
        0.5 * nodal_collection[:, :, 0]
    )
    elemental_collection = elemental_collection.at[:, :, -1].add(
        0.5 * nodal_collection[:, :, -1]
    )
    return elemental_collection


def _elements_to_nodes_batch(element_collection: jax.Array) -> jax.Array:
    node_collection = jnp.zeros(
        (
            element_collection.shape[0],
            element_collection.shape[1],
            element_collection.shape[2] + 1,
        ),
        dtype=element_collection.dtype,
    )
    node_collection = node_collection.at[:, :, :-1].add(0.5 * element_collection)
    node_collection = node_collection.at[:, :, 1:].add(0.5 * element_collection)
    return node_collection


def _find_slipping_elements_batch(
    velocity_slip: jax.Array, velocity_threshold: jax.Array
) -> jax.Array:
    abs_velocity_slip = jnp.linalg.norm(velocity_slip, axis=1)
    normalized = abs_velocity_slip / velocity_threshold - 1.0
    slipped = jnp.minimum(1.0, normalized)
    slip_function = jnp.ones_like(abs_velocity_slip)
    slip_values = jnp.abs(1.0 - slipped)
    return jnp.where(abs_velocity_slip > velocity_threshold, slip_values, slip_function)


class SnakeMuscleTorquesBlockJax(ea.NoBlockOpJax):
    def __init__(
        self,
        *,
        b_coeff: np.ndarray,
        period: float,
        base_length: float,
        _system,
    ) -> None:
        widths = _system.end_idx_in_rod_elems - _system.start_idx_in_rod_elems
        assert np.all(widths == widths[0]), (
            "SnakeMuscleTorquesBlockJax requires uniform element counts across rods."
        )
        template = ea.MuscleTorques(
            base_length=base_length,
            b_coeff=b_coeff[:-1],
            period=period,
            wave_number=2.0 * np.pi / float(b_coeff[-1]),
            phase_shift=0.0,
            direction=np.array([0.0, 1.0, 0.0]),
            rest_lengths=np.asarray(
                _system.rest_lengths[
                    _system.start_idx_in_rod_elems[0] : _system.end_idx_in_rod_elems[0]
                ]
            ),
            ramp_up_time=period,
            with_spline=True,
        )
        self.elem_indices = jnp.asarray(
            _uniform_index_matrix(
                _system.start_idx_in_rod_elems,
                _system.end_idx_in_rod_elems,
            )
        )
        self.direction = jnp.asarray(np.array([0.0, 1.0, 0.0], dtype=np.float64))
        self.s = jnp.asarray(np.asarray(template.s, dtype=np.float64))
        self.spline = jnp.asarray(np.asarray(template.my_spline, dtype=np.float64))
        self.angular_frequency = np.float64(2.0 * np.pi / period)
        self.wave_number = np.float64(2.0 * np.pi / float(b_coeff[-1]))
        self.phase_shift = np.float64(0.0)
        self.ramp_up_time = np.float64(period)

    def jax_block_operate_synchronize(
        self,
        state: dict[str, jax.Array],
        time: np.float64,
    ) -> dict[str, jax.Array]:
        dtype = state["director_collection"].dtype
        directors = _gather_tensor_batch(state["director_collection"], self.elem_indices)
        factor = jnp.minimum(
            jnp.asarray(1.0, dtype=dtype),
            jnp.asarray(time, dtype=dtype)
            / jnp.asarray(self.ramp_up_time, dtype=dtype),
        )
        torque_mag = (
            factor
            * jnp.asarray(self.spline, dtype=dtype)
            * jnp.sin(
                jnp.asarray(self.angular_frequency, dtype=dtype)
                * jnp.asarray(time, dtype=dtype)
                - jnp.asarray(self.wave_number, dtype=dtype)
                * jnp.asarray(self.s, dtype=dtype)
                + jnp.asarray(self.phase_shift, dtype=dtype)
            )
        )
        torque_local = (
            jnp.asarray(self.direction, dtype=dtype)[None, :, None]
            * torque_mag[::-1][None, None, :]
        )
        torque_local = jnp.broadcast_to(
            torque_local,
            (directors.shape[0], torque_local.shape[1], torque_local.shape[2]),
        )
        torque_world = jax.vmap(_jax_batch_matvec, in_axes=(0, 0))(
            directors, torque_local
        )
        external_torques = state["external_torques"]
        external_torques = external_torques.at[:, self.elem_indices[:, 1:]].add(
            jnp.moveaxis(torque_world[:, :, 1:], 0, 1)
        )
        previous_directors = directors[:, :, :, :-1]
        next_local = torque_local[:, :, 1:]
        previous_world = jax.vmap(_jax_batch_matvec, in_axes=(0, 0))(
            previous_directors, next_local
        )
        external_torques = external_torques.at[:, self.elem_indices[:, :-1]].add(
            -jnp.moveaxis(previous_world, 0, 1)
        )
        updated = dict(state)
        updated["external_torques"] = external_torques
        return updated


class GravityPlaneContactBlockJax(ea.NoBlockOpJax):
    def __init__(
        self,
        *,
        plane_origin: np.ndarray,
        plane_normal: np.ndarray,
        slip_velocity_tol: float,
        k: float,
        nu: float,
        kinetic_mu_array: np.ndarray,
        static_mu_array: np.ndarray,
        gravitational_acc: float,
        _system,
    ) -> None:
        del static_mu_array
        self.node_indices = jnp.asarray(
            _uniform_index_matrix(
                _system.start_idx_in_rod_nodes,
                _system.end_idx_in_rod_nodes,
            )
        )
        self.elem_indices = jnp.asarray(
            _uniform_index_matrix(
                _system.start_idx_in_rod_elems,
                _system.end_idx_in_rod_elems,
            )
        )
        self.plane_origin = jnp.asarray(np.asarray(plane_origin, dtype=np.float64))
        self.plane_normal = jnp.asarray(np.asarray(plane_normal, dtype=np.float64))
        self.gravity = jnp.asarray(
            np.array([0.0, gravitational_acc, 0.0], dtype=np.float64)
        )
        self.surface_tol = np.float64(1.0e-4)
        self.slip_velocity_tol = np.float64(slip_velocity_tol)
        self.k = np.float64(k)
        self.nu = np.float64(nu)
        self.kinetic_mu_forward = np.float64(kinetic_mu_array[0])
        self.kinetic_mu_backward = np.float64(kinetic_mu_array[1])
        self.kinetic_mu_sideways = np.float64(kinetic_mu_array[2])

    def jax_block_operate_synchronize(
        self,
        state: dict[str, jax.Array],
        time: np.float64,
    ) -> dict[str, jax.Array]:
        del time
        dtype = state["position_collection"].dtype
        position = _gather_vector_batch(state["position_collection"], self.node_indices)
        velocity = _gather_vector_batch(state["velocity_collection"], self.node_indices)
        mass = _gather_scalar_batch(state["mass"], self.node_indices)
        radius = _gather_scalar_batch(state["radius"], self.elem_indices)
        tangents = _gather_vector_batch(state["tangents"], self.elem_indices)
        internal_forces = _gather_vector_batch(
            state["internal_forces"], self.node_indices
        )
        external_forces = _gather_vector_batch(
            state["external_forces"], self.node_indices
        )

        external_forces = external_forces + jnp.asarray(self.gravity, dtype=dtype)[
            None, :, None
        ] * mass[:, None, :]

        nodal_total_forces = internal_forces + external_forces
        element_total_forces = _node_to_element_mass_or_force_batch(nodal_total_forces)
        plane_normal = jnp.asarray(self.plane_normal, dtype=dtype)[None, :, None]
        force_component_along_normal_direction = jnp.sum(
            plane_normal * element_total_forces, axis=1
        )
        forces_along_normal_direction = (
            plane_normal * force_component_along_normal_direction[:, None, :]
        )
        forces_along_normal_direction = jnp.where(
            force_component_along_normal_direction[:, None, :] > 0.0,
            0.0,
            forces_along_normal_direction,
        )
        plane_response_force = -forces_along_normal_direction

        element_position = _node_to_element_position_batch(position)
        distance_from_plane = jnp.sum(
            plane_normal
            * (
                element_position
                - jnp.asarray(self.plane_origin, dtype=dtype)[None, :, None]
            ),
            axis=1,
        )
        plane_penetration = jnp.minimum(distance_from_plane - radius, 0.0)
        elastic_force = (
            -jnp.asarray(self.k, dtype=dtype)
            * plane_normal
            * plane_penetration[:, None, :]
        )
        element_velocity = _node_to_element_velocity_batch(mass, velocity)
        normal_component_of_element_velocity = jnp.sum(
            plane_normal * element_velocity, axis=1
        )
        damping_force = (
            -jnp.asarray(self.nu, dtype=dtype)
            * plane_normal
            * normal_component_of_element_velocity[:, None, :]
        )
        plane_response_force_total = plane_response_force + elastic_force + damping_force
        no_contact = (distance_from_plane - radius) > jnp.asarray(
            self.surface_tol, dtype=dtype
        )
        plane_response_force = jnp.where(no_contact[:, None, :], 0.0, plane_response_force)
        plane_response_force_total = jnp.where(
            no_contact[:, None, :], 0.0, plane_response_force_total
        )

        plane_response_force_mag = jnp.linalg.norm(plane_response_force, axis=1)
        tangent_along_normal_direction = jnp.sum(plane_normal * tangents, axis=1)
        tangent_perpendicular_to_normal_direction = (
            tangents - plane_normal * tangent_along_normal_direction[:, None, :]
        )
        tangent_perpendicular_mag = jnp.linalg.norm(
            tangent_perpendicular_to_normal_direction, axis=1
        )
        axial_direction = tangent_perpendicular_to_normal_direction / (
            tangent_perpendicular_mag[:, None, :] + jnp.asarray(1.0e-14, dtype=dtype)
        )
        element_velocity = _node_to_element_velocity_batch(mass, velocity)
        velocity_mag_along_axial_direction = jnp.sum(
            element_velocity * axial_direction, axis=1
        )
        velocity_along_axial_direction = (
            axial_direction * velocity_mag_along_axial_direction[:, None, :]
        )
        rolling_direction = jnp.cross(axial_direction, plane_normal, axis=1)
        velocity_mag_along_rolling_direction = jnp.sum(
            element_velocity * rolling_direction, axis=1
        )
        velocity_along_rolling_direction = (
            rolling_direction * velocity_mag_along_rolling_direction[:, None, :]
        )
        slip_function_along_axial_direction = _find_slipping_elements_batch(
            velocity_along_axial_direction,
            jnp.asarray(self.slip_velocity_tol, dtype=dtype),
        )
        slip_function_along_rolling_direction = _find_slipping_elements_batch(
            velocity_along_rolling_direction,
            jnp.asarray(self.slip_velocity_tol, dtype=dtype),
        )
        kinetic_mu = jnp.where(
            velocity_mag_along_axial_direction > 0.0,
            jnp.asarray(self.kinetic_mu_forward, dtype=dtype),
            jnp.asarray(self.kinetic_mu_backward, dtype=dtype),
        )
        kinetic_friction_force_along_axial_direction = (
            -(
                1.0 - slip_function_along_axial_direction
            )[:, None, :]
            * kinetic_mu[:, None, :]
            * plane_response_force_mag[:, None, :]
            * axial_direction
        )
        kinetic_friction_force_along_rolling_direction = (
            -(
                1.0 - slip_function_along_rolling_direction
            )[:, None, :]
            * jnp.asarray(self.kinetic_mu_sideways, dtype=dtype)
            * plane_response_force_mag[:, None, :]
            * rolling_direction
        )
        total_contact_force = (
            plane_response_force_total
            + kinetic_friction_force_along_axial_direction
            + kinetic_friction_force_along_rolling_direction
        )
        external_forces = external_forces + _elements_to_nodes_batch(total_contact_force)

        updated = dict(state)
        updated["external_forces"] = _scatter_set_vector_batch(
            state["external_forces"],
            self.node_indices,
            external_forces,
        )
        return updated


class AnalyticalLinearDamperBlockJax(ea.NoBlockOpJax):
    def __init__(self, time_step: np.float64, **kwargs: object) -> None:
        damping_constant = kwargs.get("damping_constant", None)
        system = kwargs["_system"]
        assert damping_constant is not None, (
            "AnalyticalLinearDamperBlockJax currently requires damping_constant."
        )
        self.node_indices = jnp.asarray(
            _uniform_index_matrix(
                system.start_idx_in_rod_nodes,
                system.end_idx_in_rod_nodes,
            )
        )
        self.elem_indices = jnp.asarray(
            _uniform_index_matrix(
                system.start_idx_in_rod_elems,
                system.end_idx_in_rod_elems,
            )
        )
        nodal_mass = _gather_scalar_batch(
            jnp.asarray(system.mass),
            self.node_indices,
        )
        element_mass = 0.5 * (nodal_mass[:, 1:] + nodal_mass[:, :-1])
        element_mass = element_mass.at[:, 0].add(0.5 * nodal_mass[:, 0])
        element_mass = element_mass.at[:, -1].add(0.5 * nodal_mass[:, -1])
        inv_moi = _gather_tensor_batch(
            jnp.asarray(system.inv_mass_second_moment_of_inertia),
            self.elem_indices,
        )
        inv_moi_diag = jnp.stack(
            (inv_moi[:, 0, 0, :], inv_moi[:, 1, 1, :], inv_moi[:, 2, 2, :]),
            axis=1,
        )
        self.translational_damping_coefficient = np.exp(-damping_constant * time_step)
        self.rotational_damping_coefficient = jnp.exp(
            -damping_constant * time_step * element_mass[:, None, :] * inv_moi_diag
        )

    def jax_block_operate_constrain_rates(
        self,
        state: dict[str, jax.Array],
        time: np.float64,
    ) -> dict[str, jax.Array]:
        del time
        velocities = _gather_vector_batch(state["velocity_collection"], self.node_indices)
        omegas = _gather_vector_batch(state["omega_collection"], self.elem_indices)
        dilatation = _gather_scalar_batch(state["dilatation"], self.elem_indices)
        velocities = velocities * jnp.asarray(
            self.translational_damping_coefficient,
            dtype=velocities.dtype,
        )
        omegas = omegas * jnp.power(
            jnp.asarray(self.rotational_damping_coefficient, dtype=omegas.dtype),
            dilatation[:, None, :],
        )
        updated = dict(state)
        updated["velocity_collection"] = _scatter_set_vector_batch(
            state["velocity_collection"],
            self.node_indices,
            velocities,
        )
        updated["omega_collection"] = _scatter_set_vector_batch(
            state["omega_collection"],
            self.elem_indices,
            omegas,
        )
        return updated


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
    include_external_loads: bool = True,
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
    if include_external_loads:
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
        if include_external_loads:
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
    include_external_loads: bool = True,
) -> tuple[MultiSnakeJAXBlockSimulator, ea.MemoryBlockCosseratRodJax]:
    b_coeff = default_b_coeff()
    mu = base_length / (period * period * np.abs(gravitational_acc) * DEFAULT_FROUDE)
    kinetic_mu_array = np.array([mu, 1.5 * mu, 2.0 * mu], dtype=np.float64)
    static_mu_array = np.zeros(kinetic_mu_array.shape, dtype=np.float64)
    spacing = 1.5 * base_length

    ConfiguredSnakeMemoryBlock.device = device
    ConfiguredSnakeMemoryBlock.device_dtype = np.dtype(device_dtype)

    sim = MultiSnakeJAXBlockSimulator()
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

    if include_external_loads:
        sim.operate_block(ea.CosseratRod).using(
            GravityPlaneContactBlockJax,
            plane_origin=np.array([0.0, -base_length * 0.011, 0.0], dtype=np.float64),
            plane_normal=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            slip_velocity_tol=1.0e-8,
            k=1.0,
            nu=1.0e-6,
            static_mu_array=static_mu_array,
            kinetic_mu_array=kinetic_mu_array,
            gravitational_acc=gravitational_acc,
        )
        sim.operate_block(ea.CosseratRod).using(
            SnakeMuscleTorquesBlockJax,
            b_coeff=b_coeff,
            period=period,
            base_length=base_length,
        )
        sim.operate_block(ea.CosseratRod).using(
            AnalyticalLinearDamperBlockJax,
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
    include_external_loads: bool = True,
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
        "include_external_loads": include_external_loads,
    }

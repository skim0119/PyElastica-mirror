"""
Continuum Snake GPU Prototype
=============================

This example is a reduced, JAX-backed prototype of the continuum snake case.
It keeps the rod initialization and muscle-actuation parameters from the
original example, but removes callbacks, damping, and rod-plane contact so the
device-side state updates can be tested without the full host-side module stack.

The script can:

1. Run a JAX-backed reduced snake problem on CPU, Metal/MPS, or CUDA.
2. Compare the final state against a CPU PyElastica reference on the same
   reduced problem.

It is intended as a framework-validation example rather than a replacement for
the full continuum snake benchmark.
"""

from __future__ import annotations

import argparse
import time
from functools import partial

import numpy as np

import elastica as ea
from elastica._jax_calculus import (
    _jax_average as _position_average,
    _jax_difference as _position_difference,
    _jax_trapezoidal as _trapezoidal_for_single_rod,
    _jax_two_point_difference as _two_point_difference_for_single_rod,
)
from elastica._jax_linalg import (
    _jax_batch_cross as _batch_cross,
    _jax_batch_dot as _batch_dot,
    _jax_batch_matmul as _batch_matmul,
    _jax_batch_matvec as _batch_matvec,
)
from elastica._jax_rotations import (
    _jax_get_rotation_matrix as _rotation_matrix,
    _jax_inv_rotate as _inv_rotate,
)

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


class SnakeForcingReference(ea.BaseSystemCollection, ea.Forcing):
    pass


class SnakeJAXSimulator(ea.BaseSystemCollection, ea.JAXOps):
    pass


class _ConfiguredSnakeMemoryBlock(ea.MemoryBlockCosseratRodJax):
    device_dtype = np.dtype(np.float64)
    device = None

    def __init__(self, systems, system_idx_list):
        super().__init__(
            systems,
            system_idx_list,
            device_dtype=self.device_dtype,
            device=self.device,
        )


class SnakeMuscleTorquesJax(ea.NoOpsJax):
    def __init__(
        self,
        *,
        b_coeff: np.ndarray,
        period: float,
        base_length: float,
        gravitational_acc: float,
        _system,
    ) -> None:
        torque_template = ea.MuscleTorques(
            base_length=base_length,
            b_coeff=b_coeff[:-1],
            period=period,
            wave_number=2.0 * np.pi / float(b_coeff[-1]),
            phase_shift=0.0,
            direction=np.array([0.0, 1.0, 0.0]),
            rest_lengths=_system.rest_lengths,
            ramp_up_time=period,
            with_spline=True,
        )
        self.gravity = np.asarray([0.0, gravitational_acc, 0.0], dtype=np.float64)
        self.muscle_direction = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        self.muscle_s = np.asarray(torque_template.s, dtype=np.float64)
        self.muscle_spline = np.asarray(torque_template.my_spline, dtype=np.float64)
        self.muscle_angular_frequency = np.float64(2.0 * np.pi / period)
        self.muscle_wave_number = np.float64(2.0 * np.pi / float(b_coeff[-1]))
        self.muscle_phase_shift = np.float64(0.0)
        self.muscle_ramp_up_time = np.float64(period)

    def jax_operate_synchronize(self, rod_view, time):
        dtype = rod_view.position_collection.dtype
        external_forces, external_torques = self._apply_gravity_and_muscle_torques(
            time_value=time,
            director_collection=rod_view.director_collection,
            mass=rod_view.mass,
            gravity=jnp.asarray(self.gravity, dtype=dtype),
            muscle_direction=jnp.asarray(self.muscle_direction, dtype=dtype),
            muscle_s=jnp.asarray(self.muscle_s, dtype=dtype),
            muscle_spline=jnp.asarray(self.muscle_spline, dtype=dtype),
            muscle_angular_frequency=jnp.asarray(
                self.muscle_angular_frequency, dtype=dtype
            ),
            muscle_wave_number=jnp.asarray(self.muscle_wave_number, dtype=dtype),
            muscle_phase_shift=jnp.asarray(self.muscle_phase_shift, dtype=dtype),
            muscle_ramp_up_time=jnp.asarray(self.muscle_ramp_up_time, dtype=dtype),
        )
        rod_view.external_forces = external_forces
        rod_view.external_torques = external_torques
        return rod_view

    @staticmethod
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


def _node_to_element_position_jax(position_collection: jax.Array) -> jax.Array:
    return 0.5 * (position_collection[:, 1:] + position_collection[:, :-1])


def _node_to_element_velocity_jax(
    mass: jax.Array, velocity_collection: jax.Array
) -> jax.Array:
    numerator = (
        mass[jnp.newaxis, 1:] * velocity_collection[:, 1:]
        + mass[jnp.newaxis, :-1] * velocity_collection[:, :-1]
    )
    denominator = mass[jnp.newaxis, 1:] + mass[jnp.newaxis, :-1]
    return numerator / denominator


def _node_to_element_mass_or_force_jax(nodal_collection: jax.Array) -> jax.Array:
    elemental_collection = 0.5 * (
        nodal_collection[:, :-1] + nodal_collection[:, 1:]
    )
    elemental_collection = elemental_collection.at[:, 0].add(
        0.5 * nodal_collection[:, 0]
    )
    elemental_collection = elemental_collection.at[:, -1].add(
        0.5 * nodal_collection[:, -1]
    )
    return elemental_collection


def _elements_to_nodes_jax(element_collection: jax.Array) -> jax.Array:
    node_collection = jnp.zeros(
        (element_collection.shape[0], element_collection.shape[1] + 1),
        dtype=element_collection.dtype,
    )
    node_collection = node_collection.at[:, :-1].add(0.5 * element_collection)
    node_collection = node_collection.at[:, 1:].add(0.5 * element_collection)
    return node_collection


def _find_slipping_elements_jax(
    velocity_slip: jax.Array, velocity_threshold: jax.Array
) -> jax.Array:
    abs_velocity_slip = jnp.linalg.norm(velocity_slip, axis=0)
    normalized = abs_velocity_slip / velocity_threshold - 1.0
    slipped = jnp.minimum(1.0, normalized)
    slip_function = jnp.ones_like(abs_velocity_slip)
    slip_values = jnp.abs(1.0 - slipped)
    return jnp.where(abs_velocity_slip > velocity_threshold, slip_values, slip_function)


class SnakePlaneContactJax(ea.NoOpsJax):
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
        _system,
    ) -> None:
        del _system
        self.plane_origin = np.asarray(plane_origin, dtype=np.float64)
        self.plane_normal = np.asarray(plane_normal, dtype=np.float64)
        self.surface_tol = np.float64(1.0e-4)
        self.slip_velocity_tol = np.float64(slip_velocity_tol)
        self.k = np.float64(k)
        self.nu = np.float64(nu)
        self.kinetic_mu_forward = np.float64(kinetic_mu_array[0])
        self.kinetic_mu_backward = np.float64(kinetic_mu_array[1])
        self.kinetic_mu_sideways = np.float64(kinetic_mu_array[2])
        self.static_mu_forward = np.float64(static_mu_array[0])
        self.static_mu_backward = np.float64(static_mu_array[1])
        self.static_mu_sideways = np.float64(static_mu_array[2])

    def jax_operate_synchronize(self, rod_view, time):
        del time
        dtype = rod_view.position_collection.dtype
        plane_origin = jnp.asarray(self.plane_origin, dtype=dtype)
        plane_normal = jnp.asarray(self.plane_normal, dtype=dtype)
        surface_tol = jnp.asarray(self.surface_tol, dtype=dtype)
        slip_velocity_tol = jnp.asarray(self.slip_velocity_tol, dtype=dtype)
        k = jnp.asarray(self.k, dtype=dtype)
        nu = jnp.asarray(self.nu, dtype=dtype)
        kinetic_mu_forward = jnp.asarray(self.kinetic_mu_forward, dtype=dtype)
        kinetic_mu_backward = jnp.asarray(self.kinetic_mu_backward, dtype=dtype)
        kinetic_mu_sideways = jnp.asarray(self.kinetic_mu_sideways, dtype=dtype)

        nodal_total_forces = rod_view.internal_forces + rod_view.external_forces
        element_total_forces = _node_to_element_mass_or_force_jax(nodal_total_forces)
        force_component_along_normal_direction = jnp.sum(
            plane_normal[:, None] * element_total_forces, axis=0
        )
        forces_along_normal_direction = (
            plane_normal[:, None] * force_component_along_normal_direction[None, :]
        )
        forces_along_normal_direction = jnp.where(
            force_component_along_normal_direction[None, :] > 0.0,
            0.0,
            forces_along_normal_direction,
        )
        plane_response_force = -forces_along_normal_direction

        element_position = _node_to_element_position_jax(rod_view.position_collection)
        distance_from_plane = jnp.sum(
            plane_normal[:, None] * (element_position - plane_origin[:, None]), axis=0
        )
        plane_penetration = jnp.minimum(distance_from_plane - rod_view.radius, 0.0)
        elastic_force = -k * plane_normal[:, None] * plane_penetration[None, :]

        element_velocity = _node_to_element_velocity_jax(
            rod_view.mass, rod_view.velocity_collection
        )
        normal_component_of_element_velocity = jnp.sum(
            plane_normal[:, None] * element_velocity, axis=0
        )
        damping_force = -nu * plane_normal[:, None] * normal_component_of_element_velocity[None, :]

        plane_response_force_total = plane_response_force + elastic_force + damping_force
        no_contact = (distance_from_plane - rod_view.radius) > surface_tol
        plane_response_force = jnp.where(no_contact[None, :], 0.0, plane_response_force)
        plane_response_force_total = jnp.where(
            no_contact[None, :], 0.0, plane_response_force_total
        )

        external_forces = rod_view.external_forces + _elements_to_nodes_jax(
            plane_response_force_total
        )
        plane_response_force_mag = jnp.linalg.norm(plane_response_force, axis=0)

        tangent_along_normal_direction = jnp.sum(
            plane_normal[:, None] * rod_view.tangents, axis=0
        )
        tangent_perpendicular_to_normal_direction = (
            rod_view.tangents
            - plane_normal[:, None] * tangent_along_normal_direction[None, :]
        )
        tangent_perpendicular_mag = jnp.linalg.norm(
            tangent_perpendicular_to_normal_direction, axis=0
        )
        axial_direction = tangent_perpendicular_to_normal_direction / (
            tangent_perpendicular_mag[None, :] + 1.0e-14
        )

        velocity_mag_along_axial_direction = jnp.sum(
            element_velocity * axial_direction, axis=0
        )
        velocity_along_axial_direction = (
            axial_direction * velocity_mag_along_axial_direction[None, :]
        )
        velocity_sign_along_axial_direction = jnp.sign(
            velocity_mag_along_axial_direction
        )
        kinetic_mu = 0.5 * (
            self.kinetic_mu_forward * (1.0 + velocity_sign_along_axial_direction)
            + self.kinetic_mu_backward * (1.0 - velocity_sign_along_axial_direction)
        )
        kinetic_mu = jnp.asarray(kinetic_mu, dtype=dtype)
        slip_function_along_axial_direction = _find_slipping_elements_jax(
            velocity_along_axial_direction, slip_velocity_tol
        )

        rolling_direction = _batch_cross(
            axial_direction, jnp.repeat(plane_normal[:, None], axial_direction.shape[1], axis=1)
        )
        torque_arm = -plane_normal[:, None] * rod_view.radius[None, :]
        velocity_along_rolling_direction = jnp.sum(
            element_velocity * rolling_direction, axis=0
        )
        directors_transpose = jnp.transpose(rod_view.director_collection, (1, 0, 2))
        rotation_velocity = _batch_matvec(
            directors_transpose,
            _batch_cross(
                rod_view.omega_collection,
                _batch_matvec(rod_view.director_collection, torque_arm),
            ),
        )
        rotation_velocity_along_rolling_direction = jnp.sum(
            rotation_velocity * rolling_direction, axis=0
        )
        slip_velocity_mag_along_rolling_direction = (
            velocity_along_rolling_direction
            + rotation_velocity_along_rolling_direction
        )
        slip_velocity_along_rolling_direction = (
            rolling_direction * slip_velocity_mag_along_rolling_direction[None, :]
        )
        slip_function_along_rolling_direction = _find_slipping_elements_jax(
            slip_velocity_along_rolling_direction, slip_velocity_tol
        )

        unitized_total_velocity = (
            slip_velocity_along_rolling_direction + velocity_along_axial_direction
        )
        unitized_total_velocity = unitized_total_velocity / (
            jnp.linalg.norm(unitized_total_velocity + 1.0e-14, axis=0)[None, :]
        )
        kinetic_friction_force_along_axial_direction = -(
            (1.0 - slip_function_along_axial_direction)
            * kinetic_mu
            * plane_response_force_mag
            * jnp.sum(unitized_total_velocity * axial_direction, axis=0)
        )[None, :] * axial_direction
        kinetic_friction_force_along_axial_direction = jnp.where(
            no_contact[None, :], 0.0, kinetic_friction_force_along_axial_direction
        )
        external_forces = external_forces + _elements_to_nodes_jax(
            kinetic_friction_force_along_axial_direction
        )

        kinetic_friction_force_along_rolling_direction = -(
            (1.0 - slip_function_along_rolling_direction)
            * kinetic_mu_sideways
            * plane_response_force_mag
            * jnp.sum(unitized_total_velocity * rolling_direction, axis=0)
        )[None, :] * rolling_direction
        kinetic_friction_force_along_rolling_direction = jnp.where(
            no_contact[None, :], 0.0, kinetic_friction_force_along_rolling_direction
        )
        external_forces = external_forces + _elements_to_nodes_jax(
            kinetic_friction_force_along_rolling_direction
        )

        external_torques = rod_view.external_torques + _batch_matvec(
            rod_view.director_collection,
            _batch_cross(torque_arm, kinetic_friction_force_along_rolling_direction),
        )

        rod_view.external_forces = external_forces
        rod_view.external_torques = external_torques
        return rod_view


def default_b_coeff() -> np.ndarray:
    return np.array(
        [3.4e-3, 3.3e-3, 4.2e-3, 2.6e-3, 3.6e-3, 3.5e-3, 1.0],
        dtype=np.float64,
    )


def build_rod(
    n_elem: int = 50,
    base_length: float = 0.35,
    density: float = 1000.0,
    youngs_modulus: float = 1.0e6,
    poisson_ratio: float = 0.5,
) -> ea.CosseratRod:
    base_radius = base_length * 0.011
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)
    return ea.CosseratRod.straight_rod(
        n_elem,
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
    )


def build_cpu_reference_sim(
    b_coeff: np.ndarray,
    *,
    n_elem: int = 50,
    period: float = 2.0,
    base_length: float = 0.35,
    density: float = 1000.0,
    youngs_modulus: float = 1.0e6,
    poisson_ratio: float = 0.5,
    gravitational_acc: float = -9.80665,
    time_step: float = 1.0e-4,
) -> tuple[SnakeForcingReference, ea.CosseratRod]:
    class SnakeReferenceSimulator(
        ea.BaseSystemCollection, ea.Forcing, ea.Damping, ea.Contact
    ):
        pass

    sim = SnakeReferenceSimulator()
    rod = build_rod(
        n_elem=n_elem,
        base_length=base_length,
        density=density,
        youngs_modulus=youngs_modulus,
        poisson_ratio=poisson_ratio,
    )
    sim.append(rod)

    normal = np.array([0.0, 1.0, 0.0])
    wave_length = float(b_coeff[-1])
    sim.add_forcing_to(rod).using(
        ea.GravityForces,
        acc_gravity=np.array([0.0, gravitational_acc, 0.0]),
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
    ground_plane = ea.Plane(
        plane_origin=np.array([0.0, -base_length * 0.011, 0.0]),
        plane_normal=normal,
    )
    sim.append(ground_plane)
    slip_velocity_tol = 1.0e-8
    froude = 0.1
    mu = base_length / (period * period * np.abs(gravitational_acc) * froude)
    kinetic_mu_array = np.array([mu, 1.5 * mu, 2.0 * mu], dtype=np.float64)
    static_mu_array = np.zeros(kinetic_mu_array.shape, dtype=np.float64)
    sim.detect_contact_between(rod, ground_plane).using(
        ea.RodPlaneContactWithAnisotropicFriction,
        k=1.0,
        nu=1.0e-6,
        slip_velocity_tol=slip_velocity_tol,
        static_mu_array=static_mu_array,
        kinetic_mu_array=kinetic_mu_array,
    )
    sim.dampen(rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=2.0e-3,
        time_step=time_step,
    )
    sim.finalize()
    return sim, rod


def run_cpu_reference(
    b_coeff: np.ndarray,
    *,
    n_elem: int = 50,
    period: float = 2.0,
    final_time: float = 0.002,
    time_step: float = 1.0e-4,
    base_length: float = 0.35,
    density: float = 1000.0,
    youngs_modulus: float = 1.0e6,
    poisson_ratio: float = 0.5,
    gravitational_acc: float = -9.80665,
) -> tuple[dict[str, np.ndarray], float]:
    sim, rod = build_cpu_reference_sim(
        b_coeff,
        n_elem=n_elem,
        period=period,
        base_length=base_length,
        density=density,
        youngs_modulus=youngs_modulus,
        poisson_ratio=poisson_ratio,
        gravitational_acc=gravitational_acc,
        time_step=time_step,
    )
    stepper = ea.PositionVerlet()
    time_value = np.float64(0.0)
    dt = np.float64(time_step)
    total_steps = int(final_time / time_step)

    start = time.perf_counter()
    for _ in range(total_steps):
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


def build_jax_sim(
    b_coeff: np.ndarray,
    *,
    device: jax.Device,
    device_dtype: np.dtype,
    n_elem: int = 50,
    period: float = 2.0,
    base_length: float = 0.35,
    density: float = 1000.0,
    youngs_modulus: float = 1.0e6,
    poisson_ratio: float = 0.5,
    gravitational_acc: float = -9.80665,
    time_step: float = 1.0e-4,
) -> tuple[SnakeJAXSimulator, ea.MemoryBlockCosseratRodJax]:
    _ConfiguredSnakeMemoryBlock.device = device
    _ConfiguredSnakeMemoryBlock.device_dtype = np.dtype(device_dtype)

    sim = SnakeJAXSimulator()
    sim.enable_block_supports(ea.CosseratRod, _ConfiguredSnakeMemoryBlock)
    rod = build_rod(
        n_elem=n_elem,
        base_length=base_length,
        density=density,
        youngs_modulus=youngs_modulus,
        poisson_ratio=poisson_ratio,
    )
    sim.append(rod)
    sim.using(rod).operate(
        SnakeMuscleTorquesJax,
        b_coeff=b_coeff,
        period=period,
        base_length=base_length,
        gravitational_acc=gravitational_acc,
    )
    slip_velocity_tol = 1.0e-8
    froude = 0.1
    mu = base_length / (period * period * np.abs(gravitational_acc) * froude)
    kinetic_mu_array = np.array([mu, 1.5 * mu, 2.0 * mu], dtype=np.float64)
    static_mu_array = np.zeros(kinetic_mu_array.shape, dtype=np.float64)
    sim.using(rod).operate(
        SnakePlaneContactJax,
        plane_origin=np.array([0.0, -base_length * 0.011, 0.0], dtype=np.float64),
        plane_normal=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        slip_velocity_tol=slip_velocity_tol,
        k=1.0,
        nu=1.0e-6,
        static_mu_array=static_mu_array,
        kinetic_mu_array=kinetic_mu_array,
    )
    sim.using(rod).operate(
        ea.AnalyticalLinearDamperJax,
        time_step=np.float64(time_step),
        damping_constant=2.0e-3,
    )
    sim.finalize()
    block = tuple(sim.final_systems())[0]
    return sim, block


def _clone_jax_state(state: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return dict(state)


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


def run_gpu_rollout_with_stepper(
    b_coeff: np.ndarray,
    *,
    device: jax.Device,
    device_dtype: np.dtype,
    n_elem: int = 50,
    period: float = 2.0,
    final_time: float = 0.002,
    time_step: float = 1.0e-4,
) -> tuple[dict[str, jax.Array], float]:
    stepper = ea.PositionVerletGPU()
    sim, block = build_jax_sim(
        b_coeff,
        device=device,
        device_dtype=device_dtype,
        n_elem=n_elem,
        period=period,
        time_step=time_step,
    )
    initial_state = _clone_jax_state(block.jax_get_state())
    stepper.integrate(
        sim,
        time=np.float64(0.0),
        final_time=np.float64(final_time),
        dt=np.float64(time_step),
    )
    jax.block_until_ready(block.position_collection_device)

    block.jax_set_state(_clone_jax_state(initial_state))
    start = time.perf_counter()
    stepper.integrate(
        sim,
        time=np.float64(0.0),
        final_time=np.float64(final_time),
        dt=np.float64(time_step),
    )
    jax.block_until_ready(block.position_collection_device)
    elapsed = time.perf_counter() - start
    return block.jax_get_state(), elapsed


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
        default=1.000,
        help="Final simulation time of the reduced snake case.",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=1.0e-4,
        help="Time step used by both the CPU and JAX rollouts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    b_coeff = default_b_coeff()
    total_steps = int(args.final_time / args.time_step)

    backend_name, device = select_device(args.backend)
    dtype = preferred_dtype(device)
    print(f"Selected backend alias: {backend_name}")
    print(f"JAX device: {device} (platform={device.platform})")
    print(f"JAX rollout dtype: {dtype}")
    print(f"Reduced snake rollout steps: {total_steps}")

    cpu_state, cpu_elapsed = run_cpu_reference(
        b_coeff,
        n_elem=args.n_elem,
        final_time=args.final_time,
        time_step=args.time_step,
    )
    print(f"CPU reference elapsed: {cpu_elapsed:.4f} s")

    final_state_device, gpu_elapsed = run_gpu_rollout_with_stepper(
        b_coeff,
        device=device,
        device_dtype=dtype,
        n_elem=args.n_elem,
        final_time=args.final_time,
        time_step=args.time_step,
    )
    print(f"JAX rollout elapsed: {gpu_elapsed:.4f} s")

    gpu_state = jax.tree_util.tree_map(np.asarray, final_state_device)
    diffs = summarize_results(cpu_state, gpu_state)

    print("Max absolute differences vs CPU reference:")
    for key, value in diffs.items():
        print(f"  {key}: {value:.3e}")


if __name__ == "__main__":
    main()

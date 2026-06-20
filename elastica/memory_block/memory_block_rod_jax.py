"""JAX-backed memory block for Cosserat rods with explicit host/device sync."""

from __future__ import annotations

from typing import Callable, Iterable, Literal, Sequence

import numpy as np

from elastica._synchronize_periodic_boundary import (
    _synchronize_periodic_boundary_of_matrix_collection,
    _synchronize_periodic_boundary_of_scalar_collection,
    _synchronize_periodic_boundary_of_vector_collection,
)
from elastica.memory_block.utils import (
    make_block_memory_metadata,
    make_block_memory_periodic_boundary_metadata,
)
from elastica.reset_functions_for_block_structure import _reset_scalar_ghost
from elastica.rod.cosserat_rod import (
    _compute_sigma_kappa_for_blockstructure,
)
from elastica.rod.data_structures import _RodSymplecticStepperMixin
from elastica.rod.rod_base import RodBase
from elastica.typing import RodType, SystemIdxType

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without JAX
    raise ModuleNotFoundError(
        "MemoryBlockCosseratRodJax requires JAX. Install the optional GPU dependency "
        'first, for example with `uv add --optional gpu "jax[cuda13]"`.'
    ) from exc


_NODE_ATTRS: tuple[str, ...] = (
    "mass",
    "position_collection",
    "internal_forces",
    "external_forces",
    "velocity_collection",
    "acceleration_collection",
)
_ELEMENT_ATTRS: tuple[str, ...] = (
    "radius",
    "volume",
    "density",
    "lengths",
    "rest_lengths",
    "dilatation",
    "dilatation_rate",
    "tangents",
    "sigma",
    "rest_sigma",
    "internal_torques",
    "external_torques",
    "internal_stress",
    "director_collection",
    "mass_second_moment_of_inertia",
    "inv_mass_second_moment_of_inertia",
    "shear_matrix",
    "omega_collection",
    "alpha_collection",
)
_VORONOI_ATTRS: tuple[str, ...] = (
    "voronoi_dilatation",
    "rest_voronoi_lengths",
    "kappa",
    "rest_kappa",
    "internal_couple",
    "bend_matrix",
)
_SYNCABLE_ATTRS: tuple[str, ...] = _NODE_ATTRS + _ELEMENT_ATTRS + _VORONOI_ATTRS


def _jax_batch_matmul(
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


def _jax_batch_matvec(
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


def _jax_get_rotation_matrix(scale: float, axis_collection: jax.Array) -> jax.Array:
    v0, v1, v2 = axis_collection
    theta = jnp.sqrt(v0 * v0 + v1 * v1 + v2 * v2)
    theta_eps = theta + jnp.asarray(1.0e-14, dtype=axis_collection.dtype)
    v0 = v0 / theta_eps
    v1 = v1 / theta_eps
    v2 = v2 / theta_eps

    theta = theta * scale
    sin_theta = jnp.sin(theta)
    one_minus_cos_theta = 1.0 - jnp.cos(theta)

    entries = (
        1.0 - one_minus_cos_theta * (v1 * v1 + v2 * v2),
        sin_theta * v2 + one_minus_cos_theta * v0 * v1,
        -sin_theta * v1 + one_minus_cos_theta * v0 * v2,
        -sin_theta * v2 + one_minus_cos_theta * v0 * v1,
        1.0 - one_minus_cos_theta * (v0 * v0 + v2 * v2),
        sin_theta * v0 + one_minus_cos_theta * v1 * v2,
        sin_theta * v1 + one_minus_cos_theta * v0 * v2,
        -sin_theta * v0 + one_minus_cos_theta * v1 * v2,
        1.0 - one_minus_cos_theta * (v0 * v0 + v1 * v1),
    )
    return jnp.stack(entries, axis=0).reshape(3, 3, axis_collection.shape[1])


def _jax_batch_cross(
    first_vector_collection: jax.Array, second_vector_collection: jax.Array
) -> jax.Array:
    out = jnp.empty_like(first_vector_collection)
    out = out.at[0, :].set(
        first_vector_collection[1, :] * second_vector_collection[2, :]
        - first_vector_collection[2, :] * second_vector_collection[1, :]
    )
    out = out.at[1, :].set(
        first_vector_collection[2, :] * second_vector_collection[0, :]
        - first_vector_collection[0, :] * second_vector_collection[2, :]
    )
    out = out.at[2, :].set(
        first_vector_collection[0, :] * second_vector_collection[1, :]
        - first_vector_collection[1, :] * second_vector_collection[0, :]
    )
    return out


def _jax_batch_dot(
    first_vector_collection: jax.Array, second_vector_collection: jax.Array
) -> jax.Array:
    return jnp.sum(first_vector_collection * second_vector_collection, axis=0)


def _jax_position_difference(position_collection: jax.Array) -> jax.Array:
    return position_collection[:, 1:] - position_collection[:, :-1]


def _jax_position_average(vector: jax.Array) -> jax.Array:
    return 0.5 * (vector[1:] + vector[:-1])


def _jax_reset_vector_ghost(
    input_array: jax.Array, ghost_idx: jax.Array, reset_value: float = 0.0
) -> jax.Array:
    if ghost_idx.size == 0:
        return input_array
    return input_array.at[:, ghost_idx].set(
        jnp.asarray(reset_value, dtype=input_array.dtype)
    )


def _jax_reset_scalar_ghost(
    input_array: jax.Array, ghost_idx: jax.Array, reset_value: float = 0.0
) -> jax.Array:
    if ghost_idx.size == 0:
        return input_array
    return input_array.at[ghost_idx].set(
        jnp.asarray(reset_value, dtype=input_array.dtype)
    )


def _jax_sync_periodic_vector(
    input_array: jax.Array, periodic_idx: jax.Array
) -> jax.Array:
    if periodic_idx.size == 0:
        return input_array
    return input_array.at[:, periodic_idx[0, :]].set(input_array[:, periodic_idx[1, :]])


def _jax_sync_periodic_matrix(
    input_array: jax.Array, periodic_idx: jax.Array
) -> jax.Array:
    if periodic_idx.size == 0:
        return input_array
    return input_array.at[:, :, periodic_idx[0, :]].set(
        input_array[:, :, periodic_idx[1, :]]
    )


def _jax_sync_periodic_scalar(
    input_array: jax.Array, periodic_idx: jax.Array
) -> jax.Array:
    if periodic_idx.size == 0:
        return input_array
    return input_array.at[periodic_idx[0, :]].set(input_array[periodic_idx[1, :]])


def _jax_two_point_difference_for_block_structure(
    array_collection: jax.Array, ghost_idx: jax.Array
) -> jax.Array:
    array_collection = _jax_reset_vector_ghost(array_collection, ghost_idx)
    blocksize = array_collection.shape[1]
    temp_collection = jnp.zeros((3, blocksize + 1), dtype=array_collection.dtype)
    temp_collection = temp_collection.at[:, 0].set(array_collection[:, 0])
    temp_collection = temp_collection.at[:, blocksize].set(
        -array_collection[:, blocksize - 1]
    )
    temp_collection = temp_collection.at[:, 1:blocksize].set(
        array_collection[:, 1:] - array_collection[:, :-1]
    )
    return temp_collection


def _jax_trapezoidal_for_block_structure(
    array_collection: jax.Array, ghost_idx: jax.Array
) -> jax.Array:
    array_collection = _jax_reset_vector_ghost(array_collection, ghost_idx)
    blocksize = array_collection.shape[1]
    temp_collection = jnp.zeros((3, blocksize + 1), dtype=array_collection.dtype)
    temp_collection = temp_collection.at[:, 0].set(0.5 * array_collection[:, 0])
    temp_collection = temp_collection.at[:, blocksize].set(
        0.5 * array_collection[:, blocksize - 1]
    )
    temp_collection = temp_collection.at[:, 1:blocksize].set(
        0.5 * (array_collection[:, 1:] + array_collection[:, :-1])
    )
    return temp_collection


def _jax_inv_rotate(director_collection: jax.Array) -> jax.Array:
    current = director_collection[:, :, :-1]
    nxt = director_collection[:, :, 1:]

    v0 = (
        nxt[2, 0, :] * current[1, 0, :]
        + nxt[2, 1, :] * current[1, 1, :]
        + nxt[2, 2, :] * current[1, 2, :]
        - nxt[1, 0, :] * current[2, 0, :]
        - nxt[1, 1, :] * current[2, 1, :]
        - nxt[1, 2, :] * current[2, 2, :]
    )
    v1 = (
        nxt[0, 0, :] * current[2, 0, :]
        + nxt[0, 1, :] * current[2, 1, :]
        + nxt[0, 2, :] * current[2, 2, :]
        - nxt[2, 0, :] * current[0, 0, :]
        - nxt[2, 1, :] * current[0, 1, :]
        - nxt[2, 2, :] * current[0, 2, :]
    )
    v2 = (
        nxt[1, 0, :] * current[0, 0, :]
        + nxt[1, 1, :] * current[0, 1, :]
        + nxt[1, 2, :] * current[0, 2, :]
        - nxt[0, 0, :] * current[1, 0, :]
        - nxt[0, 1, :] * current[1, 1, :]
        - nxt[0, 2, :] * current[1, 2, :]
    )

    trace = (
        nxt[0, 0, :] * current[0, 0, :]
        + nxt[0, 1, :] * current[0, 1, :]
        + nxt[0, 2, :] * current[0, 2, :]
        + nxt[1, 0, :] * current[1, 0, :]
        + nxt[1, 1, :] * current[1, 1, :]
        + nxt[1, 2, :] * current[1, 2, :]
        + nxt[2, 0, :] * current[2, 0, :]
        + nxt[2, 1, :] * current[2, 1, :]
        + nxt[2, 2, :] * current[2, 2, :]
    )
    trace = jnp.clip(trace, -1.0, 3.0)
    theta = jnp.arccos(0.5 * trace - 0.5) + jnp.asarray(1.0e-14, dtype=trace.dtype)
    magnitude = -0.5 * theta / jnp.sin(theta)

    out = jnp.empty(
        (3, director_collection.shape[2] - 1), dtype=director_collection.dtype
    )
    out = out.at[0, :].set(v0 * magnitude)
    out = out.at[1, :].set(v1 * magnitude)
    out = out.at[2, :].set(v2 * magnitude)
    return out


@jax.jit
def _jax_compute_internal_forces_and_torques(
    position_collection: jax.Array,
    velocity_collection: jax.Array,
    volume: jax.Array,
    lengths: jax.Array,
    tangents: jax.Array,
    radius: jax.Array,
    rest_lengths: jax.Array,
    rest_voronoi_lengths: jax.Array,
    dilatation: jax.Array,
    dilatation_rate: jax.Array,
    voronoi_dilatation: jax.Array,
    director_collection: jax.Array,
    sigma: jax.Array,
    rest_sigma: jax.Array,
    shear_matrix: jax.Array,
    internal_stress: jax.Array,
    internal_forces: jax.Array,
    mass_second_moment_of_inertia: jax.Array,
    omega_collection: jax.Array,
    internal_torques: jax.Array,
    bend_matrix: jax.Array,
    rest_kappa: jax.Array,
    kappa: jax.Array,
    internal_couple: jax.Array,
    ghost_elems_idx: jax.Array,
    ghost_voronoi_idx: jax.Array,
    periodic_boundary_elems_idx: jax.Array,
    periodic_boundary_voronoi_idx: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    position_diff = _jax_position_difference(position_collection)
    lengths = jnp.sqrt(jnp.sum(position_diff * position_diff, axis=0)) + jnp.asarray(
        1.0e-14, dtype=position_collection.dtype
    )
    tangents = position_diff / lengths[jnp.newaxis, :]
    radius = jnp.sqrt(
        volume / lengths / jnp.asarray(jnp.pi, dtype=position_collection.dtype)
    )
    dilatation = lengths / rest_lengths
    voronoi_dilatation = _jax_position_average(lengths) / rest_voronoi_lengths

    tangents = _jax_sync_periodic_vector(tangents, periodic_boundary_elems_idx)
    lengths = _jax_sync_periodic_scalar(lengths, periodic_boundary_elems_idx)
    radius = _jax_sync_periodic_scalar(radius, periodic_boundary_elems_idx)
    dilatation = _jax_sync_periodic_scalar(dilatation, periodic_boundary_elems_idx)
    voronoi_dilatation = _jax_sync_periodic_scalar(
        voronoi_dilatation, periodic_boundary_voronoi_idx
    )

    sigma = dilatation[jnp.newaxis, :] * _jax_batch_matvec(
        director_collection, tangents
    ) - jnp.array([[0.0], [0.0], [1.0]], dtype=position_collection.dtype)
    sigma = _jax_sync_periodic_vector(sigma, periodic_boundary_elems_idx)

    internal_stress = _jax_batch_matvec(shear_matrix, sigma - rest_sigma)
    internal_stress = _jax_sync_periodic_vector(
        internal_stress, periodic_boundary_elems_idx
    )

    cosserat_internal_stress = (
        _jax_batch_matvec(
            jnp.transpose(director_collection, (1, 0, 2)), internal_stress
        )
        / dilatation[jnp.newaxis, :]
    )
    internal_forces = _jax_two_point_difference_for_block_structure(
        cosserat_internal_stress, ghost_elems_idx
    )

    kappa = _jax_inv_rotate(director_collection) / rest_voronoi_lengths[jnp.newaxis, :]
    kappa = _jax_sync_periodic_vector(kappa, periodic_boundary_voronoi_idx)

    internal_couple = _jax_batch_matvec(bend_matrix, kappa - rest_kappa)
    internal_couple = _jax_sync_periodic_vector(
        internal_couple, periodic_boundary_voronoi_idx
    )

    r_dot_v = _jax_batch_dot(position_collection, velocity_collection)
    r_plus_one_dot_v = _jax_batch_dot(
        position_collection[..., 1:], velocity_collection[..., :-1]
    )
    r_dot_v_plus_one = _jax_batch_dot(
        position_collection[..., :-1], velocity_collection[..., 1:]
    )
    dilatation_rate = (
        (r_dot_v[:-1] + r_dot_v[1:] - r_dot_v_plus_one - r_plus_one_dot_v)
        / lengths
        / rest_lengths
    )
    dilatation_rate = _jax_sync_periodic_scalar(
        dilatation_rate, periodic_boundary_elems_idx
    )

    voronoi_dilatation_inv_cube = 1.0 / (voronoi_dilatation**3)
    bend_twist_couple_2d = _jax_two_point_difference_for_block_structure(
        internal_couple * voronoi_dilatation_inv_cube[jnp.newaxis, :], ghost_voronoi_idx
    )
    bend_twist_couple_3d = _jax_trapezoidal_for_block_structure(
        _jax_batch_cross(kappa, internal_couple)
        * rest_voronoi_lengths[jnp.newaxis, :]
        * voronoi_dilatation_inv_cube[jnp.newaxis, :],
        ghost_voronoi_idx,
    )
    shear_stretch_couple = (
        _jax_batch_cross(
            _jax_batch_matvec(director_collection, tangents), internal_stress
        )
        * rest_lengths[jnp.newaxis, :]
    )
    j_omega_upon_e = (
        _jax_batch_matvec(mass_second_moment_of_inertia, omega_collection)
        / dilatation[jnp.newaxis, :]
    )
    lagrangian_transport = _jax_batch_cross(j_omega_upon_e, omega_collection)
    unsteady_dilatation = (
        j_omega_upon_e * dilatation_rate[jnp.newaxis, :] / dilatation[jnp.newaxis, :]
    )
    internal_torques = (
        bend_twist_couple_2d
        + bend_twist_couple_3d
        + shear_stretch_couple
        + lagrangian_transport
        + unsteady_dilatation
    )
    internal_torques = _jax_sync_periodic_vector(
        internal_torques, periodic_boundary_elems_idx
    )

    return (
        lengths,
        tangents,
        radius,
        dilatation,
        dilatation_rate,
        voronoi_dilatation,
        sigma,
        kappa,
        internal_stress,
        internal_couple,
        internal_forces,
        internal_torques,
    )


@jax.jit
def _jax_update_kinematics(
    position_collection: jax.Array,
    director_collection: jax.Array,
    velocity_collection: jax.Array,
    omega_collection: jax.Array,
    prefac: float,
) -> tuple[jax.Array, jax.Array]:
    position_collection = position_collection + prefac * velocity_collection
    rotation_matrix = _jax_get_rotation_matrix(prefac, omega_collection)
    director_collection = _jax_batch_matmul(rotation_matrix, director_collection)
    return position_collection, director_collection


@jax.jit
def _jax_update_dynamics(
    velocity_collection: jax.Array,
    omega_collection: jax.Array,
    acceleration_collection: jax.Array,
    alpha_collection: jax.Array,
    prefac: float,
) -> tuple[jax.Array, jax.Array]:
    velocity_collection = velocity_collection + prefac * acceleration_collection
    omega_collection = omega_collection + prefac * alpha_collection
    return velocity_collection, omega_collection


@jax.jit
def _jax_update_accelerations(
    internal_forces: jax.Array,
    external_forces: jax.Array,
    mass: jax.Array,
    inv_mass_second_moment_of_inertia: jax.Array,
    internal_torques: jax.Array,
    external_torques: jax.Array,
    dilatation: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    acceleration_collection = (internal_forces + external_forces) / mass[jnp.newaxis, :]
    alpha_collection = (
        _jax_batch_matvec(
            inv_mass_second_moment_of_inertia,
            internal_torques + external_torques,
        )
        * dilatation[jnp.newaxis, :]
    )
    return acceleration_collection, alpha_collection


@jax.jit
def _jax_zero_external_loads(
    external_forces: jax.Array, external_torques: jax.Array
) -> tuple[jax.Array, jax.Array]:
    return jnp.zeros_like(external_forces), jnp.zeros_like(external_torques)


class MemoryBlockCosseratRodJax(RodBase, _RodSymplecticStepperMixin):
    """
    JAX-backed memory block with explicit host/device synchronization.

    Unlike the NumPy memory block, this implementation does not make the original
    rods alias the block arrays. The block owns its host-side memory and only
    pushes data back to rods on explicit ``from_device`` / ``from_gpu`` calls.
    """

    allow_cpu_fallback: bool = False

    def __init__(
        self,
        systems: list[RodType],
        system_idx_list: list[SystemIdxType],
        *,
        device_dtype: str | np.dtype | None = None,
        device: jax.Device | None = None,
    ) -> None:
        self._systems = tuple(systems)
        self.n_systems = len(systems)
        self.ring_rod_flag = False
        self._device_dtype = self._normalize_device_dtype(device_dtype)
        self._initial_device = device

        system_straight_rod = []
        system_ring_rod = []
        system_idx_list_ring_rod = []
        system_idx_list_straight_rod = []
        for k, system_to_be_added in enumerate(systems):
            if system_to_be_added.ring_rod_flag:
                system_ring_rod.append(system_to_be_added)
                system_idx_list_ring_rod.append(system_idx_list[k])
                self.ring_rod_flag = True
            else:
                system_straight_rod.append(system_to_be_added)
                system_idx_list_straight_rod.append(system_idx_list[k])

        systems = system_straight_rod + system_ring_rod
        self.system_idx_list = np.array(
            system_idx_list_straight_rod + system_idx_list_ring_rod, dtype=np.int32
        )

        n_elems_straight_rods = np.array(
            [x.n_elems for x in system_straight_rod], dtype=np.int32
        )
        n_elems_ring_rods = np.array(
            [x.n_elems for x in system_ring_rod], dtype=np.int32
        )

        n_straight_rods = len(system_straight_rod)
        n_ring_rods = len(system_ring_rod)

        self.n_elems_in_rods = np.hstack((n_elems_straight_rods, n_elems_ring_rods + 2))
        self.n_rods = len(systems)
        (
            self.n_elems,
            self.ghost_nodes_idx,
            self.ghost_elems_idx,
            self.ghost_voronoi_idx,
        ) = make_block_memory_metadata(self.n_elems_in_rods)
        self.n_nodes = self.n_elems + 1
        self.n_voronoi = self.n_elems - 1

        self.start_idx_in_rod_nodes = np.hstack((0, self.ghost_nodes_idx + 1))
        self.end_idx_in_rod_nodes = np.hstack((self.ghost_nodes_idx, self.n_nodes))
        self.start_idx_in_rod_elems = np.hstack((0, self.ghost_elems_idx[1::2] + 1))
        self.end_idx_in_rod_elems = np.hstack((self.ghost_elems_idx[::2], self.n_elems))
        self.start_idx_in_rod_voronoi = np.hstack((0, self.ghost_voronoi_idx[2::3] + 1))
        self.end_idx_in_rod_voronoi = np.hstack(
            (self.ghost_voronoi_idx[::3], self.n_voronoi)
        )

        (
            _,
            self.periodic_boundary_nodes_idx,
            self.periodic_boundary_elems_idx,
            self.periodic_boundary_voronoi_idx,
        ) = make_block_memory_periodic_boundary_metadata(n_elems_ring_rods)

        if n_ring_rods != 0:
            if n_straight_rods != 0:
                self.periodic_boundary_nodes_idx += (
                    self.ghost_nodes_idx[n_straight_rods - 1] + 1
                )
                self.periodic_boundary_elems_idx += (
                    self.ghost_elems_idx[1::2][n_straight_rods - 1] + 1
                )
                self.periodic_boundary_voronoi_idx += (
                    self.ghost_voronoi_idx[2::3][n_straight_rods - 1] + 1
                )

            self.start_idx_in_rod_nodes[n_straight_rods:] = (
                self.periodic_boundary_nodes_idx[0, 0::3] + 1
            )
            self.end_idx_in_rod_nodes[n_straight_rods:] = (
                self.periodic_boundary_nodes_idx[0, 1::3]
            )
            self.start_idx_in_rod_elems[n_straight_rods:] = (
                self.periodic_boundary_elems_idx[0, 0::2] + 1
            )
            self.end_idx_in_rod_elems[n_straight_rods:] = (
                self.periodic_boundary_elems_idx[0, 1::2]
            )
            self.start_idx_in_rod_voronoi[n_straight_rods:] = (
                self.periodic_boundary_voronoi_idx[0, :] + 1
            )

        self._allocate_block_variables_in_nodes(systems)
        self._allocate_block_variables_in_elements(systems)
        self._allocate_blocks_variables_in_voronoi(systems)
        self._allocate_blocks_variables_for_symplectic_stepper(systems)

        _reset_scalar_ghost(self.mass, self.ghost_nodes_idx, 1.0)
        _reset_scalar_ghost(self.rest_lengths, self.ghost_elems_idx, 1.0)
        _reset_scalar_ghost(self.rest_voronoi_lengths, self.ghost_voronoi_idx, 1.0)

        _compute_sigma_kappa_for_blockstructure(self)

        if n_ring_rods != 0:
            for system_to_be_added in system_ring_rod:
                if np.count_nonzero(system_to_be_added.rest_sigma) == 0:
                    system_to_be_added.rest_sigma[:] = system_to_be_added.sigma[:]
                if np.count_nonzero(system_to_be_added.rest_kappa) == 0:
                    system_to_be_added.rest_kappa[:] = system_to_be_added.kappa[:]

            _synchronize_periodic_boundary_of_vector_collection(
                self.rest_sigma, self.periodic_boundary_elems_idx
            )
            _synchronize_periodic_boundary_of_vector_collection(
                self.rest_kappa, self.periodic_boundary_voronoi_idx
            )

        self._device_state: dict[str, jax.Array] = {}
        self._device_metadata: dict[str, jax.Array] = {}
        self._device_platform = (
            device.platform if device is not None else jax.default_backend()
        )
        self._device_dirty = False
        self._initialize_device_state(device=device)

    @property
    def device_platform(self) -> str:
        return self._device_platform

    @property
    def device_dtype(self) -> np.dtype:
        return self._device_dtype

    @staticmethod
    def _normalize_device_dtype(device_dtype: str | np.dtype | None) -> np.dtype:
        if device_dtype is None:
            return np.dtype(np.float64 if jax.config.x64_enabled else np.float32)
        normalized = np.dtype(device_dtype)
        if normalized == np.dtype(np.float64) and not jax.config.x64_enabled:
            raise ValueError(
                "float64 device_dtype requires JAX x64 support. Enable it with "
                '`jax.config.update("jax_enable_x64", True)` or use float32.'
            )
        if normalized not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError(
                "device_dtype must be one of float32 or float64 for MemoryBlockCosseratRodJax."
            )
        return normalized

    def _allocate_block_variables_in_nodes(self, systems: list[RodType]) -> None:
        map_scalar_dofs_in_rod_nodes = {"mass": 0}
        self.scalar_dofs_in_rod_nodes = np.zeros(
            (len(map_scalar_dofs_in_rod_nodes), self.n_nodes)
        )
        self._map_system_properties_to_block_memory(
            mapping_dict=map_scalar_dofs_in_rod_nodes,
            systems=systems,
            block_memory=self.scalar_dofs_in_rod_nodes,
            domain_type="node",
            value_type="scalar",
        )

        map_vector_dofs_in_rod_nodes = {
            "position_collection": 0,
            "internal_forces": 1,
            "external_forces": 2,
        }
        self.vector_dofs_in_rod_nodes = np.zeros(
            (len(map_vector_dofs_in_rod_nodes), 3 * self.n_nodes)
        )
        self._map_system_properties_to_block_memory(
            mapping_dict=map_vector_dofs_in_rod_nodes,
            systems=systems,
            block_memory=self.vector_dofs_in_rod_nodes,
            domain_type="node",
            value_type="vector",
        )

    def _allocate_block_variables_in_elements(self, systems: list[RodType]) -> None:
        map_scalar_dofs_in_rod_elems = {
            "radius": 0,
            "volume": 1,
            "density": 2,
            "lengths": 3,
            "rest_lengths": 4,
            "dilatation": 5,
            "dilatation_rate": 6,
        }
        self.scalar_dofs_in_rod_elems = np.zeros(
            (len(map_scalar_dofs_in_rod_elems), self.n_elems)
        )
        self._map_system_properties_to_block_memory(
            mapping_dict=map_scalar_dofs_in_rod_elems,
            systems=systems,
            block_memory=self.scalar_dofs_in_rod_elems,
            domain_type="element",
            value_type="scalar",
        )

        map_vector_dofs_in_rod_elems = {
            "tangents": 0,
            "sigma": 1,
            "rest_sigma": 2,
            "internal_torques": 3,
            "external_torques": 4,
            "internal_stress": 5,
        }
        self.vector_dofs_in_rod_elems = np.zeros(
            (len(map_vector_dofs_in_rod_elems), 3 * self.n_elems)
        )
        self._map_system_properties_to_block_memory(
            mapping_dict=map_vector_dofs_in_rod_elems,
            systems=systems,
            block_memory=self.vector_dofs_in_rod_elems,
            domain_type="element",
            value_type="vector",
        )

        map_matrix_dofs_in_rod_elems = {
            "director_collection": 0,
            "mass_second_moment_of_inertia": 1,
            "inv_mass_second_moment_of_inertia": 2,
            "shear_matrix": 3,
        }
        self.matrix_dofs_in_rod_elems = np.zeros(
            (len(map_matrix_dofs_in_rod_elems), 9 * self.n_elems)
        )
        self._map_system_properties_to_block_memory(
            mapping_dict=map_matrix_dofs_in_rod_elems,
            systems=systems,
            block_memory=self.matrix_dofs_in_rod_elems,
            domain_type="element",
            value_type="tensor",
        )

    def _allocate_blocks_variables_in_voronoi(self, systems: list[RodType]) -> None:
        map_scalar_dofs_in_rod_voronois = {
            "voronoi_dilatation": 0,
            "rest_voronoi_lengths": 1,
        }
        self.scalar_dofs_in_rod_voronois = np.zeros(
            (len(map_scalar_dofs_in_rod_voronois), self.n_voronoi)
        )
        self._map_system_properties_to_block_memory(
            mapping_dict=map_scalar_dofs_in_rod_voronois,
            systems=systems,
            block_memory=self.scalar_dofs_in_rod_voronois,
            domain_type="voronoi",
            value_type="scalar",
        )

        map_vector_dofs_in_rod_voronois = {
            "kappa": 0,
            "rest_kappa": 1,
            "internal_couple": 2,
        }
        self.vector_dofs_in_rod_voronois = np.zeros(
            (len(map_vector_dofs_in_rod_voronois), 3 * self.n_voronoi)
        )
        self._map_system_properties_to_block_memory(
            mapping_dict=map_vector_dofs_in_rod_voronois,
            systems=systems,
            block_memory=self.vector_dofs_in_rod_voronois,
            domain_type="voronoi",
            value_type="vector",
        )

        map_matrix_dofs_in_rod_voronois = {"bend_matrix": 0}
        self.matrix_dofs_in_rod_voronois = np.zeros(
            (len(map_matrix_dofs_in_rod_voronois), 9 * self.n_voronoi)
        )
        self._map_system_properties_to_block_memory(
            mapping_dict=map_matrix_dofs_in_rod_voronois,
            systems=systems,
            block_memory=self.matrix_dofs_in_rod_voronois,
            domain_type="voronoi",
            value_type="tensor",
        )

    def _allocate_blocks_variables_for_symplectic_stepper(
        self, systems: list[RodType]
    ) -> None:
        map_rate_collection = {
            "velocity_collection": 0,
            "omega_collection": 1,
            "acceleration_collection": 2,
            "alpha_collection": 3,
        }
        self.rate_collection = np.zeros((len(map_rate_collection), 3 * self.n_nodes))
        self.v_w_collection = np.lib.stride_tricks.as_strided(
            self.rate_collection[0:2], (2, 3 * self.n_nodes)
        )
        self.dvdt_dwdt_collection = np.lib.stride_tricks.as_strided(
            self.rate_collection[2:], (2, 3 * self.n_nodes)
        )

        map_rate_collection_dofs_in_rod_nodes = {
            "velocity_collection": 0,
            "acceleration_collection": 2,
        }
        self._map_system_properties_to_block_memory(
            mapping_dict=map_rate_collection_dofs_in_rod_nodes,
            systems=systems,
            block_memory=self.rate_collection,
            domain_type="node",
            value_type="vector",
        )

        map_rate_collection_dofs_in_rod_elems = {
            "omega_collection": 1,
            "alpha_collection": 3,
        }
        self._map_system_properties_to_block_memory(
            mapping_dict=map_rate_collection_dofs_in_rod_elems,
            systems=systems,
            block_memory=self.rate_collection,
            domain_type="element",
            value_type="vector",
        )

    def _map_system_properties_to_block_memory(
        self,
        mapping_dict: dict[str, int],
        systems: list[RodType],
        block_memory: np.ndarray,
        domain_type: Literal["node", "element", "voronoi"],
        value_type: Literal["scalar", "vector", "tensor"],
    ) -> None:
        start_idx_list: np.ndarray
        end_idx_list: np.ndarray
        periodic_boundary_idx: np.ndarray
        synchronize_periodic_boundary: Callable
        domain_num: int
        view_shape: tuple[int, ...]

        if domain_type == "node":
            start_idx_list = self.start_idx_in_rod_nodes.view()
            end_idx_list = self.end_idx_in_rod_nodes.view()
            domain_num = self.n_nodes
            periodic_boundary_idx = self.periodic_boundary_nodes_idx.view()
        elif domain_type == "element":
            start_idx_list = self.start_idx_in_rod_elems.view()
            end_idx_list = self.end_idx_in_rod_elems.view()
            domain_num = self.n_elems
            periodic_boundary_idx = self.periodic_boundary_elems_idx.view()
        elif domain_type == "voronoi":
            start_idx_list = self.start_idx_in_rod_voronoi.view()
            end_idx_list = self.end_idx_in_rod_voronoi.view()
            domain_num = self.n_voronoi
            periodic_boundary_idx = self.periodic_boundary_voronoi_idx.view()
        else:
            raise ValueError(
                "Incorrect domain type. Must be one of node, element, and voronoi"
            )

        if value_type == "scalar":
            view_shape = (domain_num,)
            synchronize_periodic_boundary = (
                _synchronize_periodic_boundary_of_scalar_collection
            )
        elif value_type == "vector":
            view_shape = (3, domain_num)
            synchronize_periodic_boundary = (
                _synchronize_periodic_boundary_of_vector_collection
            )
        elif value_type == "tensor":
            view_shape = (3, 3, domain_num)
            synchronize_periodic_boundary = (
                _synchronize_periodic_boundary_of_matrix_collection
            )
        else:
            raise ValueError(
                "Incorrect value type. Must be one of scalar, vector, and tensor."
            )

        for attr_name, row_idx in mapping_dict.items():
            self.__dict__[attr_name] = np.lib.stride_tricks.as_strided(
                block_memory[row_idx], shape=view_shape
            )
            for system_idx, system in enumerate(systems):
                start_idx = start_idx_list[system_idx]
                end_idx = end_idx_list[system_idx]
                self.__dict__[attr_name][..., start_idx:end_idx] = system.__dict__[
                    attr_name
                ].copy()
            synchronize_periodic_boundary(
                self.__dict__[attr_name], periodic_boundary_idx
            )

    def _normalize_attr_names(
        self, attrs: Iterable[str] | None = None
    ) -> tuple[str, ...]:
        if attrs is None:
            return _SYNCABLE_ATTRS
        return tuple(dict.fromkeys(attrs))

    def _initialize_device_state(
        self,
        attrs: Iterable[str] | None = None,
        *,
        device: jax.Device | None = None,
    ) -> None:
        target_device = device if device is not None else self._initial_device
        for attr in self._normalize_attr_names(attrs):
            device_array = jnp.asarray(
                np.asarray(getattr(self, attr)), dtype=self._device_dtype
            )
            if target_device is not None:
                device_array = jax.device_put(device_array, device=target_device)
            self._device_state[attr] = device_array
        self._update_device_metadata(device=target_device)
        if target_device is not None:
            self._device_platform = target_device.platform
        self._refresh_device_views()
        self._device_dirty = False

    def to_device(
        self,
        attrs: Iterable[str] | None = None,
        *,
        device: jax.Device | None = None,
    ) -> None:
        raise RuntimeError(
            "MemoryBlockCosseratRodJax keeps device state authoritative after "
            "initialization. `to_device()` is not supported after construction; "
            "run JAX kernels on the block and use `from_device()` for explicit host "
            "readback."
        )

    def to_gpu(self, attrs: Iterable[str] | None = None) -> None:
        self.to_device(attrs=attrs)

    def from_device(
        self,
        attrs: Iterable[str] | None = None,
        *,
        update_rods: bool = True,
    ) -> None:
        sync_attrs = self._normalize_attr_names(attrs)
        for attr in sync_attrs:
            np.copyto(getattr(self, attr), np.asarray(self._device_state[attr]))
        if update_rods:
            self._push_block_state_to_rods(sync_attrs)
        self._device_dirty = False

    def from_gpu(
        self,
        attrs: Iterable[str] | None = None,
        *,
        update_rods: bool = True,
    ) -> None:
        self.from_device(attrs=attrs, update_rods=update_rods)

    def _push_block_state_to_rods(self, attrs: Sequence[str]) -> None:
        for system_idx, system in enumerate(self._systems):
            for attr in attrs:
                start_idx, end_idx = self._get_attr_slice_indices(attr, system_idx)
                np.copyto(
                    system.__dict__[attr],
                    getattr(self, attr)[..., start_idx:end_idx],
                )

    def _get_attr_slice_indices(self, attr: str, system_idx: int) -> tuple[int, int]:
        if attr in _NODE_ATTRS:
            return (
                int(self.start_idx_in_rod_nodes[system_idx]),
                int(self.end_idx_in_rod_nodes[system_idx]),
            )
        if attr in _ELEMENT_ATTRS:
            return (
                int(self.start_idx_in_rod_elems[system_idx]),
                int(self.end_idx_in_rod_elems[system_idx]),
            )
        if attr in _VORONOI_ATTRS:
            return (
                int(self.start_idx_in_rod_voronoi[system_idx]),
                int(self.end_idx_in_rod_voronoi[system_idx]),
            )
        raise KeyError(f"Unsupported sync attribute {attr!r}")

    def _refresh_device_views(self) -> None:
        self.position_collection_device = self._device_state["position_collection"]
        self.director_collection_device = self._device_state["director_collection"]
        self.velocity_collection_device = self._device_state["velocity_collection"]
        self.omega_collection_device = self._device_state["omega_collection"]
        self.acceleration_collection_device = self._device_state[
            "acceleration_collection"
        ]
        self.alpha_collection_device = self._device_state["alpha_collection"]

    def _update_device_metadata(self, *, device: jax.Device | None) -> None:
        def put_index(array: np.ndarray) -> jax.Array:
            index_array = jnp.asarray(array, dtype=jnp.int32)
            if device is not None:
                return jax.device_put(index_array, device=device)
            return index_array

        self._device_metadata["ghost_nodes_idx"] = put_index(self.ghost_nodes_idx)
        self._device_metadata["ghost_elems_idx"] = put_index(self.ghost_elems_idx)
        self._device_metadata["ghost_voronoi_idx"] = put_index(self.ghost_voronoi_idx)
        self._device_metadata["periodic_boundary_nodes_idx"] = put_index(
            self.periodic_boundary_nodes_idx
        )
        self._device_metadata["periodic_boundary_elems_idx"] = put_index(
            self.periodic_boundary_elems_idx
        )
        self._device_metadata["periodic_boundary_voronoi_idx"] = put_index(
            self.periodic_boundary_voronoi_idx
        )

    def _device_scalar(self, value: float | np.floating) -> jax.Array:
        return jax.device_put(
            jnp.asarray(value, dtype=self._device_dtype),
            device=self.position_collection_device.device,
        )

    def compute_internal_forces_and_torques(self, time: np.float64) -> None:
        (
            lengths,
            tangents,
            radius,
            dilatation,
            dilatation_rate,
            voronoi_dilatation,
            sigma,
            kappa,
            internal_stress,
            internal_couple,
            internal_forces,
            internal_torques,
        ) = _jax_compute_internal_forces_and_torques(
            self._device_state["position_collection"],
            self._device_state["velocity_collection"],
            self._device_state["volume"],
            self._device_state["lengths"],
            self._device_state["tangents"],
            self._device_state["radius"],
            self._device_state["rest_lengths"],
            self._device_state["rest_voronoi_lengths"],
            self._device_state["dilatation"],
            self._device_state["dilatation_rate"],
            self._device_state["voronoi_dilatation"],
            self._device_state["director_collection"],
            self._device_state["sigma"],
            self._device_state["rest_sigma"],
            self._device_state["shear_matrix"],
            self._device_state["internal_stress"],
            self._device_state["internal_forces"],
            self._device_state["mass_second_moment_of_inertia"],
            self._device_state["omega_collection"],
            self._device_state["internal_torques"],
            self._device_state["bend_matrix"],
            self._device_state["rest_kappa"],
            self._device_state["kappa"],
            self._device_state["internal_couple"],
            self._device_metadata["ghost_elems_idx"],
            self._device_metadata["ghost_voronoi_idx"],
            self._device_metadata["periodic_boundary_elems_idx"],
            self._device_metadata["periodic_boundary_voronoi_idx"],
        )
        self._device_state["lengths"] = lengths
        self._device_state["tangents"] = tangents
        self._device_state["radius"] = radius
        self._device_state["dilatation"] = dilatation
        self._device_state["dilatation_rate"] = dilatation_rate
        self._device_state["voronoi_dilatation"] = voronoi_dilatation
        self._device_state["sigma"] = sigma
        self._device_state["kappa"] = kappa
        self._device_state["internal_stress"] = internal_stress
        self._device_state["internal_couple"] = internal_couple
        self._device_state["internal_forces"] = internal_forces
        self._device_state["internal_torques"] = internal_torques
        self._device_dirty = True

    def update_accelerations(self, time: np.float64, dt: np.float64) -> None:
        acceleration_collection, alpha_collection = _jax_update_accelerations(
            self._device_state["internal_forces"],
            self._device_state["external_forces"],
            self._device_state["mass"],
            self._device_state["inv_mass_second_moment_of_inertia"],
            self._device_state["internal_torques"],
            self._device_state["external_torques"],
            self._device_state["dilatation"],
        )
        self._device_state["acceleration_collection"] = acceleration_collection
        self._device_state["alpha_collection"] = alpha_collection
        self._refresh_device_views()
        self._device_dirty = True

    def update_kinematics(self, time: np.float64, prefac: np.float64) -> None:
        position_collection, director_collection = _jax_update_kinematics(
            self._device_state["position_collection"],
            self._device_state["director_collection"],
            self._device_state["velocity_collection"],
            self._device_state["omega_collection"],
            self._device_scalar(prefac),
        )
        self._device_state["position_collection"] = position_collection
        self._device_state["director_collection"] = director_collection
        self._refresh_device_views()
        self._device_dirty = True

    def update_dynamics(self, time: np.float64, prefac: np.float64) -> None:
        velocity_collection, omega_collection = _jax_update_dynamics(
            self._device_state["velocity_collection"],
            self._device_state["omega_collection"],
            self._device_state["acceleration_collection"],
            self._device_state["alpha_collection"],
            self._device_scalar(prefac),
        )
        self._device_state["velocity_collection"] = velocity_collection
        self._device_state["omega_collection"] = omega_collection
        self._refresh_device_views()
        self._device_dirty = True

    def zeroed_out_external_forces_and_torques(self, time: np.float64) -> None:
        external_forces, external_torques = _jax_zero_external_loads(
            self._device_state["external_forces"],
            self._device_state["external_torques"],
        )
        self._device_state["external_forces"] = external_forces
        self._device_state["external_torques"] = external_torques
        self._device_dirty = True

    def jax_position_verlet_run(
        self, *, time: np.float64, dt: np.float64, n_steps: int
    ) -> np.float64:
        raise NotImplementedError(
            "MemoryBlockCosseratRodJax does not yet support fully device-side "
            "Position Verlet rollout. The outer JAX loop exists now, but this block "
            "still needs device-side internal-force/torque kernels and device-side "
            "operator phases before it can replace a user Python loop."
        )

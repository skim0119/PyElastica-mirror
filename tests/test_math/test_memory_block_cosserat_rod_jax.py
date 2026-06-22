import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
CPU_DEVICE = jax.devices("cpu")[0]

from elastica.memory_block.memory_block_rod_jax import (
    MemoryBlockCosseratRodJax,
    _jax_update_accelerations,
)
from elastica.rod.cosserat_rod import (
    _compute_internal_forces,
    _compute_internal_torques,
    _update_accelerations,
)
from elastica.rod.data_structures import overload_operator_kinematic_numba
from elastica.utils import Tolerance

JAX_ATOL = 1.0e-6
JAX_RTOL = 1.0e-6


def _make_block(rod, **kwargs):
    with jax.default_device(CPU_DEVICE):
        return MemoryBlockCosseratRodJax([rod], [0], **kwargs)


class MockRod:
    def __init__(self, n_elems):
        self.n_elems = n_elems
        self.n_nodes = self.n_elems + 1
        self.n_voronoi = self.n_elems - 1
        self.ring_rod_flag = False

        rng = np.random.default_rng(42)

        self.mass = rng.random(self.n_nodes) + 1.0

        self.position_collection = rng.standard_normal((3, self.n_nodes))
        self.velocity_collection = rng.standard_normal((3, self.n_nodes))
        self.acceleration_collection = rng.standard_normal((3, self.n_nodes))
        self.internal_forces = rng.standard_normal((3, self.n_nodes))
        self.external_forces = rng.standard_normal((3, self.n_nodes))

        self.radius = rng.random(self.n_elems) + 0.1
        self.volume = rng.random(self.n_elems) + 0.1
        self.density = rng.random(self.n_elems) + 0.1
        self.lengths = rng.random(self.n_elems) + 0.1
        self.rest_lengths = self.lengths.copy()
        self.dilatation = rng.random(self.n_elems) + 1.0
        self.dilatation_rate = rng.random(self.n_elems)

        self.omega_collection = rng.standard_normal((3, self.n_elems))
        self.alpha_collection = rng.standard_normal((3, self.n_elems))
        self.tangents = rng.standard_normal((3, self.n_elems))
        self.sigma = rng.standard_normal((3, self.n_elems))
        self.rest_sigma = rng.standard_normal((3, self.n_elems))
        self.internal_torques = rng.standard_normal((3, self.n_elems))
        self.external_torques = rng.standard_normal((3, self.n_elems))
        self.internal_stress = rng.standard_normal((3, self.n_elems))

        self.director_collection = np.zeros((3, 3, self.n_elems))
        for i in range(3):
            self.director_collection[i, i, :] = 1.0
        self.mass_second_moment_of_inertia = rng.random((3, 3, self.n_elems)) + 1.0
        self.inv_mass_second_moment_of_inertia = rng.random((3, 3, self.n_elems)) + 1.0
        self.shear_matrix = rng.random((3, 3, self.n_elems)) + 1.0

        self.voronoi_dilatation = rng.random(self.n_voronoi) + 1.0
        self.rest_voronoi_lengths = rng.random(self.n_voronoi) + 0.1

        self.kappa = rng.standard_normal((3, self.n_voronoi))
        self.rest_kappa = rng.standard_normal((3, self.n_voronoi))
        self.internal_couple = rng.standard_normal((3, self.n_voronoi))

        self.bend_matrix = rng.random((3, 3, self.n_voronoi)) + 1.0


def test_memory_block_cosserat_rod_jax_to_from_device_updates_rods():
    rod = MockRod(8)
    block = _make_block(rod)

    updated_position = np.asarray(block._device_state["position_collection"]) + 3.5
    block._device_state["position_collection"] = jax.device_put(
        updated_position, device=CPU_DEVICE
    )

    updated_velocity = np.asarray(block._device_state["velocity_collection"]) - 2.0
    block._device_state["velocity_collection"] = jax.device_put(
        updated_velocity, device=CPU_DEVICE
    )

    block.from_device(attrs=("position_collection", "velocity_collection"))

    assert_array_equal(block.position_collection, updated_position)
    assert_array_equal(block.velocity_collection, updated_velocity)
    assert_array_equal(rod.position_collection, updated_position)
    assert_array_equal(rod.velocity_collection, updated_velocity)


def test_memory_block_cosserat_rod_jax_does_not_alias_original_rods():
    rod = MockRod(6)
    block = _make_block(rod)

    assert not np.shares_memory(block.position_collection, rod.position_collection)
    assert not np.shares_memory(block.velocity_collection, rod.velocity_collection)
    assert not np.shares_memory(block.director_collection, rod.director_collection)


def test_memory_block_cosserat_rod_jax_to_device_raises_after_initialization():
    rod = MockRod(6)
    block = _make_block(rod)

    with pytest.raises(RuntimeError):
        block.to_device(attrs=("position_collection",))

    with pytest.raises(RuntimeError):
        block.to_gpu(attrs=("position_collection",))


def test_memory_block_cosserat_rod_jax_respects_device_dtype():
    rod = MockRod(6)
    block = _make_block(rod, device_dtype="float32")

    assert block.device_dtype == np.dtype(np.float32)
    assert np.asarray(block._device_state["position_collection"]).dtype == np.float32
    assert np.asarray(block._device_state["director_collection"]).dtype == np.float32


def test_memory_block_cosserat_rod_jax_protocol_methods_are_state_consistent():
    rod = MockRod(6)
    block = _make_block(rod)

    with jax.default_device(CPU_DEVICE):
        state = block.jax_get_state()
        state = block.jax_kinematic_step(state, np.float64(0.0), np.float64(0.125))
        state = block.jax_compute_internal_forces_and_torques(state, np.float64(0.125))
        state = block.jax_dynamic_step(state, np.float64(0.125), np.float64(0.2))
        state = block.jax_zero_external_loads(state, np.float64(0.2))
    block.jax_set_state(state)
    block.from_device(
        attrs=(
            "position_collection",
            "director_collection",
            "lengths",
            "tangents",
            "radius",
            "dilatation",
            "dilatation_rate",
            "voronoi_dilatation",
            "sigma",
            "kappa",
            "internal_stress",
            "internal_couple",
            "internal_forces",
            "internal_torques",
            "acceleration_collection",
            "alpha_collection",
            "velocity_collection",
            "omega_collection",
            "external_forces",
            "external_torques",
        ),
        update_rods=False,
    )

    rod_ref = MockRod(6)
    ref_block = _make_block(rod_ref)
    with jax.default_device(CPU_DEVICE):
        ref_state = ref_block.jax_get_state()
        ref_state = ref_block.jax_kinematic_step(
            ref_state, np.float64(0.0), np.float64(0.125)
        )
        ref_state = ref_block.jax_compute_internal_forces_and_torques(
            ref_state, np.float64(0.125)
        )
        ref_state = ref_block.jax_dynamic_step(
            ref_state, np.float64(0.125), np.float64(0.2)
        )
        ref_state = ref_block.jax_zero_external_loads(ref_state, np.float64(0.2))
    ref_block.jax_set_state(ref_state)
    ref_block.from_device(
        attrs=(
            "position_collection",
            "director_collection",
            "lengths",
            "tangents",
            "radius",
            "dilatation",
            "dilatation_rate",
            "voronoi_dilatation",
            "sigma",
            "kappa",
            "internal_stress",
            "internal_couple",
            "internal_forces",
            "internal_torques",
            "acceleration_collection",
            "alpha_collection",
            "velocity_collection",
            "omega_collection",
            "external_forces",
            "external_torques",
        ),
        update_rods=False,
    )

    assert_allclose(
        block.position_collection,
        ref_block.position_collection,
        atol=JAX_ATOL,
        rtol=JAX_RTOL,
    )
    assert_allclose(
        block.director_collection,
        ref_block.director_collection,
        atol=JAX_ATOL,
        rtol=JAX_RTOL,
    )
    assert_allclose(block.lengths, ref_block.lengths, atol=JAX_ATOL, rtol=JAX_RTOL)
    assert_allclose(block.tangents, ref_block.tangents, atol=JAX_ATOL, rtol=JAX_RTOL)
    assert_allclose(block.radius, ref_block.radius, atol=JAX_ATOL, rtol=JAX_RTOL)
    assert_allclose(
        block.dilatation, ref_block.dilatation, atol=JAX_ATOL, rtol=JAX_RTOL
    )
    assert_allclose(
        block.dilatation_rate, ref_block.dilatation_rate, atol=JAX_ATOL, rtol=JAX_RTOL
    )
    assert_allclose(
        block.voronoi_dilatation,
        ref_block.voronoi_dilatation,
        atol=JAX_ATOL,
        rtol=JAX_RTOL,
    )
    assert_allclose(block.sigma, ref_block.sigma, atol=JAX_ATOL, rtol=JAX_RTOL)
    assert_allclose(block.kappa, ref_block.kappa, atol=JAX_ATOL, rtol=JAX_RTOL)
    assert_allclose(
        block.internal_stress, ref_block.internal_stress, atol=JAX_ATOL, rtol=JAX_RTOL
    )
    assert_allclose(
        block.internal_couple, ref_block.internal_couple, atol=JAX_ATOL, rtol=JAX_RTOL
    )
    assert_allclose(
        block.internal_forces, ref_block.internal_forces, atol=JAX_ATOL, rtol=JAX_RTOL
    )
    assert_allclose(
        block.internal_torques, ref_block.internal_torques, atol=JAX_ATOL, rtol=JAX_RTOL
    )
    assert_allclose(
        block.acceleration_collection,
        ref_block.acceleration_collection,
        atol=JAX_ATOL,
        rtol=JAX_RTOL,
    )
    assert_allclose(
        block.alpha_collection, ref_block.alpha_collection, atol=JAX_ATOL, rtol=JAX_RTOL
    )
    assert_allclose(
        block.velocity_collection,
        ref_block.velocity_collection,
        atol=JAX_ATOL,
        rtol=JAX_RTOL,
    )
    assert_allclose(
        block.omega_collection, ref_block.omega_collection, atol=JAX_ATOL, rtol=JAX_RTOL
    )
    assert_allclose(
        block.external_forces, ref_block.external_forces, atol=JAX_ATOL, rtol=JAX_RTOL
    )
    assert_allclose(
        block.external_torques, ref_block.external_torques, atol=JAX_ATOL, rtol=JAX_RTOL
    )


def test_memory_block_cosserat_rod_jax_kinematic_update_matches_numba():
    rod = MockRod(7)
    block = _make_block(rod)

    position = block.position_collection.copy()
    directors = block.director_collection.copy()
    velocity = block.velocity_collection.copy()
    omega = block.omega_collection.copy()
    prefac = 0.125

    overload_operator_kinematic_numba(
        prefac,
        position,
        directors,
        velocity,
        omega,
    )

    with jax.default_device(CPU_DEVICE):
        block.jax_set_state(
            block.jax_kinematic_step(
                block.jax_get_state(), np.float64(0.0), np.float64(prefac)
            )
        )
    block.from_device(
        attrs=("position_collection", "director_collection"), update_rods=False
    )

    assert_allclose(
        position,
        block.position_collection,
        atol=max(Tolerance.atol(), JAX_ATOL),
        rtol=JAX_RTOL,
    )
    assert_allclose(
        directors,
        block.director_collection,
        atol=max(Tolerance.atol(), JAX_ATOL),
        rtol=JAX_RTOL,
    )


def test_memory_block_cosserat_rod_jax_dynamic_update_matches_numba():
    rod = MockRod(9)
    block = _make_block(rod)

    dt = -0.3

    with jax.default_device(CPU_DEVICE):
        expected_acceleration, expected_alpha = _jax_update_accelerations(
            jax.device_put(block.internal_forces, device=CPU_DEVICE),
            jax.device_put(block.external_forces, device=CPU_DEVICE),
            jax.device_put(block.mass, device=CPU_DEVICE),
            jax.device_put(block.inv_mass_second_moment_of_inertia, device=CPU_DEVICE),
            jax.device_put(block.internal_torques, device=CPU_DEVICE),
            jax.device_put(block.external_torques, device=CPU_DEVICE),
            jax.device_put(block.dilatation, device=CPU_DEVICE),
        )
    expected_velocity, expected_omega = (
        block.velocity_collection + dt * np.asarray(expected_acceleration),
        block.omega_collection + dt * np.asarray(expected_alpha),
    )

    with jax.default_device(CPU_DEVICE):
        block.jax_set_state(
            block.jax_dynamic_step(
                block.jax_get_state(), np.float64(0.0), np.float64(dt)
            )
        )
    block.from_device(
        attrs=(
            "acceleration_collection",
            "alpha_collection",
            "velocity_collection",
            "omega_collection",
        ),
        update_rods=False,
    )

    assert_allclose(
        expected_velocity,
        block.velocity_collection,
        atol=max(Tolerance.atol(), JAX_ATOL),
        rtol=JAX_RTOL,
    )
    assert_allclose(
        expected_omega,
        block.omega_collection,
        atol=max(Tolerance.atol(), JAX_ATOL),
        rtol=JAX_RTOL,
    )
    assert_allclose(
        np.asarray(expected_acceleration),
        block.acceleration_collection,
        atol=max(Tolerance.atol(), JAX_ATOL),
        rtol=JAX_RTOL,
    )
    assert_allclose(
        np.asarray(expected_alpha),
        block.alpha_collection,
        atol=max(Tolerance.atol(), JAX_ATOL),
        rtol=JAX_RTOL,
    )

    assert_allclose(
        np.asarray(block._device_state["velocity_collection"]),
        block.velocity_collection,
        atol=max(Tolerance.atol(), JAX_ATOL),
        rtol=JAX_RTOL,
    )
    assert_allclose(
        np.asarray(block._device_state["omega_collection"]),
        block.omega_collection,
        atol=max(Tolerance.atol(), JAX_ATOL),
        rtol=JAX_RTOL,
    )


def test_memory_block_cosserat_rod_jax_update_accelerations_matches_numba():
    rod = MockRod(5)
    block = _make_block(rod)

    expected_acceleration = block.acceleration_collection.copy()
    expected_alpha = block.alpha_collection.copy()
    _update_accelerations(
        expected_acceleration,
        block.internal_forces.copy(),
        block.external_forces.copy(),
        block.mass.copy(),
        expected_alpha,
        block.inv_mass_second_moment_of_inertia.copy(),
        block.internal_torques.copy(),
        block.external_torques.copy(),
        block.dilatation.copy(),
    )

    with jax.default_device(CPU_DEVICE):
        block.jax_set_state(
            block.jax_dynamic_step(
                block.jax_get_state(), np.float64(0.0), np.float64(0.0)
            )
        )
    block.from_device(
        attrs=("acceleration_collection", "alpha_collection"), update_rods=False
    )

    assert_allclose(
        expected_acceleration,
        block.acceleration_collection,
        atol=max(Tolerance.atol(), JAX_ATOL),
        rtol=JAX_RTOL,
    )
    assert_allclose(
        expected_alpha,
        block.alpha_collection,
        atol=max(Tolerance.atol(), JAX_ATOL),
        rtol=JAX_RTOL,
    )


def test_memory_block_cosserat_rod_jax_zero_external_loads():
    rod = MockRod(4)
    block = _make_block(rod)

    with jax.default_device(CPU_DEVICE):
        block.jax_set_state(
            block.jax_zero_external_loads(block.jax_get_state(), np.float64(0.0))
        )
    block.from_device(attrs=("external_forces", "external_torques"), update_rods=False)

    assert_array_equal(block.external_forces, np.zeros_like(block.external_forces))
    assert_array_equal(block.external_torques, np.zeros_like(block.external_torques))


def test_memory_block_cosserat_rod_jax_internal_forces_and_torques_match_numba():
    rod = MockRod(6)
    block = _make_block(rod)

    expected_lengths = block.lengths.copy()
    expected_tangents = block.tangents.copy()
    expected_radius = block.radius.copy()
    expected_dilatation = block.dilatation.copy()
    expected_voronoi_dilatation = block.voronoi_dilatation.copy()
    expected_sigma = block.sigma.copy()
    expected_kappa = block.kappa.copy()
    expected_internal_stress = block.internal_stress.copy()
    expected_internal_couple = block.internal_couple.copy()
    expected_internal_forces = block.internal_forces.copy()
    expected_internal_torques = block.internal_torques.copy()
    expected_dilatation_rate = block.dilatation_rate.copy()

    _compute_internal_forces(
        block.position_collection.copy(),
        block.volume.copy(),
        expected_lengths,
        expected_tangents,
        expected_radius,
        block.rest_lengths.copy(),
        block.rest_voronoi_lengths.copy(),
        expected_dilatation,
        expected_voronoi_dilatation,
        block.director_collection.copy(),
        expected_sigma,
        block.rest_sigma.copy(),
        block.shear_matrix.copy(),
        expected_internal_stress,
        expected_internal_forces,
        block.ghost_elems_idx.copy(),
    )
    _compute_internal_torques(
        block.position_collection.copy(),
        block.velocity_collection.copy(),
        expected_tangents,
        expected_lengths,
        block.rest_lengths.copy(),
        block.director_collection.copy(),
        block.rest_voronoi_lengths.copy(),
        block.bend_matrix.copy(),
        block.rest_kappa.copy(),
        expected_kappa,
        expected_voronoi_dilatation,
        block.mass_second_moment_of_inertia.copy(),
        block.omega_collection.copy(),
        expected_internal_stress,
        expected_internal_couple,
        expected_dilatation,
        expected_dilatation_rate,
        expected_internal_torques,
        block.ghost_voronoi_idx.copy(),
    )

    with jax.default_device(CPU_DEVICE):
        block.jax_set_state(
            block.jax_compute_internal_forces_and_torques(
                block.jax_get_state(), np.float64(0.0)
            )
        )
    block.from_device(
        attrs=(
            "lengths",
            "tangents",
            "radius",
            "dilatation",
            "dilatation_rate",
            "voronoi_dilatation",
            "sigma",
            "kappa",
            "internal_stress",
            "internal_couple",
            "internal_forces",
            "internal_torques",
        ),
        update_rods=False,
    )

    assert_allclose(expected_lengths, block.lengths, atol=JAX_ATOL, rtol=JAX_RTOL)
    assert_allclose(expected_tangents, block.tangents, atol=JAX_ATOL, rtol=JAX_RTOL)
    assert_allclose(expected_radius, block.radius, atol=JAX_ATOL, rtol=JAX_RTOL)
    assert_allclose(expected_dilatation, block.dilatation, atol=JAX_ATOL, rtol=JAX_RTOL)
    assert_allclose(
        expected_voronoi_dilatation,
        block.voronoi_dilatation,
        atol=JAX_ATOL,
        rtol=JAX_RTOL,
    )
    assert_allclose(expected_sigma, block.sigma, atol=JAX_ATOL, rtol=JAX_RTOL)
    assert_allclose(expected_kappa, block.kappa, atol=JAX_ATOL, rtol=JAX_RTOL)
    assert_allclose(
        expected_internal_stress,
        block.internal_stress,
        atol=JAX_ATOL,
        rtol=JAX_RTOL,
    )
    assert_allclose(
        expected_internal_couple,
        block.internal_couple,
        atol=JAX_ATOL,
        rtol=JAX_RTOL,
    )
    assert_allclose(
        expected_internal_forces,
        block.internal_forces,
        atol=JAX_ATOL,
        rtol=JAX_RTOL,
    )
    assert_allclose(
        expected_internal_torques,
        block.internal_torques,
        atol=JAX_ATOL,
        rtol=JAX_RTOL,
    )
    assert_allclose(
        expected_dilatation_rate,
        block.dilatation_rate,
        atol=JAX_ATOL,
        rtol=JAX_RTOL,
    )

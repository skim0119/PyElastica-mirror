import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
CPU_DEVICE = jax.devices("cpu")[0]

from elastica.memory_block.memory_block_rigid_body_jax import (
    MemoryBlockRigidBodyJax,
    _jax_update_accelerations,
)
from elastica.rigidbody.rigid_body_base import RigidBodyBase

JAX_ATOL = 1.0e-6
JAX_RTOL = 1.0e-6


def _make_block(rigid_body, **kwargs):
    with jax.default_device(CPU_DEVICE):
        return MemoryBlockRigidBodyJax([rigid_body], [0], **kwargs)


class MockRigidBody(RigidBodyBase):
    def __init__(self):
        super().__init__()

        rng = np.random.default_rng(42)

        self.density = np.float64(rng.random() + 0.1)
        self.volume = np.float64(rng.random() + 0.1)
        self.mass = np.float64(rng.random() + 1.0)

        self.position_collection = rng.standard_normal((3, 1))
        self.velocity_collection = rng.standard_normal((3, 1))
        self.acceleration_collection = rng.standard_normal((3, 1))
        self.omega_collection = rng.standard_normal((3, 1))
        self.alpha_collection = rng.standard_normal((3, 1))

        self.external_forces = rng.standard_normal((3, 1))
        self.external_torques = rng.standard_normal((3, 1))

        self.director_collection = np.zeros((3, 3, 1))
        for i in range(3):
            self.director_collection[i, i, 0] = 1.0

        mass_second_moment = rng.random((3,)) + 1.0
        self.mass_second_moment_of_inertia = np.diag(mass_second_moment).reshape(
            3, 3, 1
        )
        self.inv_mass_second_moment_of_inertia = np.diag(
            1.0 / mass_second_moment
        ).reshape(3, 3, 1)


def test_memory_block_rigid_body_jax_to_from_device_updates_bodies():
    rigid_body = MockRigidBody()
    block = _make_block(rigid_body)

    updated_mass = np.asarray(block._device_state["mass"]) + 1.25
    block._device_state["mass"] = jax.device_put(updated_mass, device=CPU_DEVICE)

    updated_position = np.asarray(block._device_state["position_collection"]) + 3.5
    block._device_state["position_collection"] = jax.device_put(
        updated_position, device=CPU_DEVICE
    )

    updated_velocity = np.asarray(block._device_state["velocity_collection"]) - 2.0
    block._device_state["velocity_collection"] = jax.device_put(
        updated_velocity, device=CPU_DEVICE
    )

    block.from_device(attrs=("mass", "position_collection", "velocity_collection"))

    assert_array_equal(block.mass, updated_mass)
    assert_array_equal(block.position_collection, updated_position)
    assert_array_equal(block.velocity_collection, updated_velocity)
    assert_array_equal(np.asarray(rigid_body.mass).reshape(1), updated_mass)
    assert_array_equal(rigid_body.position_collection, updated_position)
    assert_array_equal(rigid_body.velocity_collection, updated_velocity)


def test_memory_block_rigid_body_jax_does_not_alias_original_bodies():
    rigid_body = MockRigidBody()
    block = _make_block(rigid_body)

    assert not np.shares_memory(
        block.position_collection, rigid_body.position_collection
    )
    assert not np.shares_memory(
        block.velocity_collection, rigid_body.velocity_collection
    )
    assert not np.shares_memory(
        block.director_collection, rigid_body.director_collection
    )


def test_memory_block_rigid_body_jax_to_device_raises_after_initialization():
    rigid_body = MockRigidBody()
    block = _make_block(rigid_body)

    with pytest.raises(RuntimeError):
        block.to_device(attrs=("position_collection",))

    with pytest.raises(RuntimeError):
        block.to_gpu(attrs=("position_collection",))


def test_memory_block_rigid_body_jax_respects_device_dtype():
    rigid_body = MockRigidBody()
    block = _make_block(rigid_body, device_dtype="float32")

    assert block.device_dtype == np.dtype(np.float32)
    assert np.asarray(block._device_state["position_collection"]).dtype == np.float32
    assert np.asarray(block._device_state["director_collection"]).dtype == np.float32


def test_memory_block_rigid_body_jax_protocol_methods_are_state_consistent():
    rigid_body = MockRigidBody()
    block = _make_block(rigid_body)

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
            "velocity_collection",
            "omega_collection",
            "acceleration_collection",
            "alpha_collection",
            "external_forces",
            "external_torques",
        ),
        update_rods=False,
    )

    rigid_body_ref = MockRigidBody()
    ref_block = _make_block(rigid_body_ref)
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
            "velocity_collection",
            "omega_collection",
            "acceleration_collection",
            "alpha_collection",
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
    assert_allclose(
        block.velocity_collection,
        ref_block.velocity_collection,
        atol=JAX_ATOL,
        rtol=JAX_RTOL,
    )
    assert_allclose(
        block.omega_collection,
        ref_block.omega_collection,
        atol=JAX_ATOL,
        rtol=JAX_RTOL,
    )
    assert_allclose(
        block.acceleration_collection,
        ref_block.acceleration_collection,
        atol=JAX_ATOL,
        rtol=JAX_RTOL,
    )
    assert_allclose(
        block.alpha_collection,
        ref_block.alpha_collection,
        atol=JAX_ATOL,
        rtol=JAX_RTOL,
    )
    assert_allclose(block.external_forces, 0.0, atol=JAX_ATOL, rtol=JAX_RTOL)
    assert_allclose(block.external_torques, 0.0, atol=JAX_ATOL, rtol=JAX_RTOL)


def test_memory_block_rigid_body_jax_update_accelerations_matches_rigid_body_base():
    rigid_body = MockRigidBody()
    block = _make_block(rigid_body)

    rigid_body.update_accelerations(time=np.float64(0.0), dt=np.float64(0.0))

    with jax.default_device(CPU_DEVICE):
        acceleration_collection, alpha_collection = _jax_update_accelerations(
            block._device_state["external_forces"],
            block._device_state["mass"],
            block._device_state["mass_second_moment_of_inertia"],
            block._device_state["inv_mass_second_moment_of_inertia"],
            block._device_state["omega_collection"],
            block._device_state["external_torques"],
        )

    assert_allclose(
        np.asarray(acceleration_collection),
        rigid_body.acceleration_collection,
        atol=JAX_ATOL,
        rtol=JAX_RTOL,
    )
    assert_allclose(
        np.asarray(alpha_collection),
        rigid_body.alpha_collection,
        atol=JAX_ATOL,
        rtol=JAX_RTOL,
    )

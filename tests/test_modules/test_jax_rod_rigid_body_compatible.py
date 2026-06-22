import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from elastica import (
    BaseSystemCollection,
    JAXOpsRodRigidBody,
    MemoryBlockCosseratRodJax,
    MemoryBlockRigidBodyJax,
    NoRodRigidBodyJax,
    Sphere,
)
from elastica.rod.cosserat_rod import CosseratRod


class _DummySimulator(BaseSystemCollection, JAXOpsRodRigidBody):
    pass


class _CoupleRodAndRigidBody(NoRodRigidBodyJax):
    def __init__(
        self, scale: float, *, _first_system=None, _second_system=None
    ) -> None:
        self.scale = np.float64(scale)

    def jax_operate_synchronize(self, rod_view, rigid_body_view, time):
        rod_force = self.scale * rigid_body_view.mass[None, :]
        rigid_force = self.scale * rod_view.mass[:1].sum()[None, None]
        rod_view.external_forces = rod_view.external_forces + rod_force
        rigid_body_view.external_forces = rigid_body_view.external_forces - rigid_force
        return rod_view, rigid_body_view


def _build_rod(n_elems: int = 8) -> CosseratRod:
    return CosseratRod.straight_rod(
        n_elements=n_elems,
        start=np.zeros(3),
        direction=np.array([0.0, 0.0, 1.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        base_length=1.0,
        base_radius=0.05,
        density=1_000.0,
        youngs_modulus=1.0e6,
    )


def _build_sphere() -> Sphere:
    return Sphere(
        center=np.array([0.25, 0.0, 0.0]),
        base_radius=0.05,
        density=500.0,
    )


def test_jax_rod_rigid_body_compatible_defaults_to_identity_transforms():
    simulator = _DummySimulator()
    states = ({"value": 1.0}, {"value": 2.0})

    assert simulator.jax_constrain_values(states, np.float64(0.0)) == states
    assert simulator.jax_synchronize(states, np.float64(0.0)) == states
    assert simulator.jax_constrain_rates(states, np.float64(0.0)) == states


def test_jax_rod_rigid_body_finalize_wraps_pair_views_into_stage_operator():
    with jax.default_device(jax.devices("cpu")[0]):
        simulator = _DummySimulator()
        simulator.enable_block_supports(CosseratRod, MemoryBlockCosseratRodJax)
        simulator.enable_block_supports(Sphere, MemoryBlockRigidBodyJax)

        rod = _build_rod()
        rigid_body = _build_sphere()
        simulator.append(rod)
        simulator.append(rigid_body)
        simulator.using_on(rod, rigid_body).operate(_CoupleRodAndRigidBody, 2.5)
        simulator.finalize()

        systems = tuple(simulator.final_systems())
        assert len(systems) == 2
        rod_block, rigid_body_block = systems

        initial_states = (
            rod_block.jax_get_state(),
            rigid_body_block.jax_get_state(),
        )
        updated_states = simulator.jax_synchronize(initial_states, np.float64(0.0))
        updated_rod_state, updated_rigid_body_state = updated_states

        expected_rod = np.asarray(initial_states[0]["external_forces"]).copy()
        expected_rod += 2.5 * np.asarray(initial_states[1]["mass"])[None, :]
        np.testing.assert_allclose(
            np.asarray(updated_rod_state["external_forces"]),
            expected_rod,
        )

        expected_rigid_body = np.asarray(initial_states[1]["external_forces"]).copy()
        expected_rigid_body -= (
            2.5 * np.asarray(initial_states[0]["mass"])[None, :1].sum()
        )
        np.testing.assert_allclose(
            np.asarray(updated_rigid_body_state["external_forces"]),
            expected_rigid_body,
        )

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from elastica.jax_operation import NoOpsJax
from elastica.memory_block.memory_block_rod_jax import MemoryBlockCosseratRodJax
from elastica.modules import BaseSystemCollection, JAXOps
from elastica.rod.cosserat_rod import CosseratRod


class _DummySimulator(BaseSystemCollection, JAXOps):
    pass


class _AddGravityLikeLoad(NoOpsJax):
    def __init__(self, scale: float, *, _system=None) -> None:
        self.scale = np.float64(scale)

    def jax_operate_synchronize(self, rod_view, time):
        rod_view.external_forces = (
            rod_view.external_forces + self.scale * rod_view.mass[None, :]
        )
        return rod_view


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


def test_jax_compatible_defaults_to_identity_transforms():
    simulator = _DummySimulator()
    states = ({"value": 1.0}, {"value": 2.0})

    assert simulator.jax_constrain_values(states, np.float64(0.0)) == states
    assert simulator.jax_synchronize(states, np.float64(0.0)) == states
    assert simulator.jax_constrain_rates(states, np.float64(0.0)) == states


def test_jax_ops_finalize_wraps_rod_view_into_stage_operator():
    with jax.default_device(jax.devices("cpu")[0]):
        simulator = _DummySimulator()
        simulator.enable_block_supports(CosseratRod, MemoryBlockCosseratRodJax)

        rod = _build_rod()
        simulator.append(rod)
        simulator.using(rod).operate(_AddGravityLikeLoad, 2.5)
        simulator.finalize()

        systems = tuple(simulator.final_systems())
        assert len(systems) == 1
        block = systems[0]

        initial_state = block.jax_get_state()
        updated_states = simulator.jax_synchronize((initial_state,), np.float64(0.0))
        updated_state = updated_states[0]

        expected = np.asarray(initial_state["external_forces"]).copy()
        expected += 2.5 * np.asarray(initial_state["mass"])[None, :]
        np.testing.assert_allclose(
            np.asarray(updated_state["external_forces"]),
            expected,
        )

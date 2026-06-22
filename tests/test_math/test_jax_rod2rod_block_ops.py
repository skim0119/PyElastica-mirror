from __future__ import annotations

import numpy as np

import elastica as ea


class _MarkerRodRodBlockOp(ea.NoRodRodBlockOpJax):
    def __init__(self, increment: float, *, _system, _pair_metadata) -> None:
        del _system
        self.increment = increment
        self.metadata = _pair_metadata

    def jax_rod2rod_operate_synchronize(self, state, time):
        del time
        updated = dict(state)
        external_forces = state["external_forces"]
        first_node = self.metadata.first_node_indices[:, 0]
        second_node = self.metadata.second_node_indices[:, 0]
        external_forces = external_forces.at[0, first_node].add(self.increment)
        external_forces = external_forces.at[1, second_node].add(2.0 * self.increment)
        updated["external_forces"] = external_forces
        return updated


class _RodRodBlockTestSimulator(
    ea.BaseSystemCollection,
    ea.JAXRodRodBlockOps,
):
    pass


def test_jax_rod2rod_block_op_updates_paired_rods() -> None:
    simulator = _RodRodBlockTestSimulator()
    simulator.enable_block_supports(ea.CosseratRod, ea.MemoryBlockCosseratRodJax)

    rod_one = ea.CosseratRod.straight_rod(
        4,
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        0.5,
        0.01,
        1000.0,
        youngs_modulus=1.0e6,
        shear_modulus=1.0e6 / 1.5,
    )
    rod_two = ea.CosseratRod.straight_rod(
        4,
        np.array([0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        0.5,
        0.01,
        1000.0,
        youngs_modulus=1.0e6,
        shear_modulus=1.0e6 / 1.5,
    )

    simulator.append(rod_one)
    simulator.append(rod_two)
    simulator.connect_block(rod_one, rod_two).using(_MarkerRodRodBlockOp, 3.0)
    simulator.finalize()

    block = tuple(simulator.final_systems())[0]
    state = block.jax_get_state()
    updated_states = simulator.jax_synchronize((state,), np.float64(0.0))
    updated_state = updated_states[0]

    first_node = int(block.start_idx_in_rod_nodes[0])
    second_node = int(block.start_idx_in_rod_nodes[1])

    assert np.isclose(
        updated_state["external_forces"][0, first_node], 3.0
    ), "Rod-to-rod block op did not update the first rod as expected."
    assert np.isclose(
        updated_state["external_forces"][1, second_node], 6.0
    ), "Rod-to-rod block op did not update the second rod as expected."

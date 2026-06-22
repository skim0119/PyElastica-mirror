from __future__ import annotations

import numpy as np

import elastica as ea
from benchmark._jax_snake_common import (
    AnalyticalLinearDamperBlockJax,
    ConfiguredSnakeMemoryBlock,
    GravityPlaneContactBlockJax,
    SnakeMuscleTorquesBlockJax,
)
from elastica.memory_block.memory_block_rod_jax import JAXRodView, JAXRodViewMetadata
from examples.ContinuumSnakeGPUCase.run_continuum_snake_gpu import (
    SnakeMuscleTorquesJax,
    SnakePlaneContactJax,
    build_rod,
    default_b_coeff,
)


def _single_rod_metadata(block: ConfiguredSnakeMemoryBlock) -> JAXRodViewMetadata:
    return JAXRodViewMetadata(
        0,
        slice(
            int(block.start_idx_in_rod_nodes[0]),
            int(block.end_idx_in_rod_nodes[0]),
        ),
        slice(
            int(block.start_idx_in_rod_elems[0]),
            int(block.end_idx_in_rod_elems[0]),
        ),
        slice(
            int(block.start_idx_in_rod_voronoi[0]),
            int(block.end_idx_in_rod_voronoi[0]),
        ),
    )


def test_block_snake_ops_match_single_rod_sequence() -> None:
    rod = build_rod(n_elem=10)
    ConfiguredSnakeMemoryBlock.device = None
    ConfiguredSnakeMemoryBlock.device_dtype = np.dtype(np.float64)
    block = ConfiguredSnakeMemoryBlock([rod], [0])
    base_state = block.jax_compute_internal_forces_and_torques(
        block.jax_get_state(),
        np.float64(0.0),
    )
    metadata = _single_rod_metadata(block)

    b_coeff = default_b_coeff()
    kinetic_mu = np.array([0.89226666, 1.3384, 1.7845333], dtype=np.float64)
    static_mu = np.zeros(3, dtype=np.float64)
    time = np.float64(0.3)

    rod_state = (
        SnakeMuscleTorquesJax(
            b_coeff=b_coeff,
            period=2.0,
            base_length=0.35,
            gravitational_acc=-9.80665,
            _system=rod,
        )
        .jax_operate_synchronize(JAXRodView(base_state, metadata), time)
        .commit()
    )
    rod_state = (
        SnakePlaneContactJax(
            plane_origin=np.array([0.0, -0.35 * 0.011, 0.0], dtype=np.float64),
            plane_normal=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            slip_velocity_tol=1.0e-8,
            k=1.0,
            nu=1.0e-6,
            kinetic_mu_array=kinetic_mu,
            static_mu_array=static_mu,
            _system=rod,
        )
        .jax_operate_synchronize(JAXRodView(rod_state, metadata), time)
        .commit()
    )
    rod_state = (
        ea.AnalyticalLinearDamperJax(
            time_step=np.float64(1.0e-4),
            damping_constant=2.0e-3,
            _system=rod,
        )
        .jax_operate_constrain_rates(JAXRodView(rod_state, metadata), time)
        .commit()
    )

    block_state = SnakeMuscleTorquesBlockJax(
        b_coeff=b_coeff,
        period=2.0,
        base_length=0.35,
        _system=block,
    ).jax_block_operate_synchronize(base_state, time)
    block_state = GravityPlaneContactBlockJax(
        plane_origin=np.array([0.0, -0.35 * 0.011, 0.0], dtype=np.float64),
        plane_normal=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        slip_velocity_tol=1.0e-8,
        k=1.0,
        nu=1.0e-6,
        kinetic_mu_array=kinetic_mu,
        static_mu_array=static_mu,
        gravitational_acc=-9.80665,
        _system=block,
    ).jax_block_operate_synchronize(block_state, time)
    block_state = AnalyticalLinearDamperBlockJax(
        time_step=np.float64(1.0e-4),
        damping_constant=2.0e-3,
        _system=block,
    ).jax_block_operate_constrain_rates(block_state, time)

    node_slice = metadata.node_slice
    elem_slice = metadata.element_slice

    np.testing.assert_allclose(
        np.asarray(block_state["external_forces"])[:, node_slice],
        np.asarray(rod_state["external_forces"]),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.asarray(block_state["external_torques"])[:, elem_slice],
        np.asarray(rod_state["external_torques"]),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.asarray(block_state["velocity_collection"])[:, node_slice],
        np.asarray(rod_state["velocity_collection"]),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.asarray(block_state["omega_collection"])[:, elem_slice],
        np.asarray(rod_state["omega_collection"]),
        atol=1.0e-14,
    )

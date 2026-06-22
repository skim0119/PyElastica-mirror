__doc__ = """Test scripts for rotation kernels in Elastica JAX implementation"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

CPU_DEVICE = jax.devices("cpu")[0]

from elastica._jax_rotations import _jax_get_rotation_matrix, _jax_inv_rotate
from elastica._rotations import _get_rotation_matrix, _inv_rotate
from elastica.utils import Tolerance


@pytest.mark.parametrize("axis_component, axis_idx", [(0.6, 0), (1.0, 0)])
@pytest.mark.parametrize("dt", [0.2, 1.0])
def test_jax_get_rotation_matrix_correct_rotation_about_x(axis_component, axis_idx, dt):
    vector_collection = np.zeros((3, 1))
    vector_collection[axis_idx, 0] = axis_component
    with jax.default_device(CPU_DEVICE):
        test_rot_mat = np.asarray(
            _jax_get_rotation_matrix(
                dt, jax.numpy.asarray(vector_collection, dtype=np.float64)
            )
        )
    theta = axis_component * dt
    correct_rot_mat = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta), np.sin(theta)],
            [0.0, -np.sin(theta), np.cos(theta)],
        ]
    ).reshape(3, 3, 1)

    assert test_rot_mat.shape == (3, 3, 1)
    assert_allclose(test_rot_mat, correct_rot_mat, atol=Tolerance.atol())


@pytest.mark.parametrize("ycomp", [0.7, 1.0])
@pytest.mark.parametrize("dt", [0.4, 1.0])
def test_jax_get_rotation_matrix_correct_rotation_about_y(ycomp, dt):
    vector_collection = np.array([0.0, ycomp, 0.0]).reshape(-1, 1)
    with jax.default_device(CPU_DEVICE):
        test_rot_mat = np.asarray(
            _jax_get_rotation_matrix(
                dt, jax.numpy.asarray(vector_collection, dtype=np.float64)
            )
        )
    test_theta = ycomp * dt
    correct_rot_mat = np.array(
        [
            [np.cos(test_theta), 0.0, -np.sin(test_theta)],
            [0.0, 1.0, 0.0],
            [np.sin(test_theta), 0.0, np.cos(test_theta)],
        ]
    ).reshape(3, 3, 1)

    assert test_rot_mat.shape == (3, 3, 1)
    assert_allclose(test_rot_mat, correct_rot_mat, atol=Tolerance.atol())


@pytest.mark.parametrize("zcomp", [0.5, 1.0])
@pytest.mark.parametrize("dt", [0.3, 1.0])
def test_jax_get_rotation_matrix_correct_rotation_about_z(zcomp, dt):
    vector_collection = np.array([0.0, 0.0, zcomp]).reshape(-1, 1)
    with jax.default_device(CPU_DEVICE):
        test_rot_mat = np.asarray(
            _jax_get_rotation_matrix(
                dt, jax.numpy.asarray(vector_collection, dtype=np.float64)
            )
        )
    test_theta = zcomp * dt
    correct_rot_mat = np.array(
        [
            [np.cos(test_theta), np.sin(test_theta), 0.0],
            [-np.sin(test_theta), np.cos(test_theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    ).reshape(3, 3, 1)

    assert test_rot_mat.shape == (3, 3, 1)
    assert_allclose(test_rot_mat, correct_rot_mat, atol=Tolerance.atol())


def test_jax_get_rotation_matrix_correctness_in_three_dimensions():
    vector_collection = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    vector_collection = vector_collection.reshape(-1, 1)
    theta = np.deg2rad(120.0)
    with jax.default_device(CPU_DEVICE):
        test_rot_mat = np.asarray(
            _jax_get_rotation_matrix(
                theta, jax.numpy.asarray(vector_collection, dtype=np.float64)
            )
        )
    correct_rot_mat = np.roll(np.eye(3), 1, axis=1).reshape(3, 3, 1)

    assert_allclose(test_rot_mat, correct_rot_mat, atol=Tolerance.atol())


def test_jax_get_rotation_matrix_correctness_against_canned_example():
    vector_collection = np.array([1.0, 3.2, 7.0])
    vector_collection /= np.linalg.norm(vector_collection)
    vector_collection = vector_collection.reshape(-1, 1)
    theta = np.deg2rad(76.0)
    with jax.default_device(CPU_DEVICE):
        test_rot_mat = np.asarray(
            _jax_get_rotation_matrix(
                theta, jax.numpy.asarray(vector_collection, dtype=np.float64)
            )
        )
    correct_rot_mat = np.array(
        [
            [0.254506, -0.834834, 0.488138],
            [0.915374, 0.370785, 0.156873],
            [-0.311957, 0.406903, 0.858552],
        ]
    ).T.reshape(3, 3, 1)

    assert_allclose(test_rot_mat, correct_rot_mat, atol=1e-6)


@pytest.mark.parametrize("blocksize", [32, 128, 512])
def test_jax_get_rotation_matrix_correctness_across_blocksizes(blocksize, rng):
    dim = 3
    dt = rng.random()
    vector_collection = rng.standard_normal(dim).reshape(-1, 1)
    with jax.default_device(CPU_DEVICE):
        correct_rot_mat_collection = np.asarray(
            _jax_get_rotation_matrix(
                dt, jax.numpy.asarray(vector_collection, dtype=np.float64)
            )
        )
    correct_rot_mat_collection = np.tile(correct_rot_mat_collection, blocksize)

    test_vector_collection = np.tile(vector_collection, blocksize)
    with jax.default_device(CPU_DEVICE):
        test_rot_mat_collection = np.asarray(
            _jax_get_rotation_matrix(
                dt, jax.numpy.asarray(test_vector_collection, dtype=np.float64)
            )
        )

    assert test_rot_mat_collection.shape == (3, 3, blocksize)
    assert_allclose(test_rot_mat_collection, correct_rot_mat_collection)


def test_jax_get_rotation_matrix_gives_orthonormal_matrices(rng):
    dim = 3
    blocksize = 16
    dt = rng.random()
    with jax.default_device(CPU_DEVICE):
        rot_mat = np.asarray(
            _jax_get_rotation_matrix(
                dt,
                jax.numpy.asarray(
                    rng.standard_normal((dim, blocksize)), dtype=np.float64
                ),
            )
        )

    r_rt = np.einsum("ijk,ljk->ilk", rot_mat, rot_mat)
    rt_r = np.einsum("jik,jlk->ilk", rot_mat, rot_mat)
    test_mat = np.array([np.eye(dim) for _ in range(blocksize)]).T

    assert_allclose(r_rt, test_mat, atol=Tolerance.atol())
    assert_allclose(rt_r, test_mat, atol=Tolerance.atol())


def test_jax_get_rotation_matrix_gives_unit_determinant(rng):
    dim = 3
    blocksize = 16
    dt = rng.random()
    with jax.default_device(CPU_DEVICE):
        test_rot_mat_collection = np.asarray(
            _jax_get_rotation_matrix(
                dt,
                jax.numpy.asarray(
                    rng.standard_normal((dim, blocksize)), dtype=np.float64
                ),
            )
        )
    test_det_collection = np.linalg.det(test_rot_mat_collection.T)
    correct_det_collection = 1.0 + 0.0 * test_det_collection

    assert_allclose(correct_det_collection, test_det_collection)


def test_jax_rotation_matrix_matches_numba_across_random_inputs(rng):
    blocksize = 64
    dt = rng.normal()
    vector_collection = rng.standard_normal((3, blocksize))
    correct_rot_mat = _get_rotation_matrix(dt, vector_collection)
    with jax.default_device(CPU_DEVICE):
        test_rot_mat = np.asarray(
            _jax_get_rotation_matrix(
                dt, jax.numpy.asarray(vector_collection, dtype=np.float64)
            )
        )

    assert_allclose(test_rot_mat, correct_rot_mat, atol=Tolerance.atol())


def test_jax_inv_rotate_correctness_simple_in_three_dimensions():
    rotate_from_matrix = np.eye(3).reshape(3, 3, 1)
    rotate_to_matrix = np.eye(3) @ np.roll(np.eye(3), -1, axis=1).T
    input_director_collection = np.dstack(
        (rotate_from_matrix, rotate_to_matrix.reshape(3, 3, -1))
    )

    correct_axis_collection = np.ones((3, 1)) / np.sqrt(3.0)
    with jax.default_device(CPU_DEVICE):
        test_axis_collection = np.asarray(
            _jax_inv_rotate(
                jax.numpy.asarray(input_director_collection, dtype=np.float64)
            )
        ).copy()

    correct_angle = np.deg2rad(120)
    test_angle = np.linalg.norm(test_axis_collection, axis=0)
    test_axis_collection /= test_angle

    assert_allclose(
        test_axis_collection, correct_axis_collection, atol=Tolerance.atol()
    )
    assert_allclose(test_angle, correct_angle, atol=Tolerance.atol())


def test_jax_inv_rotate_matches_numba_across_random_inputs(rng):
    blocksize = 32
    axis_collection = rng.standard_normal((3, blocksize)) * 0.2
    rotation_collection = _get_rotation_matrix(1.0, axis_collection)
    director_collection = np.zeros((3, 3, blocksize + 1), dtype=np.float64)
    director_collection[:, :, 0] = np.eye(3)
    for k in range(blocksize):
        director_collection[:, :, k + 1] = (
            rotation_collection[:, :, k] @ director_collection[:, :, k]
        )

    correct_axis_collection = _inv_rotate(director_collection)
    with jax.default_device(CPU_DEVICE):
        test_axis_collection = np.asarray(
            _jax_inv_rotate(jax.numpy.asarray(director_collection, dtype=np.float64))
        )

    assert_allclose(
        test_axis_collection, correct_axis_collection, atol=Tolerance.atol()
    )

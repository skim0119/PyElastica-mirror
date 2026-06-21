#!/usr/bin/env python3
__doc__ = """Test scripts for linear algebra helpers in Elastica JAX implementation"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

CPU_DEVICE = jax.devices("cpu")[0]

from elastica._jax_linalg import (
    _jax_batch_cross,
    _jax_batch_dot,
    _jax_batch_matmul,
    _jax_batch_matvec,
)


@pytest.mark.parametrize("blocksize", [8, 32])
def test_jax_batch_matvec(blocksize, rng):
    input_matrix_collection = rng.standard_normal((3, 3, blocksize))
    input_vector_collection = rng.standard_normal((3, blocksize))

    with jax.default_device(CPU_DEVICE):
        test_vector_collection = np.asarray(
            _jax_batch_matvec(
                jax.numpy.asarray(input_matrix_collection, dtype=np.float64),
                jax.numpy.asarray(input_vector_collection, dtype=np.float64),
            )
        )

    correct_vector_collection = [
        np.dot(input_matrix_collection[..., i], input_vector_collection[..., i])
        for i in range(blocksize)
    ]
    correct_vector_collection = np.array(correct_vector_collection).T

    assert_allclose(test_vector_collection, correct_vector_collection)


@pytest.mark.parametrize("blocksize", [8, 32])
def test_jax_batch_matmul(blocksize, rng):
    input_first_matrix_collection = rng.standard_normal((3, 3, blocksize))
    input_second_matrix_collection = rng.standard_normal((3, 3, blocksize))

    with jax.default_device(CPU_DEVICE):
        test_matrix_collection = np.asarray(
            _jax_batch_matmul(
                jax.numpy.asarray(input_first_matrix_collection, dtype=np.float64),
                jax.numpy.asarray(input_second_matrix_collection, dtype=np.float64),
            )
        )

    correct_matrix_collection = np.empty((3, 3, blocksize))
    for i in range(blocksize):
        correct_matrix_collection[..., i] = np.dot(
            input_first_matrix_collection[..., i],
            input_second_matrix_collection[..., i],
        )

    assert_allclose(test_matrix_collection, correct_matrix_collection)


@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("blocksize", [8, 32])
def test_jax_batch_cross(dim, blocksize, rng):
    input_first_vector_collection = rng.standard_normal((dim, blocksize))
    input_second_vector_collection = rng.standard_normal((dim, blocksize))

    with jax.default_device(CPU_DEVICE):
        test_vector_collection = np.asarray(
            _jax_batch_cross(
                jax.numpy.asarray(input_first_vector_collection, dtype=np.float64),
                jax.numpy.asarray(input_second_vector_collection, dtype=np.float64),
            )
        )
    correct_vector_collection = np.cross(
        input_first_vector_collection, input_second_vector_collection, axisa=0, axisb=0
    ).T

    assert_allclose(test_vector_collection, correct_vector_collection)


@pytest.mark.parametrize("blocksize", [8, 32])
def test_jax_batch_dot(blocksize, rng):
    input_first_vector_collection = rng.standard_normal((3, blocksize))
    input_second_vector_collection = rng.standard_normal((3, blocksize))

    with jax.default_device(CPU_DEVICE):
        test_vector_collection = np.asarray(
            _jax_batch_dot(
                jax.numpy.asarray(input_first_vector_collection, dtype=np.float64),
                jax.numpy.asarray(input_second_vector_collection, dtype=np.float64),
            )
        )
    correct_vector_collection = np.einsum(
        "ij,ij->j", input_first_vector_collection, input_second_vector_collection
    )

    assert_allclose(test_vector_collection, correct_vector_collection)

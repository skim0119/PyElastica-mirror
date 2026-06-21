#!/usr/bin/env python3
__doc__ = """Test scripts for calculus kernels in Elastica JAX implementation"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

CPU_DEVICE = jax.devices("cpu")[0]

from elastica._jax_calculus import (
    _jax_trapezoidal,
    _jax_two_point_difference,
    _jax_trapezoidal_for_block_structure,
    _jax_two_point_difference_for_block_structure,
)
from elastica.utils import Tolerance


class Trapezoidal:
    kernel = _jax_trapezoidal

    @staticmethod
    def oned_setup():
        blocksize = 32
        rng = np.random.default_rng(42)
        input_vector = rng.standard_normal(blocksize)

        first_element = 0.5 * input_vector[0]
        last_element = 0.5 * input_vector[-1]
        correct_vector = np.hstack(
            (first_element, 0.5 * (input_vector[1:] + input_vector[:-1]), last_element)
        )
        return input_vector, correct_vector


class Difference:
    kernel = _jax_two_point_difference

    @staticmethod
    def oned_setup():
        blocksize = 32
        rng = np.random.default_rng(42)
        input_vector = rng.standard_normal(blocksize)

        first_element = input_vector[0]
        last_element = -input_vector[-1]
        correct_vector = np.hstack(
            (first_element, (input_vector[1:] - input_vector[:-1]), last_element)
        )
        return input_vector, correct_vector


@pytest.mark.parametrize("Setup", [Trapezoidal, Difference])
@pytest.mark.parametrize("ndim", [3])
def test_jax_two_point_difference_integrity(Setup, ndim):
    input_vector_oned, correct_vector_oned = Setup.oned_setup()
    input_vector = np.repeat(input_vector_oned[np.newaxis, :], ndim, axis=0)
    with jax.default_device(CPU_DEVICE):
        test_vector = np.asarray(
            Setup.kernel(jax.numpy.asarray(input_vector, dtype=np.float64))
        )
    correct_vector = np.repeat(correct_vector_oned[np.newaxis, :], ndim, axis=0)

    assert test_vector.shape == input_vector.shape[:-1] + (input_vector.shape[-1] + 1,)
    assert_allclose(test_vector, correct_vector)


def test_jax_trapezoidal_correctness():
    blocksize = 64
    a = 0.0
    b = np.pi
    dh = (b - a) / (blocksize - 1)

    input_vector = np.repeat(
        np.sin(np.linspace(a, b, blocksize))[np.newaxis, :], 3, axis=0
    )
    with jax.default_device(CPU_DEVICE):
        test_vector = (
            np.asarray(
                _jax_trapezoidal(
                    jax.numpy.asarray(input_vector[..., 1:-1], dtype=np.float64)
                )
            )
            * dh
        )

    interior_a = a + 0.5 * dh
    interior_b = b - 0.5 * dh
    correct_vector = (
        np.repeat(
            np.sin(np.linspace(interior_a, interior_b, blocksize - 1))[np.newaxis, :],
            3,
            axis=0,
        )
        * dh
    )

    assert_allclose(np.sum(test_vector[0]), 2.0, atol=1e-3)
    assert_allclose(np.sum(test_vector[1]), 2.0, atol=1e-3)
    assert_allclose(np.sum(test_vector[2]), 2.0, atol=1e-3)
    assert_allclose(test_vector, correct_vector, atol=1e-4)


def test_jax_trapezoidal_for_block_structure_correctness(rng):
    blocksize = 30
    ghost_idx = np.array([14, 15])
    input_vector = rng.standard_normal((3, blocksize))

    with jax.default_device(CPU_DEVICE):
        correct_vector = np.hstack(
            (
                np.asarray(
                    _jax_trapezoidal(
                        jax.device_put(
                            np.asarray(
                                input_vector[..., : ghost_idx[0]], dtype=np.float64
                            ),
                            device=CPU_DEVICE,
                        )
                    )
                ),
                np.zeros((3, 1)),
                np.asarray(
                    _jax_trapezoidal(
                        jax.device_put(
                            np.asarray(
                                input_vector[..., ghost_idx[1] + 1 :], dtype=np.float64
                            ),
                            device=CPU_DEVICE,
                        )
                    )
                ),
            )
        )
    with jax.default_device(CPU_DEVICE):
        test_vector = np.asarray(
            _jax_trapezoidal_for_block_structure(
                jax.numpy.asarray(input_vector, dtype=np.float64),
                jax.numpy.asarray(ghost_idx),
            )
        )

    assert_allclose(test_vector, correct_vector, atol=Tolerance.atol())


def test_jax_two_point_difference_correctness():
    blocksize = 128
    a = 0.0
    b = np.pi
    dh = (b - a) / (blocksize - 1)
    interior_a = a + 0.5 * dh
    interior_b = b - 0.5 * dh

    input_vector = np.repeat(
        np.sin(np.linspace(a, b, blocksize))[np.newaxis, :], 3, axis=0
    )
    with jax.default_device(CPU_DEVICE):
        test_vector = (
            np.asarray(
                _jax_two_point_difference(
                    jax.numpy.asarray(input_vector[..., 1:-1], dtype=np.float64)
                )
            )
            / dh
        )
    correct_vector = np.repeat(
        np.cos(np.linspace(interior_a, interior_b, blocksize - 1))[np.newaxis, :],
        3,
        axis=0,
    )

    assert_allclose(test_vector, correct_vector, atol=1e-4)


def test_jax_two_point_difference_for_block_structure_correctness(rng):
    blocksize = 30
    ghost_idx = np.array([14, 15])
    input_vector = rng.standard_normal((3, blocksize))

    with jax.default_device(CPU_DEVICE):
        correct_vector = np.hstack(
            (
                np.asarray(
                    _jax_two_point_difference(
                        jax.device_put(
                            np.asarray(
                                input_vector[..., : ghost_idx[0]], dtype=np.float64
                            ),
                            device=CPU_DEVICE,
                        )
                    )
                ),
                np.zeros((3, 1)),
                np.asarray(
                    _jax_two_point_difference(
                        jax.device_put(
                            np.asarray(
                                input_vector[..., ghost_idx[1] + 1 :], dtype=np.float64
                            ),
                            device=CPU_DEVICE,
                        )
                    )
                ),
            )
        )
    with jax.default_device(CPU_DEVICE):
        test_vector = np.asarray(
            _jax_two_point_difference_for_block_structure(
                jax.numpy.asarray(input_vector, dtype=np.float64),
                jax.numpy.asarray(ghost_idx),
            )
        )

    assert_allclose(test_vector, correct_vector, atol=Tolerance.atol())

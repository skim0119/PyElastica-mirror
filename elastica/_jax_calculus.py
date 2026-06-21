"""JAX quadrature and difference kernels."""

from __future__ import annotations

import jax.numpy as jnp


def _jax_reset_vector_ghost(input_array, ghost_idx, reset_value: float = 0.0):
    if ghost_idx.size == 0:
        return input_array
    return input_array.at[:, ghost_idx].set(
        jnp.asarray(reset_value, dtype=input_array.dtype)
    )


def _jax_trapezoidal(array_collection):
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


def _jax_trapezoidal_for_block_structure(array_collection, ghost_idx):
    array_collection = _jax_reset_vector_ghost(array_collection, ghost_idx)
    return _jax_trapezoidal(array_collection)


def _jax_two_point_difference(array_collection):
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


def _jax_two_point_difference_for_block_structure(array_collection, ghost_idx):
    array_collection = _jax_reset_vector_ghost(array_collection, ghost_idx)
    return _jax_two_point_difference(array_collection)


def _jax_difference(vector):
    return vector[:, 1:] - vector[:, :-1]


def _jax_average(vector):
    return 0.5 * (vector[1:] + vector[:-1])


position_difference_kernel = _jax_difference
position_average = _jax_average
quadrature_kernel = _jax_trapezoidal
difference_kernel = _jax_two_point_difference
quadrature_kernel_for_block_structure = _jax_trapezoidal_for_block_structure
difference_kernel_for_block_structure = _jax_two_point_difference_for_block_structure

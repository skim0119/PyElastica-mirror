"""JAX linear algebra kernels."""

from __future__ import annotations

import jax.numpy as jnp


def _jax_batch_matvec(matrix_collection, vector_collection):
    out = jnp.empty_like(vector_collection)
    out = out.at[0, :].set(
        matrix_collection[0, 0, :] * vector_collection[0, :]
        + matrix_collection[0, 1, :] * vector_collection[1, :]
        + matrix_collection[0, 2, :] * vector_collection[2, :]
    )
    out = out.at[1, :].set(
        matrix_collection[1, 0, :] * vector_collection[0, :]
        + matrix_collection[1, 1, :] * vector_collection[1, :]
        + matrix_collection[1, 2, :] * vector_collection[2, :]
    )
    out = out.at[2, :].set(
        matrix_collection[2, 0, :] * vector_collection[0, :]
        + matrix_collection[2, 1, :] * vector_collection[1, :]
        + matrix_collection[2, 2, :] * vector_collection[2, :]
    )
    return out


def _jax_batch_matmul(first_matrix_collection, second_matrix_collection):
    out = jnp.empty_like(second_matrix_collection)
    out = out.at[0, 0, :].set(
        first_matrix_collection[0, 0, :] * second_matrix_collection[0, 0, :]
        + first_matrix_collection[0, 1, :] * second_matrix_collection[1, 0, :]
        + first_matrix_collection[0, 2, :] * second_matrix_collection[2, 0, :]
    )
    out = out.at[0, 1, :].set(
        first_matrix_collection[0, 0, :] * second_matrix_collection[0, 1, :]
        + first_matrix_collection[0, 1, :] * second_matrix_collection[1, 1, :]
        + first_matrix_collection[0, 2, :] * second_matrix_collection[2, 1, :]
    )
    out = out.at[0, 2, :].set(
        first_matrix_collection[0, 0, :] * second_matrix_collection[0, 2, :]
        + first_matrix_collection[0, 1, :] * second_matrix_collection[1, 2, :]
        + first_matrix_collection[0, 2, :] * second_matrix_collection[2, 2, :]
    )
    out = out.at[1, 0, :].set(
        first_matrix_collection[1, 0, :] * second_matrix_collection[0, 0, :]
        + first_matrix_collection[1, 1, :] * second_matrix_collection[1, 0, :]
        + first_matrix_collection[1, 2, :] * second_matrix_collection[2, 0, :]
    )
    out = out.at[1, 1, :].set(
        first_matrix_collection[1, 0, :] * second_matrix_collection[0, 1, :]
        + first_matrix_collection[1, 1, :] * second_matrix_collection[1, 1, :]
        + first_matrix_collection[1, 2, :] * second_matrix_collection[2, 1, :]
    )
    out = out.at[1, 2, :].set(
        first_matrix_collection[1, 0, :] * second_matrix_collection[0, 2, :]
        + first_matrix_collection[1, 1, :] * second_matrix_collection[1, 2, :]
        + first_matrix_collection[1, 2, :] * second_matrix_collection[2, 2, :]
    )
    out = out.at[2, 0, :].set(
        first_matrix_collection[2, 0, :] * second_matrix_collection[0, 0, :]
        + first_matrix_collection[2, 1, :] * second_matrix_collection[1, 0, :]
        + first_matrix_collection[2, 2, :] * second_matrix_collection[2, 0, :]
    )
    out = out.at[2, 1, :].set(
        first_matrix_collection[2, 0, :] * second_matrix_collection[0, 1, :]
        + first_matrix_collection[2, 1, :] * second_matrix_collection[1, 1, :]
        + first_matrix_collection[2, 2, :] * second_matrix_collection[2, 1, :]
    )
    out = out.at[2, 2, :].set(
        first_matrix_collection[2, 0, :] * second_matrix_collection[0, 2, :]
        + first_matrix_collection[2, 1, :] * second_matrix_collection[1, 2, :]
        + first_matrix_collection[2, 2, :] * second_matrix_collection[2, 2, :]
    )
    return out


def _jax_batch_cross(first_vector_collection, second_vector_collection):
    out = jnp.empty_like(first_vector_collection)
    out = out.at[0, :].set(
        first_vector_collection[1, :] * second_vector_collection[2, :]
        - first_vector_collection[2, :] * second_vector_collection[1, :]
    )
    out = out.at[1, :].set(
        first_vector_collection[2, :] * second_vector_collection[0, :]
        - first_vector_collection[0, :] * second_vector_collection[2, :]
    )
    out = out.at[2, :].set(
        first_vector_collection[0, :] * second_vector_collection[1, :]
        - first_vector_collection[1, :] * second_vector_collection[0, :]
    )
    return out


def _jax_batch_dot(first_vector_collection, second_vector_collection):
    return jnp.sum(first_vector_collection * second_vector_collection, axis=0)

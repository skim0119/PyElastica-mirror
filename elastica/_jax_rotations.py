"""JAX rotation kernels."""

from __future__ import annotations

import jax.numpy as jnp


def _jax_get_rotation_matrix(scale, axis_collection):
    v0, v1, v2 = axis_collection
    theta = jnp.sqrt(v0 * v0 + v1 * v1 + v2 * v2)
    theta_eps = theta + jnp.asarray(1.0e-14, dtype=axis_collection.dtype)
    v0 = v0 / theta_eps
    v1 = v1 / theta_eps
    v2 = v2 / theta_eps

    theta = theta * scale
    sin_theta = jnp.sin(theta)
    half_theta = 0.5 * theta
    one_minus_cos_theta = 2.0 * jnp.sin(half_theta) * jnp.sin(half_theta)

    entries = (
        1.0 - one_minus_cos_theta * (v1 * v1 + v2 * v2),
        sin_theta * v2 + one_minus_cos_theta * v0 * v1,
        -sin_theta * v1 + one_minus_cos_theta * v0 * v2,
        -sin_theta * v2 + one_minus_cos_theta * v0 * v1,
        1.0 - one_minus_cos_theta * (v0 * v0 + v2 * v2),
        sin_theta * v0 + one_minus_cos_theta * v1 * v2,
        sin_theta * v1 + one_minus_cos_theta * v0 * v2,
        -sin_theta * v0 + one_minus_cos_theta * v1 * v2,
        1.0 - one_minus_cos_theta * (v0 * v0 + v1 * v1),
    )
    return jnp.stack(entries, axis=0).reshape(3, 3, axis_collection.shape[1])


def _jax_inv_rotate(director_collection):
    """Extract relative rotation vectors between consecutive directors.

    Notes
    -----
    This routine mirrors :func:`elastica._rotations._inv_rotate`, but exact
    bitwise agreement with the Numba path is not expected in general. The map
    includes trace accumulation, ``arccos`` near 1, and ``theta / sin(theta)``,
    so backend-level floating-point differences can produce tiny discrepancies
    even when the input directors agree to machine precision.
    """
    current = director_collection[:, :, :-1]
    nxt = director_collection[:, :, 1:]

    v0 = (
        nxt[2, 0, :] * current[1, 0, :]
        + nxt[2, 1, :] * current[1, 1, :]
        + nxt[2, 2, :] * current[1, 2, :]
        - nxt[1, 0, :] * current[2, 0, :]
        - nxt[1, 1, :] * current[2, 1, :]
        - nxt[1, 2, :] * current[2, 2, :]
    )
    v1 = (
        nxt[0, 0, :] * current[2, 0, :]
        + nxt[0, 1, :] * current[2, 1, :]
        + nxt[0, 2, :] * current[2, 2, :]
        - nxt[2, 0, :] * current[0, 0, :]
        - nxt[2, 1, :] * current[0, 1, :]
        - nxt[2, 2, :] * current[0, 2, :]
    )
    v2 = (
        nxt[1, 0, :] * current[0, 0, :]
        + nxt[1, 1, :] * current[0, 1, :]
        + nxt[1, 2, :] * current[0, 2, :]
        - nxt[0, 0, :] * current[1, 0, :]
        - nxt[0, 1, :] * current[1, 1, :]
        - nxt[0, 2, :] * current[1, 2, :]
    )

    trace = (
        nxt[0, 0, :] * current[0, 0, :]
        + nxt[0, 1, :] * current[0, 1, :]
        + nxt[0, 2, :] * current[0, 2, :]
        + nxt[1, 0, :] * current[1, 0, :]
        + nxt[1, 1, :] * current[1, 1, :]
        + nxt[1, 2, :] * current[1, 2, :]
        + nxt[2, 0, :] * current[2, 0, :]
        + nxt[2, 1, :] * current[2, 1, :]
        + nxt[2, 2, :] * current[2, 2, :]
    )
    trace = jnp.clip(trace, -1.0, 3.0)
    theta = jnp.arccos(0.5 * trace - 0.5) + jnp.asarray(
        1.0e-14, dtype=director_collection.dtype
    )
    magnitude = -0.5 * theta / jnp.sin(theta)
    return jnp.stack((v0 * magnitude, v1 * magnitude, v2 * magnitude), axis=0)

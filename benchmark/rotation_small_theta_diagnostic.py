"""Small-theta diagnostic for Numba vs JAX rotation kernels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "This diagnostic requires JAX. Install the GPU extra first."
    ) from exc

jax.config.update("jax_enable_x64", True)

from elastica._jax_linalg import _jax_batch_matmul
from elastica._jax_rotations import _jax_get_rotation_matrix, _jax_inv_rotate
from elastica._linalg import _batch_matmul
from elastica._rotations import _get_rotation_matrix, _inv_rotate


def _resolve_dtype(dtype_name: str) -> np.dtype:
    assert dtype_name in ("float32", "float64"), "dtype must be float32 or float64."
    return np.dtype(np.float32 if dtype_name == "float32" else np.float64)


def _select_device(platform: str) -> jax.Device:
    assert platform in ("auto", "cpu", "mps", "cuda"), (
        "platform must be one of auto, cpu, mps, or cuda."
    )
    if platform == "auto":
        return jax.devices()[0]
    devices = jax.devices(platform)
    assert devices, f"No JAX device found for platform {platform!r}."
    return devices[0]


def _make_director(dtype: np.dtype) -> np.ndarray:
    axis = np.asarray([0.3, -0.4, 0.5], dtype=dtype)
    axis /= np.linalg.norm(axis)
    angle = dtype.type(0.7)
    rotation = _get_rotation_matrix(angle, axis.reshape(3, 1))[:, :, 0]
    return rotation.reshape(3, 3, 1)


def _compute_errors(
    theta_values: np.ndarray,
    *,
    dtype: np.dtype,
    device: jax.Device,
) -> dict[str, np.ndarray]:
    axis = np.asarray([1.0, 2.0, 3.0], dtype=dtype)
    axis /= np.linalg.norm(axis)
    director = _make_director(dtype)

    rotation_error = np.zeros_like(theta_values)
    director_error = np.zeros_like(theta_values)
    kappa_error = np.zeros_like(theta_values)

    with jax.default_device(device):
        for idx, theta in enumerate(theta_values):
            scaled_axis = (axis * theta).reshape(3, 1)

            rotation_numba = _get_rotation_matrix(dtype.type(1.0), scaled_axis)
            rotation_jax = np.asarray(
                _jax_get_rotation_matrix(
                    dtype.type(1.0),
                    jnp.asarray(scaled_axis, dtype=dtype),
                )
            )
            rotation_error[idx] = np.max(np.abs(rotation_numba - rotation_jax))

            director_numba = _batch_matmul(rotation_numba, director)
            director_jax = np.asarray(
                _jax_batch_matmul(
                    jnp.asarray(rotation_jax, dtype=dtype),
                    jnp.asarray(director, dtype=dtype),
                )
            )
            director_error[idx] = np.max(np.abs(director_numba - director_jax))

            director_pair_numba = np.concatenate((director, director_numba), axis=2)
            director_pair_jax = np.concatenate((director, director_jax), axis=2)
            kappa_numba = _inv_rotate(director_pair_numba)
            kappa_jax = np.asarray(
                _jax_inv_rotate(jnp.asarray(director_pair_jax, dtype=dtype))
            )
            kappa_error[idx] = np.max(np.abs(kappa_numba - kappa_jax))

    return {
        "rotation_error": rotation_error,
        "director_error": director_error,
        "kappa_error": kappa_error,
    }


def _plot(
    theta_values: np.ndarray,
    errors: dict[str, np.ndarray],
    output: Path,
) -> None:
    # one_minus_cos = 1.0 - np.cos(theta_values)
    one_minus_cos = 2.0 * np.sin(theta_values / 2.0) ** 2

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)

    axes[0].loglog(theta_values, one_minus_cos, label=r"$1-\cos(\theta)$")
    axes[0].loglog(theta_values, theta_values**2 / 2.0, "--", label=r"$\theta^2/2$")
    axes[0].set_xlabel(r"$\theta$")
    axes[0].set_ylabel("magnitude")
    axes[0].set_title("Small-angle scale")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].loglog(theta_values, errors["rotation_error"], label="rotation matrix")
    axes[1].loglog(theta_values, errors["director_error"], label="director update")
    axes[1].loglog(theta_values, errors["kappa_error"], label="kappa")
    axes[1].set_xlabel(r"$\theta$")
    axes[1].set_ylabel("max abs discrepancy")
    axes[1].set_title("Numba vs JAX discrepancy")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)

    print(f"wrote plot: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose small-theta discrepancies between Numba and JAX rotations."
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Number of theta samples across the log sweep.",
    )
    parser.add_argument(
        "--platform",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="JAX platform to run the diagnostic on.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
        help="Floating-point dtype for the diagnostic.",
    )
    args = parser.parse_args()
    output = Path("rotation_small_theta_diagnostic.png")

    dtype = _resolve_dtype(args.dtype)
    device = _select_device(args.platform)
    print(f"device: {device}")
    print(f"dtype: {dtype}")
    if dtype == np.dtype(np.float64) and device.platform == "mps":
        raise SystemExit(
            "MPS/MLX does not support float64. Use `--platform cpu --dtype float64` "
            "or `--platform mps --dtype float32`."
        )

    theta_values = np.logspace(-16, -2, args.n_samples, dtype=dtype)
    errors = _compute_errors(theta_values, dtype=dtype, device=device)
    _plot(theta_values, errors, output)

    for name, values in errors.items():
        max_idx = int(np.argmax(values))
        print(f"{name}: max={values[max_idx]!r} at theta={theta_values[max_idx]!r}")


if __name__ == "__main__":
    main()

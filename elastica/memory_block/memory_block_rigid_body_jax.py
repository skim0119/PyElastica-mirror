"""JAX-backed memory block for rigid bodies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np

from elastica._jax_linalg import _jax_batch_cross, _jax_batch_matmul, _jax_batch_matvec
from elastica._jax_rotations import _jax_get_rotation_matrix
from elastica.rigidbody import RigidBodyBase
from elastica.rigidbody.data_structures import _RigidRodSymplecticStepperMixin
from elastica.typing import RigidBodyType, SystemIdxType

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without JAX
    raise ModuleNotFoundError(
        "MemoryBlockRigidBodyJax requires JAX. Install the optional GPU dependency "
        'first, for example with `uv add --optional gpu "jax[cuda13]"`.'
    ) from exc


_SCALAR_ATTRS: tuple[str, ...] = ("density", "volume", "mass")
_VECTOR_ATTRS: tuple[str, ...] = (
    "position_collection",
    "external_forces",
    "external_torques",
    "velocity_collection",
    "omega_collection",
    "acceleration_collection",
    "alpha_collection",
)
_MATRIX_ATTRS: tuple[str, ...] = (
    "director_collection",
    "mass_second_moment_of_inertia",
    "inv_mass_second_moment_of_inertia",
)
_SYNCABLE_ATTRS: tuple[str, ...] = _SCALAR_ATTRS + _VECTOR_ATTRS + _MATRIX_ATTRS


@dataclass(frozen=True)
class JAXRigidBodyViewMetadata:
    block_state_idx: int
    body_slice: slice

    def slice_for_attr(self, attr: str) -> slice:
        if attr in _SYNCABLE_ATTRS:
            return self.body_slice
        raise AttributeError(f"Unsupported rigid-body-view attribute {attr!r}")


class JAXRigidBodyView:
    """Rigid-body-local facade over explicit block state for JAX operator kernels."""

    def __init__(
        self,
        state: dict[str, object],
        metadata: JAXRigidBodyViewMetadata,
        *,
        updates: dict[str, object] | None = None,
    ) -> None:
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_metadata", metadata)
        object.__setattr__(self, "_updates", {} if updates is None else dict(updates))

    def __getattr__(self, attr: str) -> object:
        if attr.startswith("_"):
            raise AttributeError(attr)
        source = self._updates.get(attr, self._state[attr])
        attr_slice = self._metadata.slice_for_attr(attr)
        return source[..., attr_slice]

    def __setattr__(self, attr: str, value: object) -> None:
        if attr.startswith("_"):
            object.__setattr__(self, attr, value)
            return
        base = self._updates.get(attr, self._state[attr])
        attr_slice = self._metadata.slice_for_attr(attr)
        self._updates[attr] = base.at[..., attr_slice].set(value)

    def commit(self) -> dict[str, object]:
        updated = dict(self._state)
        updated.update(self._updates)
        return updated


@jax.jit
def _jax_update_kinematics(
    position_collection: jax.Array,
    director_collection: jax.Array,
    velocity_collection: jax.Array,
    omega_collection: jax.Array,
    prefac: float,
) -> tuple[jax.Array, jax.Array]:
    position_collection = position_collection + prefac * velocity_collection
    rotation_matrix = _jax_get_rotation_matrix(prefac, omega_collection)
    director_collection = _jax_batch_matmul(rotation_matrix, director_collection)
    return position_collection, director_collection


@jax.jit
def _jax_update_dynamics(
    velocity_collection: jax.Array,
    omega_collection: jax.Array,
    acceleration_collection: jax.Array,
    alpha_collection: jax.Array,
    prefac: float,
) -> tuple[jax.Array, jax.Array]:
    velocity_collection = velocity_collection + prefac * acceleration_collection
    omega_collection = omega_collection + prefac * alpha_collection
    return velocity_collection, omega_collection


@jax.jit
def _jax_update_accelerations(
    external_forces: jax.Array,
    mass: jax.Array,
    mass_second_moment_of_inertia: jax.Array,
    inv_mass_second_moment_of_inertia: jax.Array,
    omega_collection: jax.Array,
    external_torques: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    acceleration_collection = external_forces / mass[jnp.newaxis, :]
    j_omega = _jax_batch_matvec(mass_second_moment_of_inertia, omega_collection)
    lagrangian_transport = _jax_batch_cross(j_omega, omega_collection)
    alpha_collection = _jax_batch_matvec(
        inv_mass_second_moment_of_inertia,
        lagrangian_transport + external_torques,
    )
    return acceleration_collection, alpha_collection


@jax.jit
def _jax_zero_external_loads(
    external_forces: jax.Array, external_torques: jax.Array
) -> tuple[jax.Array, jax.Array]:
    return jnp.zeros_like(external_forces), jnp.zeros_like(external_torques)


class MemoryBlockRigidBodyJax(RigidBodyBase, _RigidRodSymplecticStepperMixin):
    """
    JAX-backed memory block with explicit host/device synchronization.

    Unlike the NumPy memory block, this implementation does not make the original
    rigid bodies alias the block arrays. The block owns its host-side memory and
    only pushes data back to bodies on explicit ``from_device()`` calls.
    """

    allow_cpu_fallback: bool = False

    def __init__(
        self,
        systems: list[RigidBodyType],
        system_idx_list: list[SystemIdxType],
        *,
        device_dtype: str | np.dtype | None = None,
        device: jax.Device | None = None,
    ) -> None:
        self._systems = tuple(systems)
        self.n_systems = len(systems)
        self.n_elems = self.n_systems
        self.n_nodes = self.n_elems
        self.system_idx_list = np.array(system_idx_list, dtype=np.int32)
        self._device_dtype = self._normalize_device_dtype(device_dtype)
        self._initial_device = device

        self._allocate_block_variables_scalars(systems)
        self._allocate_block_variables_vectors(systems)
        self._allocate_block_variables_matrix(systems)
        self._allocate_block_variables_for_symplectic_stepper(systems)

        _RigidRodSymplecticStepperMixin.__init__(self)

        self._device_state: dict[str, jax.Array] = {}
        self._device_platform = (
            device.platform if device is not None else jax.default_backend()
        )
        self._device_dirty = False
        self._initialize_device_state(device=device)

    @property
    def device_platform(self) -> str:
        return self._device_platform

    @property
    def device_dtype(self) -> np.dtype:
        return self._device_dtype

    @staticmethod
    def _normalize_device_dtype(device_dtype: str | np.dtype | None) -> np.dtype:
        if device_dtype is None:
            return np.dtype(np.float64 if jax.config.x64_enabled else np.float32)
        normalized = np.dtype(device_dtype)
        if normalized == np.dtype(np.float64) and not jax.config.x64_enabled:
            raise ValueError(
                "float64 device_dtype requires JAX x64 support. Enable it with "
                '`jax.config.update("jax_enable_x64", True)` or use float32.'
            )
        if normalized not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError(
                "device_dtype must be one of float32 or float64 for MemoryBlockRigidBodyJax."
            )
        return normalized

    def _allocate_block_variables_scalars(self, systems: list[RigidBodyType]) -> None:
        map_scalar_dofs_in_rigid_bodies = {"density": 0, "volume": 1, "mass": 2}
        self.scalar_dofs_in_rigid_bodies = np.zeros(
            (len(map_scalar_dofs_in_rigid_bodies), self.n_elems)
        )
        self._map_system_properties_to_block_memory(
            mapping_dict=map_scalar_dofs_in_rigid_bodies,
            systems=systems,
            block_memory=self.scalar_dofs_in_rigid_bodies,
            value_type="scalar",
        )

    def _allocate_block_variables_vectors(self, systems: list[RigidBodyType]) -> None:
        map_vector_dofs_in_rigid_bodies = {
            "position_collection": 0,
            "external_forces": 1,
            "external_torques": 2,
        }
        self.vector_dofs_in_rigid_bodies = np.zeros(
            (len(map_vector_dofs_in_rigid_bodies), 3 * self.n_elems)
        )
        self._map_system_properties_to_block_memory(
            mapping_dict=map_vector_dofs_in_rigid_bodies,
            systems=systems,
            block_memory=self.vector_dofs_in_rigid_bodies,
            value_type="vector",
        )

    def _allocate_block_variables_matrix(self, systems: list[RigidBodyType]) -> None:
        map_matrix_dofs_in_rigid_bodies = {
            "director_collection": 0,
            "mass_second_moment_of_inertia": 1,
            "inv_mass_second_moment_of_inertia": 2,
        }
        self.matrix_dofs_in_rigid_bodies = np.zeros(
            (len(map_matrix_dofs_in_rigid_bodies), 9 * self.n_elems)
        )
        self._map_system_properties_to_block_memory(
            mapping_dict=map_matrix_dofs_in_rigid_bodies,
            systems=systems,
            block_memory=self.matrix_dofs_in_rigid_bodies,
            value_type="tensor",
        )

    def _allocate_block_variables_for_symplectic_stepper(
        self, systems: list[RigidBodyType]
    ) -> None:
        map_rate_collection = {
            "velocity_collection": 0,
            "omega_collection": 1,
            "acceleration_collection": 2,
            "alpha_collection": 3,
        }
        self.rate_collection = np.zeros((len(map_rate_collection), 3 * self.n_elems))
        self._map_system_properties_to_block_memory(
            mapping_dict=map_rate_collection,
            systems=systems,
            block_memory=self.rate_collection,
            value_type="vector",
        )

        self.v_w_collection = np.lib.stride_tricks.as_strided(
            self.rate_collection[0:2], (2, 3 * self.n_elems)
        )
        self.dvdt_dwdt_collection = np.lib.stride_tricks.as_strided(
            self.rate_collection[2:], (2, 3 * self.n_elems)
        )

    def _map_system_properties_to_block_memory(
        self,
        mapping_dict: dict[str, int],
        systems: list[RigidBodyType],
        block_memory: np.ndarray,
        value_type: Literal["scalar", "vector", "tensor"],
    ) -> None:
        if value_type == "scalar":
            view_shape: tuple[int, ...] = (self.n_elems,)
        elif value_type == "vector":
            view_shape = (3, self.n_elems)
        else:
            assert (
                value_type == "tensor"
            ), "value_type must be one of 'scalar', 'vector', or 'tensor'."
            view_shape = (3, 3, self.n_elems)

        for attr_name, row_idx in mapping_dict.items():
            self.__dict__[attr_name] = np.lib.stride_tricks.as_strided(
                block_memory[row_idx],
                shape=view_shape,
            )
            for system_idx, system in enumerate(systems):
                self.__dict__[attr_name][..., system_idx : system_idx + 1] = (
                    system.__dict__[attr_name]
                    if value_type == "scalar"
                    else system.__dict__[attr_name].copy()
                )

    def _normalize_attr_names(
        self, attrs: Iterable[str] | None = None
    ) -> tuple[str, ...]:
        if attrs is None:
            return _SYNCABLE_ATTRS
        return tuple(dict.fromkeys(attrs))

    def _initialize_device_state(
        self,
        attrs: Iterable[str] | None = None,
        *,
        device: jax.Device | None = None,
    ) -> None:
        target_device = device if device is not None else self._initial_device
        for attr in self._normalize_attr_names(attrs):
            host_array = np.asarray(getattr(self, attr), dtype=self._device_dtype)
            if target_device is not None:
                device_array = jax.device_put(host_array, device=target_device)
            else:
                device_array = jnp.asarray(host_array, dtype=self._device_dtype)
            self._device_state[attr] = device_array
        if target_device is not None:
            self._device_platform = target_device.platform
        self._refresh_device_views()
        self._device_dirty = False

    def to_device(
        self,
        attrs: Iterable[str] | None = None,
        *,
        device: jax.Device | None = None,
    ) -> None:
        raise RuntimeError(
            "MemoryBlockRigidBodyJax keeps device state authoritative after "
            "initialization. `to_device()` is not supported after construction; "
            "run JAX kernels on the block and use `from_device()` for explicit host "
            "readback."
        )

    def to_gpu(self, attrs: Iterable[str] | None = None) -> None:
        self.to_device(attrs=attrs)

    def from_device(
        self,
        attrs: Iterable[str] | None = None,
        *,
        update_rods: bool = True,
    ) -> None:
        sync_attrs = self._normalize_attr_names(attrs)
        for attr in sync_attrs:
            np.copyto(getattr(self, attr), np.asarray(self._device_state[attr]))
        if update_rods:
            self._push_block_state_to_systems(sync_attrs)
        self._device_dirty = False

    def from_gpu(
        self,
        attrs: Iterable[str] | None = None,
        *,
        update_rods: bool = True,
    ) -> None:
        self.from_device(attrs=attrs, update_rods=update_rods)

    def _push_block_state_to_systems(self, attrs: Sequence[str]) -> None:
        for system_idx, system in enumerate(self._systems):
            for attr in attrs:
                if attr in _SCALAR_ATTRS:
                    system.__dict__[attr] = np.asarray(
                        getattr(self, attr)[system_idx]
                    ).reshape(())
                    continue
                np.copyto(
                    system.__dict__[attr],
                    getattr(self, attr)[..., system_idx : system_idx + 1],
                )

    def _refresh_device_views(self) -> None:
        self.position_collection_device = self._device_state["position_collection"]
        self.director_collection_device = self._device_state["director_collection"]
        self.velocity_collection_device = self._device_state["velocity_collection"]
        self.omega_collection_device = self._device_state["omega_collection"]
        self.acceleration_collection_device = self._device_state[
            "acceleration_collection"
        ]
        self.alpha_collection_device = self._device_state["alpha_collection"]

    def jax_get_state(self) -> dict[str, jax.Array]:
        return dict(self._device_state)

    def jax_set_state(self, state: dict[str, jax.Array]) -> None:
        self._device_state = dict(state)
        self._refresh_device_views()
        self._device_dirty = True

    def jax_kinematic_step(
        self,
        state: dict[str, jax.Array],
        time: np.float64,
        prefac: np.float64,
    ) -> dict[str, jax.Array]:
        position_collection, director_collection = _jax_update_kinematics(
            state["position_collection"],
            state["director_collection"],
            state["velocity_collection"],
            state["omega_collection"],
            jnp.asarray(prefac, dtype=self._device_dtype),
        )
        updated = dict(state)
        updated["position_collection"] = position_collection
        updated["director_collection"] = director_collection
        return updated

    def jax_compute_internal_forces_and_torques(
        self,
        state: dict[str, jax.Array],
        time: np.float64,
    ) -> dict[str, jax.Array]:
        return dict(state)

    def jax_dynamic_step(
        self,
        state: dict[str, jax.Array],
        time: np.float64,
        dt: np.float64,
    ) -> dict[str, jax.Array]:
        acceleration_collection, alpha_collection = _jax_update_accelerations(
            state["external_forces"],
            state["mass"],
            state["mass_second_moment_of_inertia"],
            state["inv_mass_second_moment_of_inertia"],
            state["omega_collection"],
            state["external_torques"],
        )
        velocity_collection, omega_collection = _jax_update_dynamics(
            state["velocity_collection"],
            state["omega_collection"],
            acceleration_collection,
            alpha_collection,
            jnp.asarray(dt, dtype=self._device_dtype),
        )
        updated = dict(state)
        updated["acceleration_collection"] = acceleration_collection
        updated["alpha_collection"] = alpha_collection
        updated["velocity_collection"] = velocity_collection
        updated["omega_collection"] = omega_collection
        return updated

    def jax_zero_external_loads(
        self,
        state: dict[str, jax.Array],
        time: np.float64,
    ) -> dict[str, jax.Array]:
        external_forces, external_torques = _jax_zero_external_loads(
            state["external_forces"],
            state["external_torques"],
        )
        updated = dict(state)
        updated["external_forces"] = external_forces
        updated["external_torques"] = external_torques
        return updated

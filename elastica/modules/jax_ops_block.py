from __future__ import annotations

from itertools import chain
from typing import Any, Type

from elastica.jax_block_operation import NoBlockOpJax
from elastica.jax_operation import NoOpsJax
from elastica.memory_block.memory_block_rod_jax import (
    _ELEMENT_ATTRS,
    _NODE_ATTRS,
    _SYNCABLE_ATTRS,
    _VORONOI_ATTRS,
    MemoryBlockCosseratRodJax,
)
import jax
import jax.numpy as jnp
import numpy as np

from .protocol import ModuleProtocol, SystemCollectionProtocol

_BLOCK_STAGE_METHODS = (
    (
        "constrain_values",
        "jax_block_operate_constrain_values",
        "jax_per_rod_operate_constrain_values",
        "jax_operate_constrain_values",
    ),
    (
        "synchronize",
        "jax_block_operate_synchronize",
        "jax_per_rod_operate_synchronize",
        "jax_operate_synchronize",
    ),
    (
        "constrain_rates",
        "jax_block_operate_constrain_rates",
        "jax_per_rod_operate_constrain_rates",
        "jax_operate_constrain_rates",
    ),
)


class _PerRodStateView:
    def __init__(self, state: dict[str, Any], *, updates: dict[str, Any] | None = None) -> None:
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_updates", {} if updates is None else dict(updates))

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("_"):
            raise AttributeError(attr)
        return self._updates.get(attr, self._state[attr])

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr.startswith("_"):
            object.__setattr__(self, attr, value)
            return
        self._updates[attr] = value

    def commit(self) -> dict[str, Any]:
        updated = dict(self._state)
        updated.update(self._updates)
        return updated


class JAXOpsBlock(SystemCollectionProtocol):
    """
    Register pure JAX block operations and expose JAX stage transforms.

    User code normally interacts with this mixin through:

    ```python
    simulator.operate_block(ea.CosseratRod).using(MyOp, ...)
    ```

    where `MyOp` derives from `ea.NoBlockOpJax`.

    `JAXOpsBlock` supports two execution styles:

    - block-native operators implementing `jax_block_operate_*`
    - per-rod operators implementing `jax_per_rod_operate_*`

    Per-rod operators are batched across rods during `finalize()` by gathering
    uniform rod slices from the block state, applying `jax.vmap(...)`, and
    scattering the updated rod-local fields back once per stage.
    """

    _jax_block_ops_list: list[ModuleProtocol]

    def __init__(self) -> None:
        self._jax_block_ops_list = []
        super().__init__()
        self._feature_group_finalize.append(self._finalize_jax_block_ops)

    def operate_block(self, system_type) -> ModuleProtocol:  # type: ignore[no-untyped-def]
        jax_op: ModuleProtocol = _JAXBlockOp(system_type)
        self._jax_block_ops_list.append(jax_op)
        return jax_op

    def jax_synchronize(self, states, time):  # type: ignore[no-untyped-def]
        for func in self._feature_group_synchronize:
            states = func(states=states, time=time)
        return states

    def jax_constrain_values(self, states, time):  # type: ignore[no-untyped-def]
        for func in self._feature_group_constrain_values:
            states = func(states=states, time=time)
        return states

    def jax_constrain_rates(self, states, time):  # type: ignore[no-untyped-def]
        for func in chain(
            self._feature_group_constrain_rates,
            self._feature_group_damping,
        ):
            states = func(states=states, time=time)
        return states

    @classmethod
    def _wrap_jax_block_operator(
        cls,
        *,
        block_state_idx: int,
        operator: Any,
    ):
        def apply(*, states, time):  # type: ignore[no-untyped-def]
            block_state = states[block_state_idx]
            updated_state = operator(block_state, time)
            updated_states = list(states)
            updated_states[block_state_idx] = updated_state
            return tuple(updated_states)

        return apply

    @staticmethod
    def _uniform_index_matrix(
        start_idx: np.ndarray,
        end_idx: np.ndarray,
    ) -> np.ndarray:
        widths = end_idx - start_idx
        assert np.all(widths == widths[0]), (
            "Per-rod JAX block operators require uniform discretization across rods."
        )
        offsets = np.arange(int(widths[0]), dtype=np.int32)
        return start_idx[:, None].astype(np.int32) + offsets[None, :]

    @classmethod
    def _per_rod_indices(
        cls,
        block_system: MemoryBlockCosseratRodJax,
    ) -> dict[str, jax.Array]:
        return {
            "node": jnp.asarray(
                cls._uniform_index_matrix(
                    block_system.start_idx_in_rod_nodes,
                    block_system.end_idx_in_rod_nodes,
                )
            ),
            "element": jnp.asarray(
                cls._uniform_index_matrix(
                    block_system.start_idx_in_rod_elems,
                    block_system.end_idx_in_rod_elems,
                )
            ),
            "voronoi": jnp.asarray(
                cls._uniform_index_matrix(
                    block_system.start_idx_in_rod_voronoi,
                    block_system.end_idx_in_rod_voronoi,
                )
            ),
        }

    @staticmethod
    def _gather_attr(array: jax.Array, indices: jax.Array) -> jax.Array:
        if array.ndim == 1:
            return jnp.take(array, indices, axis=-1)
        if array.ndim == 2:
            return jnp.moveaxis(jnp.take(array, indices, axis=-1), 1, 0)
        if array.ndim == 3:
            return jnp.moveaxis(jnp.take(array, indices, axis=-1), 2, 0)
        raise ValueError(f"Unsupported array rank {array.ndim} for per-rod batching.")

    @staticmethod
    def _scatter_attr(array: jax.Array, indices: jax.Array, values: jax.Array) -> jax.Array:
        if array.ndim == 1:
            return array.at[indices].set(values)
        if array.ndim == 2:
            return array.at[:, indices].set(jnp.moveaxis(values, 0, 1))
        if array.ndim == 3:
            return array.at[:, :, indices].set(jnp.moveaxis(values, 0, 2))
        raise ValueError(f"Unsupported array rank {array.ndim} for per-rod batching.")

    @classmethod
    def _wrap_jax_per_rod_operator(
        cls,
        *,
        block_state_idx: int,
        block_system: MemoryBlockCosseratRodJax,
        operator: Any,
    ):
        indices = cls._per_rod_indices(block_system)
        attr_domains = {
            attr: "node"
            for attr in _NODE_ATTRS
        } | {
            attr: "element"
            for attr in _ELEMENT_ATTRS
        } | {
            attr: "voronoi"
            for attr in _VORONOI_ATTRS
        }
        attrs = tuple(_SYNCABLE_ATTRS)

        def apply(*, states, time):  # type: ignore[no-untyped-def]
            block_state = states[block_state_idx]
            local_state = {
                attr: cls._gather_attr(block_state[attr], indices[attr_domains[attr]])
                for attr in attrs
            }

            def single_apply(single_state):  # type: ignore[no-untyped-def]
                rod_view = _PerRodStateView(single_state)
                updated_view = operator(rod_view, time)
                if isinstance(updated_view, _PerRodStateView):
                    return updated_view.commit()
                if hasattr(updated_view, "commit"):
                    return updated_view.commit()
                return rod_view.commit()

            updated_local_state = jax.vmap(single_apply)(local_state)
            updated_block_state = dict(block_state)
            for attr in attrs:
                updated_block_state[attr] = cls._scatter_attr(
                    block_state[attr],
                    indices[attr_domains[attr]],
                    updated_local_state[attr],
                )
            updated_states = list(states)
            updated_states[block_state_idx] = updated_block_state
            return tuple(updated_states)

        return apply

    def _finalize_jax_block_ops(self) -> None:
        final_systems = tuple(self.final_systems())

        for jax_op in self._jax_block_ops_list:
            block_state_idx, block_system = self._find_target_block(
                final_systems,
                jax_op.target(),
            )
            staged_wrappers = []
            instantiate_with_block = False
            instantiate_with_representative_rod = False
            for stage, block_method_name, per_rod_method_name, legacy_method_name in _BLOCK_STAGE_METHODS:
                op_cls = jax_op.operator_cls()
                has_block = hasattr(op_cls, block_method_name) and getattr(
                    op_cls, block_method_name
                ) is not getattr(NoBlockOpJax, block_method_name)
                has_per_rod = hasattr(op_cls, per_rod_method_name) and getattr(
                    op_cls, per_rod_method_name
                ) is not getattr(NoBlockOpJax, per_rod_method_name)
                has_legacy = hasattr(op_cls, legacy_method_name) and getattr(
                    op_cls, legacy_method_name
                ) is not getattr(NoOpsJax, legacy_method_name)

                assert not (
                    has_block and (has_per_rod or has_legacy)
                ), (
                    f"{op_cls} mixes block and per-rod JAX block operator "
                    f"implementations for stage {stage!r}. Choose one style per stage."
                )

                if has_block:
                    instantiate_with_block = True
                    continue

                if has_per_rod:
                    instantiate_with_block = True
                    continue

                if has_legacy:
                    instantiate_with_representative_rod = True
                    continue

            assert not (
                instantiate_with_block and instantiate_with_representative_rod
            ), (
                f"{jax_op.operator_cls()} mixes block-style and legacy rod-style JAX "
                "block operator methods. Use one constructor contract."
            )

            instantiate_target = block_system
            if instantiate_with_representative_rod:
                instantiate_target = block_system._systems[0]
            op_instance = jax_op.instantiate(instantiate_target)

            for stage, block_method_name, per_rod_method_name, legacy_method_name in _BLOCK_STAGE_METHODS:
                has_block = hasattr(type(op_instance), block_method_name) and getattr(
                    type(op_instance), block_method_name
                ) is not getattr(NoBlockOpJax, block_method_name)
                has_per_rod = hasattr(type(op_instance), per_rod_method_name) and getattr(
                    type(op_instance), per_rod_method_name
                ) is not getattr(NoBlockOpJax, per_rod_method_name)
                has_legacy = hasattr(type(op_instance), legacy_method_name) and getattr(
                    type(op_instance), legacy_method_name
                ) is not getattr(NoOpsJax, legacy_method_name)

                if has_block:
                    wrapped = self._wrap_jax_block_operator(
                        block_state_idx=block_state_idx,
                        operator=getattr(op_instance, block_method_name),
                    )
                    staged_wrappers.append((stage, wrapped))
                    continue

                if has_per_rod or has_legacy:
                    method_name = per_rod_method_name if has_per_rod else legacy_method_name
                    wrapped = self._wrap_jax_per_rod_operator(
                        block_state_idx=block_state_idx,
                        block_system=block_system,
                        operator=getattr(op_instance, method_name),
                    )
                    staged_wrappers.append((stage, wrapped))

            assert staged_wrappers, (
                f"{type(op_instance)} does not define any JAX block stage methods. "
                "Implement at least one of the block-stage methods or per-rod "
                "stage methods."
            )

            for stage, wrapped in staged_wrappers:
                stage_group = self._stage_group(stage)
                stage_group.append_id(jax_op)
                stage_group.add_operators(jax_op, [wrapped])

        self._jax_block_ops_list = []
        del self._jax_block_ops_list

    def _stage_group(self, stage: str):  # type: ignore[no-untyped-def]
        assert stage in (
            "constrain_values",
            "synchronize",
            "constrain_rates",
        ), f"Unsupported JAX block operator stage {stage!r}."
        if stage == "constrain_values":
            return self._feature_group_constrain_values
        if stage == "synchronize":
            return self._feature_group_synchronize
        return self._feature_group_constrain_rates

    @staticmethod
    def _find_target_block(final_systems, target_type):  # type: ignore[no-untyped-def]
        for block_state_idx, system in enumerate(final_systems):
            if not isinstance(system, MemoryBlockCosseratRodJax):
                continue
            if isinstance(system, target_type):
                return block_state_idx, system
            if any(isinstance(subsystem, target_type) for subsystem in system._systems):
                return block_state_idx, system
        raise RuntimeError(
            "Requested JAX block operator target was not found in finalized block systems."
        )


class _JAXBlockOp:
    def __init__(self, target_type: Type[Any]) -> None:
        self._target_type = target_type
        self._op_cls: Type[Any]
        self._args: Any
        self._kwargs: Any

    def using(
        self,
        cls: Type[Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        assert issubclass(
            cls, (NoBlockOpJax, NoOpsJax)
        ), (
            f"{cls} is not a valid JAX block operator. It must derive from "
            "NoBlockOpJax or NoOpsJax."
        )
        self._op_cls = cls
        self._args = args
        self._kwargs = kwargs

    def target(self) -> Type[Any]:
        return self._target_type

    def operator_cls(self) -> Type[Any]:
        if not hasattr(self, "_op_cls"):
            raise RuntimeError(
                "No JAX block operator provided. Did you forget to call "
                "`simulator.operate_block(...).using(...)`?"
            )
        return self._op_cls

    def id(self) -> Any:
        return self._target_type

    def instantiate(self, system: Any) -> Any:
        if not hasattr(self, "_op_cls"):
            raise RuntimeError(
                "No JAX block operator provided. Did you forget to call "
                "`simulator.operate_block(...).using(...)`?"
            )
        return self._op_cls(*self._args, _system=system, **self._kwargs)

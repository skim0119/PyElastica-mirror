from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Any, Type

import numpy as np

from elastica.jax_rod2rod_block_operation import (
    JAXRodRodBlockMetadata,
    NoRodRodBlockOpJax,
)
from elastica.memory_block.memory_block_rod_jax import MemoryBlockCosseratRodJax
from elastica.typing import SystemIdxType

from .protocol import ModuleProtocol, SystemCollectionProtocol


@dataclass(frozen=True)
class _RodRodPairBlockLocation:
    block_state_idx: int
    metadata: JAXRodRodBlockMetadata


class JAXRodRodBlockOps(SystemCollectionProtocol):
    """
    Register JAX rod-to-rod block operations.

    This mixin is intended for pair interactions such as rod-rod contact that are
    applied on packed block state rather than by iterating live Python rod objects
    during the rollout.
    """

    _jax_rod2rod_block_ops_list: list[ModuleProtocol]

    def __init__(self) -> None:
        self._jax_rod2rod_block_ops_list = []
        super().__init__()
        self._feature_group_finalize.append(self._finalize_jax_rod2rod_block_ops)

    def connect_block(
        self,
        first_rod: object,
        second_rod: object,
    ) -> ModuleProtocol:
        first_sys_idx = self.get_system_index(first_rod)
        second_sys_idx = self.get_system_index(second_rod)
        jax_op: ModuleProtocol = _JAXRodRodBlockOp(first_sys_idx, second_sys_idx)
        self._jax_rod2rod_block_ops_list.append(jax_op)
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
    def _wrap_jax_rod2rod_operator(
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

    def _finalize_jax_rod2rod_block_ops(self) -> None:
        final_systems = tuple(self.final_systems())

        for jax_op in self._jax_rod2rod_block_ops_list:
            location = self._find_pair_block_location(
                final_systems,
                jax_op.first_id(),
                jax_op.second_id(),
            )
            op_instance = jax_op.instantiate(
                final_systems[location.block_state_idx],
                location.metadata,
            )
            method = getattr(op_instance, "jax_rod2rod_operate_synchronize", None)
            assert method is not None, (
                f"{type(op_instance)} does not define "
                "`jax_rod2rod_operate_synchronize`."
            )
            if getattr(type(op_instance), "jax_rod2rod_operate_synchronize") is getattr(
                NoRodRodBlockOpJax, "jax_rod2rod_operate_synchronize"
            ):
                raise AssertionError(
                    f"{type(op_instance)} must override "
                    "`jax_rod2rod_operate_synchronize`."
                )
            wrapped = self._wrap_jax_rod2rod_operator(
                block_state_idx=location.block_state_idx,
                operator=method,
            )
            self._feature_group_synchronize.append_id(jax_op)
            self._feature_group_synchronize.add_operators(jax_op, [wrapped])

        self._jax_rod2rod_block_ops_list = []
        del self._jax_rod2rod_block_ops_list

    @staticmethod
    def _uniform_index_matrix(start: int, end: int) -> np.ndarray:
        return np.arange(start, end, dtype=np.int32)[None, :]

    @classmethod
    def _find_pair_block_location(
        cls,
        final_systems: tuple[object, ...],
        first_sys_idx: SystemIdxType,
        second_sys_idx: SystemIdxType,
    ) -> _RodRodPairBlockLocation:
        for block_state_idx, system in enumerate(final_systems):
            if not isinstance(system, MemoryBlockCosseratRodJax):
                continue
            first_matches = np.where(system.system_idx_list == first_sys_idx)[0]
            second_matches = np.where(system.system_idx_list == second_sys_idx)[0]
            if first_matches.size == 0 or second_matches.size == 0:
                continue

            first_local_idx = int(first_matches[0])
            second_local_idx = int(second_matches[0])
            metadata = JAXRodRodBlockMetadata(
                first_system_indices=np.asarray([first_sys_idx], dtype=np.int32),
                second_system_indices=np.asarray([second_sys_idx], dtype=np.int32),
                first_node_indices=cls._uniform_index_matrix(
                    int(system.start_idx_in_rod_nodes[first_local_idx]),
                    int(system.end_idx_in_rod_nodes[first_local_idx]),
                ),
                second_node_indices=cls._uniform_index_matrix(
                    int(system.start_idx_in_rod_nodes[second_local_idx]),
                    int(system.end_idx_in_rod_nodes[second_local_idx]),
                ),
                first_element_indices=cls._uniform_index_matrix(
                    int(system.start_idx_in_rod_elems[first_local_idx]),
                    int(system.end_idx_in_rod_elems[first_local_idx]),
                ),
                second_element_indices=cls._uniform_index_matrix(
                    int(system.start_idx_in_rod_elems[second_local_idx]),
                    int(system.end_idx_in_rod_elems[second_local_idx]),
                ),
                first_voronoi_indices=cls._uniform_index_matrix(
                    int(system.start_idx_in_rod_voronoi[first_local_idx]),
                    int(system.end_idx_in_rod_voronoi[first_local_idx]),
                ),
                second_voronoi_indices=cls._uniform_index_matrix(
                    int(system.start_idx_in_rod_voronoi[second_local_idx]),
                    int(system.end_idx_in_rod_voronoi[second_local_idx]),
                ),
            )
            return _RodRodPairBlockLocation(
                block_state_idx=block_state_idx,
                metadata=metadata,
            )

        raise RuntimeError(
            "Requested rod pair was not found in a common MemoryBlockCosseratRodJax."
        )


class _JAXRodRodBlockOp:
    def __init__(
        self, first_sys_idx: SystemIdxType, second_sys_idx: SystemIdxType
    ) -> None:
        self._first_sys_idx = first_sys_idx
        self._second_sys_idx = second_sys_idx
        self._op_cls: Type[Any]
        self._args: Any
        self._kwargs: Any

    def using(
        self,
        cls: Type[Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        assert issubclass(cls, NoRodRodBlockOpJax), (
            f"{cls} is not a valid JAX rod-to-rod block operator. "
            "It must derive from NoRodRodBlockOpJax."
        )
        self._op_cls = cls
        self._args = args
        self._kwargs = kwargs

    def instantiate(
        self,
        system: Any,
        metadata: JAXRodRodBlockMetadata,
    ) -> Any:
        if not hasattr(self, "_op_cls"):
            raise RuntimeError(
                "No JAX rod-to-rod block operator provided. Did you forget to call "
                "`simulator.connect_block(...).using(...)`?"
            )
        return self._op_cls(
            *self._args,
            _system=system,
            _pair_metadata=metadata,
            **self._kwargs,
        )

    def id(self) -> tuple[SystemIdxType, SystemIdxType]:
        return (self._first_sys_idx, self._second_sys_idx)

    def first_id(self) -> SystemIdxType:
        return self._first_sys_idx

    def second_id(self) -> SystemIdxType:
        return self._second_sys_idx

from __future__ import annotations

from itertools import chain
from typing import Any, Type

import numpy as np

from elastica.jax_operation import NoOpsJax
from elastica.memory_block.memory_block_rod_jax import (
    JAXRodView,
    JAXRodViewMetadata,
    MemoryBlockCosseratRodJax,
)
from elastica.typing import SystemIdxType

from .protocol import ModuleProtocol, SystemCollectionProtocol

_STAGE_METHODS = (
    ("constrain_values", "jax_operate_constrain_values"),
    ("synchronize", "jax_operate_synchronize"),
    ("constrain_rates", "jax_operate_constrain_rates"),
)


class JAXOps(SystemCollectionProtocol):
    """
    Register pure JAX rod-local operators and expose JAX stage transforms.
    """

    _jax_ops_list: list[ModuleProtocol]

    def __init__(self) -> None:
        self._jax_ops_list = []
        super().__init__()
        self._feature_group_finalize.append(self._finalize_jax_ops)

    def using(self, system) -> ModuleProtocol:  # type: ignore[no-untyped-def]
        sys_idx = self.get_system_index(system)
        jax_op: ModuleProtocol = _JAXOp(sys_idx)
        self._jax_ops_list.append(jax_op)
        return jax_op

    @classmethod
    def _wrap_jax_rod_operator(
        cls,
        *,
        metadata: JAXRodViewMetadata,
        operator: Any,
    ):
        def apply(*, states, time):  # type: ignore[no-untyped-def]
            block_state = states[metadata.block_state_idx]
            rod_view = JAXRodView(block_state, metadata)
            updated_view = operator(rod_view, time)
            committed_state = (
                updated_view.commit()
                if isinstance(updated_view, JAXRodView)
                else rod_view.commit()
            )
            updated_states = list(states)
            updated_states[metadata.block_state_idx] = committed_state
            return tuple(updated_states)

        return apply

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

    def _finalize_jax_ops(self) -> None:
        final_systems = tuple(self.final_systems())
        self._jax_ops_list.sort(key=lambda x: x.id())

        for jax_op in self._jax_ops_list:
            sys_id = jax_op.id()
            op_instance = jax_op.instantiate(self[sys_id])
            metadata = self._make_jax_rod_view_metadata(final_systems, sys_id)
            staged_wrappers = []
            for stage, method_name in _STAGE_METHODS:
                method = getattr(op_instance, method_name, None)
                if method is None:
                    continue
                if getattr(type(op_instance), method_name) is getattr(
                    NoOpsJax, method_name
                ):
                    continue
                wrapped = self._wrap_jax_rod_operator(
                    metadata=metadata,
                    operator=method,
                )
                staged_wrappers.append((stage, wrapped))

            assert staged_wrappers, (
                f"{type(op_instance)} does not define any JAX stage methods. "
                "Implement at least one of "
                "`jax_operate_constrain_values`, "
                "`jax_operate_synchronize`, or "
                "`jax_operate_constrain_rates`."
            )

            for stage, wrapped in staged_wrappers:
                stage_group = self._stage_group(stage)
                stage_group.append_id(jax_op)
                stage_group.add_operators(jax_op, [wrapped])

        self._jax_ops_list = []
        del self._jax_ops_list

    def _stage_group(self, stage: str):  # type: ignore[no-untyped-def]
        assert stage in (
            "constrain_values",
            "synchronize",
            "constrain_rates",
        ), f"Unsupported JAX operator stage {stage!r}."
        if stage == "constrain_values":
            return self._feature_group_constrain_values
        if stage == "synchronize":
            return self._feature_group_synchronize
        return self._feature_group_constrain_rates

    def _make_jax_rod_view_metadata(
        self,
        final_systems: tuple[object, ...],
        sys_id: SystemIdxType,
    ) -> JAXRodViewMetadata:
        for block_state_idx, system in enumerate(final_systems):
            if not isinstance(system, MemoryBlockCosseratRodJax):
                continue
            matches = np.where(system.system_idx_list == sys_id)[0]
            if matches.size == 0:
                continue
            local_idx = int(matches[0])
            return JAXRodViewMetadata(
                block_state_idx=block_state_idx,
                node_slice=slice(
                    int(system.start_idx_in_rod_nodes[local_idx]),
                    int(system.end_idx_in_rod_nodes[local_idx]),
                ),
                element_slice=slice(
                    int(system.start_idx_in_rod_elems[local_idx]),
                    int(system.end_idx_in_rod_elems[local_idx]),
                ),
                voronoi_slice=slice(
                    int(system.start_idx_in_rod_voronoi[local_idx]),
                    int(system.end_idx_in_rod_voronoi[local_idx]),
                ),
            )
        raise RuntimeError(
            "JAX operator target system was not found in a MemoryBlockCosseratRodJax. "
            "Enable JAX block support for the rod type before finalize()."
        )


class _JAXOp:
    def __init__(self, sys_idx: SystemIdxType) -> None:
        self._sys_idx = sys_idx
        self._op_cls: Type[Any]
        self._args: Any
        self._kwargs: Any

    def operate(
        self,
        cls: Type[Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        assert issubclass(
            cls, NoOpsJax
        ), f"{cls} is not a valid JAX operator. It must derive from NoOpsJax."
        self._op_cls = cls
        self._args = args
        self._kwargs = kwargs

    def id(self) -> SystemIdxType:
        return self._sys_idx

    def instantiate(self, system: Any) -> Any:
        if not hasattr(self, "_op_cls"):
            raise RuntimeError(
                "No JAX operator provided. Did you forget to call "
                "`simulator.using(system).operate(...)`?"
            )
        return self._op_cls(*self._args, _system=system, **self._kwargs)

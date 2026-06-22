from __future__ import annotations

from itertools import chain
from typing import Any, Type

import numpy as np

from elastica.jax_rod_rigid_body_operation import NoRodRigidBodyJax
from elastica.memory_block.memory_block_rigid_body_jax import (
    JAXRigidBodyView,
    JAXRigidBodyViewMetadata,
    MemoryBlockRigidBodyJax,
)
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


class JAXOpsRodRigidBody(SystemCollectionProtocol):
    """
    Register pure-JAX mixed rod/rigid-body operators and expose JAX stage transforms.
    """

    _jax_rod_rigid_body_ops_list: list[ModuleProtocol]

    def __init__(self) -> None:
        self._jax_rod_rigid_body_ops_list = []
        super().__init__()
        self._feature_group_finalize.append(self._finalize_jax_rod_rigid_body_ops)

    def using_on(
        self,
        rod_system,
        rigid_body_system,
    ) -> ModuleProtocol:  # type: ignore[no-untyped-def]
        rod_sys_idx = self.get_system_index(rod_system)
        rigid_body_sys_idx = self.get_system_index(rigid_body_system)
        jax_op: ModuleProtocol = _JAXRodRigidBodyOp(rod_sys_idx, rigid_body_sys_idx)
        self._jax_rod_rigid_body_ops_list.append(jax_op)
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
    def _wrap_jax_rod_rigid_body_operator(
        cls,
        *,
        rod_metadata: JAXRodViewMetadata,
        rigid_body_metadata: JAXRigidBodyViewMetadata,
        operator: Any,
    ):
        def apply(*, states, time):  # type: ignore[no-untyped-def]
            rod_state = states[rod_metadata.block_state_idx]
            rigid_body_state = states[rigid_body_metadata.block_state_idx]
            rod_view = JAXRodView(rod_state, rod_metadata)
            rigid_body_view = JAXRigidBodyView(rigid_body_state, rigid_body_metadata)
            updated_views = operator(rod_view, rigid_body_view, time)

            if updated_views is None:
                committed_rod_state = rod_view.commit()
                committed_rigid_body_state = rigid_body_view.commit()
            else:
                updated_rod_view, updated_rigid_body_view = updated_views
                committed_rod_state = (
                    updated_rod_view.commit()
                    if isinstance(updated_rod_view, JAXRodView)
                    else rod_view.commit()
                )
                committed_rigid_body_state = (
                    updated_rigid_body_view.commit()
                    if isinstance(updated_rigid_body_view, JAXRigidBodyView)
                    else rigid_body_view.commit()
                )

            updated_states = list(states)
            updated_states[rod_metadata.block_state_idx] = committed_rod_state
            updated_states[rigid_body_metadata.block_state_idx] = (
                committed_rigid_body_state
            )
            return tuple(updated_states)

        return apply

    def _finalize_jax_rod_rigid_body_ops(self) -> None:
        final_systems = tuple(self.final_systems())

        for jax_op in self._jax_rod_rigid_body_ops_list:
            rod_metadata = self._make_jax_rod_view_metadata(
                final_systems, jax_op.rod_id()
            )
            rigid_body_metadata = self._make_jax_rigid_body_view_metadata(
                final_systems, jax_op.rigid_body_id()
            )
            op_instance = jax_op.instantiate(
                rod_system=self[jax_op.rod_id()],
                rigid_body_system=self[jax_op.rigid_body_id()],
            )

            staged_wrappers = []
            for stage, method_name in _STAGE_METHODS:
                method = getattr(op_instance, method_name, None)
                if method is None:
                    continue
                if getattr(type(op_instance), method_name) is getattr(
                    NoRodRigidBodyJax, method_name
                ):
                    continue
                wrapped = self._wrap_jax_rod_rigid_body_operator(
                    rod_metadata=rod_metadata,
                    rigid_body_metadata=rigid_body_metadata,
                    operator=method,
                )
                staged_wrappers.append((stage, wrapped))

            assert staged_wrappers, (
                f"{type(op_instance)} does not define any mixed JAX stage methods. "
                "Implement at least one of "
                "`jax_operate_constrain_values`, "
                "`jax_operate_synchronize`, or "
                "`jax_operate_constrain_rates`."
            )

            for stage, wrapped in staged_wrappers:
                stage_group = self._stage_group(stage)
                stage_group.append_id(jax_op)
                stage_group.add_operators(jax_op, [wrapped])

        self._jax_rod_rigid_body_ops_list = []
        del self._jax_rod_rigid_body_ops_list

    def _stage_group(self, stage: str):  # type: ignore[no-untyped-def]
        assert stage in (
            "constrain_values",
            "synchronize",
            "constrain_rates",
        ), f"Unsupported mixed JAX operator stage {stage!r}."
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
            "Mixed JAX operator target rod was not found in a MemoryBlockCosseratRodJax. "
            "Enable JAX block support for the rod type before finalize()."
        )

    def _make_jax_rigid_body_view_metadata(
        self,
        final_systems: tuple[object, ...],
        sys_id: SystemIdxType,
    ) -> JAXRigidBodyViewMetadata:
        for block_state_idx, system in enumerate(final_systems):
            if not isinstance(system, MemoryBlockRigidBodyJax):
                continue
            matches = np.where(system.system_idx_list == sys_id)[0]
            if matches.size == 0:
                continue
            local_idx = int(matches[0])
            return JAXRigidBodyViewMetadata(
                block_state_idx=block_state_idx,
                body_slice=slice(local_idx, local_idx + 1),
            )
        raise RuntimeError(
            "Mixed JAX operator target rigid body was not found in a MemoryBlockRigidBodyJax. "
            "Enable JAX block support for the rigid-body type before finalize()."
        )


class _JAXRodRigidBodyOp:
    def __init__(
        self,
        rod_sys_idx: SystemIdxType,
        rigid_body_sys_idx: SystemIdxType,
    ) -> None:
        self._rod_sys_idx = rod_sys_idx
        self._rigid_body_sys_idx = rigid_body_sys_idx
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
            cls, NoRodRigidBodyJax
        ), f"{cls} is not a valid mixed JAX operator. It must derive from NoRodRigidBodyJax."
        self._op_cls = cls
        self._args = args
        self._kwargs = kwargs

    def id(self) -> tuple[SystemIdxType, SystemIdxType]:
        return self._rod_sys_idx, self._rigid_body_sys_idx

    def rod_id(self) -> SystemIdxType:
        return self._rod_sys_idx

    def rigid_body_id(self) -> SystemIdxType:
        return self._rigid_body_sys_idx

    def instantiate(
        self,
        *,
        rod_system: Any,
        rigid_body_system: Any,
    ) -> Any:
        if not hasattr(self, "_op_cls"):
            raise RuntimeError(
                "No mixed JAX operator provided. Did you forget to call "
                "`simulator.using_on(rod, rigid_body).operate(...)`?"
            )
        return self._op_cls(
            *self._args,
            _first_system=rod_system,
            _second_system=rigid_body_system,
            **self._kwargs,
        )

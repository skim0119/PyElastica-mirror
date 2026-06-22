from __future__ import annotations

from itertools import chain
from typing import Any, Type

from elastica.jax_block_operation import NoBlockOpJax
from elastica.memory_block.memory_block_rod_jax import MemoryBlockCosseratRodJax

from .protocol import ModuleProtocol, SystemCollectionProtocol

_BLOCK_STAGE_METHODS = (
    ("constrain_values", "jax_block_operate_constrain_values"),
    ("synchronize", "jax_block_operate_synchronize"),
    ("constrain_rates", "jax_block_operate_constrain_rates"),
)


class JAXOpsBlock(SystemCollectionProtocol):
    """
    Register pure JAX block-wide operators and expose JAX stage transforms.
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

    def _finalize_jax_block_ops(self) -> None:
        final_systems = tuple(self.final_systems())

        for jax_op in self._jax_block_ops_list:
            block_state_idx, block_system = self._find_target_block(
                final_systems,
                jax_op.target(),
            )
            op_instance = jax_op.instantiate(block_system)
            staged_wrappers = []
            for stage, method_name in _BLOCK_STAGE_METHODS:
                method = getattr(op_instance, method_name, None)
                if method is None:
                    continue
                if getattr(type(op_instance), method_name) is getattr(
                    NoBlockOpJax, method_name
                ):
                    continue
                wrapped = self._wrap_jax_block_operator(
                    block_state_idx=block_state_idx,
                    operator=method,
                )
                staged_wrappers.append((stage, wrapped))

            assert staged_wrappers, (
                f"{type(op_instance)} does not define any JAX block stage methods. "
                "Implement at least one of "
                "`jax_block_operate_constrain_values`, "
                "`jax_block_operate_synchronize`, or "
                "`jax_block_operate_constrain_rates`."
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
            cls, NoBlockOpJax
        ), f"{cls} is not a valid JAX block operator. It must derive from NoBlockOpJax."
        self._op_cls = cls
        self._args = args
        self._kwargs = kwargs

    def target(self) -> Type[Any]:
        return self._target_type

    def id(self) -> Any:
        return self._target_type

    def instantiate(self, system: Any) -> Any:
        if not hasattr(self, "_op_cls"):
            raise RuntimeError(
                "No JAX block operator provided. Did you forget to call "
                "`simulator.operate_block(...).using(...)`?"
            )
        return self._op_cls(*self._args, _system=system, **self._kwargs)

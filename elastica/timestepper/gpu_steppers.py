"""GPU-oriented Position Verlet timestepper."""

from collections.abc import Iterable
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np


JAXPyTree = Any


class JAXBlock(Protocol):
    """
    Minimal block interface required for a JAX-owned timestep loop.

    Notes
    -----
    All ``jax_*`` methods below must behave as pure transforms on PyTree state.
    They must not mutate Python object fields when used inside a JAX loop.
    The only host-side mutation point is ``jax_set_state(...)`` after rollout.
    """

    def jax_get_state(self) -> JAXPyTree: ...

    def jax_set_state(self, state: JAXPyTree) -> None: ...

    def jax_kinematic_step(
        self,
        state: JAXPyTree,
        time: np.float64,
        prefac: np.float64,
    ) -> JAXPyTree: ...

    def jax_dynamic_step(
        self,
        state: JAXPyTree,
        time: np.float64,
        dt: np.float64,
    ) -> JAXPyTree: ...

    def jax_compute_internal_forces_and_torques(
        self,
        state: JAXPyTree,
        time: np.float64,
    ) -> JAXPyTree: ...

    def jax_zero_external_loads(
        self,
        state: JAXPyTree,
        time: np.float64,
    ) -> JAXPyTree: ...


class JAXCompatibleSystems(Protocol):
    """
    Minimal collection interface required for a JAX-owned timestep loop.

    Notes
    -----
    Constraint/forcing/contact/damping phases must also be exposed as pure state
    transforms. Host callbacks are intentionally excluded from the loop contract.
    """

    def final_systems(self) -> Iterable[JAXBlock]: ...

    def jax_constrain_values(
        self,
        states: tuple[JAXPyTree, ...],
        time: np.float64,
    ) -> tuple[JAXPyTree, ...]: ...

    def jax_synchronize(
        self,
        states: tuple[JAXPyTree, ...],
        time: np.float64,
    ) -> tuple[JAXPyTree, ...]: ...

    def jax_constrain_rates(
        self,
        states: tuple[JAXPyTree, ...],
        time: np.float64,
    ) -> tuple[JAXPyTree, ...]: ...


class PositionVerletGPU:
    """Dedicated Position Verlet integrator for device-backed systems."""

    def __init__(self) -> None:
        self._compiled_rollout_cache: dict[tuple[int, int], Any] = {}

    @staticmethod
    def _body_fn(
        _,
        carry,
        *,
        systems: tuple[JAXBlock, ...],
        system_collection: JAXCompatibleSystems,
        dt_jax: jax.Array,
        half_dt_jax: jax.Array,
    ):
        step_time, step_states = carry

        step_states = tuple(
            system.jax_kinematic_step(state, step_time, half_dt_jax)
            for system, state in zip(systems, step_states)
        )
        step_time = step_time + half_dt_jax

        step_states = system_collection.jax_constrain_values(step_states, step_time)

        step_states = tuple(
            system.jax_compute_internal_forces_and_torques(state, step_time)
            for system, state in zip(systems, step_states)
        )

        step_states = system_collection.jax_synchronize(step_states, step_time)

        step_states = tuple(
            system.jax_dynamic_step(state, step_time, dt_jax)
            for system, state in zip(systems, step_states)
        )

        step_states = system_collection.jax_constrain_rates(step_states, step_time)

        step_states = tuple(
            system.jax_kinematic_step(state, step_time, half_dt_jax)
            for system, state in zip(systems, step_states)
        )
        step_time = step_time + half_dt_jax

        step_states = system_collection.jax_constrain_values(step_states, step_time)

        step_states = tuple(
            system.jax_zero_external_loads(state, step_time)
            for system, state in zip(systems, step_states)
        )

        return step_time, step_states

    def _get_compiled_rollout(
        self,
        system_collection: JAXCompatibleSystems,
        systems: tuple[JAXBlock, ...],
        n_steps: int,
    ):
        cache_key = (id(system_collection), n_steps)
        if cache_key in self._compiled_rollout_cache:
            return self._compiled_rollout_cache[cache_key]

        def rollout(
            time_jax: jax.Array,
            states: tuple[JAXPyTree, ...],
            dt_jax: jax.Array,
            half_dt_jax: jax.Array,
        ) -> tuple[jax.Array, tuple[JAXPyTree, ...]]:
            def body_fn(step_idx: int, carry):  # type: ignore[no-untyped-def]
                return self._body_fn(
                    step_idx,
                    carry,
                    systems=systems,
                    system_collection=system_collection,
                    dt_jax=dt_jax,
                    half_dt_jax=half_dt_jax,
                )

            return jax.lax.fori_loop(
                0,
                n_steps,
                body_fn,
                (time_jax, states),
            )

        compiled_rollout = jax.jit(rollout)
        self._compiled_rollout_cache[cache_key] = compiled_rollout
        return compiled_rollout

    @staticmethod
    def _reference_device_from_states(
        systems: tuple[JAXBlock, ...],
        states: tuple[JAXPyTree, ...],
    ) -> jax.Device:
        first_system = systems[0]
        if hasattr(first_system, "position_collection_device"):
            return first_system.position_collection_device.device

        for leaf in jax.tree_util.tree_leaves(states):
            if hasattr(leaf, "devices"):
                return next(iter(leaf.devices()))
            if hasattr(leaf, "device"):
                return leaf.device

        return jax.devices()[0]

    @staticmethod
    def _reference_dtype_from_states(
        states: tuple[JAXPyTree, ...],
    ) -> np.dtype:
        for leaf in jax.tree_util.tree_leaves(states):
            if hasattr(leaf, "dtype"):
                return np.dtype(leaf.dtype)
        return np.dtype(np.float64)

    def integrate(
        self,
        SystemCollection: JAXCompatibleSystems,
        time: np.float64 | float,
        final_time: np.float64 | float,
        dt: np.float64 | float,
    ) -> np.float64:
        """
        Integrate Position Verlet steps from ``time`` to ``final_time`` with step ``dt``.
        """
        assert dt > 0.0, "dt must be positive."
        assert final_time >= time, "final_time must be greater than or equal to time."

        simulation_time = np.float64(time)
        target_time = np.float64(final_time)
        simulation_dt = np.float64(dt)
        duration = float(target_time - simulation_time)
        n_steps = int(np.round(duration / float(simulation_dt)))
        assert np.isclose(
            simulation_time + n_steps * simulation_dt, target_time
        ), "final_time - time must be an integer multiple of dt."

        systems = tuple(SystemCollection.final_systems())
        states = tuple(system.jax_get_state() for system in systems)
        reference_device = self._reference_device_from_states(systems, states)
        reference_dtype = self._reference_dtype_from_states(states)
        dt_jax = jax.device_put(
            np.asarray(simulation_dt, dtype=reference_dtype),
            device=reference_device,
        )
        half_dt_jax = jax.device_put(
            np.asarray(0.5 * simulation_dt, dtype=reference_dtype),
            device=reference_device,
        )
        time_jax = jax.device_put(
            np.asarray(simulation_time, dtype=reference_dtype),
            device=reference_device,
        )
        compiled_rollout = self._get_compiled_rollout(
            system_collection=SystemCollection,
            systems=systems,
            n_steps=n_steps,
        )
        final_time_jax, final_states = compiled_rollout(
            time_jax,
            states,
            dt_jax,
            half_dt_jax,
        )

        for system, state in zip(systems, final_states):
            system.jax_set_state(state)

        return np.float64(final_time_jax)

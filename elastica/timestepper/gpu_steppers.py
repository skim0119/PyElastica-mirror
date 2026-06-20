"""GPU-oriented symplectic timesteppers with explicit host/device sync."""

from collections.abc import Iterable

import numpy as np

from elastica.timestepper.symplectic_steppers import PositionVerlet


class PositionVerletGPU(PositionVerlet):
    """
    Position Verlet stepper for device-backed block systems.

    Parameters
    ----------
    sync_host_operations : bool
        If ``True``, host-only operators such as forcing/contact/damping/callbacks are
        allowed by synchronizing the device state back to the host around each host
        phase. If ``False``, the stepper raises when those phases are populated.
    sync_callbacks : bool
        If ``True``, synchronize state back to host before callback execution.
    """

    def __init__(
        self,
        *,
        sync_host_operations: bool = False,
        sync_callbacks: bool = False,
    ) -> None:
        super().__init__()
        self.sync_host_operations = sync_host_operations
        self.sync_callbacks = sync_callbacks

    def _has_host_operators(self, system_collection) -> bool:  # type: ignore[no-untyped-def]
        return any(
            bool(getattr(group, "_operator_ids", ()))
            for group in (
                system_collection._feature_group_constrain_values,
                system_collection._feature_group_constrain_rates,
                system_collection._feature_group_synchronize,
                system_collection._feature_group_damping,
                system_collection._feature_group_callback,
            )
        )

    def _has_callbacks(self, system_collection) -> bool:  # type: ignore[no-untyped-def]
        return bool(
            getattr(system_collection._feature_group_callback, "_operator_ids", ())
        )

    def _sync_systems_from_device(
        self, systems: Iterable[object], attrs: Iterable[str] | None = None
    ) -> None:
        for system in systems:
            if hasattr(system, "from_device"):
                system.from_device(attrs=attrs)  # type: ignore[attr-defined]

    def _sync_systems_to_device(
        self, systems: Iterable[object], attrs: Iterable[str] | None = None
    ) -> None:
        for system in systems:
            if hasattr(system, "to_device"):
                system.to_device(attrs=attrs)  # type: ignore[attr-defined]

    def step(
        self,
        SystemCollection,
        time: np.float64 | float,
        dt: np.float64 | float,
    ) -> np.float64:
        if self._has_host_operators(SystemCollection) and not self.sync_host_operations:
            raise RuntimeError(
                "PositionVerletGPU encountered host-side operators. Re-run with "
                "`sync_host_operations=True` for explicit host/device transfers, or port "
                "those operators to the device path first."
            )

        simulation_time = np.float64(time)
        simulation_dt = np.float64(dt)
        systems = tuple(SystemCollection.final_systems())

        for kin_prefactor, kin_step, dyn_step in self.steps_and_prefactors[:-1]:
            for system in systems:
                kin_step(system, simulation_time, simulation_dt)

            simulation_time += kin_prefactor(simulation_dt)

            if self.sync_host_operations:
                self._sync_systems_from_device(
                    systems,
                    attrs=(
                        "position_collection",
                        "director_collection",
                        "velocity_collection",
                        "omega_collection",
                    ),
                )
            SystemCollection.constrain_values(simulation_time)
            if self.sync_host_operations:
                self._sync_systems_to_device(
                    systems,
                    attrs=("position_collection", "director_collection"),
                )

            for system in systems:
                system.compute_internal_forces_and_torques(simulation_time)

            if self.sync_host_operations:
                self._sync_systems_from_device(
                    systems,
                    attrs=(
                        "position_collection",
                        "director_collection",
                        "velocity_collection",
                        "omega_collection",
                        "external_forces",
                        "external_torques",
                    ),
                )
            SystemCollection.synchronize(simulation_time)
            if self.sync_host_operations:
                self._sync_systems_to_device(
                    systems, attrs=("external_forces", "external_torques")
                )

            for system in systems:
                dyn_step(system, simulation_time, simulation_dt)

            if self.sync_host_operations:
                self._sync_systems_from_device(
                    systems,
                    attrs=("velocity_collection", "omega_collection"),
                )
            SystemCollection.constrain_rates(simulation_time)
            if self.sync_host_operations:
                self._sync_systems_to_device(
                    systems,
                    attrs=(
                        "velocity_collection",
                        "omega_collection",
                        "acceleration_collection",
                        "alpha_collection",
                    ),
                )

        last_kin_prefactor = self.steps_and_prefactors[-1][0]
        last_kin_step = self.steps_and_prefactors[-1][1]

        for system in systems:
            last_kin_step(system, simulation_time, simulation_dt)
        simulation_time += last_kin_prefactor(simulation_dt)

        if self.sync_host_operations:
            self._sync_systems_from_device(
                systems,
                attrs=("position_collection", "director_collection"),
            )
        SystemCollection.constrain_values(simulation_time)
        if self.sync_host_operations:
            self._sync_systems_to_device(
                systems,
                attrs=("position_collection", "director_collection"),
            )

        if self.sync_callbacks or (
            self.sync_host_operations and self._has_callbacks(SystemCollection)
        ):
            self._sync_systems_from_device(systems)
        SystemCollection.apply_callbacks(
            simulation_time, round(simulation_time / simulation_dt)
        )

        for system in systems:
            system.zeroed_out_external_forces_and_torques(simulation_time)

        return simulation_time

    def run(
        self,
        SystemCollection,
        time: np.float64 | float,
        dt: np.float64 | float,
        n_steps: int,
    ) -> np.float64:
        """
        Execute multiple Position Verlet steps inside one device-side rollout.

        This path is intended to replace user-side Python loops of the form::

            for _ in range(n_steps):
                time = stepper.step(sim, time, dt)

        with a single JAX-backed call. The system collection must therefore be fully
        device-compatible: no host operators, no host callbacks, and a single final
        system exposing a ``jax_position_verlet_run(...)`` method.
        """
        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")

        if self.sync_host_operations or self.sync_callbacks:
            raise RuntimeError(
                "PositionVerletGPU.run requires a fully device-side path. Disable "
                "`sync_host_operations` and `sync_callbacks`, and port those phases "
                "to JAX before using the rollout API."
            )

        if self._has_host_operators(SystemCollection):
            raise RuntimeError(
                "PositionVerletGPU.run encountered host-side operators. Port "
                "constraints/forcing/contact/damping/callbacks to the device path "
                "before using the JAX rollout API."
            )

        systems = tuple(SystemCollection.final_systems())
        if len(systems) != 1:
            raise NotImplementedError(
                "PositionVerletGPU.run currently supports exactly one final system."
            )

        system = systems[0]
        rollout = getattr(system, "jax_position_verlet_run", None)
        if rollout is None:
            raise TypeError(
                "Final system does not expose `jax_position_verlet_run(...)`, which "
                "is required for device-side rollout."
            )

        final_time = rollout(
            time=np.float64(time),
            dt=np.float64(dt),
            n_steps=int(n_steps),
        )
        return np.float64(final_time)

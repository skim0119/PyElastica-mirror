from __future__ import annotations

from typing import TypeAlias

import numpy as np

from elastica.memory_block.memory_block_rod_jax import (
    JAXRodView,
    MemoryBlockCosseratRodJax,
)

JAXTime: TypeAlias = np.float64
JAXVectorArray: TypeAlias = np.ndarray
JAXScalarArray: TypeAlias = np.ndarray


class NoOpsJax:
    """
    Empty template for pure-JAX rod-local operators.

    Implement any subset of:
    - `jax_operate_constrain_values(rod_view, time)`
    - `jax_operate_synchronize(rod_view, time)`
    - `jax_operate_constrain_rates(rod_view, time)`
    """

    def jax_operate_constrain_values(
        self,
        rod_view: JAXRodView,
        time: JAXTime,
    ) -> JAXRodView:
        return rod_view

    def jax_operate_synchronize(
        self,
        rod_view: JAXRodView,
        time: JAXTime,
    ) -> JAXRodView:
        return rod_view

    def jax_operate_constrain_rates(
        self,
        rod_view: JAXRodView,
        time: JAXTime,
    ) -> JAXRodView:
        return rod_view


class OneEndFixedJax(NoOpsJax):
    """JAX equivalent of fixing the first node and first director."""

    def __init__(self, *, _system: object) -> None:
        self.fixed_position_collection: JAXVectorArray = np.asarray(
            _system.position_collection[..., 0].copy()
        )
        self.fixed_directors_collection: np.ndarray = np.asarray(
            _system.director_collection[..., 0].copy()
        )

    def jax_operate_constrain_values(
        self,
        rod_view: JAXRodView,
        time: JAXTime,
    ) -> JAXRodView:
        rod_view.position_collection = rod_view.position_collection.at[:, 0].set(
            self.fixed_position_collection
        )
        rod_view.director_collection = rod_view.director_collection.at[:, :, 0].set(
            self.fixed_directors_collection
        )
        return rod_view

    def jax_operate_constrain_rates(
        self,
        rod_view: JAXRodView,
        time: JAXTime,
    ) -> JAXRodView:
        import jax.numpy as jnp

        rod_view.velocity_collection = rod_view.velocity_collection.at[:, 0].set(
            jnp.zeros(3, dtype=rod_view.velocity_collection.dtype)
        )
        rod_view.omega_collection = rod_view.omega_collection.at[:, 0].set(
            jnp.zeros(3, dtype=rod_view.omega_collection.dtype)
        )
        return rod_view


class EndpointForcesJax(NoOpsJax):
    """JAX equivalent of ramped endpoint forces."""

    def __init__(
        self,
        start_force: JAXVectorArray,
        end_force: JAXVectorArray,
        ramp_up_time: float,
        *,
        _system: object,
    ) -> None:
        del _system
        assert ramp_up_time > 0.0, "ramp_up_time must be positive."
        self.start_force: JAXVectorArray = np.asarray(start_force, dtype=np.float64)
        self.end_force: JAXVectorArray = np.asarray(end_force, dtype=np.float64)
        self.ramp_up_time: np.float64 = np.float64(ramp_up_time)

    def jax_operate_synchronize(
        self,
        rod_view: JAXRodView,
        time: JAXTime,
    ) -> JAXRodView:
        import jax.numpy as jnp

        factor = jnp.minimum(
            jnp.asarray(1.0, dtype=rod_view.external_forces.dtype),
            jnp.asarray(time, dtype=rod_view.external_forces.dtype)
            / jnp.asarray(self.ramp_up_time, dtype=rod_view.external_forces.dtype),
        )
        rod_view.external_forces = rod_view.external_forces.at[:, 0].add(
            jnp.asarray(self.start_force, dtype=rod_view.external_forces.dtype) * factor
        )
        rod_view.external_forces = rod_view.external_forces.at[:, -1].add(
            jnp.asarray(self.end_force, dtype=rod_view.external_forces.dtype) * factor
        )
        return rod_view


class AnalyticalLinearDamperJax(NoOpsJax):
    """JAX equivalent of AnalyticalLinearDamper."""

    def __init__(self, time_step: np.float64, **kwargs: object) -> None:
        damping_constant = kwargs.get("damping_constant", None)
        uniform_damping_constant = kwargs.get("uniform_damping_constant", None)
        translational_damping_constant = kwargs.get(
            "translational_damping_constant", None
        )
        rotational_damping_constant = kwargs.get("rotational_damping_constant", None)
        system = kwargs["_system"]

        provided_params = [
            p
            for p in (
                damping_constant,
                uniform_damping_constant,
                translational_damping_constant,
                rotational_damping_constant,
            )
            if p is not None
        ]

        if len(provided_params) == 1 and damping_constant is not None:
            nodal_mass = system.mass
            self._translational_damping_coefficient = np.exp(
                -damping_constant * time_step
            )
            if system.ring_rod_flag:
                element_mass = nodal_mass
            else:
                element_mass = 0.5 * (nodal_mass[1:] + nodal_mass[:-1])
                element_mass[0] += 0.5 * nodal_mass[0]
                element_mass[-1] += 0.5 * nodal_mass[-1]
            self._rotational_damping_coefficient = np.exp(
                -damping_constant
                * time_step
                * element_mass
                * np.diagonal(system.inv_mass_second_moment_of_inertia).T
            )
        elif len(provided_params) == 1 and uniform_damping_constant is not None:
            coeff = np.exp(-uniform_damping_constant * time_step)
            self._translational_damping_coefficient = coeff
            self._rotational_damping_coefficient = coeff
        elif (
            len(provided_params) == 2
            and translational_damping_constant is not None
            and rotational_damping_constant is not None
        ):
            nodal_mass = system.mass
            self._translational_damping_coefficient = np.exp(
                -translational_damping_constant / nodal_mass * time_step
            )
            inv_moi = np.diagonal(system.inv_mass_second_moment_of_inertia).T
            self._rotational_damping_coefficient = np.exp(
                -rotational_damping_constant * inv_moi * time_step
            )
        else:
            raise ValueError(
                "AnalyticalLinearDamperJax requires the same parameterization as "
                "AnalyticalLinearDamper."
            )

    def jax_operate_constrain_rates(
        self,
        rod_view: JAXRodView,
        time: JAXTime,
    ) -> JAXRodView:
        import jax.numpy as jnp

        del time
        rod_view.velocity_collection = rod_view.velocity_collection * jnp.asarray(
            self._translational_damping_coefficient,
            dtype=rod_view.velocity_collection.dtype,
        )
        rod_view.omega_collection = rod_view.omega_collection * jnp.power(
            jnp.asarray(
                self._rotational_damping_coefficient,
                dtype=rod_view.omega_collection.dtype,
            ),
            rod_view.dilatation,
        )
        return rod_view


class GravityAnalyticalDamperJax(NoOpsJax):
    """
    Example JAX operator combining gravity loading with analytical rate damping.
    """

    def __init__(
        self,
        *,
        acc_gravity: JAXVectorArray | None = None,
        time_step: float,
        uniform_damping_constant: float | None = None,
        damping_constant: float | None = None,
        translational_damping_constant: float | None = None,
        rotational_damping_constant: float | None = None,
        _system: MemoryBlockCosseratRodJax,
    ) -> None:
        if acc_gravity is None:
            acc_gravity = np.array([0.0, -9.80665, 0.0], dtype=np.float64)
        self.acc_gravity: JAXVectorArray = np.asarray(acc_gravity)

        provided_params = [
            p
            for p in (
                damping_constant,
                uniform_damping_constant,
                translational_damping_constant,
                rotational_damping_constant,
            )
            if p is not None
        ]

        if len(provided_params) == 1 and damping_constant is not None:
            nodal_mass = _system.mass
            self._translational_damping_coefficient: float | JAXScalarArray = np.exp(
                -damping_constant * time_step
            )
            if _system.ring_rod_flag:
                element_mass = nodal_mass
            else:
                element_mass = 0.5 * (nodal_mass[1:] + nodal_mass[:-1])
                element_mass[0] += 0.5 * nodal_mass[0]
                element_mass[-1] += 0.5 * nodal_mass[-1]
            self._rotational_damping_coefficient: float | JAXScalarArray = np.exp(
                -damping_constant
                * time_step
                * element_mass
                * np.diagonal(_system.inv_mass_second_moment_of_inertia).T
            )
        elif len(provided_params) == 1 and uniform_damping_constant is not None:
            self._translational_damping_coefficient = np.exp(
                -uniform_damping_constant * time_step
            )
            self._rotational_damping_coefficient = (
                self._translational_damping_coefficient
            )
        elif (
            len(provided_params) == 2
            and translational_damping_constant is not None
            and rotational_damping_constant is not None
        ):
            nodal_mass = _system.mass
            self._translational_damping_coefficient = np.exp(
                -translational_damping_constant / nodal_mass * time_step
            )
            inv_moi = np.diagonal(_system.inv_mass_second_moment_of_inertia).T
            self._rotational_damping_coefficient = np.exp(
                -rotational_damping_constant * inv_moi * time_step
            )
        else:
            raise ValueError(
                "GravityAnalyticalDamperJax requires one valid AnalyticalLinearDamper "
                "parameterization."
            )

    def jax_operate_synchronize(
        self,
        rod_view: JAXRodView,
        time: JAXTime,
    ) -> JAXRodView:
        rod_view.external_forces = (
            rod_view.external_forces
            + self.acc_gravity[:, None] * rod_view.mass[None, :]
        )
        return rod_view

    def jax_operate_constrain_rates(
        self,
        rod_view: JAXRodView,
        time: JAXTime,
    ) -> JAXRodView:
        import jax.numpy as jnp

        rod_view.velocity_collection = (
            rod_view.velocity_collection * self._translational_damping_coefficient
        )
        rod_view.omega_collection = rod_view.omega_collection * jnp.power(
            jnp.asarray(self._rotational_damping_coefficient),
            rod_view.dilatation,
        )
        return rod_view

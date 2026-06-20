from __future__ import annotations

from typing import Any

import numpy as np


class NoOpsJax:
    """
    Empty template for pure-JAX rod-local operators.

    Implement any subset of:
    - `jax_operate_constrain_values(rod_view, time)`
    - `jax_operate_synchronize(rod_view, time)`
    - `jax_operate_constrain_rates(rod_view, time)`
    """

    def jax_operate_constrain_values(self, rod_view, time):
        return rod_view

    def jax_operate_synchronize(self, rod_view, time):
        return rod_view

    def jax_operate_constrain_rates(self, rod_view, time):
        return rod_view


class GravityAnalyticalDamperJax(NoOpsJax):
    """
    Example JAX operator combining gravity loading with analytical rate damping.
    """

    def __init__(
        self,
        *,
        acc_gravity: np.ndarray | None = None,
        time_step: float,
        uniform_damping_constant: float | None = None,
        damping_constant: float | None = None,
        translational_damping_constant: float | None = None,
        rotational_damping_constant: float | None = None,
        _system: Any,
    ) -> None:
        if acc_gravity is None:
            acc_gravity = np.array([0.0, -9.80665, 0.0], dtype=np.float64)
        self.acc_gravity = np.asarray(acc_gravity)

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
            self._translational_damping_coefficient = np.exp(
                -damping_constant * time_step
            )
            if _system.ring_rod_flag:
                element_mass = nodal_mass
            else:
                element_mass = 0.5 * (nodal_mass[1:] + nodal_mass[:-1])
                element_mass[0] += 0.5 * nodal_mass[0]
                element_mass[-1] += 0.5 * nodal_mass[-1]
            self._rotational_damping_coefficient = np.exp(
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

    def jax_operate_synchronize(self, rod_view, time):
        rod_view.external_forces = (
            rod_view.external_forces
            + self.acc_gravity[:, None] * rod_view.mass[None, :]
        )
        return rod_view

    def jax_operate_constrain_rates(self, rod_view, time):
        rod_view.velocity_collection = (
            rod_view.velocity_collection * self._translational_damping_coefficient
        )
        rod_view.omega_collection = rod_view.omega_collection * np.power(
            self._rotational_damping_coefficient,
            rod_view.dilatation,
        )
        return rod_view

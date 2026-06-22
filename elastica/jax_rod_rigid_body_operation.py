from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np

JAXTime: TypeAlias = np.float64


class NoRodRigidBodyJax:
    """
    User-facing base class for pure-JAX rod-to-rigid-body interaction operators.

    Implement any subset of:
    - `jax_operate_constrain_values(rod_view, rigid_body_view, time)`
    - `jax_operate_synchronize(rod_view, rigid_body_view, time)`
    - `jax_operate_constrain_rates(rod_view, rigid_body_view, time)`

    Notes
    -----
    Each stage receives one rod-local view and one rigid-body-local view. Mutate
    those views and return either:
    - `(rod_view, rigid_body_view)`, or
    - `None`, in which case the wrapper commits both mutated views.
    """

    def jax_operate_constrain_values(
        self,
        rod_view: Any,
        rigid_body_view: Any,
        time: JAXTime,
    ) -> tuple[Any, Any] | None:
        """Apply a mixed rod/rigid-body value constraint stage."""
        return rod_view, rigid_body_view

    def jax_operate_synchronize(
        self,
        rod_view: Any,
        rigid_body_view: Any,
        time: JAXTime,
    ) -> tuple[Any, Any] | None:
        """Apply a mixed rod/rigid-body synchronize stage."""
        return rod_view, rigid_body_view

    def jax_operate_constrain_rates(
        self,
        rod_view: Any,
        rigid_body_view: Any,
        time: JAXTime,
    ) -> tuple[Any, Any] | None:
        """Apply a mixed rod/rigid-body rate constraint stage."""
        return rod_view, rigid_body_view

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np

JAXTime: TypeAlias = np.float64
JAXBlockState: TypeAlias = dict[str, Any]


class NoBlockOpJax:
    """
    Empty template for pure-JAX block-wide operators.

    Implement any subset of:
    - `jax_block_operate_constrain_values(state, time)`
    - `jax_block_operate_synchronize(state, time)`
    - `jax_block_operate_constrain_rates(state, time)`
    """

    def jax_block_operate_constrain_values(
        self,
        state: JAXBlockState,
        time: JAXTime,
    ) -> JAXBlockState:
        return state

    def jax_block_operate_synchronize(
        self,
        state: JAXBlockState,
        time: JAXTime,
    ) -> JAXBlockState:
        return state

    def jax_block_operate_constrain_rates(
        self,
        state: JAXBlockState,
        time: JAXTime,
    ) -> JAXBlockState:
        return state

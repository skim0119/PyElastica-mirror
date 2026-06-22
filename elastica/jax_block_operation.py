from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np

JAXTime: TypeAlias = np.float64
JAXBlockState: TypeAlias = dict[str, Any]


class NoBlockOpJax:
    """
    User-facing base class for custom JAX block operations.

    `NoBlockOpJax` supports two authoring styles.

    1. Block-wide methods
       Operate directly on the full packed JAX block state. Use this when the
       operation is naturally expressed on the entire block, for example a
       uniform block-wide damping pass or a custom transform that does not need
       rod-local slicing semantics.

    2. Per-rod methods
       Operate on one rod-shaped view at a time, but are automatically batched
       across every rod in the block during `finalize()`. Use this when the
       logic is conceptually "same operator for every rod" and you want to keep
       rod-local syntax while still avoiding Python-side iteration during the
       JAX rollout.

    The simulator-side lowering logic decides how to execute the operator:

    - `jax_block_operate_*` methods are registered as block-native transforms.
    - `jax_per_rod_operate_*` methods are gathered, `vmap`'d across rods, and
      scattered back to the block state once per stage.

    Notes
    -----
    Do not implement both block-wide and per-rod variants for the same stage in
    one class. Choose one representation per stage.

    Implement any subset of:
    - `jax_block_operate_constrain_values(state, time)`
    - `jax_block_operate_synchronize(state, time)`
    - `jax_block_operate_constrain_rates(state, time)`
    - `jax_per_rod_operate_constrain_values(rod_view, time)`
    - `jax_per_rod_operate_synchronize(rod_view, time)`
    - `jax_per_rod_operate_constrain_rates(rod_view, time)`
    """

    def jax_block_operate_constrain_values(
        self,
        state: JAXBlockState,
        time: JAXTime,
    ) -> JAXBlockState:
        """Apply a block-wide value constraint stage."""
        return state

    def jax_block_operate_synchronize(
        self,
        state: JAXBlockState,
        time: JAXTime,
    ) -> JAXBlockState:
        """Apply a block-wide synchronize stage."""
        return state

    def jax_block_operate_constrain_rates(
        self,
        state: JAXBlockState,
        time: JAXTime,
    ) -> JAXBlockState:
        """Apply a block-wide rate constraint stage."""
        return state

    def jax_per_rod_operate_constrain_values(
        self,
        rod_view: Any,
        time: JAXTime,
    ) -> Any:
        """Apply a per-rod value constraint stage that will be batched with `vmap`."""
        return rod_view

    def jax_per_rod_operate_synchronize(
        self,
        rod_view: Any,
        time: JAXTime,
    ) -> Any:
        """Apply a per-rod synchronize stage that will be batched with `vmap`."""
        return rod_view

    def jax_per_rod_operate_constrain_rates(
        self,
        rod_view: Any,
        time: JAXTime,
    ) -> Any:
        """Apply a per-rod rate constraint stage that will be batched with `vmap`."""
        return rod_view

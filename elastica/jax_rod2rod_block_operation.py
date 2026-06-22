from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np

JAXTime: TypeAlias = np.float64
JAXBlockState: TypeAlias = dict[str, Any]


@dataclass(frozen=True)
class JAXRodRodBlockMetadata:
    """
    Pair metadata describing two rods embedded in one JAX memory block.

    Arrays are 2D with shape `(n_pairs, local_width)` so one operator instance can
    conceptually act on one or more rod pairs. The current scaffold uses this
    metadata to expose node-, element-, and voronoi-domain indices for each pair.
    """

    first_system_indices: np.ndarray
    second_system_indices: np.ndarray
    first_node_indices: np.ndarray
    second_node_indices: np.ndarray
    first_element_indices: np.ndarray
    second_element_indices: np.ndarray
    first_voronoi_indices: np.ndarray
    second_voronoi_indices: np.ndarray


class NoRodRodBlockOpJax:
    """
    User-facing base class for custom JAX rod-to-rod block operations.

    This API is intended for interactions such as rod-rod contact or custom pair
    constraints that conceptually operate on two rods at once, but should still be
    lowered onto a packed `MemoryBlockCosseratRodJax`.

    During `finalize()`, the mixin injects:

    - `_system`: the owning `MemoryBlockCosseratRodJax`
    - `_pair_metadata`: a `JAXRodRodBlockMetadata` instance describing the paired
      rod-local slices inside that block

    Implement:
    - `jax_rod2rod_operate_synchronize(state, time)`

    and return an updated block state dictionary.
    """

    def jax_rod2rod_operate_synchronize(
        self,
        state: JAXBlockState,
        time: JAXTime,
    ) -> JAXBlockState:
        """Apply a rod-to-rod synchronize-stage transform on packed block state."""
        return state

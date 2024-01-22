# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Mapping of global sample index to shard and relative sample index."""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


class Spanner:
    """Given a list of shards, construct a mapping of global index to shard and relative index.

    Args:
        shard_sizes (NDArray[np.int64]): Number of samples in each shard.
        span_size (int): Size of the divisions of the sample space. Defaults to ``1 << 10``.
    """

    def __init__(self, shard_sizes: NDArray[np.int64], span_size: int = 1 << 10) -> None:
        self.shard_sizes = shard_sizes
        self.span_size = span_size
        self.num_samples = sum(shard_sizes)
        self.shard_bounds = np.concatenate([np.zeros(1, np.int64), shard_sizes.cumsum()])

        overflow = self.num_samples % span_size
        underflow = span_size - overflow if overflow else 0
        self.shard_sizes[-1] += underflow

        sample_shards = np.repeat(np.arange(len(shard_sizes)), self.shard_sizes)
        sample_shards = sample_shards.reshape(-1, span_size)
        span_lowest_shards = sample_shards.min(1)
        span_highest_shards = sample_shards.max(1)

        self.spans = []
        for low, high in zip(span_lowest_shards, span_highest_shards):
            shards = np.arange(low, high + 1)
            self.spans.append(shards)

        self.shard_sizes[-1] -= underflow

    def __getitem__(self, index: int) -> Tuple[int, int]:
        """Map global sample index to shard and relative sample index.

        Args:
            index (int): Global sample index.

        Returns:
            Tuple[int, int]: Shard and relative sample index.
        """
        if not (0 <= index < self.num_samples):
            raise ValueError(f'Invalid sample index `{index}`: 0 <= {index} < {self.num_samples}')

        span = index // self.span_size
        for shard in self.spans[span]:
            shard_start = self.shard_bounds[shard]
            shard_stop = self.shard_bounds[shard + 1]
            if shard_start <= index < shard_stop:
                return shard, int(index - shard_start)

        raise RuntimeError('Internal error: shards were indexed incorrectly')


class MegaSpanner:
    def __init__(self, all_shard_sizes: list[list[NDArray[np.int64]]],
                 dummy_shards: set[int], span_size: int = 1 << 10, ) -> None:
        spanner_levels = max(len(sizes) for sizes in all_shard_sizes)
        # spanner_size_sum = [sizes[0].sum() for sizes in all_shard_sizes]
        #  if len(sizes) < spanner_level else [spanner_size_sum]
        self._spanners = [
            Spanner(
                np.concatenate([sizes[spanner_level] for sizes in all_shard_sizes],
                               axis=0),
                span_size=span_size,
            )
            for spanner_level in range(spanner_levels)
        ]
        self._shard_offsets = [0] + [
            sum(len(sizes[spanner_level]) for sizes in all_shard_sizes)
            for spanner_level in range(spanner_levels)
        ][:-1]
        self._dummy_shards = dummy_shards

    def __getitem__(self, index: int) -> Tuple[int, int]:
        all_items = []
        for spanner_level, spanner in enumerate(self._spanners):
            shard_id, shard_sample_id = spanner[index]
            true_shard_id = shard_id + self._shard_offsets[spanner_level]
            if true_shard_id in self._dummy_shards:
                continue
            all_items.append((true_shard_id, shard_sample_id))
        return all_items


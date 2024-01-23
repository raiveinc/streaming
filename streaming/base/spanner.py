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



class GroupSpanner:
    # We can pass the total number of sample in order to gain some compute time
    def __init__(self, group_shard_sizes: list[NDArray[np.int64]], span_size: int = 1 << 10) -> None:
        self.group_shard_sizes = group_shard_sizes
        self.span_size = span_size
        self.group_bounds = np.array([sum(shard_sizes[0]) for shard_sizes in group_shard_sizes], np.int64)
        self.group_sample_offest = np.concatenate([np.zeros(1, np.int64), self.group_bounds.cumsum()])
        print("GROUP SAMPLE OFFSET", self.group_sample_offest)
        self.num_samples = sum(self.group_bounds)
        # As there might not be many groups, this might be quite useless...
        self.global_group_spanner = Spanner(self.group_bounds, span_size)
        # Is this an abomination? Yes it is
        self.group_spanner = np.empty(len(group_shard_sizes), dtype=object)
        for group_id, group in enumerate(group_shard_sizes):
            self.group_spanner[group_id] = np.empty(len(group), dtype=object)
            for shard_list_id, shard_list in enumerate(group):
                self.group_spanner[group_id][shard_list_id] = Spanner(shard_list, span_size)


        # Contains the number of shards len
        self.shards_per_group = np.empty(len(group_shard_sizes), np.int64)
        for group_id, shard_sizes in enumerate(group_shard_sizes):
            total = 0
            for shard_size in shard_sizes:
                total += len(shard_size)
            self.shards_per_group[group_id] = total
        self.shards_offset_per_group = np.concatenate([np.zeros(1, np.int64), self.shards_per_group.cumsum()])

    def __getitem__(self, index: int) -> Tuple[int, int]:
        group_id, _ = self.global_group_spanner[index]
        # TODO; This might be totally unoptimized as there is a few number of groups
        shard_offset = self.shards_offset_per_group[group_id]
        ret = []
        group_offset = 0
        group_spanners = self.group_spanner[group_id]
        sample_offset = self.group_sample_offest[group_id]

        spanner_index = index - sample_offset
        for group_id, spanner in enumerate(group_spanners):
            shard_id, index_in_shard = spanner[spanner_index]
            absolute_shard_id = group_offset + shard_id + shard_offset
            # TODO: If there is an ugly bug check the index in shard
            ret.append((absolute_shard_id, index_in_shard))

        if len(ret) > 0:
            return ret
        raise RuntimeError('Internal error: shards were indexed incorrectly')


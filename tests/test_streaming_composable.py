# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import math
import os
from typing import Tuple

import pytest

from streaming.base import ComposableStream, Source, StreamingDataLoader, StreamingDataset
from tests.common.utils import convert_to_mds


@pytest.mark.parametrize('batch_size', [4])
@pytest.mark.parametrize('seed', [2222])
@pytest.mark.parametrize('shuffle', [True])
@pytest.mark.parametrize('num_workers', [6])
@pytest.mark.parametrize('num_canonical_nodes', [8])
@pytest.mark.usefixtures('local_remote_dir')
@pytest.mark.parametrize('batching_method', ['per_stream', 'random'])
def test_dataloader_per_composable_stream_batching(local_remote_dir: Tuple[str,
                                                                           str], batch_size: int,
                                                   seed: int, shuffle: bool, num_workers: int,
                                                   num_canonical_nodes: int, batching_method: str):
    local, remote = local_remote_dir
    local1 = os.path.join(local, 'stream1')
    local2 = os.path.join(local, 'stream2')
    remote1 = os.path.join(remote, 'stream1')
    remote2 = os.path.join(remote, 'stream2')

    local3 = os.path.join(local, 'stream3')
    local4 = os.path.join(local, 'stream4')
    remote3 = os.path.join(remote, 'stream3')
    remote4 = os.path.join(remote, 'stream4')

    stream1_size = 200
    stream2_size = 800

    convert_to_mds(out_root=remote1,
                   dataset_name='sequencedataset',
                   num_samples=stream1_size,
                   size_limit=1 << 8)
    convert_to_mds(out_root=remote2,
                   dataset_name='numberandsaydataset',
                   num_samples=stream1_size,
                   size_limit=1 << 8)

    convert_to_mds(out_root=remote3,
                   dataset_name='sequencedataset',
                   num_samples=stream2_size,
                   size_limit=1 << 8)
    convert_to_mds(out_root=remote4,
                   dataset_name='numberandsaydataset',
                   num_samples=stream2_size,
                   size_limit=1 << 8)

    streams = [
        ComposableStream(sources=[
            Source(local=local1, remote=remote1),
            Source(local=local2, remote=remote2),
        ]),
        ComposableStream(
            sources=[Source(local=local3, remote=remote3),
                     Source(local=local4, remote=remote4)]),
    ]

    # Build StreamingDataset
    dataset = StreamingDataset(streams=streams,
                               shuffle=shuffle,
                               batch_size=batch_size,
                               shuffle_seed=seed,
                               num_canonical_nodes=num_canonical_nodes,
                               batching_method=batching_method)

    # Build DataLoader
    dataloader = StreamingDataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers)

    total_batches_stream_1 = stream1_size // batch_size if stream1_size % num_canonical_nodes == 0 else (
        stream1_size + (num_canonical_nodes - stream1_size % num_canonical_nodes)) // batch_size
    total_batches_stream_2 = stream2_size // batch_size if stream2_size % num_canonical_nodes == 0 else (
        stream2_size + (num_canonical_nodes - stream2_size % num_canonical_nodes)) // batch_size
    total_batches = total_batches_stream_1 + total_batches_stream_2

    batches_seen = 0
    for _ in dataloader:
        batches_seen += 1

    assert batches_seen == total_batches


@pytest.mark.parametrize('batch_size', [4])
@pytest.mark.parametrize('seed', [2222])
@pytest.mark.parametrize('shuffle', [True])
@pytest.mark.parametrize('num_workers', [6])
@pytest.mark.parametrize('num_canonical_nodes', [8])
@pytest.mark.usefixtures('local_remote_dir')
@pytest.mark.parametrize('batching_method', ['stratified'])
def test_dataloader_per_composable_stream_batching_stratified(local_remote_dir: Tuple[str, str],
                                                              batch_size: int, seed: int,
                                                              shuffle: bool, num_workers: int,
                                                              num_canonical_nodes: int,
                                                              batching_method: str):
    local, remote = local_remote_dir
    local1 = os.path.join(local, 'stream1')
    local2 = os.path.join(local, 'stream2')
    remote1 = os.path.join(remote, 'stream1')
    remote2 = os.path.join(remote, 'stream2')

    local3 = os.path.join(local, 'stream3')
    local4 = os.path.join(local, 'stream4')
    remote3 = os.path.join(remote, 'stream3')
    remote4 = os.path.join(remote, 'stream4')

    num_stream_1_samples = 200
    num_stream_2_samples = 800

    convert_to_mds(out_root=remote1,
                   dataset_name='sequencedataset',
                   num_samples=num_stream_1_samples,
                   size_limit=1 << 8)
    convert_to_mds(out_root=remote2,
                   dataset_name='numberandsaydataset',
                   num_samples=num_stream_1_samples,
                   size_limit=1 << 8)

    convert_to_mds(out_root=remote3,
                   dataset_name='sequencedataset',
                   num_samples=num_stream_2_samples,
                   size_limit=1 << 8)
    convert_to_mds(out_root=remote4,
                   dataset_name='numberandsaydataset',
                   num_samples=num_stream_2_samples,
                   size_limit=1 << 8)

    streams = [
        ComposableStream(sources=[
            Source(local=local1, remote=remote1),
            Source(local=local2, remote=remote2),
        ]),
        ComposableStream(
            sources=[Source(local=local3, remote=remote3),
                     Source(local=local4, remote=remote4)]),
    ]

    # Build StreamingDataset
    dataset = StreamingDataset(streams=streams,
                               shuffle=shuffle,
                               batch_size=batch_size,
                               shuffle_seed=seed,
                               num_canonical_nodes=num_canonical_nodes,
                               batching_method=batching_method)

    # Build DataLoader
    dataloader = StreamingDataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers)

    # Ensure that the samples seen in each batch are proportional to the stream sizes.
    total_samples = num_stream_1_samples + num_stream_2_samples
    stream_1_batch_part = round(batch_size * (num_stream_1_samples / total_samples))
    stream_2_batch_part = batch_size - stream_1_batch_part

    # The total number of possible batches is the minimum of the batch parts from each stream.
    # Total number of samples will be padded to be divisible by NCN.
    total_stream_1_batches = num_stream_1_samples // stream_1_batch_part \
        if num_stream_1_samples % num_canonical_nodes == 0 else (
        num_stream_1_samples +
        (num_canonical_nodes - num_stream_1_samples % num_canonical_nodes)) // stream_1_batch_part
    total_stream_2_batches = num_stream_2_samples // stream_2_batch_part \
        if num_stream_2_samples % num_canonical_nodes == 0 else (
        num_stream_2_samples +
        (num_canonical_nodes - num_stream_2_samples % num_canonical_nodes)) // stream_2_batch_part
    total_batches = min(total_stream_1_batches, total_stream_2_batches)
    batches_seen = 0
    for _ in dataloader:
        batches_seen += 1

    print(
        f"batches_seen: {batches_seen}, total_batches: {total_batches}, dataloader len: {len(dataloader)}"
    )
    assert batches_seen == total_batches

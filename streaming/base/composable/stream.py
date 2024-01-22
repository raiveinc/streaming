# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray
from streaming.base.format.base.reader import Reader
from streaming.base.composable import ComposableReader
from streaming.base.stream import Base, Stream
from streaming.base.world import World


@dataclass
class Source(object):
    """Stream sources.

    Args:
        remote (str, optional): Remote path or directory to download the dataset from. If ``None``,
            its data must exist locally. Defaults to ``None``.
        local (str, optional): Local working directory to download shards to. This is where shards
            are cached while they are being used. Uses a temp directory if not set. Defaults to
            ``None``.
    """
    remote: Optional[str] = None
    local: Optional[str] = None


class ComposableStream(Base):

    def __init__(self,
                 *,
                 sources: List[Source] = [],
                 split: Optional[str] = None,
                 proportion: Optional[float] = None,
                 repeat: Optional[float] = None,
                 choose: Optional[int] = None,
                 download_retry: Optional[int] = None,
                 download_timeout: Optional[float] = None,
                 validate_hash: Optional[str] = None,
                 keep_zip: Optional[bool] = None) -> None:

        super().__init__(split=split,
                         proportion=proportion,
                         repeat=repeat,
                         choose=choose,
                         download_retry=download_retry,
                         download_timeout=download_timeout,
                         validate_hash=validate_hash,
                         keep_zip=keep_zip)
        self.streams = {}
        for key, source in enumerate(sources):
            self.streams[key] = Stream(remote=source.remote,
                                       local=source.local,
                                       split=split,
                                       proportion=proportion,
                                       repeat=repeat,
                                       choose=choose,
                                       download_retry=download_retry,
                                       download_timeout=download_timeout,
                                       validate_hash=validate_hash,
                                       keep_zip=keep_zip)

    def apply_default(self, default: dict) -> None:
        for stream in self.streams.values():
            stream.apply_default(default)

    # def prepare_shard(self, composable_shard: ComposableReader) -> int:
    #     delta = 0
    #     for key, stream in self.streams.items():
    #         shard = composable_shard.get_shard(key)
    #         delta += stream.prepare_shard(shard)
    #     return delta

    def get_shards(self, world: World, allow_unsafe_types: bool,) -> list[Reader]:
        return [stream.get_shards(world, allow_unsafe_types) for stream in self.streams.values()]

    def get_listing(self) -> Dict[int, List[str]]:
        listing = {}
        for key, stream in self.streams.items():
            listing[key] = stream.get_listing()
        return listing

    def _get_safe_keep_zip(self) -> Dict[int, bool]:
        safe_keep_zip = {}
        for key, stream in self.streams.items():
            safe_keep_zip[key] = stream.safe_keep_zip
        return safe_keep_zip

    # def set_up_local(self, shards: List[ComposableReader],
    #                  cache_usage_per_shard: NDArray[np.int64]) -> None:
    #     composable_listing = self.get_listing()
    #     composable_safe_keep_zip = self._get_safe_keep_zip()
    #     for i, composable_shard in enumerate(shards):
    #         cache_usage_per_shard[i] = composable_shard.set_up_local(composable_listing,
    #                                                                  composable_safe_keep_zip)

    def get_index_size(self) -> int:
        st_size = 0
        for stream in self.streams.values():
            st_size += stream.get_index_size()
        return st_size

    def stream_local(self) -> List[str]:
        local = []
        for stream in self.streams.values():
            local.append(stream.stream_local())
        return local

    def stream_remote(self) -> List[Optional[str]]:
        remote = []
        for stream in self.streams.values():
            remote.append(stream.stream_remote())
        return remote

# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import List, Optional

from streaming.base.stream import Stream


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


class ComposableStream:

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

        self.streams = [
            Stream(remote=source.remote,
                   local=source.local,
                   split=split,
                   proportion=proportion,
                   repeat=repeat,
                   choose=choose,
                   download_retry=download_retry,
                   download_timeout=download_timeout,
                   validate_hash=validate_hash,
                   keep_zip=keep_zip) for source in sources
        ]

    def list(self) -> list[Stream]:
        return self.streams

    def __len__(self) -> int:
        return len(self.streams)

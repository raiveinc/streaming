# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Set

from streaming.base.format import BaseReader, Reader


class ComposableReader(BaseReader):
    """ComposableReader are multiple Reader objects composed together.

    This reader is composed of multiple shard readers, they must all be in sync.
    Each shard reader is responsible for reading a specific shard of data.

    Args:
        samples (int): The number of samples to read.
        shard_dict (Dict[int, Reader]): A dictionary mapping shard keys to their respective readers.
    """

    def __init__(self, *, samples: int, shard_dict: Dict[int, Reader]) -> None:
        super().__init__(samples=samples)
        self.shard_dict = shard_dict

    def get_shard(self, key: int) -> Reader:
        """Returns the reader for the given shard key.

        Parameters:
            key (int): The shard key.

        Returns:
            Reader: The reader for the given shard key.
        """
        return self.shard_dict[key]

    def is_raw_file_present(self) -> bool:
        """Checks if the raw file is present for all shards.

        Returns:
            bool: True if the raw file is present for all shards, False otherwise.
        """
        for shard in self.shard_dict.values():
            if not shard.is_raw_file_present():
                return False
        return True

    def set_up_local(self, listing: Dict[int, Set[str]], safe_keep_zip: Dict[int, bool]) -> int:
        """Bring what shard files are present to a consistent state, returning whether present for
        each reader.

        Parameters:
            listing (Dict[int, Set[str]]): Mapping shard keys to their respective listings.
            safe_keep_zip (Dict[int, bool]): Mapping shard keys to stream's safe_keep_zip.

        Returns:
            int: The total size of all shards after setup.
        """
        size = 0
        for key, shard in self.shard_dict.items():
            size += shard.set_up_local(listing[key], safe_keep_zip[key])
        return size

    def evict(self) -> None:
        """Evicts all shards."""
        for shard in self.shard_dict.values():
            shard.evict()

    def get_raw_size(self) -> int:
        """Returns the total raw size of all shards.

        Returns:
            int: The total raw size of all shards.
        """
        size = 0
        for shard in self.shard_dict.values():
            size += shard.get_raw_size()
        return size

    def get_zip_size(self) -> int:
        """Returns the total zip size of all shards.

        Returns:
            int: The total zip size of all shards.
        """
        size = 0
        for shard in self.shard_dict.values():
            size += shard.get_zip_size()
        return size

    def get_max_size(self) -> int:
        """Returns the maximum size among all shards.

        Returns:
            int: The maximum size among all shards.
        """
        size = 0
        for shard in self.shard_dict.values():
            size += shard.get_max_size()
        return size

    def get_persistent_size(self, keep_zip: bool) -> int:
        size = 0
        for shard in self.shard_dict.values():
            size += shard.get_persistent_size(keep_zip)
        return size

    def validate(self, allow_unsafe_types: bool) -> None:
        for shard in self.shard_dict.values():
            shard.validate(allow_unsafe_types)

    def decode_sample(self, data: List[bytes]) -> Dict[str, Any]:
        ret = {}
        for index, shard in enumerate(self.shard_dict.values()):
            ret.update(shard.decode_sample(data[index]))
        return ret

    def get_sample_data(self, idx: int) -> List[bytes]:
        bytes_list = []
        for shard in self.shard_dict.values():
            bytes_list.append(shard.get_sample_data(idx))
        return bytes_list

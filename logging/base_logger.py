from __future__ import annotations
from abc import ABC, abstractmethod


class BaseLogger(ABC):
    def __init__(self, log_per_batch: bool, log_per_epoch: bool):
        self.log_per_batch = log_per_batch
        self.log_per_epoch = log_per_epoch

    @abstractmethod
    def log_batch(self, step: int, flops: int, cumulative_flops: int, epoch: int | None = None):
        ...

    @abstractmethod
    def log_epoch(self, epoch: int, flops: int, cumulative_flops: int):
        ...

    @abstractmethod
    def close(self):
        ...

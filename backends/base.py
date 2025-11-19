from abc import ABC, abstractmethod


class BaseBackend(ABC):
    def __init__(self, model, logger=None):
        self.model = model
        self.logger = logger
        self.total_flops: int = 0
        self._last_batch_flops: int = 0
        self._batch_idx: int = 0
        self._epoch_idx: int = 0

    @abstractmethod
    def start(self):
        """Aggancia gli hook"""
        ...

    @abstractmethod
    def stop(self):
        """Rimuove gli hook"""
        ...

    def get_total_flops(self) -> int:
        return int(self.total_flops)

    def get_last_batch_flops(self) -> int:
        return int(self._last_batch_flops)

    def set_epoch(self, epoch: int):
        """Opzionale: permette di tracciare l'epoch corrente nei log."""
        self._epoch_idx = epoch

from .base import BaseBackend
from .torch_backend import TorchBackend


def create_backend(model, backend: str, logger=None) -> BaseBackend:
    """
    Se backend='auto' prova a riconoscere il tipo di modello.
    Per ora gestiamo solo PyTorch.
    """
    if backend in ("torch", "auto"):
        try:
            import torch
            from torch.nn import Module
            if isinstance(model, Module):
                return TorchBackend(model, logger=logger)
        except ImportError:
            if backend == "torch":
                raise

    raise ValueError(f"Impossibile determinare il backend per il modello: {type(model)}")

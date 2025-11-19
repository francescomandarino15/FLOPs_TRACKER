import torch
import torch.nn as nn

from .base import BaseBackend


class TorchBackend(BaseBackend):
    """
    Backend per modelli PyTorch.

    - Gestisce DataParallel / DDP (usa .module)
    - Conta i FLOPs di Conv2d e Linear
    - Logga automaticamente per batch tramite logger (se presente)
    """

    def __init__(self, model: nn.Module, logger=None):
        # DataParallel 
        if isinstance(model, (nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model = model.module

        super().__init__(model, logger=logger)
        self._layer_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._root_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._current_batch_flops: int = 0

    def start(self):
        # Hook sui layer che vogliamo tracciare
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                h = module.register_forward_hook(self._layer_hook)
                self._layer_handles.append(h)
            # TODO: aggiungere qui Pooling, Normalization, RNN, Transformer, ecc.

        # Hook sul modello root per identificare inizio/fine batch
        pre_h = self.model.register_forward_pre_hook(self._on_batch_start)
        post_h = self.model.register_forward_hook(self._on_batch_end)
        self._root_handles.extend([pre_h, post_h])

    def stop(self):
        for h in self._layer_handles:
            h.remove()
        for h in self._root_handles:
            h.remove()
        self._layer_handles.clear()
        self._root_handles.clear()

    # ---------- Hook di supporto ---------- #

    def _on_batch_start(self, module, input):
        # inizio di un nuovo batch: azzero conteggio locale
        self._current_batch_flops = 0

    def _on_batch_end(self, module, input, output):
        # fine batch: aggiorno contatori e loggo
        batch_flops = self._current_batch_flops
        self._last_batch_flops = batch_flops
        self.total_flops += batch_flops
        self._batch_idx += 1

        if self.logger is not None and hasattr(self.logger, "log_batch"):
            self.logger.log_batch(
                step=self._batch_idx,
                flops=batch_flops,
                cumulative_flops=self.total_flops,
                epoch=self._epoch_idx,
            )

    def _layer_hook(self, layer, input, output):
        x = input[0]
        y = output

        if isinstance(layer, nn.Conv2d):
            flops = self._conv2d_flops(layer, x, y)
        elif isinstance(layer, nn.Linear):
            flops = self._linear_flops(layer, x, y)
        else:
            flops = 0

        self._current_batch_flops += flops

    # ---------- Formule FLOPs ---------- #

    def _conv2d_flops(self, conv: nn.Conv2d, x, y):
        """
        FLOPs Conv2d:
        per elemento di output: 2 * (C_in / groups) * K_h * K_w
        Totale = batch * H_out * W_out * C_out * 2 * (C_in/groups) * K_h * K_w
        """
        batch_size = x.shape[0]
        C_in = conv.in_channels
        C_out = conv.out_channels
        K_h, K_w = conv.kernel_size
        H_out, W_out = y.shape[2], y.shape[3]
        groups = conv.groups

        flops_per_out = 2 * (C_in // groups) * K_h * K_w
        num_out_elements = batch_size * C_out * H_out * W_out
        return int(flops_per_out * num_out_elements)

    def _linear_flops(self, linear: nn.Linear, x, y):
        """
        FLOPs Linear:
        ~ 2 * in_features * out_features per sample
        """
        batch_size = x.shape[0]
        in_f = linear.in_features
        out_f = linear.out_features
        return int(batch_size * 2 * in_f * out_f)

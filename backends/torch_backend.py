import torch
import torch.nn as nn

from .base import BaseBackend


class TorchBackend(BaseBackend):
    """
    Backend per modelli PyTorch.
    - Gestisce DataParallel / DDP (usa .module)
    - Conta i FLOPs di:
        * Conv1d / Conv2d / Conv3d
        * ConvTranspose1d / 2d / 3d
        * Linear
        * Pooling (Max/Avg/Adaptive)
        * Normalization (BatchNorm, LayerNorm, GroupNorm, InstanceNorm)
        * RNN / LSTM / GRU
        * MultiheadAttention
        * Embedding / EmbeddingBag
    - Logga per batch tramite logger (se presente)
    """

    def __init__(self, model: nn.Module, logger=None):
        # DataParallel 
        if isinstance(model, (nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model = model.module

        super().__init__(model, logger=logger)
        self._layer_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._root_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._current_batch_flops: int = 0

    # ---------------- START / STOP ---------------- #

    def start(self):
        # Hook sui layer che vogliamo tracciare
        for module in self.model.modules():
            if isinstance(
                module,
                (
                    nn.Conv1d,
                    nn.Conv2d,
                    nn.Conv3d,
                    nn.ConvTranspose1d,
                    nn.ConvTranspose2d,
                    nn.ConvTranspose3d,
                    nn.Linear,
                    nn.MaxPool1d,
                    nn.MaxPool2d,
                    nn.MaxPool3d,
                    nn.AvgPool1d,
                    nn.AvgPool2d,
                    nn.AvgPool3d,
                    nn.AdaptiveAvgPool1d,
                    nn.AdaptiveAvgPool2d,
                    nn.AdaptiveAvgPool3d,
                    nn.AdaptiveMaxPool1d,
                    nn.AdaptiveMaxPool2d,
                    nn.AdaptiveMaxPool3d,
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                    nn.LayerNorm,
                    nn.GroupNorm,
                    nn.InstanceNorm1d,
                    nn.InstanceNorm2d,
                    nn.InstanceNorm3d,
                    nn.RNN,
                    nn.LSTM,
                    nn.GRU,
                    nn.MultiheadAttention,
                    nn.Embedding,
                    nn.EmbeddingBag,
                ),
            ):
                h = module.register_forward_hook(self._layer_hook)
                self._layer_handles.append(h)
            # NOTA: Dropout, PixelShuffle, Padding ecc. hanno FLOPs ~0 → li ignoriamo.

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

    # ---------------- HOOK DI BATCH ---------------- #

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

    # ---------------- HOOK DEI LAYER ---------------- #

    def _layer_hook(self, layer, input, output):
        x = input[0]
        y = output

        flops = 0

        # --- CONV --- #
        if isinstance(layer, nn.Conv1d):
            flops = self._conv1d_flops(layer, x, y)
        elif isinstance(layer, nn.Conv2d):
            flops = self._conv2d_flops(layer, x, y)
        elif isinstance(layer, nn.Conv3d):
            flops = self._conv3d_flops(layer, x, y)
        elif isinstance(layer, nn.ConvTranspose1d):
            flops = self._convtranspose1d_flops(layer, x, y)
        elif isinstance(layer, nn.ConvTranspose2d):
            flops = self._convtranspose2d_flops(layer, x, y)
        elif isinstance(layer, nn.ConvTranspose3d):
            flops = self._convtranspose3d_flops(layer, x, y)

        # --- LINEAR --- #
        elif isinstance(layer, nn.Linear):
            flops = self._linear_flops(layer, x, y)

        # --- POOLING --- #
        elif isinstance(layer, (nn.MaxPool1d, nn.AvgPool1d)):
            flops = self._pool1d_flops(layer, x, y)
        elif isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
            flops = self._pool2d_flops(layer, x, y)
        elif isinstance(layer, (nn.MaxPool3d, nn.AvgPool3d, nn.AdaptiveAvgPool3d, nn.AdaptiveMaxPool3d)):
            flops = self._pool3d_flops(layer, x, y)
        elif isinstance(layer, (nn.AdaptiveAvgPool1d, nn.AdaptiveMaxPool1d)):
            flops = self._pool1d_flops(layer, x, y)
        elif isinstance(layer, (nn.AdaptiveAvgPool3d, nn.AdaptiveMaxPool3d)):
            flops = self._pool3d_flops(layer, x, y)

        # --- NORMALIZATION --- #
        elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            flops = self._batchnorm_flops(layer, x, y)
        elif isinstance(layer, nn.LayerNorm):
            flops = self._layernorm_flops(layer, x, y)
        elif isinstance(layer, nn.GroupNorm):
            flops = self._groupnorm_flops(layer, x, y)
        elif isinstance(layer, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            flops = self._instancenorm_flops(layer, x, y)

        # --- RNN / LSTM / GRU --- #
        elif isinstance(layer, (nn.RNN, nn.LSTM, nn.GRU)):
            flops = self._rnn_flops(layer, x, y)

        # --- MULTIHEAD ATTENTION --- #
        elif isinstance(layer, nn.MultiheadAttention):
            flops = self._mha_flops(layer, input, output)

        # --- EMBEDDING --- #
        elif isinstance(layer, nn.Embedding):
            flops = self._embedding_flops(layer, x, y)
        elif isinstance(layer, nn.EmbeddingBag):
            flops = self._embeddingbag_flops(layer, x, y)

        # Dropout, PixelShuffle, Padding, Shuffle ecc. -> flops ~ 0, ignorati

        self._current_batch_flops += int(flops)

    # ---------------- FORMULE FLOPs ---------------- #
    # Conv

    def _conv1d_flops(self, conv: nn.Conv1d, x, y):
        batch_size = x.shape[0]
        C_in = conv.in_channels
        C_out = conv.out_channels
        K = conv.kernel_size[0]
        L_out = y.shape[2]
        groups = conv.groups
        flops_per_out = 2 * (C_in // groups) * K
        num_out_elements = batch_size * C_out * L_out
        return flops_per_out * num_out_elements

    def _conv2d_flops(self, conv: nn.Conv2d, x, y):
        batch_size = x.shape[0]
        C_in = conv.in_channels
        C_out = conv.out_channels
        K_h, K_w = conv.kernel_size
        H_out, W_out = y.shape[2], y.shape[3]
        groups = conv.groups
        flops_per_out = 2 * (C_in // groups) * K_h * K_w
        num_out_elements = batch_size * C_out * H_out * W_out
        return flops_per_out * num_out_elements

    def _conv3d_flops(self, conv: nn.Conv3d, x, y):
        batch_size = x.shape[0]
        C_in = conv.in_channels
        C_out = conv.out_channels
        K_d, K_h, K_w = conv.kernel_size
        D_out, H_out, W_out = y.shape[2], y.shape[3], y.shape[4]
        groups = conv.groups
        flops_per_out = 2 * (C_in // groups) * K_d * K_h * K_w
        num_out_elements = batch_size * C_out * D_out * H_out * W_out
        return flops_per_out * num_out_elements

    def _convtranspose1d_flops(self, conv: nn.ConvTranspose1d, x, y):
        # formula analoga alla conv
        batch_size = x.shape[0]
        C_in = conv.in_channels
        C_out = conv.out_channels
        K = conv.kernel_size[0]
        L_out = y.shape[2]
        groups = conv.groups
        flops_per_out = 2 * (C_in // groups) * K
        num_out_elements = batch_size * C_out * L_out
        return flops_per_out * num_out_elements

    def _convtranspose2d_flops(self, conv: nn.ConvTranspose2d, x, y):
        batch_size = x.shape[0]
        C_in = conv.in_channels
        C_out = conv.out_channels
        K_h, K_w = conv.kernel_size
        H_out, W_out = y.shape[2], y.shape[3]
        groups = conv.groups
        flops_per_out = 2 * (C_in // groups) * K_h * K_w
        num_out_elements = batch_size * C_out * H_out * W_out
        return flops_per_out * num_out_elements

    def _convtranspose3d_flops(self, conv: nn.ConvTranspose3d, x, y):
        batch_size = x.shape[0]
        C_in = conv.in_channels
        C_out = conv.out_channels
        K_d, K_h, K_w = conv.kernel_size
        D_out, H_out, W_out = y.shape[2], y.shape[3], y.shape[4]
        groups = conv.groups
        flops_per_out = 2 * (C_in // groups) * K_d * K_h * K_w
        num_out_elements = batch_size * C_out * D_out * H_out * W_out
        return flops_per_out * num_out_elements

    # Linear

    def _linear_flops(self, linear: nn.Linear, x, y):
        # supponiamo input shape (batch_size, in_features)
        batch_size = x.shape[0]
        in_f = linear.in_features
        out_f = linear.out_features
        return batch_size * 2 * in_f * out_f

    # Pooling 

    def _pool1d_flops(self, layer, x, y):
        batch_size, C, L_out = y.shape
        if hasattr(layer, "kernel_size"):
            k = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
        else:
            # Adaptive pool: kernel implicito
            L_in = x.shape[2]
            k = L_in // L_out if L_out > 0 else 1
        # MaxPool ~ (k-1) confronti, AvgPool ~ k somme
        k_eff = max(k, 1)
        flops_per_out = k_eff
        return batch_size * C * L_out * flops_per_out

    def _pool2d_flops(self, layer, x, y):
        batch_size, C, H_out, W_out = y.shape
        if hasattr(layer, "kernel_size"):
            if isinstance(layer.kernel_size, int):
                K_h = K_w = layer.kernel_size
            else:
                K_h, K_w = layer.kernel_size
        else:
            # Adaptive pool: kernel implicito
            H_in, W_in = x.shape[2], x.shape[3]
            K_h = max(H_in // H_out, 1)
            K_w = max(W_in // W_out, 1)
        k_eff = max(K_h * K_w, 1)
        flops_per_out = k_eff
        return batch_size * C * H_out * W_out * flops_per_out

    def _pool3d_flops(self, layer, x, y):
        batch_size, C, D_out, H_out, W_out = y.shape
        if hasattr(layer, "kernel_size"):
            ks = layer.kernel_size
            if isinstance(ks, int):
                K_d = K_h = K_w = ks
            else:
                K_d, K_h, K_w = ks
        else:
            D_in, H_in, W_in = x.shape[2], x.shape[3], x.shape[4]
            K_d = max(D_in // D_out, 1)
            K_h = max(H_in // H_out, 1)
            K_w = max(W_in // W_out, 1)
        k_eff = max(K_d * K_h * K_w, 1)
        flops_per_out = k_eff
        return batch_size * C * D_out * H_out * W_out * flops_per_out

    # Normalization (stima: ~4 FLOPs per elemento)

    def _batchnorm_flops(self, layer, x, y):
        # assumiamo che y abbia stessa shape di x
        num_elements = y.numel()
        return 4 * num_elements

    def _layernorm_flops(self, layer, x, y):
        num_elements = y.numel()
        return 4 * num_elements

    def _groupnorm_flops(self, layer, x, y):
        num_elements = y.numel()
        return 4 * num_elements

    def _instancenorm_flops(self, layer, x, y):
        num_elements = y.numel()
        return 4 * num_elements

    # RNN / LSTM / GRU (stima classica per gate)

    def _rnn_flops(self, layer, x, y):
        # x shape: (seq_len, batch, input_size) o (batch, seq_len, input_size)
        batch_first = getattr(layer, "batch_first", False)
        if batch_first:
            batch_size, seq_len, input_size = x.shape
        else:
            seq_len, batch_size, input_size = x.shape

        hidden_size = layer.hidden_size
        num_layers = layer.num_layers
        num_directions = 2 if layer.bidirectional else 1

        # per gate: Wx (in*hid) + Wh (hid*hid) -> 2*(in*hid + hid*hid) FLOPs
        if isinstance(layer, nn.LSTM):
            num_gates = 4
        elif isinstance(layer, nn.GRU):
            num_gates = 3
        else:  # nn.RNN
            num_gates = 1

        flops_per_timestep = 2 * num_gates * (input_size * hidden_size + hidden_size * hidden_size)
        timesteps = seq_len * num_layers * num_directions
        return batch_size * timesteps * flops_per_timestep

    # MultiheadAttention (stima semplificata)

    def _mha_flops(self, layer: nn.MultiheadAttention, input, output):
        # input: (q, k, v, ...) tipicamente (L, N, E) / (S, N, E)
        q = input[0]
        k = input[1] if len(input) > 1 and input[1] is not None else q
        v = input[2] if len(input) > 2 and input[2] is not None else q

        # L: lunghezza query, S: lunghezza key/value, N: batch, E: embed_dim
        L, N, E = q.shape
        S = k.shape[0]

        num_heads = layer.num_heads
        d_k = E // num_heads

        # Q, K, V projection: ~3 * (2 * E * E * L * N)
        flops_qkv = 3 * 2 * E * E * L * N

        # attention scores: Q_h K_h^T per head: 2*L*S*d_k
        flops_scores = num_heads * 2 * L * S * d_k

        # attention * V: 2*L*S*d_k per head
        flops_attn_v = num_heads * 2 * L * S * d_k

        # output projection: 2*E*E*L*N
        flops_out = 2 * E * E * L * N

        return flops_qkv + flops_scores + flops_attn_v + flops_out

    # Embedding (lookup: contiamo come 1 FLOP per valore estratto)

    def _embedding_flops(self, layer: nn.Embedding, x, y):
        # x: (batch, seq_len) o (seq_len, batch)
        num_indices = x.numel()
        emb_dim = layer.embedding_dim
        return num_indices * emb_dim

    def _embeddingbag_flops(self, layer: nn.EmbeddingBag, x, y):
        # somma di embeddings per "bag"
        # stima: numel(x) * emb_dim
        num_indices = x.numel()
        emb_dim = layer.embedding_dim
        return num_indices * emb_dim

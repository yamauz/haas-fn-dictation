"""Pure-PyTorch stand-in for flash-linear-attention's ``KimiDeltaAttention`` (fla 0.5.2).

Haas-ja-v1.0 uses KDA only in its small prediction network (3.9M parameters),
but fla runs it on Triton kernels, which need CUDA. This module reimplements
the same layer with plain tensor ops so the model runs on macOS (CPU / MPS).

Parameter names and shapes match fla exactly, so the checkpoint loads as is.
The math follows fla's ``fused_recurrent_kda`` kernel with the options
``KimiDeltaAttention`` passes (qk L2 norm, gate and beta sigmoid in kernel,
``state_v_first=True``). The recurrent form is exact for any sequence length;
it is only slower than fla's chunked kernel, which matters for training, not
for decoding, where the prediction network sees one label at a time.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ShortConvolution(nn.Conv1d):
    """Causal depthwise conv with SiLU, matching ``fla.modules.ShortConvolution``.

    The cache is ``[B, D, W]`` and holds the last ``W`` inputs, newest last.
    """

    def __init__(self, hidden_size: int, kernel_size: int, bias: bool = False):
        super().__init__(
            hidden_size,
            hidden_size,
            kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
        )

    def forward(self, x: torch.Tensor, cache: torch.Tensor | None = None):
        batch_size, seq_len, dim = x.shape
        width = self.kernel_size[0]
        if cache is None:
            cache = x.new_zeros(batch_size, dim, width)
        # [B, D, W + T]: the oldest cache column is never inside a window.
        xx = torch.cat([cache.to(x.dtype), x.transpose(1, 2)], dim=-1)
        windows = xx[..., 1:].unfold(-1, width, 1)  # [B, D, T, W]
        y = (windows * self.weight[:, 0, None, :]).sum(-1)
        if self.bias is not None:
            y = y + self.bias[:, None]
        return F.silu(y).transpose(1, 2), xx[..., -width:].contiguous()


class RMSNormGated(nn.Module):
    """``rmsnorm(x) * weight * sigmoid(g)``, matching ``FusedRMSNormGated(activation="sigmoid")``."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float() * torch.sigmoid(g.float())).to(dtype)


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


class KimiDeltaAttention(nn.Module):
    """Same constructor, parameters, cache protocol and outputs as fla's layer."""

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int | None = None,
        mode: str = "chunk",
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int | None = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        if not use_short_conv:
            raise ValueError("only use_short_conv=True is implemented")
        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.conv_size = conv_size
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = num_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.q_conv1d = ShortConvolution(self.key_dim, conv_size, conv_bias)
        self.k_conv1d = ShortConvolution(self.key_dim, conv_size, conv_bias)
        self.v_conv1d = ShortConvolution(self.value_dim, conv_size, conv_bias)

        gate_dim = self.num_v_heads * self.head_k_dim
        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, gate_dim, bias=False),
        )
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        # Overwritten by the checkpoint.
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.dt_bias = nn.Parameter(torch.zeros(gate_dim))
        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = RMSNormGated(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        **kwargs,
    ):
        if attention_mask is not None:
            raise NotImplementedError("attention_mask is not used by the Haas prediction network")
        batch_size, seq_len, _ = hidden_states.shape
        H, HV, K, V = self.num_heads, self.num_v_heads, self.head_k_dim, self.head_v_dim

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx] or None

        conv_q = conv_k = conv_v = None
        if last_state is not None and last_state.get("conv_state") is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]
        x = hidden_states
        q, conv_q = self.q_conv1d(self.q_proj(x), conv_q)
        k, conv_k = self.k_conv1d(self.k_proj(x), conv_k)
        v, conv_v = self.v_conv1d(self.v_proj(x), conv_v)

        q = _l2norm(q.float().view(batch_size, seq_len, H, K)) * K**-0.5
        k = _l2norm(k.float().view(batch_size, seq_len, H, K))
        if HV != H:
            q = q.repeat_interleave(HV // H, dim=2)
            k = k.repeat_interleave(HV // H, dim=2)
        v = v.float().view(batch_size, seq_len, HV, V)
        g = self.f_proj(x).float().view(batch_size, seq_len, HV, K) + self.dt_bias.float().view(
            HV, K
        )
        g = -self.A_log.float().exp()[:, None] * F.softplus(g)  # log-space decay, [B, T, HV, K]
        beta = torch.sigmoid(self.b_proj(x).float())  # [B, T, HV]
        if self.allow_neg_eigval:
            beta = beta * 2

        # Value-first state [B, HV, V, K], as fla stores it with state_v_first=True.
        state = None if last_state is None else last_state.get("recurrent_state")
        S = x.new_zeros(batch_size, HV, V, K, dtype=torch.float32)
        if state is not None:
            S = S + state.float().reshape(batch_size, HV, V, K)
        outputs = []
        for t in range(seq_len):
            k_t = k[:, t]
            S = S * g[:, t].exp()[:, :, None, :]
            v_t = (v[:, t] - (S * k_t[:, :, None, :]).sum(-1)) * beta[:, t, :, None]
            S = S + v_t[..., None] * k_t[:, :, None, :]
            outputs.append((S * q[:, t, :, None, :]).sum(-1))
        o = torch.stack(outputs, dim=1)  # [B, T, HV, V]

        if past_key_values is not None:
            past_key_values.update(
                layer_idx=self.layer_idx,
                recurrent_state=S,
                conv_state=(conv_q, conv_k, conv_v),
                offset=seq_len,
            )

        gate = self.g_proj(x).view(batch_size, seq_len, HV, V)
        o = self.o_norm(o.to(x.dtype), gate).reshape(batch_size, seq_len, self.value_dim)
        return self.o_proj(o), None, past_key_values


def install() -> None:
    """Make ``haas_asr`` build its prediction network with this layer. Call before loading."""
    import haas_asr.prednet

    haas_asr.prednet.load_kda_class = lambda: KimiDeltaAttention

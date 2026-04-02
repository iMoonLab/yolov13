# Ultralytics AGPL-3.0 License

from __future__ import annotations

from typing import Optional, Tuple

import torch

import flash_attn_turing as flash_attn_gpu


def _maybe_contiguous(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q, k, v = [_maybe_contiguous(x) for x in (q, k, v)]
    out, lse = flash_attn_gpu.fwd(q, k, v, softmax_scale, causal)
    return out, lse


def _flash_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dout, q, k, v, out, lse = [_maybe_contiguous(x) for x in (dout, q, k, v, out, lse)]
    dq, dk, dv = flash_attn_gpu.bwd(q, k, v, out, lse, dout, softmax_scale, causal)
    return dq, dk, dv


class _FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, softmax_scale: Optional[float], causal: bool):
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_forward(q, k, v, softmax_scale, causal)
        if any(x.requires_grad for x in (q, k, v)):
            ctx.save_for_backward(q, k, v, out, lse)
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _flash_attn_backward(dout, q, k, v, out, lse, ctx.softmax_scale, ctx.causal)
        return dq, dk, dv, None, None


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    return _FlashAttnFunc.apply(q, k, v, softmax_scale, causal)

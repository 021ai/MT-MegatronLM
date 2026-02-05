# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Optional
from unittest.mock import MagicMock

import torch
from packaging import version

from megatron.core.utils import experimental_fn, null_decorator

try:
    import triton
    import triton.language as tl

    if version.parse(triton.__version__) < version.parse("3.4.0") and not torch.cuda.is_available():
        HAVE_TRITON = False
    else:
        HAVE_TRITON = tl.constexpr(version.parse(triton.__version__) >= version.parse("2.0.0"))
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    triton.autotune = null_decorator
    triton.heuristics = null_decorator
    tl = MagicMock()

from megatron.core.fusions.fused_mla_yarn_rope_apply import _get_thd_token_idx

# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_H": 1}),
#         triton.Config({"BLOCK_H": 2}),
#         triton.Config({"BLOCK_H": 4}),
#         triton.Config({"BLOCK_H": 8}),
#         triton.Config({"BLOCK_H": 16}),
#         triton.Config({"BLOCK_H": 32}),
#         triton.Config({"BLOCK_H": 64}),
#         triton.Config({"BLOCK_H": 128}),
#     ],
#     key=["emb_dim", "head_num"],
#     restore_value=["Q"],
# )
@triton.jit
def _rotary_fwd_q_kernel(
    Q,
    COS,
    SIN,
    qk_head_dim,
    emb_dim: tl.constexpr,
    head_num: tl.constexpr,
    batch_size,
    seq_num,
    cu_seqlens_q,
    stride_x_seq,
    stride_x_nheads,
    cp_rank,
    cp_size,
    BLOCK_H: tl.constexpr,
):
    """
    Triton kernel of the forward pass for applying YARN RoPE to MLA's query.
    This kernel inplace modifies the input tensor Q.

    Input:
        Q: [seq_len, batch_size, head_num, qk_head_dim + emb_dim]
            or [total_seq_len, head_num, qk_head_dim + emb_dim]
        COS/SIN: [max_seq_len, emb_dim]

        batch_size: batch size for sbhd format, not used for thd format
        seq_num: number of sequences for thd format, not used for sbhd format
        cu_seqlens_q: [seq_num + 1] accumulated sequence lengths for thd format
    """
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)

    if cu_seqlens_q is None:
        token_idx = pid_m // batch_size
    else:
        token_idx = _get_thd_token_idx(cu_seqlens_q, pid_m, seq_num, cp_rank, cp_size)

    cos_left = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
    sin_left = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
    cos_right = tl.load(COS + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
    sin_right = tl.load(SIN + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
    cos_left = cos_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    sin_left = sin_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    cos_right = cos_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    sin_right = sin_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)

    Q = Q + pid_m * stride_x_seq + pid_head * BLOCK_H * stride_x_nheads

    x_off = tl.arange(0, BLOCK_H)[:, None] * stride_x_nheads + qk_head_dim
    mask = x_off < head_num * stride_x_nheads
    # x1 = t[..., 0::2], x2 = t[..., 1::2]
    # x_1_off = x_off + tl.arange(0, emb_dim // 2)[None, :] * 2
    # x_2_off = x_1_off + 1
    # x1 = t[..., 0:d/2], x2 = t[..., d/2:]
    x_1_off = x_off + tl.arange(0, emb_dim // 2)[None, :]
    x_2_off = x_1_off + emb_dim // 2
    x_1 = tl.load(Q + x_1_off, mask=mask)
    x_2 = tl.load(Q + x_2_off, mask=mask)

    x_left = x_1 * cos_left - x_2 * sin_left
    x_right = x_2 * cos_right + x_1 * sin_right

    x_left_off = x_off + tl.arange(0, emb_dim // 2)[None, :]
    x_right_off = x_left_off + emb_dim // 2
    tl.store(Q + x_left_off, x_left, mask=mask)
    tl.store(Q + x_right_off, x_right, mask=mask)


# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_H": 1}),
#         triton.Config({"BLOCK_H": 2}),
#         triton.Config({"BLOCK_H": 4}),
#         triton.Config({"BLOCK_H": 8}),
#         triton.Config({"BLOCK_H": 16}),
#         triton.Config({"BLOCK_H": 32}),
#         triton.Config({"BLOCK_H": 64}),
#         triton.Config({"BLOCK_H": 128}),
#     ],
#     key=["emb_dim", "head_num"],
#     restore_value=["DO"],
# )
@triton.jit
def _rotary_bwd_q_kernel(
    DO,
    COS,
    SIN,
    qk_head_dim,
    emb_dim: tl.constexpr,
    head_num: tl.constexpr,
    batch_size,
    seq_num,
    cu_seqlens_q,
    stride_x_seq,
    stride_x_nheads,
    cp_rank,
    cp_size,
    BLOCK_H: tl.constexpr,
):
    """
    Triton kernel of the backward pass for applying YARN RoPE to MLA's query.
    This kernel inplace modifies the input tensor DO.

    Input:
        DO: [seq_len, batch_size, head_num, qk_head_dim + emb_dim]
            or [total_seq_len, head_num, qk_head_dim + emb_dim]
        COS/SIN: [max_seq_len, emb_dim]

        batch_size, seq_num, and cu_seqlens_q are the same as in the forward pass
    """
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)

    if cu_seqlens_q is None:
        token_idx = pid_m // batch_size
    else:
        token_idx = _get_thd_token_idx(cu_seqlens_q, pid_m, seq_num, cp_rank, cp_size)

    cos_left = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
    sin_left = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
    cos_right = tl.load(COS + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
    sin_right = tl.load(SIN + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
    cos_left = cos_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    sin_left = sin_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    cos_right = cos_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    sin_right = sin_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)

    DO = DO + pid_m * stride_x_seq + pid_head * BLOCK_H * stride_x_nheads

    x_off = tl.arange(0, BLOCK_H)[:, None] * stride_x_nheads + qk_head_dim
    mask = x_off < head_num * stride_x_nheads
    x_left_off = x_off + tl.arange(0, emb_dim // 2)[None, :]
    x_right_off = x_left_off + emb_dim // 2
    x_left = tl.load(DO + x_left_off, mask=mask)
    x_right = tl.load(DO + x_right_off, mask=mask)

    x_1 = x_left * cos_left + x_right * sin_right
    x_2 = -x_left * sin_left + x_right * cos_right

    # x_1_off = x_off + tl.arange(0, emb_dim // 2)[None, :] * 2
    # x_2_off = x_1_off + 1
    x_1_off = x_off + tl.arange(0, emb_dim // 2)[None, :]
    x_2_off = x_1_off + emb_dim // 2
    tl.store(DO + x_1_off, x_1, mask=mask)
    tl.store(DO + x_2_off, x_2, mask=mask)
    
    
    

# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_H": 1}),
#         triton.Config({"BLOCK_H": 2}),
#         triton.Config({"BLOCK_H": 4}),
#         triton.Config({"BLOCK_H": 8}),
#         triton.Config({"BLOCK_H": 16}),
#         triton.Config({"BLOCK_H": 32}),
#         triton.Config({"BLOCK_H": 64}),
#         triton.Config({"BLOCK_H": 128}),
#     ],
#     key=["emb_dim", "k_dim", "v_dim", "head_num"],
# )
@triton.jit
def _rotary_fwd_kv_kernel(
    KV,
    K_POS_EMB,
    O_KEY,
    O_VALUE,
    COS,
    SIN,
    emb_dim: tl.constexpr,
    k_dim: tl.constexpr,
    v_dim: tl.constexpr,
    head_num: tl.constexpr,
    batch_size,
    seq_num,
    cu_seqlens_kv,
    stride_kv_seq,
    stride_kv_nheads,
    stride_emb_seq,
    stride_k_seq,
    stride_k_nheads,
    stride_v_seq,
    stride_v_nheads,
    cp_rank,
    cp_size,
    BLOCK_H: tl.constexpr,
):
    """
    Triton kernel of the forward pass for applying YARN RoPE to MLA's key and value.
    It splits the input tensor KV into key and value,
    and concatenates the processed RoPE to the key.

    Input:
        KV: [seq_len, batch_size, head_num, k_dim + v_dim]
            or [total_seq_len, head_num, k_dim + v_dim]
        K_POS_EMB: [seq_len, batch_size, emb_dim] or [total_seq_len, emb_dim]
        COS/SIN: [max_seq_len, emb_dim]

        batch_size: batch size for sbhd format, not used for thd format
        seq_num: number of sequences for thd format, not used for sbhd format
        cu_seqlens_kv: [seq_num + 1] accumulated sequence lengths for thd format

    Output:
        O_KEY: [seq_len, batch_size, head_num, emb_dim + k_dim]
            or [total_seq_len, head_num, emb_dim + k_dim]
        O_VALUE: [seq_len, batch_size, head_num, v_dim] or [total_seq_len, head_num, v_dim]
    """
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)

    if cu_seqlens_kv is None:
        token_idx = pid_m // batch_size
    else:
        token_idx = _get_thd_token_idx(cu_seqlens_kv, pid_m, seq_num, cp_rank, cp_size)

    cos_left = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
    sin_left = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
    cos_right = tl.load(COS + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
    sin_right = tl.load(SIN + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))

    KV_ptr = KV + pid_m * stride_kv_seq + pid_head * BLOCK_H * stride_kv_nheads
    kv_off = tl.arange(0, BLOCK_H)[:, None] * stride_kv_nheads
    mask = kv_off < head_num * stride_kv_nheads
    k_in_off = kv_off + tl.arange(0, k_dim)[None, :]
    v_in_off = kv_off + k_dim + tl.arange(0, v_dim)[None, :]
    k = tl.load(KV_ptr + k_in_off, mask=mask)
    v = tl.load(KV_ptr + v_in_off, mask=mask)

    K_ptr = O_KEY + pid_m * stride_k_seq + pid_head * BLOCK_H * stride_k_nheads
    V_ptr = O_VALUE + pid_m * stride_v_seq + pid_head * BLOCK_H * stride_v_nheads

    k_out_off = tl.arange(0, BLOCK_H)[:, None] * stride_k_nheads + tl.arange(0, k_dim)[None, :]
    v_out_off = tl.arange(0, BLOCK_H)[:, None] * stride_v_nheads + tl.arange(0, v_dim)[None, :]
    tl.store(K_ptr + k_out_off, k, mask=mask)
    tl.store(V_ptr + v_out_off, v, mask=mask)

    EMB = K_POS_EMB + pid_m * stride_emb_seq
    # x1 = t[..., 0::2], x2 = t[..., 1::2]
    # x_1 = tl.load(EMB + tl.arange(0, emb_dim // 2) * 2)
    # x_2 = tl.load(EMB + tl.arange(0, emb_dim // 2) * 2 + 1)
    x_1 = tl.load(EMB + tl.arange(0, emb_dim // 2))
    x_2 = tl.load(EMB + tl.arange(0, emb_dim // 2) + emb_dim // 2)
    x_left = x_1 * cos_left - x_2 * sin_left
    x_right = x_2 * cos_right + x_1 * sin_right
    x_left = x_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    x_right = x_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)

    x_left_off = (
        tl.arange(0, BLOCK_H)[:, None] * stride_k_nheads
        + k_dim
        + tl.arange(0, emb_dim // 2)[None, :]
    )
    x_right_off = x_left_off + emb_dim // 2
    tl.store(K_ptr + x_left_off, x_left, mask=mask)
    tl.store(K_ptr + x_right_off, x_right, mask=mask)


# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_H": 1}),
#         triton.Config({"BLOCK_H": 2}),
#         triton.Config({"BLOCK_H": 4}),
#         triton.Config({"BLOCK_H": 8}),
#         triton.Config({"BLOCK_H": 16}),
#         triton.Config({"BLOCK_H": 32}),
#         triton.Config({"BLOCK_H": 64}),
#         triton.Config({"BLOCK_H": 128}),
#     ],
#     key=["emb_dim", "k_dim", "v_dim", "head_num"],
# )
@triton.jit
def _rotary_bwd_kv_kernel(
    dK,
    dV,
    dKV,
    dEMB,
    COS,
    SIN,
    emb_dim: tl.constexpr,
    k_dim: tl.constexpr,
    v_dim: tl.constexpr,
    head_num: tl.constexpr,
    batch_size,
    seq_num,
    cu_seqlens_kv,
    stride_dk_seq,
    stride_dk_nheads,
    stride_dv_seq,
    stride_dv_nheads,
    stride_dkv_seq,
    stride_dkv_nheads,
    stride_demb_seq,
    cp_rank,
    cp_size,
    BLOCK_H: tl.constexpr,
):
    """
    Triton kernel of the backward pass for applying YARN RoPE to MLA's key and value.

    Input:
        dK: [seq_len, batch_size, head_num, emb_dim + k_dim]
            or [total_seq_len, head_num, emb_dim + k_dim]
        dV: [seq_len, batch_size, head_num, v_dim] or [total_seq_len, head_num, v_dim]
        COS/SIN: [max_seq_len, emb_dim]

        batch_size, seq_num, and cu_seqlens_kv are the same as in the forward pass

    Output:
        dKV: [seq_len, batch_size, head_num, k_dim + v_dim]
            or [total_seq_len, head_num, k_dim + v_dim]
        dEMB: [seq_len, batch_size, emb_dim] or [total_seq_len, emb_dim]
    """
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)

    if cu_seqlens_kv is None:
        token_idx = pid_m // batch_size
    else:
        token_idx = _get_thd_token_idx(cu_seqlens_kv, pid_m, seq_num, cp_rank, cp_size)

    dKV_ptr = dKV + pid_m * stride_dkv_seq + pid_head * BLOCK_H * stride_dkv_nheads
    dkv_off = tl.arange(0, BLOCK_H)[:, None] * stride_dkv_nheads
    mask = dkv_off < head_num * stride_dkv_nheads
    dk_out_off = dkv_off + tl.arange(0, k_dim)[None, :]
    dv_out_off = dkv_off + k_dim + tl.arange(0, v_dim)[None, :]

    dK_ptr = dK + pid_m * stride_dk_seq + pid_head * BLOCK_H * stride_dk_nheads
    dV_ptr = dV + pid_m * stride_dv_seq + pid_head * BLOCK_H * stride_dv_nheads
    dk_in_off = tl.arange(0, BLOCK_H)[:, None] * stride_dk_nheads + tl.arange(0, k_dim)[None, :]
    dv_in_off = tl.arange(0, BLOCK_H)[:, None] * stride_dv_nheads + tl.arange(0, v_dim)[None, :]
    dk = tl.load(dK_ptr + dk_in_off, mask=mask)
    dv = tl.load(dV_ptr + dv_in_off, mask=mask)
    tl.store(dKV_ptr + dk_out_off, dk, mask=mask)
    tl.store(dKV_ptr + dv_out_off, dv, mask=mask)

    if pid_head == 0:
        x_left_accum = tl.zeros((BLOCK_H, emb_dim // 2), dtype=tl.float32)
        x_right_accum = tl.zeros((BLOCK_H, emb_dim // 2), dtype=tl.float32)
        for i in tl.static_range(triton.cdiv(head_num, BLOCK_H)):
            dK_ptr = dK + pid_m * stride_dk_seq + i * BLOCK_H * stride_dk_nheads
            x_off = tl.arange(0, BLOCK_H)[:, None] * stride_dk_nheads + k_dim
            mask = x_off < head_num * stride_dk_nheads
            x_left_off = x_off + tl.arange(0, emb_dim // 2)[None, :]
            x_right_off = x_left_off + emb_dim // 2
            x_left = tl.load(dK_ptr + x_left_off, mask=mask)
            x_right = tl.load(dK_ptr + x_right_off, mask=mask)
            x_left_accum += x_left
            x_right_accum += x_right
        x_left_accum = tl.sum(x_left_accum, axis=0)
        x_right_accum = tl.sum(x_right_accum, axis=0)
        x_left_accum = x_left_accum.to(dEMB.dtype.element_ty)
        x_right_accum = x_right_accum.to(dEMB.dtype.element_ty)

        cos_left = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
        sin_left = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
        cos_right = tl.load(COS + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
        sin_right = tl.load(SIN + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))

        x_1 = x_left_accum * cos_left + x_right_accum * sin_right
        x_2 = -x_left_accum * sin_left + x_right_accum * cos_right
        dEMB_ptr = dEMB + pid_m * stride_demb_seq
        tl.store(dEMB_ptr + tl.arange(0, emb_dim // 2) , x_1)
        tl.store(dEMB_ptr + tl.arange(0, emb_dim // 2) + emb_dim // 2, x_2)

import megatron.core.fusions.fused_mla_yarn_rope_apply
from transformer_engine.musa.pytorch.utils import replace_attr
print('replace rotary done')
replace_attr(megatron.core.fusions.fused_mla_yarn_rope_apply, "rotary_fwd_q_kernel", _rotary_fwd_q_kernel)
replace_attr(megatron.core.fusions.fused_mla_yarn_rope_apply, "rotary_bwd_q_kernel", _rotary_bwd_q_kernel)
replace_attr(megatron.core.fusions.fused_mla_yarn_rope_apply, "rotary_fwd_kv_kernel", _rotary_fwd_kv_kernel)
replace_attr(megatron.core.fusions.fused_mla_yarn_rope_apply, "rotary_bwd_kv_kernel", _rotary_bwd_kv_kernel)


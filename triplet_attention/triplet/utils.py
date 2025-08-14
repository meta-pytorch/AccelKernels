# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch


def get_triplet_tensor_core_tflops(
    bs, seq_len, num_heads, num_kv_heads, head_dim, w1, w2
):
    # Pre-compute TFLOPs for tokens in range [0, w1), where it's 2D casual attention for w1 and w2
    QK_GEMM_W1 = bs * num_heads * w1 * head_dim * w1 * w1 * 2
    QK_GEMM_W1 *= 0.5  # 0.5 is the casual mask ratio for w1
    QK_GEMM_W1 *= 0.5  # 0.5 is the casual mask ratio for w2
    total_tflops_w1 = (QK_GEMM_W1 * 2) / 1e12

    # Pre-compute TFLOPs for tokens in range [w1, w2), where it's SWA for w1 and casual attention for w2
    QK_GEMM_W2 = bs * num_heads * (w2 - w1) * head_dim * w1 * w2 * 2
    QK_GEMM_W2 *= 0.5  # 0.5 is the casual mask ratio for w2
    total_tflops_w2 = (QK_GEMM_W2 * 2) / 1e12

    # Causal attention for w1 and w2
    if seq_len <= w1:
        QK_GEMM = bs * num_heads * seq_len * head_dim * seq_len * seq_len * 2
        QK_GEMM *= 0.5  # 0.5 is the casual mask ratio for w1
        QK_GEMM *= 0.5  # 0.5 is the casual mask ratio for w2
        return (QK_GEMM * 2) / 1e12
    # Causal attention for w2
    elif seq_len <= w2:
        current_seq_len = seq_len - w1
        QK_GEMM = bs * num_heads * current_seq_len * head_dim * w1 * current_seq_len * 2
        QK_GEMM *= 0.5  # 0.5 is the casual mask ratio for w2
        return (QK_GEMM * 2) / 1e12 + total_tflops_w1
    else:
        current_seq_len = seq_len - w1 - w2
        QK_GEMM = bs * num_heads * current_seq_len * head_dim * w1 * w2 * 2
        return (QK_GEMM * 2) / 1e12 + total_tflops_w1 + total_tflops_w2


def compute_sqnr(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-30,
) -> float:
    """Compute SQNR in dB.

    Args:
        x: The reference tensor.
        y: The de-quantized tensor.
        eps: A small value to avoid division by zero.

    Returns:
        SQNR in dB.
    """
    error = x - y
    signal = x * x
    error_power = error * error
    ratio = signal.mean() / (error_power.mean() + eps)
    sqnr_db = 10.0 * torch.log10(ratio).item()
    return round(sqnr_db, 2)

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe


import sys

import pytest
import torch

from triplet.utils import compute_sqnr

from triplet.ops.tlx.fwd_ws_pingpong import triplet_tlx_fwd_ws
from triplet.ops.triton.fwd import triplet_fwd as triplet_triton_fwd


@pytest.fixture(autouse=True)
def deterministic_seed():
    """Ensure reproducibility for every test."""
    torch.manual_seed(7)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("N", [512, 1024, 2028])
@pytest.mark.parametrize("H", [128])
@pytest.mark.parametrize("D", [128])
@pytest.mark.parametrize("w1", [16, 32])
@pytest.mark.parametrize("w2", [128, 256])
def test_triplet_tlx_fwd_ws_pipelined(B, N, H, D, w1, w2):
    device = torch.accelerator.current_accelerator()

    Hkv = 1
    print(f"Testing B={B}, N={N}, H={H}, D={D}, w1={w1} w2={w2}")

    # Create normalized tensors as required
    Q = torch.randn(B, N, H, D, device=device, dtype=torch.bfloat16).normal_(
        mean=0.0, std=0.5
    )
    K1 = torch.randn(B, N, Hkv, D, device=device, dtype=torch.bfloat16).normal_(
        mean=0.0, std=0.5
    )
    K2 = torch.randn_like(K1).normal_(mean=0.0, std=0.5)
    V1 = torch.randn_like(K1).normal_(mean=0.0, std=0.5)
    V2 = torch.randn_like(K1).normal_(mean=0.0, std=0.5)

    # Normalize tensors
    Q = Q / torch.norm(Q, dim=-1, keepdim=True)
    K1 = K1 / torch.norm(K1, dim=-1, keepdim=True)
    K2 = K2 / torch.norm(K2, dim=-1, keepdim=True)
    V1 = V1 / torch.norm(V1, dim=-1, keepdim=True)
    V2 = V2 / torch.norm(V2, dim=-1, keepdim=True)

    # Test the pipelined triplet_tlx_fwd_ws kernel
    out_pipelined, _ = triplet_tlx_fwd_ws(Q, K1, K2, V1, V2, w1=w1, w2=w2)

    # out_tlx_baseline, _ = tlx_fwd_baseline(Q, K1, K2, V1, V2, w1=w1, w2=w2)

    # Ground truth implementation
    out_ref, _ = triplet_triton_fwd(
        Q,
        K1,
        K2,
        V1,
        V2,
        w1=w1,
        w2=w2,
    )

    # Compute SQNR loss vs reference
    sqnr_ref = compute_sqnr(out_pipelined, out_ref)
    assert (
        sqnr_ref > 30.0
    ), f"SQNR vs reference should be larger than 30.0. Got: {sqnr_ref}"

    # Element-wise comparison with reference
    torch.testing.assert_close(out_pipelined, out_ref, atol=1e-2, rtol=0.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))

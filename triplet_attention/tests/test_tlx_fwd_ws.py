# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe


import sys



import pytest
import torch

from triplet.ops.pytorch.triplet_attention import triplet_attn_fwd_ref
from triplet.ops.tlx.fwd_ws import triplet_tlx_fwd_ws
from triplet.utils import compute_sqnr



@pytest.fixture(autouse=True)
def deterministic_seed():
    """Ensure reproducibility for every test."""
    torch.manual_seed(7)


@pytest.mark.parametrize("B", [1])
@pytest.mark.parametrize("N", [128, 256, 512])
@pytest.mark.parametrize("H", [64])
@pytest.mark.parametrize("D", [128])
@pytest.mark.parametrize("w1", [16, 32])
@pytest.mark.parametrize("w2", [64, 128, 256])
def test_triplet_tlx_fwd_ws(B, N, H, D, w1, w2):
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

    # Test the triplet_tlx_fwd_ws kernel
    out, _ = triplet_tlx_fwd_ws(Q, K1, K2, V1, V2, w1=w1, w2=w2)

    # Ground truth implementation
    out_ref = triplet_attn_fwd_ref(
        Q,
        K1,
        K2,
        V1,
        V2,
        w1=w1,
        w2=w2,
        use_fp32=False,
        disable_kv_bias=True,
    )

    # Compute SQNR loss
    sqnr = compute_sqnr(out, out_ref)
    assert (
        sqnr > 25.0
    ), f"SQNR should be larger than 30.0 for out and out_ref Got: {sqnr}"

    # Element-wise comparison
    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=0.0)

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))

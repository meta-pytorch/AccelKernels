# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe


import sys

import pytest
import torch
from two_simplicial.ops.pytorch.two_simplicial_attention import two_simplicial_attn_fwd_ref

from two_simplicial.ops.triton.fwd import two_simplicial_fwd
from two_simplicial.utils import compute_sqnr, assert_diff


@pytest.fixture(autouse=True)
def deterministic_seed():
    """Ensure reproducibility for every test."""
    torch.manual_seed(7)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("N", [128, 256, 512])
@pytest.mark.parametrize("H", [16])
@pytest.mark.parametrize("D", [128])
@pytest.mark.parametrize("w1", [32, 256])
@pytest.mark.parametrize("w2", [4, 128, 256])
def test_two_simplicial_triton_fwd(B, N, H, D, w1, w2):
    device = torch.accelerator.current_accelerator()

    Hkv = 1
    print(f"Testing B={B}, N={N}, H={H}, D={D}, w1={w1} w2={w2}")

    Q = torch.randn(B, N, H, D, device=device, dtype=torch.bfloat16).normal_(
        mean=0.0, std=0.5
    )
    K1 = torch.randn(B, N, Hkv, D, device=device, dtype=torch.bfloat16).normal_(
        mean=0.0, std=0.5
    )
    K2 = torch.randn_like(K1).normal_(mean=0.0, std=0.5)
    V1 = torch.randn_like(K1).normal_(mean=0.0, std=0.5)
    V2 = torch.randn_like(K1).normal_(mean=0.0, std=0.5)

    out, _ = two_simplicial_fwd(Q, K1, K2, V1, V2, w1=w1, w2=w2)

    out_ref = two_simplicial_attn_fwd_ref(
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

    sqnr = compute_sqnr(out, out_ref)
    assert sqnr > 40.0, (
        f"SQNR should be larger than 30.0 for out and out_ref Got: {sqnr}"
    )
    assert_diff(out, out_ref)


if __name__ == "__main__":
    # test_kv_expand(4, 256, 4, 128, 4)
    sys.exit(pytest.main([__file__, "-v", "-s"]))

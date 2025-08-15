# Fast Triplet Attention

High-performance GPU Kernels of Triplet Attention

Paper: https://arxiv.org/pdf/2507.02754

Blog: "TBD"

## Installation

```bash
# Install from source
git clone <repository-url>
cd triplet_attention
pip install -e .
```

## Quick Start

### Using Triton Implementation

```python
import torch
from triplet.ops.triton.fwd import triplet_fwd

# Setup tensors
batch_size, seq_len, num_heads, head_dim = 4, 1024, 64, 128
device = torch.cuda.current_device()

Q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
K1 = torch.randn(batch_size, seq_len, 1, head_dim, device=device, dtype=torch.bfloat16)
K2 = torch.randn(batch_size, seq_len, 1, head_dim, device=device, dtype=torch.bfloat16)
V1 = torch.randn(batch_size, seq_len, 1, head_dim, device=device, dtype=torch.bfloat16)
V2 = torch.randn(batch_size, seq_len, 1, head_dim, device=device, dtype=torch.bfloat16)

# Run triplet attention with windowed attention (w1=32, w2=256)
output, _ = triplet_fwd(Q, K1, K2, V1, V2, w1=32, w2=256)
```

### Using TLX Implementation

```python
from triplet.ops.tlx.fwd_ws import triplet_tlx_fwd_ws

# High-performance TensorLX implementation
output, _ = triplet_tlx_fwd_ws(Q, K1, K2, V1, V2, w1=32, w2=256)
```

### Using PyTorch Reference Implementation

```python
from triplet.ops.pytorch.triplet_attention import triplet_attn_fwd_ref

# Reference implementation with different variants
output = triplet_attn_fwd_ref(
    Q, K1, K2, V1, V2,
    w1=32, w2=256,
    use_fp32=True,
    disable_kv_bias=True
)
```

## Performance

### Benchmarking

Run performance benchmarks:

```bash
# Forward pass benchmarks
python benchmarks/bench_fwd.py
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v
```

## License

Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

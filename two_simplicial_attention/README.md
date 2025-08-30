# Fast Two Simplicial Attention

High-performance GPU Kernels of Two Simplicial Attention

as proposed in: https://arxiv.org/abs/1909.00668
Optimized in: https://arxiv.org/pdf/2507.02754

Blog: "TBD"

## Installation

```bash
# Install from source
git clone <repository-url>
cd two_simplicial_attention
pip install -e .
```

## Quick Start

### Using Triton Implementation

```python
import torch
from two_simplicial.ops.triton.fwd import two_simplicial_fwd

# Setup tensors
batch_size, seq_len, num_heads, head_dim = 4, 1024, 64, 128
device = torch.cuda.current_device()

Q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
K1 = torch.randn(batch_size, seq_len, 1, head_dim, device=device, dtype=torch.bfloat16)
K2 = torch.randn(batch_size, seq_len, 1, head_dim, device=device, dtype=torch.bfloat16)
V1 = torch.randn(batch_size, seq_len, 1, head_dim, device=device, dtype=torch.bfloat16)
V2 = torch.randn(batch_size, seq_len, 1, head_dim, device=device, dtype=torch.bfloat16)

output, _ = two_simplicial_fwd(Q, K1, K2, V1, V2, w1=32, w2=256)
```

### Using TLX Implementation

```python
from two_simplicial.ops.tlx.fwd_ws import two_simplicial_tlx_fwd_ws

# High-performance TLX implementation
output, _ = two_simplicial_tlx_fwd_ws(Q, K1, K2, V1, V2, w1=32, w2=256)
```

### Using PyTorch Reference Implementation

```python
from two_simplicial.ops.pytorch.two_simplicial_attention import two_simplicial_attn_fwd_ref

# Reference implementation
output = two_simplicial_attn_fwd_ref(
    Q, K1, K2, V1, V2,
    w1=32, w2=256,
    use_fp32=True,
    disable_kv_bias=True
)
```

## Performance

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

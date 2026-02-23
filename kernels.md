# Metal LLM Kernel Reference

Data flow, buffer shapes, and dispatch patterns for every kernel in this repo.

---

## Notation

```
B   = batch size
T   = sequence length (tokens)
C   = embedding dimension (n_embed)
nh  = number of attention heads
hs  = head size = C / nh
V   = vocab size
4C  = MLP hidden size (GPT-2 uses 4×)
```

---

## Forward Pass Data Flow

```
tokens [B, T]
    │
    ▼ embedding_forward + positional_embedding_add
    │
residual [B, T, C]
    │
    ├─ for each transformer layer ─────────────────────────────────┐
    │                                                              │
    │   ┌── layernorm_forward → [B, T, C]                         │
    │   │                                                          │
    │   │   QKV projection (MPS matmul) → [B, T, 3C]             │
    │   │       split → Q [B,nh,T,hs]  K [B,nh,T,hs]  V [B,nh,T,hs]
    │   │                                                          │
    │   │   (optional) rope_forward on Q, K                       │
    │   │                                                          │
    │   │   attention_scores → att [B, nh, T, T]                  │
    │   │   softmax_inplace  → att [B, nh, T, T]  (in-place)      │
    │   │   attention_mix    → attn_out [B, nh, T, hs]            │
    │   │                                                          │
    │   │   output projection (MPS) → [B, T, C]                   │
    │   │                                                          │
    │   ├── residual_layernorm (fused add + norm) → [B, T, C]     │
    │   │                                                          │
    │   │   MLP up projection (MPS) → [B, T, 4C]                  │
    │   │   gelu_forward (or silu_forward) → [B, T, 4C]           │
    │   │   MLP down projection (MPS) → [B, T, C]                 │
    │   │                                                          │
    │   └── residual_add → residual [B, T, C]                     │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
    │
    ▼ layernorm_forward (final norm)
    │
    ▼ lm_head (MPS matmul) → logits [B, T, V]
    │
    ▼ argmax_sample (or top-k / temperature sampling)
    │
  token [int]
```

---

## Kernel Reference Table

### embedding.metal

| Kernel | Grid | Threadgroup | Shared mem |
|--------|------|-------------|------------|
| `embedding_forward` | `(B*T, C)` | `(1, C)` | none |
| `positional_embedding_add` | `(B*T, C)` | `(1, C)` | none |

**Buffers:**
```
embedding_forward:
  [0] weight  float [vocab, C]    read
  [1] tokens  int   [B*T]         read
  [2] out     float [B*T, C]      write
  [3] C       int   constant
```

---

### layernorm.metal

| Kernel | Grid | Threadgroup | Shared mem |
|--------|------|-------------|------------|
| `layernorm_forward` | `(N,)` | `(min(C,1024),)` | `gsz × float` |
| `rmsnorm_forward` | `(N,)` | `(min(C,1024),)` | `gsz × float` |

**Buffers:**
```
layernorm_forward:
  [0] x      float [N, C]     read
  [1] gamma  float [C]         read
  [2] beta   float [C]         read
  [3] out    float [N, C]      write
  [4] mean   float [N]         write (optional, pass nil at inference)
  [5] rstd   float [N]         write (optional, pass nil at inference)
  [6] C      int   constant
  [7] eps    float constant     (typically 1e-5)
```

**Note:** Dispatch one threadgroup per row. Threadgroup memory = `gsz * sizeof(float)`.

---

### attention.metal

| Kernel | Grid | Threadgroup | Shared mem |
|--------|------|-------------|------------|
| `attention_scores` | `(B*nh, T, T)` | `(1,1,1)` | none |
| `softmax_inplace` | `(B*nh*T,)` one TG per row | `(min(T,1024),)` | `gsz × float` |
| `attention_mix` | `(B*nh, T, hs)` | `(1,1,1)` | none |
| `rope_forward` | `(B*nh, T, hs/2)` | `(1,1,1)` | none |

**Buffers:**
```
attention_scores:
  [0] Q    float [B*nh, T, hs]  read
  [1] K    float [B*nh, T, hs]  read
  [2] att  float [B*nh, T, T]   write
  [3] T    int   constant
  [4] hs   int   constant

softmax_inplace:
  [0] x    float [N, T]         read+write (in-place)
  [1] T    int   constant

attention_mix:
  [0] att  float [B*nh, T, T]   read
  [1] V    float [B*nh, T, hs]  read
  [2] out  float [B*nh, T, hs]  write
  [3] T    int   constant
  [4] hs   int   constant

rope_forward:
  [0] x      float [B*nh, T, hs]     read+write
  [1] freqs  float [T, hs]            read  (precomputed cos/sin interleaved)
  [2] T      int   constant
  [3] hs     int   constant
```

---

### activation.metal

All kernels: Grid = `(N,)`, Threadgroup = `(min(N, 1024),)`, no shared memory.

| Kernel | Formula |
|--------|---------|
| `gelu_forward` | `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))` |
| `silu_forward` | `x / (1 + e^{-x})` |
| `swiglu_forward` | `silu(gate) * up` |
| `relu_forward` | `max(0, x)` |
| `sigmoid_forward` | `1 / (1 + e^{-x})` |

---

### residual.metal

| Kernel | Grid | Threadgroup | Shared mem |
|--------|------|-------------|------------|
| `residual_add` | `(N,)` | `(1024,)` | none |
| `residual_layernorm` | `(N_rows,)` | `(min(C,1024),)` | `gsz × float` |
| `residual_rmsnorm` | `(N_rows,)` | `(min(C,1024),)` | `gsz × float` |

---

### sampling.metal

All kernels: single threadgroup over vocab.

| Kernel | Output |
|--------|--------|
| `argmax_sample` | winning token index (int) |
| `topk_mask` | logits with sub-threshold values → -inf |
| `softcap_logits` | logits ∈ (-cap, cap) via tanh squeeze |
| `temperature_scale` | logits /= temperature |

**Typical sampling pipeline:**
```
1. softcap_logits    (optional, Gemma models)
2. temperature_scale (optional, T ≠ 1.0)
3. topk_mask         (optional, top-k sampling)
4. softmax_inplace   (convert logits → probs)
5. argmax_sample     (greedy) OR sample on CPU
```

---

### flash_attention.metal

Fused single-pass attention. No `[B,nh,T,T]` matrix is ever allocated.

| Kernel | Grid | Threadgroup | Shared mem |
|--------|------|-------------|------------|
| `flash_attention_forward` | `(B*nh, T/Br)` | `(Br,)` | `5 tiles × Br or Bc × hs × float` |

**When to use vs. naive attention:**

| Sequence length | Recommendation |
|-----------------|----------------|
| T ≤ 512 | Naive attention is fine |
| T > 512 | Use flash_attention (saves memory, often faster) |
| T > 2048 | Flash attention is essential |

---

## Threadgroup Memory Sizing Guide

For `layernorm_forward` and `softmax_inplace`:
```swift
// threadgroup memory needed (bytes)
let tgMem = min(C, 1024) * MemoryLayout<Float>.size
encoder.setThreadgroupMemoryLength(tgMem, index: 0)
```

For `argmax_sample`, two threadgroup buffers:
```swift
// float buffer for max values
encoder.setThreadgroupMemoryLength(tgSize * 4, index: 0)
// int buffer for indices
encoder.setThreadgroupMemoryLength(tgSize * 4, index: 1)
```

---

## Common Pitfalls

**1. Forgetting `threadgroup_barrier`**
Every shared-memory reduction needs a barrier between write and read phases.
Missing barriers → non-deterministic garbage.

**2. Integer overflow in index math**
At large `T` and `C`, `bh * T * T` can overflow `int`.
Cast to `long` or `uint64_t` in the kernel for safety.

**3. Causal mask off-by-one**
The condition is `k > q` (strict), not `k >= q`.
Token at position `q` can attend to itself.

**4. Dispatch vs. dispatchThreadgroups**
- `dispatchThreads`: Metal automatically handles boundary threads (use this)
- `dispatchThreadgroups`: You control groups exactly, may waste threads at boundaries

**5. Shared memory limits**
M1/M2: 32KB max threadgroup memory.
At `C=1024`, `gsz=1024`: `1024 * 4 = 4KB` — well within limits.
At `C=4096` (LLaMA 7B), use `gsz=1024` and loop over channels.

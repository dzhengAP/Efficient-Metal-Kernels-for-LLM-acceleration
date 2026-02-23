# metal-llm

> Karpathy-style Metal compute kernels for LLM inference on Apple Silicon.

One file per concept. No magic. If you can read it in 5 minutes, it's simple enough.

```
embedding → layernorm → [attn_scores → softmax → attn_mix] → gelu → residual → sample
```

---

## Philosophy

- **One kernel = one idea.** Read any file in isolation and understand it completely.
- **Float32 first.** Add half precision only when you need speed.
- **Heavy matmul goes to MPS.** Apple spent years on that. These kernels cover everything around it.
- **No framework magic.** Bare Metal + MPSMatrixMultiplication + a bit of Swift glue.

---

## File Structure

```
metal-llm/
├── kernels/
│   ├── embedding.metal          # token id → vector lookup
│   ├── layernorm.metal          # normalize rows to mean=0, std=1
│   ├── attention.metal          # scores, softmax, weighted sum
│   ├── activation.metal         # gelu
│   ├── residual.metal           # fused residual add + layernorm
│   ├── sampling.metal           # argmax / top-k sampler
│   └── flash_attention.metal    # fused QKV (no T×T materialization)
├── swift/
│   ├── LLMPipeline.swift        # dispatch order, ties it all together
│   └── MetalContext.swift       # device, queue, library setup
├── tests/
│   └── kernel_tests.swift       # numerically verify each kernel vs numpy
├── docs/
│   └── kernels.md               # annotated diagram of data flow
└── README.md
```

---

## Quick Start

```swift
let pipeline = LLMPipeline(modelPath: "gpt2.bin", device: MTLCreateSystemDefaultDevice()!)
let tokens = pipeline.encode("The meaning of life is")
let output = pipeline.generate(tokens, maxNewTokens: 50)
print(pipeline.decode(output))
```

---

## Kernel Overview

| Kernel | Input | Output | Notes |
|--------|-------|--------|-------|
| `embedding_forward` | `[B,T]` token ids | `[B,T,C]` vectors | simple table lookup |
| `layernorm_forward` | `[N,C]` | `[N,C]` | one threadgroup per row |
| `attention_scores` | Q,K `[B,nh,T,hs]` | att `[B,nh,T,T]` | causal mask built-in |
| `softmax_inplace` | `[N,T]` | `[N,T]` | safe softmax (subtract max) |
| `attention_mix` | att,V `[B,nh,T,T/hs]` | `[B,nh,T,hs]` | weighted sum |
| `gelu_forward` | `[N]` | `[N]` | tanh approximation |
| `residual_layernorm` | x,y `[N,C]` | res,out `[N,C]` | fused, saves bandwidth |
| `argmax_sample` | logits `[vocab]` | token `int` | greedy decode |
| `flash_attention` | Q,K,V `[B,nh,T,hs]` | `[B,nh,T,hs]` | no T×T matrix, O(1) memory |

---

## Requirements

- macOS 13+ / iOS 16+
- Metal-capable Apple Silicon (M1 or later recommended)
- Xcode 14+

---

## License

MIT

// attention.metal
// Causal self-attention — the heart of a transformer.
//
// Three steps:
//   1. attention_scores:  att[b,h,q,k] = Q[b,h,q] · K[b,h,k] / sqrt(hs)
//   2. softmax_inplace:   att[b,h,q,:] = softmax(att[b,h,q,:])   (causal masked)
//   3. attention_mix:     out[b,h,q,c] = sum_k att[b,h,q,k] * V[b,h,k,c]
//
// We keep them separate so each is readable in isolation.
// For production, fuse into flash_attention.metal.
//
// Notation:
//   B  = batch size
//   T  = sequence length
//   nh = number of heads
//   hs = head size  (C / nh)
//   C  = total channels (nh * hs)

#include <metal_stdlib>
using namespace metal;

// ----------------------------------------------------------------
// Step 1: Scaled dot-product scores with causal mask
//
// att[bh, q, k] = Q[bh,q] · K[bh,k] / sqrt(hs)   if k <= q
//               = -inf                               if k >  q
//
// Dispatch: grid = (B*nh, T, T)  — one thread per (head, query, key)
// ----------------------------------------------------------------
kernel void attention_scores(
    device const float*  Q      [[buffer(0)]],   // [B*nh, T, hs]
    device const float*  K      [[buffer(1)]],   // [B*nh, T, hs]
    device       float*  att    [[buffer(2)]],   // [B*nh, T, T]
    constant     int&    T      [[buffer(3)]],
    constant     int&    hs     [[buffer(4)]],
    uint3 pos [[thread_position_in_grid]]          // (bh, q, k)
) {
    int bh = pos.x;
    int q  = pos.y;
    int k  = pos.z;

    // causal: token at position q can only attend to positions <= q
    if (k > q) {
        att[bh * T * T + q * T + k] = -INFINITY;
        return;
    }

    float score = 0.0f;
    int qoff = bh * T * hs + q * hs;
    int koff = bh * T * hs + k * hs;
    for (int i = 0; i < hs; i++)
        score += Q[qoff + i] * K[koff + i];

    att[bh * T * T + q * T + k] = score * rsqrt((float)hs);
}

// ----------------------------------------------------------------
// Step 2: Softmax in place, one row at a time
//
// Uses the "safe softmax" trick: subtract max before exp.
//   softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
// Numerically identical but never overflows.
//
// Dispatch: grid = (N,), threadgroup = (gsz,)
//           N = B * nh * T  (one row per query position per head)
// ----------------------------------------------------------------
kernel void softmax_inplace(
    device       float*  x      [[buffer(0)]],   // [N, T] — modified in place
    constant     int&    T      [[buffer(1)]],
    uint  row [[threadgroup_position_in_grid]],
    uint  lid [[thread_position_in_threadgroup]],
    uint  gsz [[threads_per_threadgroup]],
    threadgroup float*   smem   [[threadgroup(0)]]   // gsz floats
) {
    device float* row_ptr = x + row * T;

    // ---- find max ----
    float local_max = -INFINITY;
    for (int i = lid; i < T; i += gsz)
        local_max = max(local_max, row_ptr[i]);
    smem[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = gsz >> 1; s > 0; s >>= 1) {
        if (lid < s) smem[lid] = max(smem[lid], smem[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = smem[0];

    // ---- exp and partial sum ----
    float local_sum = 0.0f;
    for (int i = lid; i < T; i += gsz) {
        float e = exp(row_ptr[i] - row_max);
        row_ptr[i] = e;
        local_sum += e;
    }
    smem[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = gsz >> 1; s > 0; s >>= 1) {
        if (lid < s) smem[lid] += smem[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / smem[0];

    // ---- normalize ----
    for (int i = lid; i < T; i += gsz)
        row_ptr[i] *= inv_sum;
}

// ----------------------------------------------------------------
// Step 3: Weighted sum of values
//
// out[bh, q, c] = sum_{k=0}^{q} att[bh,q,k] * V[bh,k,c]
//
// Dispatch: grid = (B*nh, T, hs) — one thread per (head, query, channel)
// ----------------------------------------------------------------
kernel void attention_mix(
    device const float*  att    [[buffer(0)]],   // [B*nh, T, T]
    device const float*  V      [[buffer(1)]],   // [B*nh, T, hs]
    device       float*  out    [[buffer(2)]],   // [B*nh, T, hs]
    constant     int&    T      [[buffer(3)]],
    constant     int&    hs     [[buffer(4)]],
    uint3 pos [[thread_position_in_grid]]          // (bh, q, c)
) {
    int bh = pos.x, q = pos.y, c = pos.z;

    float val = 0.0f;
    int att_base = bh * T * T + q * T;
    int v_base   = bh * T * hs;
    for (int k = 0; k <= q; k++)
        val += att[att_base + k] * V[v_base + k * hs + c];

    out[bh * T * hs + q * hs + c] = val;
}

// ----------------------------------------------------------------
// Rotary Positional Embedding (RoPE) — used in LLaMA / Mistral.
// Applied to Q and K before computing attention scores.
//
// Rotates pairs of channels (i, i+hs/2) by an angle that depends
// on the token position. The key insight: rotations are preserved
// under dot products, so relative position is baked into attention.
//
// Dispatch: grid = (B*nh, T, hs/2) — one thread per pair
// ----------------------------------------------------------------
kernel void rope_forward(
    device       float*  x       [[buffer(0)]],   // [B*nh, T, hs] — modified in place
    device const float*  freqs   [[buffer(1)]],   // [T, hs/2] — precomputed cos/sin interleaved
    constant     int&    T       [[buffer(2)]],
    constant     int&    hs      [[buffer(3)]],
    uint3 pos [[thread_position_in_grid]]           // (bh, t, pair_idx)
) {
    int bh  = pos.x;
    int t   = pos.y;
    int p   = pos.z;                               // pair index in [0, hs/2)

    int offset = bh * T * hs + t * hs;
    float x0 = x[offset + p];
    float x1 = x[offset + p + hs / 2];

    float cos_val = freqs[t * hs + 2 * p];
    float sin_val = freqs[t * hs + 2 * p + 1];

    x[offset + p]          = x0 * cos_val - x1 * sin_val;
    x[offset + p + hs / 2] = x0 * sin_val + x1 * cos_val;
}

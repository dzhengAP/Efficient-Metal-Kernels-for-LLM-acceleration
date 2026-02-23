// residual.metal
// Residual connections and fused operations.
//
// Residual connections (x = x + sublayer(x)) are the reason transformers
// can be trained 100+ layers deep. They give gradients a highway back
// to early layers without passing through every weight matrix.
//
// We provide:
//   1. Plain residual add               — simplest form
//   2. Fused residual add + layernorm   — saves memory bandwidth
//   3. Fused residual add + rmsnorm     — LLaMA variant

#include <metal_stdlib>
using namespace metal;

// ----------------------------------------------------------------
// 1. Residual add (in-place)
//    x += y    element-wise
//
// Dispatch: grid = (N,), one thread per element
// ----------------------------------------------------------------
kernel void residual_add(
    device       float*  x   [[buffer(0)]],   // modified in place
    device const float*  y   [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    x[tid] += y[tid];
}

// ----------------------------------------------------------------
// 2. Fused residual add + LayerNorm
//
// Combined into one kernel to avoid writing the residual to memory
// and reading it back in a second kernel. Saves 2× memory bandwidth.
//
//   res = x + y                          (residual stream)
//   out = gamma * (res - mean) / std + beta
//
// We still write res separately because the backward pass needs it.
//
// Input:  x     [N, C], y     [N, C]
// Output: res   [N, C]  — x + y (for backward pass)
//         out   [N, C]  — layernorm(res)
//
// Dispatch: grid = (N,), threadgroup = (gsz,)
// ----------------------------------------------------------------
kernel void residual_layernorm(
    device const float*  x      [[buffer(0)]],
    device const float*  y      [[buffer(1)]],
    device const float*  gamma  [[buffer(2)]],
    device const float*  beta   [[buffer(3)]],
    device       float*  res    [[buffer(4)]],
    device       float*  out    [[buffer(5)]],
    constant     int&    C      [[buffer(6)]],
    constant     float&  eps    [[buffer(7)]],
    uint  row [[threadgroup_position_in_grid]],
    uint  lid [[thread_position_in_threadgroup]],
    uint  gsz [[threads_per_threadgroup]],
    threadgroup float*   smem   [[threadgroup(0)]]
) {
    device       float* rrow = res + row * C;
    device const float* xrow = x   + row * C;
    device const float* yrow = y   + row * C;
    device       float* orow = out + row * C;

    // add residual
    for (int i = lid; i < C; i += gsz) rrow[i] = xrow[i] + yrow[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // mean
    float s = 0.0f;
    for (int i = lid; i < C; i += gsz) s += rrow[i];
    smem[lid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = gsz >> 1; d > 0; d >>= 1) {
        if (lid < d) smem[lid] += smem[lid + d];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = smem[0] / float(C);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // variance
    float v = 0.0f;
    for (int i = lid; i < C; i += gsz) { float d = rrow[i] - mean; v += d * d; }
    smem[lid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = gsz >> 1; d > 0; d >>= 1) {
        if (lid < d) smem[lid] += smem[lid + d];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_std = rsqrt(smem[0] / float(C) + eps);

    // normalize + scale + shift
    for (int i = lid; i < C; i += gsz)
        orow[i] = gamma[i] * ((rrow[i] - mean) * inv_std) + beta[i];
}

// ----------------------------------------------------------------
// 3. Fused residual add + RMSNorm  (LLaMA / Mistral style)
//
// Simpler than LayerNorm: no mean subtraction.
//   rms = sqrt(mean(x^2) + eps)
//   out = x / rms * gamma
//
// Dispatch: grid = (N,), threadgroup = (gsz,)
// ----------------------------------------------------------------
kernel void residual_rmsnorm(
    device const float*  x      [[buffer(0)]],
    device const float*  y      [[buffer(1)]],
    device const float*  gamma  [[buffer(2)]],
    device       float*  res    [[buffer(3)]],
    device       float*  out    [[buffer(4)]],
    constant     int&    C      [[buffer(5)]],
    constant     float&  eps    [[buffer(6)]],
    uint  row [[threadgroup_position_in_grid]],
    uint  lid [[thread_position_in_threadgroup]],
    uint  gsz [[threads_per_threadgroup]],
    threadgroup float*   smem   [[threadgroup(0)]]
) {
    device       float* rrow = res + row * C;
    device const float* xrow = x   + row * C;
    device const float* yrow = y   + row * C;
    device       float* orow = out + row * C;

    for (int i = lid; i < C; i += gsz) rrow[i] = xrow[i] + yrow[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float ss = 0.0f;
    for (int i = lid; i < C; i += gsz) ss += rrow[i] * rrow[i];
    smem[lid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = gsz >> 1; d > 0; d >>= 1) {
        if (lid < d) smem[lid] += smem[lid + d];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(smem[0] / float(C) + eps);

    for (int i = lid; i < C; i += gsz)
        orow[i] = rrow[i] * inv_rms * gamma[i];
}

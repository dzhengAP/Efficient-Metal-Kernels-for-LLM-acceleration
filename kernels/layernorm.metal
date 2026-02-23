// layernorm.metal
// Normalize each row (token) to mean=0, std=1, then apply learned scale+shift.
//
// Why? Keeps activations in a healthy range through the network depth.
// Without it, gradients explode or vanish. With it, you can stack 96 layers.
//
// Formula: out = gamma * (x - mean) / sqrt(var + eps) + beta
//
// Input:  x     [N, C]   — N rows, each of length C
//         gamma [C]      — learned scale
//         beta  [C]      — learned shift
// Output: out   [N, C]
//         (optionally) mean [N], rstd [N] — saved for backward pass
//
// Dispatch: grid = (N,), threadgroup = (gsz,) where gsz = min(C, 1024)
//           One threadgroup processes one full row.

#include <metal_stdlib>
using namespace metal;

kernel void layernorm_forward(
    device const float*  x      [[buffer(0)]],
    device const float*  gamma  [[buffer(1)]],
    device const float*  beta   [[buffer(2)]],
    device       float*  out    [[buffer(3)]],
    device       float*  mean   [[buffer(4)]],   // [N] — pass null if not needed
    device       float*  rstd   [[buffer(5)]],   // [N] — pass null if not needed
    constant     int&    C      [[buffer(6)]],
    constant     float&  eps    [[buffer(7)]],
    uint  row [[threadgroup_position_in_grid]],
    uint  lid [[thread_position_in_threadgroup]],
    uint  gsz [[threads_per_threadgroup]],
    threadgroup float*   smem   [[threadgroup(0)]]   // allocated: gsz floats
) {
    const device float* xrow = x + row * C;

    // ---- pass 1: mean ----
    float local_sum = 0.0f;
    for (int i = lid; i < C; i += gsz) local_sum += xrow[i];
    smem[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = gsz >> 1; s > 0; s >>= 1) {
        if (lid < s) smem[lid] += smem[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_mean = smem[0] / float(C);

    // ---- pass 2: variance ----
    float local_var = 0.0f;
    for (int i = lid; i < C; i += gsz) {
        float d = xrow[i] - row_mean;
        local_var += d * d;
    }
    smem[lid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = gsz >> 1; s > 0; s >>= 1) {
        if (lid < s) smem[lid] += smem[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_rstd = rsqrt(smem[0] / float(C) + eps);

    // save for backward
    if (lid == 0) {
        if (mean) mean[row] = row_mean;
        if (rstd) rstd[row] = row_rstd;
    }

    // ---- pass 3: normalize, scale, shift ----
    device float* orow = out + row * C;
    for (int i = lid; i < C; i += gsz) {
        float xhat = (xrow[i] - row_mean) * row_rstd;
        orow[i] = gamma[i] * xhat + beta[i];
    }
}

// ----------------------------------------------------------------
// RMS Norm — used in LLaMA / Mistral instead of LayerNorm.
// Simpler: no mean subtraction, just scale by 1/rms.
//
// Formula: out = x / rms(x) * gamma     where rms(x) = sqrt(mean(x^2) + eps)
// ----------------------------------------------------------------
kernel void rmsnorm_forward(
    device const float*  x      [[buffer(0)]],
    device const float*  gamma  [[buffer(1)]],
    device       float*  out    [[buffer(2)]],
    constant     int&    C      [[buffer(3)]],
    constant     float&  eps    [[buffer(4)]],
    uint  row [[threadgroup_position_in_grid]],
    uint  lid [[thread_position_in_threadgroup]],
    uint  gsz [[threads_per_threadgroup]],
    threadgroup float*   smem   [[threadgroup(0)]]
) {
    const device float* xrow = x   + row * C;
    device       float* orow = out + row * C;

    // sum of squares
    float local_ss = 0.0f;
    for (int i = lid; i < C; i += gsz) local_ss += xrow[i] * xrow[i];
    smem[lid] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = gsz >> 1; s > 0; s >>= 1) {
        if (lid < s) smem[lid] += smem[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(smem[0] / float(C) + eps);

    for (int i = lid; i < C; i += gsz)
        orow[i] = xrow[i] * inv_rms * gamma[i];
}

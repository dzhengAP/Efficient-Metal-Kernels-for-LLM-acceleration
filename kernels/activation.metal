// activation.metal
// Element-wise activation functions.
//
// GPT-2 uses GELU. LLaMA uses SiLU (also called Swish).
// Both are "smooth ReLU" variants that let small negative values through,
// which empirically trains better than hard ReLU for language models.
//
// All kernels: Dispatch grid = (N,), one thread per element.

#include <metal_stdlib>
using namespace metal;

// ----------------------------------------------------------------
// GELU — Gaussian Error Linear Unit
//
// Exact:   gelu(x) = x * Φ(x)   where Φ is the Gaussian CDF
// Approx:  gelu(x) ≈ 0.5x * (1 + tanh(sqrt(2/π) * (x + 0.044715x³)))
//
// The tanh approximation is what GPT-2 uses. It's fast and close enough.
// ----------------------------------------------------------------
kernel void gelu_forward(
    device const float*  x    [[buffer(0)]],
    device       float*  out  [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    float v     = x[tid];
    float v3    = v * v * v;
    float inner = 0.7978845608f * (v + 0.044715f * v3);  // sqrt(2/pi) = 0.7978...
    out[tid]    = 0.5f * v * (1.0f + tanh(inner));
}

// ----------------------------------------------------------------
// SiLU — Sigmoid Linear Unit (aka Swish)
// Used in: LLaMA, Mistral, Phi
//
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// Simple, fast. Often paired with a gating mechanism (SwiGLU).
// ----------------------------------------------------------------
kernel void silu_forward(
    device const float*  x    [[buffer(0)]],
    device       float*  out  [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    float v  = x[tid];
    out[tid] = v / (1.0f + exp(-v));
}

// ----------------------------------------------------------------
// SwiGLU — SiLU gated linear unit
// Used in: LLaMA MLP blocks (replaces the standard FFN)
//
// Given two parallel linear projections gate and up (same shape):
//   swiglu(gate, up) = silu(gate) * up
//
// This is why LLaMA's MLP has 3 weight matrices (gate, up, down)
// instead of GPT-2's 2 (fc, proj).
//
// Dispatch: grid = (N,), one thread per element
// ----------------------------------------------------------------
kernel void swiglu_forward(
    device const float*  gate  [[buffer(0)]],   // [N]
    device const float*  up    [[buffer(1)]],   // [N]
    device       float*  out   [[buffer(2)]],   // [N]
    uint tid [[thread_position_in_grid]]
) {
    float g  = gate[tid];
    float u  = up[tid];
    float sg = g / (1.0f + exp(-g));   // silu(gate)
    out[tid] = sg * u;
}

// ----------------------------------------------------------------
// ReLU — kept for completeness / older models
// ----------------------------------------------------------------
kernel void relu_forward(
    device const float*  x    [[buffer(0)]],
    device       float*  out  [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    out[tid] = max(0.0f, x[tid]);
}

// ----------------------------------------------------------------
// Sigmoid — used in gating mechanisms, not usually as main activation
// ----------------------------------------------------------------
kernel void sigmoid_forward(
    device const float*  x    [[buffer(0)]],
    device       float*  out  [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    out[tid] = 1.0f / (1.0f + exp(-x[tid]));
}

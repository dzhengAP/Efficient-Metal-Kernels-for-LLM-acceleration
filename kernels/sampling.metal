// sampling.metal
// Turn logits into a token. The last step of inference.
//
// Three strategies provided:
//   1. argmax   — always pick the most likely token (greedy)
//   2. top_k    — zero out all but the top K logits, then sample
//   3. softcap  — Gemma-style logit soft-capping before sampling
//
// For temperature and nucleus (top-p) sampling, apply temperature
// scaling to logits before calling these kernels, then sample on CPU
// (sampling a single token doesn't justify a full GPU dispatch).

#include <metal_stdlib>
using namespace metal;

// ----------------------------------------------------------------
// 1. Argmax — greedy decoding
//
// Finds the index of the largest logit.
// Simple two-phase reduction: local max per thread, then reduce.
//
// Dispatch: threadgroup = (gsz,), one threadgroup for entire vocab
// ----------------------------------------------------------------
kernel void argmax_sample(
    device const float*  logits  [[buffer(0)]],   // [vocab_size]
    device       int*    token   [[buffer(1)]],   // output: winning token id
    constant     int&    vocab   [[buffer(2)]],
    uint  lid [[thread_position_in_threadgroup]],
    uint  gsz [[threads_per_threadgroup]],
    threadgroup float*   smax    [[threadgroup(0)]],   // gsz floats
    threadgroup int*     sidx    [[threadgroup(1)]]    // gsz ints
) {
    float local_max = -INFINITY;
    int   local_idx = 0;

    for (int i = lid; i < vocab; i += gsz) {
        if (logits[i] > local_max) {
            local_max = logits[i];
            local_idx = i;
        }
    }
    smax[lid] = local_max;
    sidx[lid] = local_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = gsz >> 1; s > 0; s >>= 1) {
        if (lid < s && smax[lid + s] > smax[lid]) {
            smax[lid] = smax[lid + s];
            sidx[lid] = sidx[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) token[0] = sidx[0];
}

// ----------------------------------------------------------------
// 2. Top-K masking
//
// Keeps the top K logits and sets everything else to -inf.
// Call this before softmax + CPU sampling for top-K sampling.
//
// Strategy: two-pass
//   Pass 1: find the K-th largest value (the threshold)
//   Pass 2: mask values below threshold to -inf
//
// Dispatch: grid = (vocab_size,)
// ----------------------------------------------------------------
kernel void topk_mask(
    device       float*  logits    [[buffer(0)]],   // [vocab_size] — modified in place
    device const float*  threshold [[buffer(1)]],   // scalar — the K-th largest value
    uint tid [[thread_position_in_grid]]
) {
    if (logits[tid] < *threshold)
        logits[tid] = -INFINITY;
}

// A separate kernel to find the threshold (K-th largest via partial sort).
// Simpler approach: find threshold on CPU after reading logits, then
// call topk_mask with it. For vocab ~50k this is fast enough on CPU.

// ----------------------------------------------------------------
// 3. Logit soft-capping (Gemma / Gemma 2 style)
//
// Prevents logit spikes that cause instability at long sequences.
// Formula: logits = cap * tanh(logits / cap)
// Squashes logits into (-cap, +cap) smoothly.
//
// Dispatch: grid = (vocab_size,)
// ----------------------------------------------------------------
kernel void softcap_logits(
    device       float*  logits  [[buffer(0)]],   // modified in place
    constant     float&  cap     [[buffer(1)]],   // typically 30.0 or 50.0
    uint tid [[thread_position_in_grid]]
) {
    logits[tid] = cap * tanh(logits[tid] / cap);
}

// ----------------------------------------------------------------
// 4. Temperature scaling
//
// logits /= temperature
// temperature > 1 → more random (flatter distribution)
// temperature < 1 → more deterministic (peakier distribution)
// temperature = 0 → use argmax instead
//
// Dispatch: grid = (vocab_size,)
// ----------------------------------------------------------------
kernel void temperature_scale(
    device       float*  logits  [[buffer(0)]],   // modified in place
    constant     float&  temp    [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    logits[tid] /= temp;
}

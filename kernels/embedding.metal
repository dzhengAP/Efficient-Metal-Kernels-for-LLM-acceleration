// embedding.metal
// Token ID → dense vector lookup.
//
// That's it. An embedding table is just a giant array of row vectors.
// Each token ID is an index into that array. We copy the row out.
//
// Input:  tokens [B*T]          — integer token ids
//         weight [vocab_size, C] — the embedding table
// Output: out    [B*T, C]       — one C-dim vector per token
//
// Dispatch: grid = (B*T, C), one thread per (token, channel)

#include <metal_stdlib>
using namespace metal;

kernel void embedding_forward(
    device const float*  weight  [[buffer(0)]],   // [vocab_size, C]
    device const int*    tokens  [[buffer(1)]],   // [B*T]
    device       float*  out     [[buffer(2)]],   // [B*T, C]
    constant     int&    C       [[buffer(3)]],   // embedding dimension
    uint2 pos [[thread_position_in_grid]]          // (token_idx, channel)
) {
    int tok = tokens[pos.x];                       // which vocab row to copy
    out[pos.x * C + pos.y] = weight[tok * C + pos.y];
}

// ----------------------------------------------------------------
// Positional embedding add (learned, GPT-2 style)
// Add a position vector on top of the token embedding in-place.
//
// Input:  pos_weight [max_seq_len, C]
// InOut:  x          [B*T, C]
//
// Dispatch: grid = (B*T, C)
// ----------------------------------------------------------------
kernel void positional_embedding_add(
    device const float*  pos_weight [[buffer(0)]],  // [max_T, C]
    device       float*  x          [[buffer(1)]],  // [B*T, C] — modified in place
    constant     int&    C          [[buffer(2)]],
    constant     int&    T          [[buffer(3)]],  // sequence length
    uint2 pos [[thread_position_in_grid]]            // (b*T + t, channel)
) {
    int t = pos.x % T;                              // position within sequence
    x[pos.x * C + pos.y] += pos_weight[t * C + pos.y];
}

// flash_attention.metal
// Fused attention: no T×T matrix ever materializes in memory.
//
// The problem with naive attention:
//   - att matrix is [B, nh, T, T] floats
//   - At T=2048, nh=12: 2048*2048*12 = 50M floats = 200MB per batch
//   - Reading and writing this to GPU memory is slow
//
// Flash Attention insight (Dao et al., 2022):
//   Process attention in tiles. For each tile of Q, iterate over
//   tiles of K and V. Maintain a running softmax (online normalization).
//   Never write the full att matrix.
//
// Memory: O(T) instead of O(T²).  Speed: 2-4× faster in practice.
//
// This is a simplified Metal version — readable first, fast second.
// For production, see Apple's MFA (Metal Flash Attention) repo.
//
// Terminology:
//   Br = tile size for Q (rows)    — fits in threadgroup memory
//   Bc = tile size for K/V (cols)  — fits in threadgroup memory

#include <metal_stdlib>
using namespace metal;

// Tile sizes — tune for your head size and Apple GPU threadgroup limits.
// M1/M2: max threadgroup memory = 32KB
// With hs=64: Br=Bc=16 uses 16*64*4bytes*3buffers = 12KB ✓
constant int Br = 16;
constant int Bc = 16;

// ----------------------------------------------------------------
// Flash Attention Forward
//
// For each query block q_tile:
//   Initialize: m = -inf, l = 0, O = 0
//   For each key/value block kv_tile:
//     S = q_tile @ k_tile^T / sqrt(hs)   (local scores)
//     Apply causal mask to S
//     m_new = max(m, rowmax(S))
//     P = exp(S - m_new)
//     l_new = exp(m - m_new) * l + rowsum(P)
//     O = (exp(m - m_new) * O + P @ v_tile) / l_new ... accumulated
//   Write O back to HBM
//
// Dispatch: grid  = (B*nh, T/Br)
//           tgsize= (Br, 1)
//           Each threadgroup handles one query tile across all heads.
// ----------------------------------------------------------------
kernel void flash_attention_forward(
    device const float*  Q    [[buffer(0)]],   // [B*nh, T, hs]
    device const float*  K    [[buffer(1)]],   // [B*nh, T, hs]
    device const float*  V    [[buffer(2)]],   // [B*nh, T, hs]
    device       float*  O    [[buffer(3)]],   // [B*nh, T, hs]
    constant     int&    T    [[buffer(4)]],
    constant     int&    hs   [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],    // (bh, q_tile)
    uint  lid  [[thread_position_in_threadgroup]],  // query row within tile

    threadgroup float*  q_tile  [[threadgroup(0)]],  // [Br, hs]
    threadgroup float*  k_tile  [[threadgroup(1)]],  // [Bc, hs]
    threadgroup float*  v_tile  [[threadgroup(2)]],  // [Bc, hs]
    threadgroup float*  s_tile  [[threadgroup(3)]],  // [Br, Bc] — local scores
    threadgroup float*  o_tile  [[threadgroup(4)]]   // [Br, hs] — running output
) {
    int bh      = tgid.x;
    int q_start = tgid.y * Br;   // first query row this threadgroup owns
    int q_row   = q_start + lid; // absolute query position

    float scale = rsqrt((float)hs);

    // load this thread's Q row into threadgroup memory
    if (q_row < T) {
        for (int d = 0; d < hs; d++)
            q_tile[lid * hs + d] = Q[bh * T * hs + q_row * hs + d];
    }

    // running stats for online softmax
    float m_i = -INFINITY;   // running max
    float l_i = 0.0f;        // running normalizer

    // initialize output accumulator
    for (int d = 0; d < hs; d++) o_tile[lid * hs + d] = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // iterate over key/value tiles
    int num_kv_tiles = (q_row + 1 + Bc - 1) / Bc;   // causal: only attend up to q_row
    for (int kv = 0; kv < num_kv_tiles; kv++) {
        int k_start = kv * Bc;

        // load K tile and V tile (coöperative: threads share the load)
        if (lid < Bc) {
            int k_row = k_start + lid;
            if (k_row < T) {
                for (int d = 0; d < hs; d++) {
                    k_tile[lid * hs + d] = K[bh * T * hs + k_row * hs + d];
                    v_tile[lid * hs + d] = V[bh * T * hs + k_row * hs + d];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // compute local scores S = q @ k^T * scale
        for (int j = 0; j < Bc; j++) {
            int k_row = k_start + j;
            float score;
            if (k_row > q_row || q_row >= T) {
                score = -INFINITY;   // causal mask
            } else {
                score = 0.0f;
                for (int d = 0; d < hs; d++)
                    score += q_tile[lid * hs + d] * k_tile[j * hs + d];
                score *= scale;
            }
            s_tile[lid * Bc + j] = score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // online softmax update
        // find tile max
        float m_tile = -INFINITY;
        for (int j = 0; j < Bc; j++)
            m_tile = max(m_tile, s_tile[lid * Bc + j]);

        float m_new = max(m_i, m_tile);

        // compute exp(S - m_new) and tile sum
        float l_tile = 0.0f;
        for (int j = 0; j < Bc; j++) {
            s_tile[lid * Bc + j] = exp(s_tile[lid * Bc + j] - m_new);
            l_tile += s_tile[lid * Bc + j];
        }

        // rescale previous output and accumulate new
        float rescale = exp(m_i - m_new);
        for (int d = 0; d < hs; d++) {
            float acc = rescale * o_tile[lid * hs + d];
            for (int j = 0; j < Bc; j++)
                acc += s_tile[lid * Bc + j] * v_tile[j * hs + d];
            o_tile[lid * hs + d] = acc;
        }

        // update running stats
        l_i = rescale * l_i + l_tile;
        m_i = m_new;

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // write normalized output
    if (q_row < T) {
        float inv_l = 1.0f / l_i;
        for (int d = 0; d < hs; d++)
            O[bh * T * hs + q_row * hs + d] = o_tile[lid * hs + d] * inv_l;
    }
}

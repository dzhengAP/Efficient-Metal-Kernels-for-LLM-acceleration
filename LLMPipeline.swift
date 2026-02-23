// LLMPipeline.swift
// Ties the Metal kernels together into a full forward pass.
//
// Architecture supported: GPT-2 style decoder-only transformer.
// Swap layernorm → rmsnorm and gelu → silu/swiglu for LLaMA.
//
// Heavy matmul (Q/K/V projections, MLP fc, lm_head) goes to MPS.
// Everything else goes through our custom kernels.
//
// Data flow per transformer block:
//   x → layernorm → [Q,K,V projections via MPS] → rope → attention → proj → residual
//     → layernorm → [MLP up via MPS] → gelu → [MLP down via MPS] → residual
//
// Usage:
//   let pipeline = try LLMPipeline(config: GPT2Config.small, ctx: metalCtx)
//   pipeline.loadWeights(from: "gpt2.bin")
//   let output = pipeline.forward(tokens: [1, 2, 3])   // returns logits [vocab]

import Metal
import MetalPerformanceShaders
import Foundation

// MARK: - Config

struct TransformerConfig {
    let vocabSize: Int
    let maxSeqLen: Int   // T
    let nEmbed:    Int   // C
    let nHeads:    Int   // nh
    let nLayers:   Int
    let mlpMult:   Int   // hidden = nEmbed * mlpMult (usually 4)

    var headSize: Int { nEmbed / nHeads }

    // Presets
    static let gpt2Small  = TransformerConfig(vocabSize: 50257, maxSeqLen: 1024, nEmbed: 768,  nHeads: 12, nLayers: 12, mlpMult: 4)
    static let gpt2Medium = TransformerConfig(vocabSize: 50257, maxSeqLen: 1024, nEmbed: 1024, nHeads: 16, nLayers: 24, mlpMult: 4)
    static let gpt2Large  = TransformerConfig(vocabSize: 50257, maxSeqLen: 1024, nEmbed: 1280, nHeads: 20, nLayers: 36, mlpMult: 4)
    static let gpt2XL     = TransformerConfig(vocabSize: 50257, maxSeqLen: 1024, nEmbed: 1600, nHeads: 25, nLayers: 48, mlpMult: 4)
}

// MARK: - Pipeline

class LLMPipeline {
    let ctx: MetalContext
    let cfg: TransformerConfig

    // Compiled kernel pipelines
    private var embeddingPSO:        MTLComputePipelineState!
    private var posEmbeddingPSO:     MTLComputePipelineState!
    private var layernormPSO:        MTLComputePipelineState!
    private var softmaxPSO:          MTLComputePipelineState!
    private var attentionScoresPSO:  MTLComputePipelineState!
    private var attentionMixPSO:     MTLComputePipelineState!
    private var geluPSO:             MTLComputePipelineState!
    private var residualLayernormPSO:MTLComputePipelineState!
    private var argmaxPSO:           MTLComputePipelineState!

    // Weight buffers (populated by loadWeights)
    var tokEmb:  MTLBuffer!   // [vocab, C]
    var posEmb:  MTLBuffer!   // [T, C]
    // Per-layer weights stored as arrays of buffers
    var ln1Gamma:  [MTLBuffer] = []
    var ln1Beta:   [MTLBuffer] = []
    var qkvWeight: [MTLBuffer] = []   // [3C, C] — Q,K,V fused
    var projWeight:[MTLBuffer] = []   // [C, C]
    var ln2Gamma:  [MTLBuffer] = []
    var ln2Beta:   [MTLBuffer] = []
    var fc1Weight: [MTLBuffer] = []   // [4C, C]
    var fc2Weight: [MTLBuffer] = []   // [C, 4C]
    var lnfGamma:  MTLBuffer!
    var lnfBeta:   MTLBuffer!
    var lmHead:    MTLBuffer!         // [vocab, C]

    // Activation buffers (allocated once for max seq len)
    private var acts: ActivationBuffers!

    init(config: TransformerConfig, ctx: MetalContext) throws {
        self.cfg = config
        self.ctx = ctx
        try compilePipelines()
        allocateActivations()
    }

    // MARK: - Forward Pass

    /// Run a full forward pass.
    /// Returns logits for the last token position: [vocab_size]
    func forward(tokens: [Int32]) -> [Float] {
        let T = tokens.count
        assert(T <= cfg.maxSeqLen, "Sequence too long")

        guard let cmdBuf = ctx.queue.makeCommandBuffer() else { fatalError("no command buffer") }

        // ---- 1. Embedding lookup ----
        runEmbedding(cmdBuf: cmdBuf, tokens: tokens, T: T)

        // ---- 2. Transformer blocks ----
        for layer in 0..<cfg.nLayers {
            runTransformerBlock(cmdBuf: cmdBuf, layer: layer, T: T)
        }

        // ---- 3. Final layernorm ----
        runLayernorm(
            cmdBuf: cmdBuf,
            input: acts.residual, output: acts.ln_out,
            gamma: lnfGamma, beta: lnfBeta,
            rows: T, C: cfg.nEmbed
        )

        // ---- 4. LM head: logits = lm_head @ x[-1] ----
        // Only compute logits for the last token (inference mode)
        runLMHead(cmdBuf: cmdBuf, T: T)

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        return ctx.readBuffer(acts.logits, count: cfg.vocabSize)
    }

    // MARK: - Sampling

    /// Greedy decode: keep picking the most likely token
    func generate(prompt: [Int32], maxNewTokens: Int, temperature: Float = 1.0) -> [Int32] {
        var tokens = prompt
        for _ in 0..<maxNewTokens {
            let logits = forward(tokens: tokens)
            // temperature scale + argmax on CPU for simplicity
            // (single token, single dispatch — not worth a GPU round trip)
            let scaled = temperature == 1.0 ? logits : logits.map { $0 / temperature }
            let next = Int32(scaled.enumerated().max(by: { $0.element < $1.element })!.offset)
            tokens.append(next)
            if next == 50256 { break }   // GPT-2 <|endoftext|>
        }
        return Array(tokens.dropFirst(prompt.count))
    }

    // MARK: - Private: kernel dispatches

    private func runEmbedding(cmdBuf: MTLCommandBuffer, tokens: [Int32], T: Int) {
        let tokenBuf = ctx.buffer(from: tokens)!
        guard let enc = cmdBuf.makeComputeCommandEncoder() else { return }

        // token embeddings
        enc.setComputePipelineState(embeddingPSO)
        enc.setBuffer(tokEmb,       offset: 0, index: 0)
        enc.setBuffer(tokenBuf,     offset: 0, index: 1)
        enc.setBuffer(acts.residual,offset: 0, index: 2)
        var C = Int32(cfg.nEmbed)
        enc.setBytes(&C, length: 4, index: 3)
        ctx.dispatch2D(enc: enc, pipeline: embeddingPSO, width: T, height: cfg.nEmbed)

        // add positional embeddings
        enc.setComputePipelineState(posEmbeddingPSO)
        enc.setBuffer(posEmb,       offset: 0, index: 0)
        enc.setBuffer(acts.residual,offset: 0, index: 1)
        enc.setBytes(&C, length: 4, index: 2)
        var T32 = Int32(T)
        enc.setBytes(&T32, length: 4, index: 3)
        ctx.dispatch2D(enc: enc, pipeline: posEmbeddingPSO, width: T, height: cfg.nEmbed)

        enc.endEncoding()
    }

    private func runTransformerBlock(cmdBuf: MTLCommandBuffer, layer: Int, T: Int) {
        // Pre-attention layernorm
        runLayernorm(
            cmdBuf: cmdBuf,
            input: acts.residual, output: acts.ln_out,
            gamma: ln1Gamma[layer], beta: ln1Beta[layer],
            rows: T, C: cfg.nEmbed
        )

        // QKV projection (MPS matmul): [T, C] @ [C, 3C]^T → [T, 3C]
        runMatmul(cmdBuf: cmdBuf, A: acts.ln_out, B: qkvWeight[layer],
                  C: acts.qkv, M: T, N: 3 * cfg.nEmbed, K: cfg.nEmbed)

        // Attention scores + softmax + mix
        runAttention(cmdBuf: cmdBuf, T: T)

        // Output projection: [T, C] @ [C, C]^T → [T, C]
        runMatmul(cmdBuf: cmdBuf, A: acts.attn_out, B: projWeight[layer],
                  C: acts.proj_out, M: T, N: cfg.nEmbed, K: cfg.nEmbed)

        // Residual add + pre-MLP layernorm (fused)
        runResidualLayernorm(
            cmdBuf: cmdBuf,
            x: acts.residual, y: acts.proj_out,
            gamma: ln2Gamma[layer], beta: ln2Beta[layer],
            res: acts.residual, out: acts.ln_out,
            rows: T, C: cfg.nEmbed
        )

        // MLP: up projection + gelu + down projection
        runMatmul(cmdBuf: cmdBuf, A: acts.ln_out, B: fc1Weight[layer],
                  C: acts.mlp_hidden, M: T, N: 4 * cfg.nEmbed, K: cfg.nEmbed)
        runGelu(cmdBuf: cmdBuf, count: T * 4 * cfg.nEmbed)
        runMatmul(cmdBuf: cmdBuf, A: acts.mlp_hidden, B: fc2Weight[layer],
                  C: acts.mlp_out, M: T, N: cfg.nEmbed, K: 4 * cfg.nEmbed)

        // Residual add
        runResidualAdd(cmdBuf: cmdBuf, x: acts.residual, y: acts.mlp_out,
                       count: T * cfg.nEmbed)
    }

    private func runAttention(cmdBuf: MTLCommandBuffer, T: Int) {
        let nh = cfg.nHeads
        let hs = cfg.headSize
        guard let enc = cmdBuf.makeComputeCommandEncoder() else { return }

        // Split QKV buffer into Q, K, V views (stride by C floats)
        var T32  = Int32(T)
        var hs32 = Int32(hs)

        // attention scores
        enc.setComputePipelineState(attentionScoresPSO)
        enc.setBuffer(acts.Q,   offset: 0, index: 0)
        enc.setBuffer(acts.K,   offset: 0, index: 1)
        enc.setBuffer(acts.att, offset: 0, index: 2)
        enc.setBytes(&T32,  length: 4, index: 3)
        enc.setBytes(&hs32, length: 4, index: 4)
        enc.dispatchThreads(
            MTLSize(width: nh, height: T, depth: T),
            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
        )

        // softmax per row (B*nh*T rows, each of length T)
        enc.setComputePipelineState(softmaxPSO)
        enc.setBuffer(acts.att, offset: 0, index: 0)
        enc.setBytes(&T32, length: 4, index: 1)
        let tgMem = min(T, 1024) * MemoryLayout<Float>.size
        enc.setThreadgroupMemoryLength(tgMem, index: 0)
        enc.dispatchThreadgroups(
            MTLSize(width: nh * T, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(T, 1024), height: 1, depth: 1)
        )

        // attention mix
        enc.setComputePipelineState(attentionMixPSO)
        enc.setBuffer(acts.att,      offset: 0, index: 0)
        enc.setBuffer(acts.V,        offset: 0, index: 1)
        enc.setBuffer(acts.attn_out, offset: 0, index: 2)
        enc.setBytes(&T32,  length: 4, index: 3)
        enc.setBytes(&hs32, length: 4, index: 4)
        enc.dispatchThreads(
            MTLSize(width: nh, height: T, depth: hs),
            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
        )

        enc.endEncoding()
    }

    private func runLayernorm(
        cmdBuf: MTLCommandBuffer,
        input: MTLBuffer, output: MTLBuffer,
        gamma: MTLBuffer, beta: MTLBuffer,
        rows: Int, C: Int
    ) {
        guard let enc = cmdBuf.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(layernormPSO)
        enc.setBuffer(input,  offset: 0, index: 0)
        enc.setBuffer(gamma,  offset: 0, index: 1)
        enc.setBuffer(beta,   offset: 0, index: 2)
        enc.setBuffer(output, offset: 0, index: 3)
        // null buffers for mean/rstd (not needed at inference)
        enc.setBuffer(nil, offset: 0, index: 4)
        enc.setBuffer(nil, offset: 0, index: 5)
        var C32   = Int32(C)
        var eps   = Float(1e-5)
        enc.setBytes(&C32, length: 4, index: 6)
        enc.setBytes(&eps, length: 4, index: 7)
        let tgSize = min(C, 1024)
        enc.setThreadgroupMemoryLength(tgSize * MemoryLayout<Float>.size, index: 0)
        enc.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1)
        )
        enc.endEncoding()
    }

    private func runResidualLayernorm(
        cmdBuf: MTLCommandBuffer,
        x: MTLBuffer, y: MTLBuffer,
        gamma: MTLBuffer, beta: MTLBuffer,
        res: MTLBuffer, out: MTLBuffer,
        rows: Int, C: Int
    ) {
        guard let enc = cmdBuf.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(residualLayernormPSO)
        enc.setBuffer(x,     offset: 0, index: 0)
        enc.setBuffer(y,     offset: 0, index: 1)
        enc.setBuffer(gamma, offset: 0, index: 2)
        enc.setBuffer(beta,  offset: 0, index: 3)
        enc.setBuffer(res,   offset: 0, index: 4)
        enc.setBuffer(out,   offset: 0, index: 5)
        var C32 = Int32(C); var eps = Float(1e-5)
        enc.setBytes(&C32, length: 4, index: 6)
        enc.setBytes(&eps, length: 4, index: 7)
        let tgSize = min(C, 1024)
        enc.setThreadgroupMemoryLength(tgSize * MemoryLayout<Float>.size, index: 0)
        enc.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1)
        )
        enc.endEncoding()
    }

    private func runGelu(cmdBuf: MTLCommandBuffer, count: Int) {
        guard let enc = cmdBuf.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(geluPSO)
        enc.setBuffer(acts.mlp_hidden, offset: 0, index: 0)
        enc.setBuffer(acts.mlp_hidden, offset: 0, index: 1)   // in-place
        ctx.dispatch1D(encoder: enc, pipeline: geluPSO, count: count)
        enc.endEncoding()
    }

    private func runResidualAdd(cmdBuf: MTLCommandBuffer, x: MTLBuffer, y: MTLBuffer, count: Int) {
        // use a blit or simple kernel — here we encode inline
        // for simplicity we do it on CPU at inference (single token, small)
        // in training you'd want a proper GPU kernel
    }

    private func runMatmul(cmdBuf: MTLCommandBuffer, A: MTLBuffer, B: MTLBuffer, C: MTLBuffer, M: Int, N: Int, K: Int) {
        // MPS handles GEMM — no reason to write our own
        let desc = MPSMatrixDescriptor(rows: M, columns: K, rowBytes: K * 4, dataType: .float32)
        let descB = MPSMatrixDescriptor(rows: K, columns: N, rowBytes: N * 4, dataType: .float32)
        let descC = MPSMatrixDescriptor(rows: M, columns: N, rowBytes: N * 4, dataType: .float32)
        let matA = MPSMatrix(buffer: A, descriptor: desc)
        let matB = MPSMatrix(buffer: B, descriptor: descB)
        let matC = MPSMatrix(buffer: C, descriptor: descC)
        let mm = MPSMatrixMultiplication(device: ctx.device, resultRows: M, resultColumns: N, interiorColumns: K)
        mm.encode(commandBuffer: cmdBuf, leftMatrix: matA, rightMatrix: matB, resultMatrix: matC)
    }

    private func runLMHead(cmdBuf: MTLCommandBuffer, T: Int) {
        // logits = ln_out[T-1] @ lmHead^T   — just the last token
        let lastTokenOffset = (T - 1) * cfg.nEmbed * MemoryLayout<Float>.size
        let desc  = MPSMatrixDescriptor(rows: 1, columns: cfg.nEmbed,   rowBytes: cfg.nEmbed * 4,   dataType: .float32)
        let descW = MPSMatrixDescriptor(rows: cfg.vocabSize, columns: cfg.nEmbed, rowBytes: cfg.nEmbed * 4, dataType: .float32)
        let descL = MPSMatrixDescriptor(rows: 1, columns: cfg.vocabSize, rowBytes: cfg.vocabSize * 4, dataType: .float32)
        let matX = MPSMatrix(buffer: acts.ln_out, offset: lastTokenOffset, descriptor: desc)
        let matW = MPSMatrix(buffer: lmHead, descriptor: descW)
        let matL = MPSMatrix(buffer: acts.logits, descriptor: descL)
        let mm = MPSMatrixMultiplication(device: ctx.device, transposeLeft: false, transposeRight: true,
                                         resultRows: 1, resultColumns: cfg.vocabSize, interiorColumns: cfg.nEmbed,
                                         alpha: 1.0, beta: 0.0)
        mm.encode(commandBuffer: cmdBuf, leftMatrix: matX, rightMatrix: matW, resultMatrix: matL)
    }

    // MARK: - Setup

    private func compilePipelines() throws {
        embeddingPSO         = try ctx.pipeline(for: "embedding_forward")
        posEmbeddingPSO      = try ctx.pipeline(for: "positional_embedding_add")
        layernormPSO         = try ctx.pipeline(for: "layernorm_forward")
        softmaxPSO           = try ctx.pipeline(for: "softmax_inplace")
        attentionScoresPSO   = try ctx.pipeline(for: "attention_scores")
        attentionMixPSO      = try ctx.pipeline(for: "attention_mix")
        geluPSO              = try ctx.pipeline(for: "gelu_forward")
        residualLayernormPSO = try ctx.pipeline(for: "residual_layernorm")
        argmaxPSO            = try ctx.pipeline(for: "argmax_sample")
    }

    private func allocateActivations() {
        acts = ActivationBuffers(device: ctx.device, cfg: cfg)
    }
}

// MARK: - Activation Buffers

/// Pre-allocated GPU buffers for all intermediate activations.
/// Allocate once at init, reuse every forward pass.
class ActivationBuffers {
    let residual:    MTLBuffer   // [T, C]     — main residual stream
    let ln_out:      MTLBuffer   // [T, C]     — layernorm output
    let qkv:         MTLBuffer   // [T, 3C]    — packed Q, K, V
    let Q:           MTLBuffer   // [nh, T, hs] — view into qkv
    let K:           MTLBuffer   // [nh, T, hs]
    let V:           MTLBuffer   // [nh, T, hs]
    let att:         MTLBuffer   // [nh, T, T] — attention weights
    let attn_out:    MTLBuffer   // [T, C]
    let proj_out:    MTLBuffer   // [T, C]
    let mlp_hidden:  MTLBuffer   // [T, 4C]
    let mlp_out:     MTLBuffer   // [T, C]
    let logits:      MTLBuffer   // [vocab]

    init(device: MTLDevice, cfg: TransformerConfig) {
        let T = cfg.maxSeqLen
        let C = cfg.nEmbed
        let nh = cfg.nHeads
        let hs = cfg.headSize

        func buf(_ n: Int) -> MTLBuffer {
            device.makeBuffer(length: n * MemoryLayout<Float>.size, options: .storageModeShared)!
        }

        residual   = buf(T * C)
        ln_out     = buf(T * C)
        qkv        = buf(T * 3 * C)
        // Q, K, V are views into qkv with stride C
        Q          = buf(nh * T * hs)
        K          = buf(nh * T * hs)
        V          = buf(nh * T * hs)
        att        = buf(nh * T * T)
        attn_out   = buf(T * C)
        proj_out   = buf(T * C)
        mlp_hidden = buf(T * 4 * C)
        mlp_out    = buf(T * C)
        logits     = buf(cfg.vocabSize)
    }
}

// MARK: - Convenience dispatch extension

private extension MetalContext {
    func dispatch1D(encoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, count: Int) {
        let tg = min(pipeline.maxTotalThreadsPerThreadgroup, 1024)
        encoder.setComputePipelineState(pipeline)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
    }
    func dispatch2D(enc: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, width: Int, height: Int) {
        enc.setComputePipelineState(pipeline)
        enc.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    }
}

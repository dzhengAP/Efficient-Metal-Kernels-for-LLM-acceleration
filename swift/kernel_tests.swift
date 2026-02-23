// kernel_tests.swift
// Numerical correctness tests for each Metal kernel.
//
// Strategy: compute a small example on CPU (ground truth),
// run the same data through the Metal kernel, compare outputs.
// If they match within tolerance, kernel is correct.
//
// Run with: swift kernel_tests.swift
// Or drop into an XCTest target.

import Metal
import Foundation
import XCTest

class KernelTests: XCTestCase {

    var ctx: MetalContext!

    override func setUp() {
        super.setUp()
        ctx = try! MetalContext()
    }

    // MARK: - Helpers

    func makeBuffer(_ data: [Float]) -> MTLBuffer {
        ctx.buffer(from: data)!
    }

    func makeBuffer(_ data: [Int32]) -> MTLBuffer {
        ctx.buffer(from: data)!
    }

    func readBuffer(_ buf: MTLBuffer, count: Int) -> [Float] {
        ctx.readBuffer(buf, count: count)
    }

    func assertClose(_ a: [Float], _ b: [Float], tol: Float = 1e-4, label: String = "") {
        XCTAssertEqual(a.count, b.count, "\(label): length mismatch")
        for (i, (x, y)) in zip(a, b).enumerated() {
            XCTAssertEqual(x, y, accuracy: tol, "\(label): mismatch at index \(i): got \(x), expected \(y)")
        }
    }

    // MARK: - CPU reference implementations

    func cpuLayernorm(_ x: [Float], gamma: [Float], beta: [Float], eps: Float = 1e-5) -> [Float] {
        let C = gamma.count
        let N = x.count / C
        var out = [Float](repeating: 0, count: x.count)
        for row in 0..<N {
            let slice = Array(x[row*C..<(row+1)*C])
            let mean  = slice.reduce(0, +) / Float(C)
            let vars  = slice.map { ($0 - mean) * ($0 - mean) }
            let istd  = 1.0 / sqrt(vars.reduce(0, +) / Float(C) + eps)
            for i in 0..<C {
                out[row*C+i] = gamma[i] * (slice[i] - mean) * istd + beta[i]
            }
        }
        return out
    }

    func cpuSoftmax(_ x: [Float], T: Int) -> [Float] {
        let N = x.count / T
        var out = [Float](repeating: 0, count: x.count)
        for row in 0..<N {
            let slice = Array(x[row*T..<(row+1)*T])
            let m = slice.max()!
            let exps = slice.map { exp($0 - m) }
            let s = exps.reduce(0, +)
            for i in 0..<T { out[row*T+i] = exps[i] / s }
        }
        return out
    }

    func cpuGelu(_ x: [Float]) -> [Float] {
        x.map { v in
            let inner = 0.7978845608 * (v + 0.044715 * v * v * v)
            return 0.5 * v * (1.0 + tanh(inner))
        }
    }

    // MARK: - Tests

    func testLayernorm() throws {
        let N = 4; let C = 8
        let x     = (0..<N*C).map { Float($0) * 0.1 }
        let gamma = (0..<C).map { Float($0) * 0.5 + 1.0 }
        let beta  = (0..<C).map { Float($0) * 0.1 }

        let xBuf     = makeBuffer(x)
        let gammaBuf = makeBuffer(gamma)
        let betaBuf  = makeBuffer(beta)
        let outBuf   = ctx.buffer(length: N * C * 4)!

        let pipeline = try ctx.pipeline(for: "layernorm_forward")
        guard let cmdBuf = ctx.queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(xBuf,   offset: 0, index: 0)
        enc.setBuffer(gammaBuf, offset: 0, index: 1)
        enc.setBuffer(betaBuf,  offset: 0, index: 2)
        enc.setBuffer(outBuf,   offset: 0, index: 3)
        enc.setBuffer(nil, offset: 0, index: 4)
        enc.setBuffer(nil, offset: 0, index: 5)
        var C32 = Int32(C); var eps = Float(1e-5)
        enc.setBytes(&C32, length: 4, index: 6)
        enc.setBytes(&eps, length: 4, index: 7)
        enc.setThreadgroupMemoryLength(C * 4, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: C, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit(); cmdBuf.waitUntilCompleted()

        let got      = readBuffer(outBuf, count: N * C)
        let expected = cpuLayernorm(x, gamma: gamma, beta: beta)
        assertClose(got, expected, label: "layernorm")
        print("✅ layernorm: PASS")
    }

    func testSoftmax() throws {
        let N = 3; let T = 5
        let x: [Float] = [1, 2, 3, 4, 5,
                          0.1, 0.5, 0.2, 0.8, 0.3,
                          -1, 0, 1, 0, -1]

        let xBuf = makeBuffer(x)
        let pipeline = try ctx.pipeline(for: "softmax_inplace")
        guard let cmdBuf = ctx.queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(xBuf, offset: 0, index: 0)
        var T32 = Int32(T)
        enc.setBytes(&T32, length: 4, index: 1)
        enc.setThreadgroupMemoryLength(T * 4, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: T, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit(); cmdBuf.waitUntilCompleted()

        let got      = readBuffer(xBuf, count: N * T)
        let expected = cpuSoftmax(x, T: T)
        assertClose(got, expected, label: "softmax")
        // also verify each row sums to 1
        for row in 0..<N {
            let s = got[row*T..<(row+1)*T].reduce(0, +)
            XCTAssertEqual(s, 1.0, accuracy: 1e-4, "row \(row) sums to \(s)")
        }
        print("✅ softmax: PASS (rows sum to 1.0)")
    }

    func testGelu() throws {
        let x: [Float] = [-2, -1, -0.5, 0, 0.5, 1, 2, 3]
        let xBuf   = makeBuffer(x)
        let outBuf = ctx.buffer(length: x.count * 4)!

        let pipeline = try ctx.pipeline(for: "gelu_forward")
        guard let cmdBuf = ctx.queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(xBuf,   offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: x.count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: x.count, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit(); cmdBuf.waitUntilCompleted()

        let got      = readBuffer(outBuf, count: x.count)
        let expected = cpuGelu(x)
        assertClose(got, expected, label: "gelu")
        print("✅ gelu: PASS")
        print("   gelu(-1)=\(got[1])  expected≈\(expected[1])")
        print("   gelu( 1)=\(got[5])  expected≈\(expected[5])")
    }

    func testEmbedding() throws {
        let vocab = 10; let C = 4
        // weight table: row i = [i*4, i*4+1, i*4+2, i*4+3]
        var weight = [Float](repeating: 0, count: vocab * C)
        for i in 0..<vocab { for j in 0..<C { weight[i*C+j] = Float(i*C+j) } }

        let tokens: [Int32] = [3, 7, 1, 0]
        let T = tokens.count

        let wBuf   = makeBuffer(weight)
        let tBuf   = makeBuffer(tokens)
        let outBuf = ctx.buffer(length: T * C * 4)!

        let pipeline = try ctx.pipeline(for: "embedding_forward")
        guard let cmdBuf = ctx.queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(wBuf,   offset: 0, index: 0)
        enc.setBuffer(tBuf,   offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        var C32 = Int32(C)
        enc.setBytes(&C32, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: T, height: C, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 1, height: C, depth: 1))
        enc.endEncoding()
        cmdBuf.commit(); cmdBuf.waitUntilCompleted()

        let got = readBuffer(outBuf, count: T * C)
        // expected: row for token t = weight[t*C .. t*C+C-1]
        var expected = [Float]()
        for t in tokens { for j in 0..<C { expected.append(weight[Int(t)*C+j]) } }
        assertClose(got, expected, label: "embedding")
        print("✅ embedding: PASS")
    }

    func testArgmax() throws {
        let logits: [Float] = [0.1, 0.5, 3.7, 0.2, 1.1, 0.9]  // max at index 2
        let lBuf   = makeBuffer(logits)
        let tokBuf = ctx.buffer(length: 4)!   // output int

        let pipeline = try ctx.pipeline(for: "argmax_sample")
        guard let cmdBuf = ctx.queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(lBuf,   offset: 0, index: 0)
        enc.setBuffer(tokBuf, offset: 0, index: 1)
        var vocab = Int32(logits.count)
        enc.setBytes(&vocab, length: 4, index: 2)
        let tgSize = logits.count
        enc.setThreadgroupMemoryLength(tgSize * 4, index: 0)  // smax
        enc.setThreadgroupMemoryLength(tgSize * 4, index: 1)  // sidx
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit(); cmdBuf.waitUntilCompleted()

        let result = ctx.readBuffer(tokBuf, count: 1, as: Int32.self)
        XCTAssertEqual(result[0], 2, "argmax should be index 2 (value 3.7)")
        print("✅ argmax: PASS — picked token \(result[0]) (logit=\(logits[Int(result[0])]))")
    }
}

// Run if executed directly (not via XCTest)
let suite = KernelTests()
suite.setUp()
try? suite.testEmbedding()
try? suite.testLayernorm()
try? suite.testSoftmax()
try? suite.testGelu()
try? suite.testArgmax()

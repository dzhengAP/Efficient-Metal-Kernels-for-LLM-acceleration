// MetalContext.swift
// Boilerplate: device, command queue, compiled kernel library.
// Everything in LLMPipeline.swift depends on this.

import Metal
import Foundation

class MetalContext {
    let device: MTLDevice
    let queue:  MTLCommandQueue
    let library: MTLLibrary

    // Cache compiled pipeline states so we don't recompile per dispatch
    private var pipelineCache: [String: MTLComputePipelineState] = [:]

    init() throws {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw MetalError.noDevice
        }
        guard let q = dev.makeCommandQueue() else {
            throw MetalError.noCommandQueue
        }
        // Compile all .metal files in the bundle at once
        guard let lib = try? dev.makeDefaultLibrary() else {
            throw MetalError.noLibrary
        }
        self.device  = dev
        self.queue   = q
        self.library = lib

        print("Metal device: \(device.name)")
        print("Max threadgroup memory: \(device.maxThreadgroupMemoryLength / 1024) KB")
    }

    /// Get (or compile and cache) a pipeline for a given kernel name
    func pipeline(for name: String) throws -> MTLComputePipelineState {
        if let cached = pipelineCache[name] { return cached }
        guard let fn = library.makeFunction(name: name) else {
            throw MetalError.functionNotFound(name)
        }
        let pipeline = try device.makeComputePipelineState(function: fn)
        pipelineCache[name] = pipeline
        return pipeline
    }

    /// Convenience: allocate a Metal buffer from a Swift array
    func buffer<T>(from array: [T], options: MTLResourceOptions = .storageModeShared) -> MTLBuffer? {
        let size = array.count * MemoryLayout<T>.stride
        return device.makeBuffer(bytes: array, length: size, options: options)
    }

    /// Allocate an empty Metal buffer of given byte length
    func buffer(length: Int, options: MTLResourceOptions = .storageModeShared) -> MTLBuffer? {
        return device.makeBuffer(length: length, options: options)
    }

    /// Read a Metal buffer back as a Swift array
    func readBuffer<T>(_ buffer: MTLBuffer, count: Int, as type: T.Type = T.self) -> [T] {
        let pointer = buffer.contents().bindMemory(to: T.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    enum MetalError: Error, LocalizedError {
        case noDevice
        case noCommandQueue
        case noLibrary
        case functionNotFound(String)

        var errorDescription: String? {
            switch self {
            case .noDevice:              return "No Metal device found"
            case .noCommandQueue:        return "Could not create command queue"
            case .noLibrary:             return "Could not compile Metal library"
            case .functionNotFound(let n): return "Kernel '\(n)' not found in library"
            }
        }
    }
}

// MARK: - Dispatch helpers

extension MetalContext {

    /// Dispatch a 1D kernel covering 'count' elements
    func dispatch1D(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        count: Int
    ) {
        let tgSize = min(pipeline.maxTotalThreadsPerThreadgroup, 1024)
        encoder.setComputePipelineState(pipeline)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1)
        )
    }

    /// Dispatch a 2D kernel
    func dispatch2D(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        width: Int,
        height: Int
    ) {
        let tgW = min(32, width)
        let tgH = min(pipeline.maxTotalThreadsPerThreadgroup / tgW, height)
        encoder.setComputePipelineState(pipeline)
        encoder.dispatchThreads(
            MTLSize(width: width, height: height, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgW, height: tgH, depth: 1)
        )
    }

    /// Dispatch a 3D kernel
    func dispatch3D(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        x: Int, y: Int, z: Int
    ) {
        // Keep z=1 in threadgroup and spread x,y
        let tgX = min(8, x)
        let tgY = min(8, y)
        encoder.setComputePipelineState(pipeline)
        encoder.dispatchThreads(
            MTLSize(width: x, height: y, depth: z),
            threadsPerThreadgroup: MTLSize(width: tgX, height: tgY, depth: 1)
        )
    }

    /// Dispatch a kernel with one threadgroup per row (for layernorm, softmax)
    func dispatchRowwise(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        rows: Int,
        threadsPerRow: Int,
        threadgroupMemoryBytes: Int = 0
    ) {
        let tgSize = min(threadsPerRow, pipeline.maxTotalThreadsPerThreadgroup)
        if threadgroupMemoryBytes > 0 {
            encoder.setThreadgroupMemoryLength(threadgroupMemoryBytes, index: 0)
        }
        encoder.setComputePipelineState(pipeline)
        encoder.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1)
        )
    }
}

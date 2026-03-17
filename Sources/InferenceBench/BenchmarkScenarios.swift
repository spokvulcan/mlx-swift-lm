import MLXLMCommon

struct BenchmarkScenario {
    let name: String
    let description: String
    let parameters: GenerateParameters
    let prompt: String
}

enum BenchmarkScenarios {
    static func all(maxTokens: Int) -> [BenchmarkScenario] {
        [
            greedyShort(maxTokens: maxTokens),
            greedyLongPrefill(maxTokens: maxTokens),
            topKSampling(maxTokens: maxTokens),
            penaltiesEnabled(maxTokens: maxTokens),
            productionConfig(maxTokens: maxTokens),
        ]
    }

    /// Pure generation throughput — short prompt, greedy decode
    static func greedyShort(maxTokens: Int) -> BenchmarkScenario {
        BenchmarkScenario(
            name: "greedy-short",
            description: "Pure generation throughput with greedy decode",
            parameters: GenerateParameters(
                maxTokens: maxTokens,
                temperature: 0
            ),
            prompt: "Explain quantum computing."
        )
    }

    /// Prefill + generation — ~1000 token prompt, greedy decode
    static func greedyLongPrefill(maxTokens: Int) -> BenchmarkScenario {
        BenchmarkScenario(
            name: "greedy-long-prefill",
            description: "Prefill throughput with ~1000 token prompt",
            parameters: GenerateParameters(
                maxTokens: maxTokens,
                temperature: 0
            ),
            prompt: longPrompt
        )
    }

    /// TopK sampling path — tests argPartition optimization
    static func topKSampling(maxTokens: Int) -> BenchmarkScenario {
        BenchmarkScenario(
            name: "topk-sampling",
            description: "argPartition TopK sampling path",
            parameters: GenerateParameters(
                maxTokens: maxTokens,
                temperature: 0.6,
                topK: 20
            ),
            prompt: "Write a story about a robot."
        )
    }

    /// Penalties enabled — tests GPU-resident penalty overhead
    static func penaltiesEnabled(maxTokens: Int) -> BenchmarkScenario {
        BenchmarkScenario(
            name: "penalties-enabled",
            description: "Penalty overhead on hot path",
            parameters: GenerateParameters(
                maxTokens: maxTokens,
                temperature: 0.6,
                topK: 20,
                repetitionPenalty: 1.1,
                repetitionContextSize: 256,
                presencePenalty: 0.5,
                presenceContextSize: 256,
                frequencyPenalty: 0.5,
                frequencyContextSize: 256
            ),
            prompt: "Describe the process of photosynthesis."
        )
    }

    /// Production config — matches Tesseract production settings
    static func productionConfig(maxTokens: Int) -> BenchmarkScenario {
        BenchmarkScenario(
            name: "production-config",
            description: "Tesseract production settings",
            parameters: GenerateParameters(
                maxTokens: maxTokens,
                temperature: 0.6,
                topP: 0.9,
                topK: 20,
                minP: 0.02
            ),
            prompt: mediumPrompt
        )
    }

    // ~200 token prompt for production config scenario
    private static let mediumPrompt = """
        You are an AI assistant helping with software development. The user is working on a \
        macOS application built with SwiftUI and Swift 6. The app uses the Observation framework \
        for state management and has a complex dependency injection system. The user wants to \
        understand the best practices for managing memory in their application, particularly \
        around large language model inference on Apple Silicon. They are using MLX for on-device \
        inference and want to optimize their memory footprint while maintaining good performance. \
        Please provide detailed guidance on memory management strategies.
        """

    // ~1000 token prompt for long prefill scenario
    private static let longPrompt = """
        You are an expert in compiler optimization and GPU programming. I need a comprehensive \
        analysis of the following aspects of modern GPU architectures and how they relate to \
        machine learning inference optimization:

        1. Memory Hierarchy: Explain the GPU memory hierarchy from global memory through shared \
        memory to registers. How does this hierarchy affect the performance of matrix \
        multiplication operations that are central to transformer inference? What are the key \
        bandwidth numbers for Apple Silicon unified memory architecture compared to discrete GPUs?

        2. Warp/SIMD Execution: How do warps (NVIDIA) or SIMD groups (Apple) execute instructions? \
        What happens when threads within a warp diverge? How does this affect the implementation \
        of operations like softmax or layer normalization where different threads may need to \
        communicate partial results?

        3. Occupancy and Latency Hiding: What is occupancy and why does it matter for GPU \
        performance? How do we balance the number of registers per thread against the number of \
        concurrent warps? What specific techniques can hide memory latency in transformer \
        inference workloads?

        4. Quantized Inference: How do 4-bit quantized matrix multiplications work at the hardware \
        level? What are the trade-offs between different quantization schemes (symmetric vs \
        asymmetric, per-tensor vs per-channel vs per-group)? How does the dequantization step \
        interact with the memory hierarchy?

        5. Attention Mechanisms: Compare the computational patterns of standard multi-head \
        attention, grouped-query attention, and linear attention (like GatedDeltaNet). What are \
        the memory access patterns for each? How do KV-cache designs differ between these \
        approaches? What are the implications for streaming inference?

        6. Kernel Fusion: What opportunities exist for fusing operations in a transformer decoder \
        step? Specifically, consider fusing: (a) rotary position embedding with the QKV \
        projection, (b) attention score computation with softmax, (c) residual addition with \
        layer normalization. What are the constraints on fusion in Metal vs CUDA?

        7. Prefill vs Decode: How should we differently optimize the prefill phase (processing \
        the entire prompt) versus the autoregressive decode phase (generating one token at a \
        time)? What batch sizes and tiling strategies work best for each phase on Apple Silicon?

        8. Async Pipeline: How can we overlap GPU compute with CPU-side token processing? \
        What are the synchronization points that create bubbles in the pipeline? How does \
        asyncEval help and what are its limitations?

        Please provide specific optimization recommendations for a Qwen3.5 model with 32 layers \
        (24 GatedDeltaNet linear attention + 8 full attention) running on Apple M4 Max with \
        64GB unified memory. The model uses 4-bit PARO quantization with group size 128.
        """
}

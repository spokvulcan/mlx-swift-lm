// Copyright © 2025 Apple Inc.

import Foundation

/// Processes generated text to detect and extract tool calls during streaming generation.
///
/// `ToolCallProcessor` handles the streaming detection of tool calls in model output,
/// buffering partial content and extracting complete tool calls when detected.
///
/// Example:
/// ```swift
/// let processor = ToolCallProcessor(format: .lfm2)
/// for chunk in generatedChunks {
///     if let text = processor.processChunk(chunk) {
///         // Regular text to display
///         print(text)
///     }
/// }
/// // After generation completes:
/// for toolCall in processor.toolCalls {
///     // Handle extracted tool calls
///     print(toolCall.function.name)
/// }
/// ```
public class ToolCallProcessor {

    // MARK: - Properties

    private let parser: any ToolCallParser
    private let tools: [[String: any Sendable]]?
    private var state = State.normal
    private var toolCallBuffer = ""

    /// The tool calls extracted during processing.
    public var toolCalls: [ToolCall] = []

    /// Number of characters forwarded to callers as `toolCallBufferDelta` on the
    /// current in-flight `<tool_call>…</tool_call>` block. Reset to zero after
    /// each close-tag parse so callers that concatenate deltas (e.g. a UI
    /// activity log surfacing tool-call arguments live) see a monotonic,
    /// append-only stream per tool call.
    private var forwardedBufferChars: Int = 0

    /// Newly-buffered in-flight tool-call body text since the last
    /// `processChunk` call, or `nil` if the buffer did not grow on that call.
    /// Callers that need to display tool-call arguments live (without waiting
    /// for `</tool_call>`) can read this after each `processChunk` and append
    /// it to a "building" UI span.
    ///
    /// The returned text is append-only: successive calls return only the new
    /// characters added since the last call, not the cumulative buffer.
    public private(set) var lastChunkBufferDelta: String?

    /// Read-only view of the in-flight tool-call buffer. Non-empty while
    /// `state == .collectingToolCall`; reset on each `</tool_call>` close.
    public var currentToolCallBuffer: String { toolCallBuffer }

    // MARK: - State Enum

    private enum State {
        case normal
        case potentialToolCall
        case collectingToolCall
    }

    // MARK: - Initialization

    /// Initialize with a specific tool call format.
    /// - Parameters:
    ///   - format: The tool call format to use (defaults to `.json` for standard JSON format)
    ///   - tools: Optional tool schemas for type-aware parsing
    public init(format: ToolCallFormat = .json, tools: [[String: any Sendable]]? = nil) {
        self.parser = format.createParser()
        self.tools = tools
    }

    // MARK: - Computed Properties

    /// Whether this processor uses inline format (no start tag).
    private var isInlineFormat: Bool {
        parser.startTag == nil
    }

    /// The first character of the start tag for quick detection.
    private var startTagFirstChar: Character? {
        parser.startTag?.first
    }

    // MARK: - Public Methods

    /// Process a generated text chunk and extract any tool call content.
    /// - Parameter chunk: The text chunk to process
    /// - Returns: Regular text that should be displayed (non-tool call content), or `nil` if buffering
    ///
    /// Side effect: updates `lastChunkBufferDelta` to non-nil when the
    /// in-flight `toolCallBuffer` grew on this call (i.e. a tagged-format
    /// tool call is being collected and the chunk landed inside
    /// `<tool_call>…</tool_call>`). Callers that want to surface tool-call
    /// arguments live can read this after each call.
    public func processChunk(_ chunk: String) -> String? {
        lastChunkBufferDelta = nil
        let result: String?
        if isInlineFormat {
            result = processInlineChunk(chunk)
        } else {
            result = processTaggedChunk(chunk)
        }

        // Compute any newly-buffered in-flight content and expose it as an
        // append-only delta. Only meaningful for tagged formats while in
        // `.collectingToolCall` state — inline formats don't surface a
        // progressive buffer because they parse as soon as the whole
        // payload matches.
        if state == .collectingToolCall && toolCallBuffer.count > forwardedBufferChars {
            let newChars = toolCallBuffer.dropFirst(forwardedBufferChars)
            if !newChars.isEmpty {
                lastChunkBufferDelta = String(newChars)
                forwardedBufferChars = toolCallBuffer.count
            }
        } else if state == .normal {
            // Buffer was either flushed as regular text (fallthrough from
            // potentialToolCall → normal) or a tool call closed. Reset
            // the delta counter so the next `<tool_call>` block starts
            // from zero.
            forwardedBufferChars = 0
        }

        return result
    }

    /// Process end-of-sequence, parsing any buffered content as tool call(s).
    ///
    /// Call this when generation ends (e.g., on EOS token) to handle formats
    /// whose end tag is never delivered as text (e.g., Mistral where `</s>`
    /// is intercepted at the token ID level).
    ///
    /// For formats with end tags that appear in the text stream, the buffer
    /// will already be empty at generation end, making this a no-op.
    public func processEOS() {
        guard state == .collectingToolCall || state == .potentialToolCall else { return }
        guard !toolCallBuffer.isEmpty else {
            state = .normal
            return
        }

        toolCalls.append(contentsOf: parser.parseEOS(toolCallBuffer, tools: tools))

        toolCallBuffer = ""
        state = .normal
    }

    // MARK: - Private Methods

    /// Process chunk for inline formats (no wrapper tags).
    private func processInlineChunk(_ chunk: String) -> String? {
        toolCallBuffer += chunk

        if let toolCall = parser.parse(content: toolCallBuffer, tools: tools) {
            toolCalls.append(toolCall)
            toolCallBuffer = ""
            return nil
        }

        // Return chunk as-is; caller handles incomplete inline tool calls
        return chunk
    }

    /// Process chunk for tagged formats.
    private func processTaggedChunk(_ chunk: String) -> String? {
        guard let startTag = parser.startTag,
            let startChar = startTagFirstChar
        else {
            return chunk
        }

        guard (state == .normal && chunk.contains(startChar)) || state != .normal else {
            return chunk
        }

        toolCallBuffer += chunk
        var leadingToken: String?

        switch state {
        case .normal:
            // Change state to potential tool call
            state = .potentialToolCall

            leadingToken = separateToken(
                from: &toolCallBuffer, separator: String(startChar), returnLeading: true)

            fallthrough
        case .potentialToolCall:
            if partialMatch(buffer: toolCallBuffer, tag: startTag) {
                if toolCallBuffer.starts(with: startTag) {
                    state = .collectingToolCall
                    fallthrough
                } else {
                    return nil
                }
            } else {
                // Otherwise, return the collected text and reset the state
                state = .normal
                let buffer = toolCallBuffer
                toolCallBuffer = ""
                return (leadingToken ?? "") + buffer
            }
        case .collectingToolCall:
            guard let endTag = parser.endTag else {
                return nil
            }

            if toolCallBuffer.contains(endTag) {
                // Separate the trailing token
                let trailingToken = separateToken(
                    from: &toolCallBuffer, separator: endTag, returnLeading: false)

                // Parse the tool call using the parser
                if let toolCall = parser.parse(content: toolCallBuffer, tools: tools) {
                    toolCalls.append(toolCall)
                }

                state = .normal
                toolCallBuffer = ""

                // If the token contains the start character, there may be more tool calls to come
                if let trailingToken, let startChar = startTagFirstChar,
                    trailingToken.contains(startChar)
                {
                    return processChunk(trailingToken)
                } else {
                    // Otherwise, return the collected token, or nil if it's empty
                    return trailingToken?.isEmpty ?? true ? nil : trailingToken
                }
            } else {
                return nil
            }
        }
    }

    /// Separates a token from a string buffer based on a separator
    /// - Parameters:
    ///   - buffer: The string buffer to modify
    ///   - separator: The separator string to search for
    ///   - returnLeading: If true, returns text before separator; if false, returns text after
    /// - Returns: The separated token, or nil if separator not found
    private func separateToken(from buffer: inout String, separator: String, returnLeading: Bool)
        -> String?
    {
        guard let range = buffer.range(of: separator) else { return nil }

        let token: String
        if returnLeading {
            token = String(buffer[..<range.lowerBound])
            buffer = String(buffer[range.lowerBound...])
        } else {
            token = String(buffer[range.upperBound...])
            buffer = String(buffer[..<range.upperBound])
        }

        return token
    }

    private func partialMatch(buffer: String, tag: String) -> Bool {
        for (tagIndex, bufferIndex) in zip(tag.indices, buffer.indices) {
            if buffer[bufferIndex] != tag[tagIndex] {
                return false
            }
        }

        return true
    }
}

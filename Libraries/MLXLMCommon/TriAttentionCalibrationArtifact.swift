// Copyright © 2026 Apple Inc.

import Foundation
import ZIPFoundation

public struct TriAttentionCalibrationHeadKey: Sendable, Codable, Hashable {
    public let layerIndex: Int
    public let headIndex: Int

    public init(layerIndex: Int, headIndex: Int) {
        self.layerIndex = layerIndex
        self.headIndex = headIndex
    }
}

public indirect enum TriAttentionCalibrationMetadataValue: Sendable, Codable, Hashable {
    case string(String)
    case integer(Int)
    case bool(Bool)
    case array([TriAttentionCalibrationMetadataValue])
}

public struct TriAttentionCalibrationMetadata: Sendable, Codable, Hashable {
    public let sampledHeads: [TriAttentionCalibrationHeadKey]
    public let headDim: Int?
    public let ropeStyle: String?
    public let modelName: String?
    public let additionalMetadata: [String: TriAttentionCalibrationMetadataValue]

    public init(
        sampledHeads: [TriAttentionCalibrationHeadKey],
        headDim: Int?,
        ropeStyle: String?,
        modelName: String?,
        additionalMetadata: [String: TriAttentionCalibrationMetadataValue] = [:]
    ) {
        self.sampledHeads = sampledHeads
        self.headDim = headDim
        self.ropeStyle = ropeStyle
        self.modelName = modelName
        self.additionalMetadata = additionalMetadata
    }
}

public struct TriAttentionCalibrationHeadStats: Sendable, Codable, Hashable {
    public let qMeanReal: [Float]
    public let qMeanImag: [Float]
    public let qAbsMean: [Float]

    public init(
        qMeanReal: [Float],
        qMeanImag: [Float],
        qAbsMean: [Float]
    ) {
        self.qMeanReal = qMeanReal
        self.qMeanImag = qMeanImag
        self.qAbsMean = qAbsMean
    }
}

public struct TriAttentionCalibrationArtifact: Sendable, Codable, Hashable {
    public let metadata: TriAttentionCalibrationMetadata
    public let statsByHead: [TriAttentionCalibrationHeadKey: TriAttentionCalibrationHeadStats]

    public init(
        metadata: TriAttentionCalibrationMetadata,
        statsByHead: [TriAttentionCalibrationHeadKey: TriAttentionCalibrationHeadStats]
    ) {
        self.metadata = metadata
        self.statsByHead = statsByHead
    }

    public static func load(contentsOf url: URL) throws -> Self {
        try TriAttentionCalibrationArtifactParser(url: url).parse()
    }
}

public enum TriAttentionCalibrationArtifactError: LocalizedError, Equatable {
    case unreadableArchive(URL)
    case missingArchiveEntry(String)
    case unsupportedFormatVersion(String)
    case unsupportedByteOrder(String)
    case unsupportedPickleProtocol(Int)
    case truncatedPickle
    case unsupportedPickleOpcode(UInt8)
    case invalidPickle(String)
    case unsupportedGlobal(module: String, name: String)
    case unsupportedPersistentReference(String)
    case unsupportedStorageType(String)
    case unsupportedStorageLocation(String)
    case missingStorage(String)
    case invalidStorageByteCount(expected: Int, actual: Int)
    case unsupportedTensorLayout(String)
    case invalidTopLevelPayload
    case missingMetadata
    case missingStats
    case invalidMetadata(String)
    case invalidStatsKey(String)
    case invalidStatsEntry(String)

    public var errorDescription: String? {
        switch self {
        case .unreadableArchive(let url):
            return "TriAttention calibration artifact is not a readable torch archive: \(url.path)"
        case .missingArchiveEntry(let entry):
            return "TriAttention calibration artifact is missing archive entry: \(entry)"
        case .unsupportedFormatVersion(let version):
            return "Unsupported TriAttention calibration archive format version: \(version)"
        case .unsupportedByteOrder(let order):
            return "Unsupported TriAttention calibration archive byte order: \(order)"
        case .unsupportedPickleProtocol(let protocolVersion):
            return "Unsupported TriAttention calibration pickle protocol: \(protocolVersion)"
        case .truncatedPickle:
            return "TriAttention calibration pickle payload is truncated"
        case .unsupportedPickleOpcode(let opcode):
            return String(format: "Unsupported TriAttention calibration pickle opcode: 0x%02x", opcode)
        case .invalidPickle(let detail):
            return "Invalid TriAttention calibration pickle payload: \(detail)"
        case .unsupportedGlobal(let module, let name):
            return "Unsupported TriAttention calibration pickle global: \(module).\(name)"
        case .unsupportedPersistentReference(let detail):
            return "Unsupported TriAttention calibration pickle persistent reference: \(detail)"
        case .unsupportedStorageType(let storageType):
            return "Unsupported TriAttention calibration storage type: \(storageType)"
        case .unsupportedStorageLocation(let location):
            return "Unsupported TriAttention calibration storage location: \(location)"
        case .missingStorage(let key):
            return "TriAttention calibration archive is missing storage payload data/\(key)"
        case .invalidStorageByteCount(let expected, let actual):
            return "TriAttention calibration storage byte count mismatch: expected \(expected), got \(actual)"
        case .unsupportedTensorLayout(let detail):
            return "Unsupported TriAttention calibration tensor layout: \(detail)"
        case .invalidTopLevelPayload:
            return "TriAttention calibration artifact top-level payload is not the expected dictionary"
        case .missingMetadata:
            return "TriAttention calibration artifact payload is missing metadata"
        case .missingStats:
            return "TriAttention calibration artifact payload is missing stats"
        case .invalidMetadata(let detail):
            return "TriAttention calibration artifact metadata is invalid: \(detail)"
        case .invalidStatsKey(let key):
            return "TriAttention calibration artifact stats key is invalid: \(key)"
        case .invalidStatsEntry(let key):
            return "TriAttention calibration artifact stats entry is invalid: \(key)"
        }
    }
}

private struct TriAttentionCalibrationArtifactParser {
    private let url: URL
    private let archive: Archive
    private let archiveRoot: String

    init(url: URL) throws {
        self.url = url
        let archive: Archive
        do {
            archive = try Archive(url: url, accessMode: .read)
        } catch {
            throw TriAttentionCalibrationArtifactError.unreadableArchive(url)
        }
        self.archive = archive
        self.archiveRoot = try Self.resolveArchiveRoot(in: archive)
    }

    func parse() throws -> TriAttentionCalibrationArtifact {
        let formatVersion = try readStringEntry(at: archivePath(".format_version")).trimmingCharacters(
            in: .whitespacesAndNewlines)
        guard formatVersion == "1" else {
            throw TriAttentionCalibrationArtifactError.unsupportedFormatVersion(formatVersion)
        }

        let byteOrder = try readStringEntry(at: archivePath("byteorder")).trimmingCharacters(
            in: .whitespacesAndNewlines)
        guard byteOrder == "little" else {
            throw TriAttentionCalibrationArtifactError.unsupportedByteOrder(byteOrder)
        }

        let pickleData = try readEntry(at: archivePath("data.pkl"))
        var decoder = TorchPickleDecoder(
            data: pickleData,
            storageLoader: { storage in
                try loadTensor(for: storage)
            }
        )
        let pickleValue = try decoder.decode()

        return try buildArtifact(from: pickleValue)
    }

    private static func resolveArchiveRoot(in archive: Archive) throws -> String {
        let candidatePaths = archive.map(\.path)
        guard let dataPath = candidatePaths.first(where: { $0 == "data.pkl" || $0.hasSuffix("/data.pkl") }) else {
            throw TriAttentionCalibrationArtifactError.missingArchiveEntry("data.pkl")
        }
        guard dataPath.hasSuffix("data.pkl") else {
            throw TriAttentionCalibrationArtifactError.missingArchiveEntry("data.pkl")
        }
        let suffix = "/data.pkl"
        if dataPath == "data.pkl" {
            return ""
        }
        guard dataPath.hasSuffix(suffix) else {
            throw TriAttentionCalibrationArtifactError.missingArchiveEntry("data.pkl")
        }
        return String(dataPath.dropLast(suffix.count))
    }

    private func archivePath(_ relativePath: String) -> String {
        archiveRoot.isEmpty ? relativePath : "\(archiveRoot)/\(relativePath)"
    }

    private func readStringEntry(at path: String) throws -> String {
        let data = try readEntry(at: path)
        guard let string = String(data: data, encoding: .utf8) else {
            throw TriAttentionCalibrationArtifactError.invalidPickle(
                "Archive entry \(path) is not valid UTF-8"
            )
        }
        return string
    }

    private func readEntry(at path: String) throws -> Data {
        guard let entry = archive[path] else {
            throw TriAttentionCalibrationArtifactError.missingArchiveEntry(path)
        }

        var data = Data()
        _ = try archive.extract(entry) { chunk in
            data.append(chunk)
        }
        return data
    }

    private func loadTensor(for storage: TorchStorageReference) throws -> TorchTensor {
        guard storage.storageType == .float32 else {
            throw TriAttentionCalibrationArtifactError.unsupportedStorageType(storage.storageType.rawValue)
        }
        guard storage.location == "cpu" else {
            throw TriAttentionCalibrationArtifactError.unsupportedStorageLocation(storage.location)
        }

        let storagePath = archivePath("data/\(storage.key)")
        guard let entry = archive[storagePath] else {
            throw TriAttentionCalibrationArtifactError.missingStorage(storage.key)
        }

        var rawData = Data()
        _ = try archive.extract(entry) { chunk in
            rawData.append(chunk)
        }

        let expectedBytes = storage.elementCount * MemoryLayout<Float>.stride
        guard rawData.count == expectedBytes else {
            throw TriAttentionCalibrationArtifactError.invalidStorageByteCount(
                expected: expectedBytes,
                actual: rawData.count
            )
        }

        return TorchTensor(values: try decodeFloat32s(rawData), elementCount: storage.elementCount)
    }

    private func decodeFloat32s(_ data: Data) throws -> [Float] {
        guard data.count.isMultiple(of: MemoryLayout<Float>.stride) else {
            throw TriAttentionCalibrationArtifactError.invalidPickle(
                "Storage byte count \(data.count) is not divisible by 4"
            )
        }

        let elementCount = data.count / MemoryLayout<Float>.stride
        var floats: [Float] = []
        floats.reserveCapacity(elementCount)
        var index = data.startIndex
        for _ in 0 ..< elementCount {
            let b0 = UInt32(data[index])
            let b1 = UInt32(data[data.index(index, offsetBy: 1)])
            let b2 = UInt32(data[data.index(index, offsetBy: 2)])
            let b3 = UInt32(data[data.index(index, offsetBy: 3)])
            let bits = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
            floats.append(Float(bitPattern: bits))
            index = data.index(index, offsetBy: 4)
        }
        return floats
    }

    private func buildArtifact(from root: PickleValue) throws -> TriAttentionCalibrationArtifact {
        guard case .dictionary(let payload) = root else {
            throw TriAttentionCalibrationArtifactError.invalidTopLevelPayload
        }

        guard let metadataValue = payload.entries["metadata"] else {
            throw TriAttentionCalibrationArtifactError.missingMetadata
        }
        guard let statsValue = payload.entries["stats"] else {
            throw TriAttentionCalibrationArtifactError.missingStats
        }

        let metadata = try buildMetadata(from: metadataValue)
        let stats = try buildStats(from: statsValue)
        return TriAttentionCalibrationArtifact(metadata: metadata, statsByHead: stats)
    }

    private func buildMetadata(from value: PickleValue) throws -> TriAttentionCalibrationMetadata {
        guard case .dictionary(let metadata) = value else {
            throw TriAttentionCalibrationArtifactError.invalidMetadata("metadata is not a dictionary")
        }

        guard let sampledHeadsValue = metadata.entries["sampled_heads"] else {
            throw TriAttentionCalibrationArtifactError.invalidMetadata("missing sampled_heads")
        }
        let sampledHeads = try extractSampledHeads(sampledHeadsValue)
        let headDim = try metadata.entries["head_dim"].map(extractInt)
        let ropeStyle = try metadata.entries["rope_style"].map(extractString)
        let modelName = try metadata.entries["model_name"].map(extractString)

        var additionalMetadata: [String: TriAttentionCalibrationMetadataValue] = [:]
        for (key, entry) in metadata.entries {
            switch key {
            case "sampled_heads", "head_dim", "rope_style", "model_name":
                continue
            default:
                if let converted = convertMetadataValue(entry) {
                    additionalMetadata[key] = converted
                }
            }
        }

        return TriAttentionCalibrationMetadata(
            sampledHeads: sampledHeads,
            headDim: headDim,
            ropeStyle: ropeStyle,
            modelName: modelName,
            additionalMetadata: additionalMetadata
        )
    }

    private func buildStats(
        from value: PickleValue
    ) throws -> [TriAttentionCalibrationHeadKey: TriAttentionCalibrationHeadStats] {
        guard case .dictionary(let statsDictionary) = value else {
            throw TriAttentionCalibrationArtifactError.invalidStatsEntry("stats is not a dictionary")
        }

        var statsByHead: [TriAttentionCalibrationHeadKey: TriAttentionCalibrationHeadStats] = [:]
        statsByHead.reserveCapacity(statsDictionary.entries.count)
        for (rawKey, rawValue) in statsDictionary.entries {
            let headKey = try parseStatsKey(rawKey)
            guard case .dictionary(let statEntry) = rawValue else {
                throw TriAttentionCalibrationArtifactError.invalidStatsEntry(rawKey)
            }

            let qMeanReal = try extractTensor(named: "q_mean_real", from: statEntry, statsKey: rawKey)
            let qMeanImag = try extractTensor(named: "q_mean_imag", from: statEntry, statsKey: rawKey)
            let qAbsMean = try extractTensor(named: "q_abs_mean", from: statEntry, statsKey: rawKey)

            guard qMeanReal.values.count == qMeanImag.values.count,
                  qMeanReal.values.count == qAbsMean.values.count
            else {
                throw TriAttentionCalibrationArtifactError.invalidStatsEntry(rawKey)
            }

            statsByHead[headKey] = TriAttentionCalibrationHeadStats(
                qMeanReal: qMeanReal.values,
                qMeanImag: qMeanImag.values,
                qAbsMean: qAbsMean.values
            )
        }
        return statsByHead
    }

    private func extractSampledHeads(
        _ value: PickleValue
    ) throws -> [TriAttentionCalibrationHeadKey] {
        guard let items = value.collectionItems else {
            throw TriAttentionCalibrationArtifactError.invalidMetadata("sampled_heads is not a list")
        }

        return try items.map { pairValue in
            guard let pair = pairValue.collectionItems, pair.count == 2 else {
                throw TriAttentionCalibrationArtifactError.invalidMetadata(
                    "sampled_heads entry is not a pair"
                )
            }
            return TriAttentionCalibrationHeadKey(
                layerIndex: try extractInt(pair[0]),
                headIndex: try extractInt(pair[1])
            )
        }
    }

    private func extractInt(_ value: PickleValue) throws -> Int {
        guard case .integer(let integerValue) = value else {
            throw TriAttentionCalibrationArtifactError.invalidMetadata("expected integer metadata value")
        }
        return integerValue
    }

    private func extractString(_ value: PickleValue) throws -> String {
        guard case .string(let stringValue) = value else {
            throw TriAttentionCalibrationArtifactError.invalidMetadata("expected string metadata value")
        }
        return stringValue
    }

    private func convertMetadataValue(
        _ value: PickleValue
    ) -> TriAttentionCalibrationMetadataValue? {
        switch value {
        case .string(let stringValue):
            return .string(stringValue)
        case .integer(let integerValue):
            return .integer(integerValue)
        case .bool(let boolValue):
            return .bool(boolValue)
        case .list(let listValue):
            let items = listValue.items.compactMap(convertMetadataValue)
            guard items.count == listValue.items.count else { return nil }
            return .array(items)
        case .tuple(let items):
            let converted = items.compactMap(convertMetadataValue)
            guard converted.count == items.count else { return nil }
            return .array(converted)
        default:
            return nil
        }
    }

    private func parseStatsKey(_ key: String) throws -> TriAttentionCalibrationHeadKey {
        guard key.hasPrefix("layer"),
              let separator = key.range(of: "_head")
        else {
            throw TriAttentionCalibrationArtifactError.invalidStatsKey(key)
        }

        let layerPart = key[key.index(key.startIndex, offsetBy: "layer".count)..<separator.lowerBound]
        let headPart = key[separator.upperBound...]
        guard let layerIndex = Int(layerPart), let headIndex = Int(headPart) else {
            throw TriAttentionCalibrationArtifactError.invalidStatsKey(key)
        }

        return TriAttentionCalibrationHeadKey(layerIndex: layerIndex, headIndex: headIndex)
    }

    private func extractTensor(
        named name: String,
        from dictionary: PickleDictionary,
        statsKey: String
    ) throws -> TorchTensor {
        guard let value = dictionary.entries[name], case .tensor(let tensor) = value else {
            throw TriAttentionCalibrationArtifactError.invalidStatsEntry(statsKey)
        }
        return tensor
    }
}

private enum TorchStorageType: String {
    case float32 = "FloatStorage"
}

private struct TorchStorageReference {
    let storageType: TorchStorageType
    let key: String
    let location: String
    let elementCount: Int
}

private struct TorchTensor: Sendable {
    let values: [Float]
    let elementCount: Int
}

private final class PickleList: @unchecked Sendable {
    var items: [PickleValue]

    init(items: [PickleValue] = []) {
        self.items = items
    }
}

private final class PickleDictionary: @unchecked Sendable {
    var entries: [String: PickleValue]

    init(entries: [String: PickleValue] = [:]) {
        self.entries = entries
    }
}

private final class PickleOrderedDictionary: @unchecked Sendable {
    var entries: [(PickleValue, PickleValue)]

    init(entries: [(PickleValue, PickleValue)] = []) {
        self.entries = entries
    }
}

private enum PickleGlobal {
    case rebuildTensorV2
    case floatStorage
    case orderedDict
}

private indirect enum PickleValue: Sendable {
    case integer(Int)
    case bool(Bool)
    case string(String)
    case list(PickleList)
    case dictionary(PickleDictionary)
    case tuple([PickleValue])
    case global(PickleGlobal)
    case storageReference(TorchStorageReference)
    case tensor(TorchTensor)
    case orderedDictionary(PickleOrderedDictionary)

    var collectionItems: [PickleValue]? {
        switch self {
        case .list(let listValue):
            return listValue.items
        case .tuple(let tupleValue):
            return tupleValue
        default:
            return nil
        }
    }
}

private enum StackItem {
    case mark
    case value(PickleValue)
}

private struct TorchPickleDecoder {
    private enum OpCode {
        static let mark: UInt8 = 0x28
        static let stop: UInt8 = 0x2e
        static let binInt: UInt8 = 0x4a
        static let binInt1: UInt8 = 0x4b
        static let binInt2: UInt8 = 0x4d
        static let binPersId: UInt8 = 0x51
        static let reduce: UInt8 = 0x52
        static let binUnicode: UInt8 = 0x58
        static let emptyList: UInt8 = 0x5d
        static let appends: UInt8 = 0x65
        static let global: UInt8 = 0x63
        static let binGet: UInt8 = 0x68
        static let longBinGet: UInt8 = 0x6a
        static let binput: UInt8 = 0x71
        static let longBinput: UInt8 = 0x72
        static let tuple: UInt8 = 0x74
        static let setItems: UInt8 = 0x75
        static let emptyDict: UInt8 = 0x7d
        static let proto: UInt8 = 0x80
        static let tuple1: UInt8 = 0x85
        static let newFalse: UInt8 = 0x89
        static let emptyTuple: UInt8 = 0x29
    }

    private var reader: PickleByteReader
    private var stack: [StackItem] = []
    private var memo: [Int: PickleValue] = [:]
    private let storageLoader: (TorchStorageReference) throws -> TorchTensor

    init(
        data: Data,
        storageLoader: @escaping (TorchStorageReference) throws -> TorchTensor
    ) {
        self.reader = PickleByteReader(data: data)
        self.storageLoader = storageLoader
    }

    mutating func decode() throws -> PickleValue {
        while true {
            let opcode = try reader.readByte()
            switch opcode {
            case OpCode.proto:
                let protocolVersion = Int(try reader.readByte())
                guard protocolVersion == 2 else {
                    throw TriAttentionCalibrationArtifactError.unsupportedPickleProtocol(protocolVersion)
                }
            case OpCode.emptyDict:
                push(.dictionary(PickleDictionary()))
            case OpCode.emptyList:
                push(.list(PickleList()))
            case OpCode.emptyTuple:
                push(.tuple([]))
            case OpCode.mark:
                stack.append(.mark)
            case OpCode.binUnicode:
                let byteCount = Int(try reader.readUInt32())
                let stringValue = try reader.readString(byteCount: byteCount)
                push(.string(stringValue))
            case OpCode.binInt1:
                push(.integer(Int(try reader.readByte())))
            case OpCode.binInt2:
                push(.integer(Int(try reader.readUInt16())))
            case OpCode.binInt:
                push(.integer(Int(try reader.readInt32())))
            case OpCode.newFalse:
                push(.bool(false))
            case OpCode.binput:
                let index = Int(try reader.readByte())
                memo[index] = try peekValue()
            case OpCode.longBinput:
                let index = Int(try reader.readUInt32())
                memo[index] = try peekValue()
            case OpCode.binGet:
                let index = Int(try reader.readByte())
                guard let value = memo[index] else {
                    throw TriAttentionCalibrationArtifactError.invalidPickle(
                        "missing memo entry \(index)"
                    )
                }
                push(value)
            case OpCode.longBinGet:
                let index = Int(try reader.readUInt32())
                guard let value = memo[index] else {
                    throw TriAttentionCalibrationArtifactError.invalidPickle(
                        "missing memo entry \(index)"
                    )
                }
                push(value)
            case OpCode.appends:
                let items = try popValuesAfterMark()
                guard case .list(let listValue) = try popValue() else {
                    throw TriAttentionCalibrationArtifactError.invalidPickle(
                        "APPENDS expected list target"
                    )
                }
                listValue.items.append(contentsOf: items)
                push(.list(listValue))
            case OpCode.setItems:
                let items = try popValuesAfterMark()
                guard case .dictionary(let dictionary) = try popValue() else {
                    throw TriAttentionCalibrationArtifactError.invalidPickle(
                        "SETITEMS expected dictionary target"
                    )
                }
                guard items.count.isMultiple(of: 2) else {
                    throw TriAttentionCalibrationArtifactError.invalidPickle(
                        "SETITEMS expected key/value pairs"
                    )
                }
                var index = 0
                while index < items.count {
                    guard case .string(let key) = items[index] else {
                        throw TriAttentionCalibrationArtifactError.invalidPickle(
                            "SETITEMS keys must be strings"
                        )
                    }
                    dictionary.entries[key] = items[index + 1]
                    index += 2
                }
                push(.dictionary(dictionary))
            case OpCode.tuple:
                push(.tuple(try popValuesAfterMark()))
            case OpCode.tuple1:
                push(.tuple([try popValue()]))
            case OpCode.global:
                let module = try reader.readLineString()
                let name = try reader.readLineString()
                push(try decodeGlobal(module, name: name))
            case OpCode.binPersId:
                let persistentID = try popValue()
                push(try decodePersistentReference(persistentID))
            case OpCode.reduce:
                let arguments = try popValue()
                let callable = try popValue()
                push(try reduce(callable: callable, arguments: arguments))
            case OpCode.stop:
                let value = try popValue()
                guard stack.isEmpty else {
                    throw TriAttentionCalibrationArtifactError.invalidPickle(
                        "stack not empty after STOP"
                    )
                }
                return value
            default:
                throw TriAttentionCalibrationArtifactError.unsupportedPickleOpcode(opcode)
            }
        }
    }

    private mutating func push(_ value: PickleValue) {
        stack.append(.value(value))
    }

    private mutating func popValue() throws -> PickleValue {
        guard let item = stack.popLast() else {
            throw TriAttentionCalibrationArtifactError.truncatedPickle
        }
        guard case .value(let value) = item else {
            throw TriAttentionCalibrationArtifactError.invalidPickle("unexpected MARK on value stack")
        }
        return value
    }

    private func peekValue() throws -> PickleValue {
        guard let item = stack.last else {
            throw TriAttentionCalibrationArtifactError.truncatedPickle
        }
        guard case .value(let value) = item else {
            throw TriAttentionCalibrationArtifactError.invalidPickle("unexpected MARK on value stack")
        }
        return value
    }

    private mutating func popValuesAfterMark() throws -> [PickleValue] {
        guard let markIndex = stack.lastIndex(where: {
            if case .mark = $0 { return true }
            return false
        }) else {
            throw TriAttentionCalibrationArtifactError.invalidPickle("missing MARK")
        }

        let tail = stack[(markIndex + 1)...]
        var values: [PickleValue] = []
        values.reserveCapacity(tail.count)
        for item in tail {
            guard case .value(let value) = item else {
                throw TriAttentionCalibrationArtifactError.invalidPickle("nested MARK in marked span")
            }
            values.append(value)
        }
        stack.removeSubrange(markIndex...)
        return values
    }

    private func decodeGlobal(_ module: String, name: String) throws -> PickleValue {
        switch (module, name) {
        case ("torch._utils", "_rebuild_tensor_v2"):
            return .global(.rebuildTensorV2)
        case ("torch", "FloatStorage"):
            return .global(.floatStorage)
        case ("collections", "OrderedDict"):
            return .global(.orderedDict)
        default:
            throw TriAttentionCalibrationArtifactError.unsupportedGlobal(module: module, name: name)
        }
    }

    private func decodePersistentReference(_ value: PickleValue) throws -> PickleValue {
        guard case .tuple(let items) = value, items.count == 5 else {
            throw TriAttentionCalibrationArtifactError.unsupportedPersistentReference(
                "unexpected persistent reference payload"
            )
        }

        guard case .string(let marker) = items[0], marker == "storage" else {
            throw TriAttentionCalibrationArtifactError.unsupportedPersistentReference(
                "unexpected persistent reference marker"
            )
        }
        guard case .global(.floatStorage) = items[1] else {
            throw TriAttentionCalibrationArtifactError.unsupportedPersistentReference(
                "unsupported storage class"
            )
        }
        guard case .string(let key) = items[2],
              case .string(let location) = items[3],
              case .integer(let elementCount) = items[4]
        else {
            throw TriAttentionCalibrationArtifactError.unsupportedPersistentReference(
                "invalid persistent reference fields"
            )
        }

        return .storageReference(
            TorchStorageReference(
                storageType: .float32,
                key: key,
                location: location,
                elementCount: elementCount
            )
        )
    }

    private func reduce(callable: PickleValue, arguments: PickleValue) throws -> PickleValue {
        guard case .tuple(let tupleArguments) = arguments else {
            throw TriAttentionCalibrationArtifactError.invalidPickle("REDUCE expected tuple arguments")
        }

        switch callable {
        case .global(.orderedDict):
            guard tupleArguments.isEmpty else {
                throw TriAttentionCalibrationArtifactError.invalidPickle(
                    "OrderedDict reduce expected empty tuple"
                )
            }
            return .orderedDictionary(PickleOrderedDictionary())

        case .global(.rebuildTensorV2):
            return try rebuildTensor(from: tupleArguments)

        default:
            throw TriAttentionCalibrationArtifactError.invalidPickle(
                "REDUCE called with unsupported callable"
            )
        }
    }

    private func rebuildTensor(from arguments: [PickleValue]) throws -> PickleValue {
        guard arguments.count == 6 else {
            throw TriAttentionCalibrationArtifactError.invalidPickle(
                "_rebuild_tensor_v2 expected 6 arguments"
            )
        }

        guard case .storageReference(let storage) = arguments[0] else {
            throw TriAttentionCalibrationArtifactError.invalidPickle(
                "_rebuild_tensor_v2 expected storage reference"
            )
        }
        guard case .integer(let storageOffset) = arguments[1],
              storageOffset >= 0
        else {
            throw TriAttentionCalibrationArtifactError.invalidPickle(
                "_rebuild_tensor_v2 expected non-negative storage offset"
            )
        }
        guard let sizeItems = arguments[2].collectionItems,
              let strideItems = arguments[3].collectionItems,
              case .bool(false) = arguments[4],
              case .orderedDictionary = arguments[5]
        else {
            throw TriAttentionCalibrationArtifactError.invalidPickle(
                "_rebuild_tensor_v2 arguments have unsupported shape"
            )
        }
        let size = try sizeItems.map(extractInteger)
        let stride = try strideItems.map(extractInteger)

        guard size.count == 1, stride.count == 1, stride[0] == 1 else {
            throw TriAttentionCalibrationArtifactError.unsupportedTensorLayout(
                "only contiguous 1D float tensors are supported"
            )
        }

        let tensorStorage = try storageLoader(storage)
        let valueCount = size[0]
        let endIndex = storageOffset + valueCount
        guard valueCount >= 0, endIndex <= tensorStorage.values.count else {
            throw TriAttentionCalibrationArtifactError.invalidPickle(
                "tensor slice exceeds storage bounds"
            )
        }

        let values = Array(tensorStorage.values[storageOffset..<endIndex])
        return .tensor(TorchTensor(values: values, elementCount: valueCount))
    }

    private func extractInteger(_ value: PickleValue) throws -> Int {
        guard case .integer(let integerValue) = value else {
            throw TriAttentionCalibrationArtifactError.invalidPickle("expected integer in tuple")
        }
        return integerValue
    }
}

private struct PickleByteReader {
    let data: Data
    private(set) var offset: Int = 0

    mutating func readByte() throws -> UInt8 {
        guard offset < data.count else {
            throw TriAttentionCalibrationArtifactError.truncatedPickle
        }
        let value = data[data.index(data.startIndex, offsetBy: offset)]
        offset += 1
        return value
    }

    mutating func readUInt16() throws -> UInt16 {
        let b0 = UInt16(try readByte())
        let b1 = UInt16(try readByte())
        return b0 | (b1 << 8)
    }

    mutating func readUInt32() throws -> UInt32 {
        let b0 = UInt32(try readByte())
        let b1 = UInt32(try readByte())
        let b2 = UInt32(try readByte())
        let b3 = UInt32(try readByte())
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    }

    mutating func readInt32() throws -> Int32 {
        Int32(bitPattern: try readUInt32())
    }

    mutating func readString(byteCount: Int) throws -> String {
        guard byteCount >= 0, offset + byteCount <= data.count else {
            throw TriAttentionCalibrationArtifactError.truncatedPickle
        }
        let start = data.index(data.startIndex, offsetBy: offset)
        let end = data.index(start, offsetBy: byteCount)
        let slice = data[start..<end]
        offset += byteCount
        guard let string = String(data: slice, encoding: .utf8) else {
            throw TriAttentionCalibrationArtifactError.invalidPickle("invalid UTF-8 string")
        }
        return string
    }

    mutating func readLineString() throws -> String {
        let startOffset = offset
        while offset < data.count {
            let byte = data[data.index(data.startIndex, offsetBy: offset)]
            offset += 1
            if byte == 0x0a {
                let start = data.index(data.startIndex, offsetBy: startOffset)
                let end = data.index(data.startIndex, offsetBy: offset - 1)
                let slice = data[start..<end]
                guard let string = String(data: slice, encoding: .utf8) else {
                    throw TriAttentionCalibrationArtifactError.invalidPickle("invalid GLOBAL string")
                }
                return string
            }
        }
        throw TriAttentionCalibrationArtifactError.truncatedPickle
    }
}

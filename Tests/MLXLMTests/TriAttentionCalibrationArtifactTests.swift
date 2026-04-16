// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import XCTest

final class TriAttentionCalibrationArtifactTests: XCTestCase {

    private func fixtureURL() throws -> URL {
        if let url = Bundle.module.url(
            forResource: "triattention-minimal-valid",
            withExtension: "pt"
        ) {
            return url
        }
        if let url = Bundle.module.resourceURL?.appendingPathComponent(
            "TriAttention/triattention-minimal-valid.pt",
            isDirectory: false
        ), FileManager.default.fileExists(atPath: url.path) {
            return url
        }

        XCTFail("Missing TriAttention fixture resource")
        throw TriAttentionCalibrationArtifactError.missingArchiveEntry("triattention-minimal-valid.pt")
    }

    func testTriAttentionCalibrationArtifactLoadsUpstreamTorchFixture() throws {
        let artifact = try TriAttentionCalibrationArtifact.load(contentsOf: fixtureURL())

        XCTAssertEqual(
            artifact.metadata.sampledHeads,
            [
                TriAttentionCalibrationHeadKey(layerIndex: 0, headIndex: 0),
                TriAttentionCalibrationHeadKey(layerIndex: 1, headIndex: 2),
            ]
        )
        XCTAssertEqual(artifact.metadata.headDim, 8)
        XCTAssertEqual(artifact.metadata.ropeStyle, "half")
        XCTAssertEqual(artifact.metadata.modelName, "fixture/qwen3.5-paro-test")
        XCTAssertEqual(artifact.metadata.additionalMetadata["num_traces"], .integer(2))
        XCTAssertEqual(artifact.metadata.additionalMetadata["dtype"], .string("bfloat16"))
        XCTAssertEqual(artifact.metadata.additionalMetadata["use_chat_template"], .bool(false))

        let firstHead = artifact.statsByHead[TriAttentionCalibrationHeadKey(layerIndex: 0, headIndex: 0)]
        XCTAssertEqual(firstHead?.qMeanReal, [0.25, 1.5, -2.0, 3.25])
        XCTAssertEqual(firstHead?.qMeanImag, [4.5, -5.75, 6.125, -7.0])
        XCTAssertEqual(firstHead?.qAbsMean, [8.0, 9.5, 10.25, 11.75])

        let secondHead = artifact.statsByHead[TriAttentionCalibrationHeadKey(layerIndex: 1, headIndex: 2)]
        XCTAssertEqual(secondHead?.qMeanReal, [-0.5, 0.0, 0.5, 1.0])
        XCTAssertEqual(secondHead?.qMeanImag, [1.5, 2.0, 2.5, 3.0])
        XCTAssertEqual(secondHead?.qAbsMean, [3.5, 4.0, 4.5, 5.0])
    }

    func testTriAttentionCalibrationArtifactRejectsMalformedArchive() throws {
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("triattention-invalid-\(UUID().uuidString).pt", isDirectory: false)
        defer { try? FileManager.default.removeItem(at: tempURL) }
        try Data("not a torch zip archive".utf8).write(to: tempURL)

        XCTAssertThrowsError(try TriAttentionCalibrationArtifact.load(contentsOf: tempURL)) { error in
            guard case TriAttentionCalibrationArtifactError.unreadableArchive(let url) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(url.path, tempURL.path)
        }
    }
}

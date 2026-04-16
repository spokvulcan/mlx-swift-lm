// Copyright © 2026 Apple Inc.

import CryptoKit
import Foundation

public struct TriAttentionCalibrationArtifactIdentity: Sendable, Codable, Hashable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    public static func sha256(of artifactData: Data) -> Self {
        let digest = SHA256.hash(data: artifactData)
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        return Self(rawValue: hex)
    }
}

public enum TriAttentionImplementationVersion: String, Sendable, Codable {
    case v1
}

public struct TriAttentionConfiguration: Sendable, Codable, Hashable {
    public static let v1BudgetTokens = 12_000
    public static let v1Disabled = Self(
        enabled: false,
        budgetTokens: v1BudgetTokens,
        calibrationArtifactIdentity: nil,
        implementationVersion: .v1
    )

    public let enabled: Bool
    public let budgetTokens: Int
    public let calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity?
    public let implementationVersion: TriAttentionImplementationVersion

    public init(
        enabled: Bool,
        budgetTokens: Int = Self.v1BudgetTokens,
        calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity? = nil,
        implementationVersion: TriAttentionImplementationVersion = .v1
    ) {
        self.enabled = enabled
        self.budgetTokens = budgetTokens
        self.calibrationArtifactIdentity = calibrationArtifactIdentity
        self.implementationVersion = implementationVersion
    }

    public func withCalibrationArtifactIdentity(
        _ calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity?
    ) -> Self {
        Self(
            enabled: enabled,
            budgetTokens: budgetTokens,
            calibrationArtifactIdentity: calibrationArtifactIdentity,
            implementationVersion: implementationVersion
        )
    }
}

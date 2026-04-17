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

public enum TriAttentionPrefixProtectionMode: String, Sendable, Codable, Hashable {
    case protectNone
    case protectStablePrefixOnly
    case protectAllPrefill
}

public struct TriAttentionConfiguration: Sendable, Codable, Hashable {
    public static let v1BudgetTokens = 12_000
    public static let v1Disabled = Self(
        enabled: false,
        budgetTokens: v1BudgetTokens,
        calibrationArtifactIdentity: nil,
        implementationVersion: .v1,
        prefixProtectionMode: .protectStablePrefixOnly
    )

    public let enabled: Bool
    public let budgetTokens: Int
    public let calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity?
    public let implementationVersion: TriAttentionImplementationVersion
    public let prefixProtectionMode: TriAttentionPrefixProtectionMode

    public init(
        enabled: Bool,
        budgetTokens: Int = Self.v1BudgetTokens,
        calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity? = nil,
        implementationVersion: TriAttentionImplementationVersion = .v1,
        prefixProtectionMode: TriAttentionPrefixProtectionMode = .protectStablePrefixOnly
    ) {
        self.enabled = enabled
        self.budgetTokens = budgetTokens
        self.calibrationArtifactIdentity = calibrationArtifactIdentity
        self.implementationVersion = implementationVersion
        self.prefixProtectionMode = prefixProtectionMode
    }

    public func withCalibrationArtifactIdentity(
        _ calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity?
    ) -> Self {
        Self(
            enabled: enabled,
            budgetTokens: budgetTokens,
            calibrationArtifactIdentity: calibrationArtifactIdentity,
            implementationVersion: implementationVersion,
            prefixProtectionMode: prefixProtectionMode
        )
    }

    enum CodingKeys: String, CodingKey {
        case enabled
        case budgetTokens
        case calibrationArtifactIdentity
        case implementationVersion
        case prefixProtectionMode
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        enabled = try container.decode(Bool.self, forKey: .enabled)
        budgetTokens = try container.decode(Int.self, forKey: .budgetTokens)
        calibrationArtifactIdentity = try container.decodeIfPresent(
            TriAttentionCalibrationArtifactIdentity.self,
            forKey: .calibrationArtifactIdentity
        )
        implementationVersion = try container.decode(
            TriAttentionImplementationVersion.self,
            forKey: .implementationVersion
        )
        prefixProtectionMode = try container.decodeIfPresent(
            TriAttentionPrefixProtectionMode.self,
            forKey: .prefixProtectionMode
        ) ?? .protectNone
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(enabled, forKey: .enabled)
        try container.encode(budgetTokens, forKey: .budgetTokens)
        try container.encodeIfPresent(
            calibrationArtifactIdentity,
            forKey: .calibrationArtifactIdentity
        )
        try container.encode(implementationVersion, forKey: .implementationVersion)
        try container.encode(prefixProtectionMode, forKey: .prefixProtectionMode)
    }
}

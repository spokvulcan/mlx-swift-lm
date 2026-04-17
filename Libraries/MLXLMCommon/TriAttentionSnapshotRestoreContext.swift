import Foundation

public struct TriAttentionSnapshotRestoreContext {
    public let expectedConfiguration: TriAttentionConfiguration
    public let runtimeState: TriAttentionQwen35RuntimeState

    public init(
        expectedConfiguration: TriAttentionConfiguration,
        runtimeState: TriAttentionQwen35RuntimeState
    ) {
        self.expectedConfiguration = expectedConfiguration
        self.runtimeState = runtimeState
    }
}

public protocol TriAttentionSnapshotRestoreContextProviding {
    func triAttentionSnapshotRestoreContext(
        configuration: TriAttentionConfiguration,
        artifact: TriAttentionCalibrationArtifact?
    ) -> TriAttentionSnapshotRestoreContext?
}

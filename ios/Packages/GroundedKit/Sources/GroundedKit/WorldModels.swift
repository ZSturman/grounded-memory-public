import Foundation

/// The learner's current estimate; none of these scores imply calibrated probability.
public struct WorldSnapshot: Codable, Equatable, Sendable {
    public var learnerId: String
    public var episodeId: String?
    public var location: WorldLocation
    /// Absent from servers that predate control modes.
    public var executive: WorldExecutive?
    public var attention: WorldQuestion?
    public var questions: [WorldQuestion]
    public var beliefs: [WorldBelief]
    public var predictions: [WorldPrediction]
    public var recentEvents: [WorldEvent]
    public var status: String
}

public struct WorldLocation: Codable, Equatable, Sendable {
    public var placeId: String?
    public var label: String
    public var certainty: String
    public var candidates: [PlaceCandidate]
}

/// The learner's fixed control mode and what it asks the body (the person holding the phone) to do.
public struct WorldExecutive: Codable, Equatable, Sendable {
    public var mode: String
    public var reason: String
    public var request: String?
}

public struct PlaceCandidate: Codable, Equatable, Identifiable, Sendable {
    public var id: String
    public var label: String
    public var similarity: Double
}

public struct WorldQuestion: Codable, Equatable, Identifiable, Sendable {
    public var id: String
    public var episodeId: String
    public var version: Int
    public var kind: String
    public var text: String
    public var reason: String
    public var evidenceRefs: [String]
    public var region: Region?
    public var proposition: WorldProposition
    public var status: String
}

public struct WorldProposition: Codable, Equatable, Sendable {
    public var placeId: String?
    public var entityId: String?
    public var eventKey: String?
    public var beliefId: String?
    public var label: String?
}

public struct WorldBelief: Codable, Equatable, Identifiable, Sendable {
    public var id: String
    public var subjectLabel: String
    public var predicate: String
    public var value: String
    public var status: String
    public var certainty: String
    public var version: Int
    public var supporting: [String]
    public var conflicting: [String]
}

public struct WorldPrediction: Codable, Equatable, Identifiable, Sendable {
    public var id: String
    public var label: String
    public var start: Double
    public var due: Double
    public var status: String
}

public struct WorldEvent: Codable, Equatable, Identifiable, Sendable {
    public var id: String
    public var label: String
    public var kind: String
    public var surprising: Bool
}

public struct WorldReply: Codable, Equatable, Sendable {
    public var id = UUID().uuidString
    public var questionVersion: Int
    public var text: String
    public var response: String
    public var teacherId = "local-human"
    public var entityId: String?
    public var expectedEvent: String?
    public var horizonSeconds: Double = 30

    public init(question: WorldQuestion, text: String, response: String = "answer", entityId: String? = nil,
                expectedEvent: String? = nil, horizonSeconds: Double = 30) {
        self.questionVersion = question.version
        self.text = text
        self.response = response
        self.entityId = entityId
        self.expectedEvent = expectedEvent
        self.horizonSeconds = horizonSeconds
    }
}

public struct WorldAnswerResult: Codable, Sendable {
    public struct Choice: Codable, Identifiable, Sendable { public var id: String; public var label: String }
    public var needsClarification: Bool?
    public var matches: [Choice]?
}

public struct AudioSample: Sendable {
    public var id = UUID().uuidString
    public var wav: Data
    public var timestamp: Double
    public var clockId: String
    public var sourceId: String
    public var playback: Bool

    public init(wav: Data, timestamp: Double, clockId: String, sourceId: String, playback: Bool) {
        self.wav = wav; self.timestamp = timestamp; self.clockId = clockId
        self.sourceId = sourceId; self.playback = playback
    }
}

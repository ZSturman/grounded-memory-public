import Foundation

/// Mirrors `lean()` in backend/grounded_memory/api.py. Feature vectors never reach the phone.
public struct SessionState: Codable, Equatable, Sendable {
    public var id: String
    public var mode: String
    public var status: String
    public var paused: Bool
    public var world: WorldSnapshot?
    public var context: PlaceContext?
    public var placeGuess: Match?
    public var relations: [ObservedRelation]?
    public var question: Question?
    public var latest: Latest
}

public struct PlaceContext: Codable, Equatable, Sendable {
    public var placeId: String
    public var label: String
}

public struct Latest: Codable, Equatable, Sendable {
    public var mediaTime: Double?
    public var frame: String?
    public var regions: [Region]
}

public struct Region: Codable, Equatable, Identifiable, Sendable {
    public var id: String
    /// x, y, width, height normalized to the upright (portrait) frame that was sent.
    public var bbox: [Double]
    public var asset: String
    public var salience: Double
    public var matches: [Match]?
    public var sensorId: String?
    public var prompt: Prompt?

    public var best: Match? { matches?.first }
}

public struct Match: Codable, Equatable, Sendable {
    public var id: String
    public var label: String
    public var appearance: Double
    public var score: Double
    /// same / compatible / unverified: how the remembered example's sensor relates to this one.
    public var sensors: String?
}

/// How the person shaped an outline: SAM options run from smallest (0) to largest (options - 1).
public struct Prompt: Codable, Equatable, Sendable {
    public var granularity: Int?
    public var options: Int?
}

/// A geometric relation measured from pose and depth; it becomes a claim only when confirmed.
public struct ObservedRelation: Codable, Equatable, Identifiable, Sendable {
    public var id: String
    public var subject: String
    public var predicate: String
    public var object: String

    public var sentence: String { "\(subject) \(predicate.replacingOccurrences(of: "_", with: " ")) \(object)" }
}

/// The capture device, sent with every sample so evidence from different lenses is never conflated.
public struct SensorDescriptor: Codable, Equatable, Sendable {
    public var modality = "video"
    public var source: String
    public var projection: String
    public var device: String
    public var platform: String
    public var camera: String
    public var lens: String
    public var resolution: [Int]
    public var fps: Int
    public var depth: String?

    public init(
        source: String, projection: String, device: String, platform: String, camera: String, lens: String,
        resolution: [Int], fps: Int, depth: String?
    ) {
        (self.source, self.projection, self.device, self.platform) = (source, projection, device, platform)
        (self.camera, self.lens, self.resolution, self.fps, self.depth) = (camera, lens, resolution, fps, depth)
    }
}

/// Pose (column-major 4x4), intrinsics (column-major 3x3) and sensor image size for one frame.
public struct CameraPose: Codable, Equatable, Sendable {
    public var transform: [Float]
    public var intrinsics: [Float]
    public var resolution: [Double]

    public init(transform: [Float], intrinsics: [Float], resolution: [Double]) {
        (self.transform, self.intrinsics, self.resolution) = (transform, intrinsics, resolution)
    }
}

/// One time-aligned sample: a frame plus whichever optional streams this device has.
public struct Sample: Sendable {
    public var id = UUID().uuidString
    public var clockId = "legacy"
    public var trackingFrameId = "legacy"
    public var jpeg: Data
    public var mediaTime: Double
    public var camera: CameraPose?
    /// Little-endian uint16 millimetres and uint8 confidence, sensor orientation.
    public var depth: Data?
    public var confidence: Data?
    public var depthSize: [Int]?

    public init(jpeg: Data, mediaTime: Double, camera: CameraPose?, depth: Data?, confidence: Data?, depthSize: [Int]?) {
        (self.jpeg, self.mediaTime, self.camera) = (jpeg, mediaTime, camera)
        (self.depth, self.confidence, self.depthSize) = (depth, confidence, depthSize)
    }
}

public struct Question: Codable, Equatable, Identifiable, Sendable {
    public var id: String
    public var text: String
    public var region: Region
    public var mediaTime: Double
    public var frame: String
    /// The frozen frame with this question's outline drawn on it.
    public var preview: String?
}

/// A question put aside during capture, answerable later from its frozen evidence.
public struct SavedQuestion: Codable, Equatable, Identifiable, Sendable {
    public var id: String
    public var text: String
    public var region: Region
    public var preview: String?
    public var place: String
    public var reason: String?
    public var status: String?
}

/// Proposed teaching terms; each is a substring of what the person said and still needs review.
public struct Interpretation: Codable, Equatable, Sendable {
    public var label: String
    public var categories: [String]
    public var aliases: [String]

    public init(label: String, categories: [String] = [], aliases: [String] = []) {
        self.label = label
        self.categories = categories
        self.aliases = aliases
    }
}

public struct Entity: Codable, Equatable, Identifiable, Sendable {
    public struct Example: Codable, Equatable, Sendable {
        public var asset: String?
    }

    public var id: String
    public var kind: String
    public var label: String
    public var examples: [Example]?
}

/// Teaching and place answers either succeed or ask whether a same-named entity is the same thing.
public struct TeachResult: Codable, Equatable, Sendable {
    public var needsClarification: Bool?
    public var entityId: String?
    public var matches: [Entity]?
    public var context: PlaceContext?
}

public enum Resolution: String, Codable, Sendable {
    case same, sameKind = "same_kind", subtype, unrelated
}

public enum JSON {
    public static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }()

    public static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        return encoder
    }()
}

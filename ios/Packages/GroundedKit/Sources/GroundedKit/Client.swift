import Foundation

public struct APIError: LocalizedError, Equatable {
    public var status: Int
    public var message: String
    public var body = Data()
    public var errorDescription: String? { message }
}

/// HTTPS client for the Mac backend. Pairing is a signed cookie kept by the shared cookie store.
public final class GroundedClient: @unchecked Sendable {
    public let baseURL: URL
    private let session: URLSession
    public let outbox: ObservationOutbox?
    private let delivery = DeliveryGate()

    public init(baseURL: URL, authority: Data) {
        self.baseURL = baseURL
        self.outbox = try? ObservationOutbox(server: baseURL)
        let configuration = URLSessionConfiguration.default
        configuration.httpCookieStorage = .shared
        configuration.timeoutIntervalForRequest = 90
        session = URLSession(configuration: configuration, delegate: TrustDelegate(anchor: authority), delegateQueue: nil)
    }

    /// Downloads the proxy's public root before it can be pinned. Show its fingerprint for comparison.
    public static func fetchAuthority(from baseURL: URL) async throws -> Data {
        let session = URLSession(configuration: .ephemeral, delegate: TrustDelegate(anchor: nil), delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        let (data, response) = try await session.data(from: baseURL.appending(path: "api/ca"))
        guard (response as? HTTPURLResponse)?.statusCode == 200, let pem = String(data: data, encoding: .utf8),
            let der = Authority.der(fromPEM: pem)
        else { throw APIError(status: 0, message: "This server did not provide a certificate. Is it running make phone?") }
        return der
    }

    public func isPaired() async throws -> Bool {
        struct Status: Decodable { var paired: Bool }
        return try await request("api/status", as: Status.self).paired
    }

    public func pair(code: String) async throws {
        _ = try await raw("api/pair", body: ["code": code])
    }

    public func createLiveSession() async throws -> String {
        struct Created: Decodable { var id: String }
        return try await request("api/v2/episodes", body: ["mode": "live"], as: Created.self).id
    }

    public func sendFrame(session id: String, jpeg: Data, mediaTime: Double, recording: String) async throws -> SessionState {
        var components = URLComponents(url: baseURL.appending(path: "api/sessions/\(id)/frames"), resolvingAgainstBaseURL: false)!
        components.queryItems = [.init(name: "media_time", value: String(mediaTime)), .init(name: "segment", value: recording)]
        var request = URLRequest(url: components.url!)
        request.httpMethod = "POST"
        request.setValue("image/jpeg", forHTTPHeaderField: "Content-Type")
        request.httpBody = jpeg
        return try JSON.decoder.decode(SessionState.self, from: try await perform(request))
    }

    /// Ask about the object under a tapped point of the latest analysed frame (portrait-normalized).
    public func point(session id: String, at point: SIMD2<Double>) async throws -> SessionState {
        try await request("api/sessions/\(id)/point", body: ["x": point.x, "y": point.y], as: SessionState.self)
    }

    /// Sends a frame with its optional pose and depth streams; returns the analysis.
    public func send(_ sample: Sample, session id: String, recording: String, sensor: SensorDescriptor) async throws
        -> SessionState
    {
        struct Meta: Encodable {
            var mediaTime: Double; var recording: String; var sensor: SensorDescriptor
            var camera: CameraPose?; var depthSize: [Int]?
            var observationId: String; var clockId: String; var trackingFrameId: String
        }
        let meta = Meta(
            mediaTime: sample.mediaTime, recording: recording, sensor: sensor,
            camera: sample.camera, depthSize: sample.depth == nil ? nil : sample.depthSize,
            observationId: sample.id, clockId: sample.clockId, trackingFrameId: sample.trackingFrameId)
        var form = Multipart()
        form.add("meta", data: try JSON.encoder.encode(meta), type: "application/json")
        form.add("image", data: sample.jpeg, type: "image/jpeg", filename: "frame.jpg")
        if let depth = sample.depth { form.add("depth", data: depth, filename: "depth") }
        if sample.depth != nil, let confidence = sample.confidence { form.add("confidence", data: confidence, filename: "confidence") }
        var request = URLRequest(url: baseURL.appending(path: "api/v2/episodes/\(id)/samples"))
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(form.boundary)", forHTTPHeaderField: "Content-Type")
        request.httpBody = form.finished()
        request.timeoutInterval = 15
        let data = try await deliver(request, id: sample.id)
        return try JSON.decoder.decode(SessionState.self, from: data)
    }

    /// Correct the current question's outline: add include/exclude points, step smaller/larger, or reset.
    public func refine(
        session id: String, question: String, points: [(Double, Double, Bool)] = [], granularity: String? = nil,
        reset: Bool = false
    ) async throws -> SessionState {
        struct Body: Encodable { var questionId: String; var points: [[Point]]; var granularity: String?; var reset: Bool }
        enum Point: Encodable {
            case number(Double), flag(Bool)
            func encode(to encoder: Encoder) throws {
                var container = encoder.singleValueContainer()
                switch self {
                case .number(let value): try container.encode(value)
                case .flag(let value): try container.encode(value)
                }
            }
        }
        let body = Body(
            questionId: question, points: points.map { [.number($0.0), .number($0.1), .flag($0.2)] },
            granularity: granularity, reset: reset)
        return try await request("api/sessions/\(id)/question/refine", body: body, as: SessionState.self)
    }

    public func confirmRelation(_ id: String) async throws {
        _ = try await raw("api/relations/\(id)/confirm", body: [String: String]())
    }

    public func state(session id: String) async throws -> SessionState {
        try await request("api/sessions/\(id)/state", as: SessionState.self)
    }

    public func setPlace(session id: String, label: String, resolution: Resolution? = nil, previousId: String? = nil)
        async throws -> TeachResult
    {
        struct Body: Encodable { var label: String; var resolution: Resolution?; var previousId: String? }
        return try await request(
            "api/sessions/\(id)/context", body: Body(label: label, resolution: resolution, previousId: previousId),
            as: TeachResult.self)
    }

    public func interpret(_ text: String) async throws -> Interpretation {
        try await request("api/interpret", body: ["text": text], as: Interpretation.self)
    }

    public func teach(
        session id: String, question: String, text: String, terms: Interpretation, resolution: Resolution? = nil,
        previousId: String? = nil
    ) async throws -> TeachResult {
        struct Body: Encodable {
            var questionId: String; var text: String; var label: String; var categories: [String]; var aliases: [String]
            var resolution: Resolution?; var previousId: String?
        }
        let body = Body(
            questionId: question, text: text, label: terms.label, categories: terms.categories, aliases: terms.aliases,
            resolution: resolution, previousId: previousId)
        return try await request("api/sessions/\(id)/teach", body: body, as: TeachResult.self)
    }

    public func control(session id: String, _ action: String, reason: String? = nil) async throws {
        var body = ["action": action]
        body["reason"] = reason
        _ = try await raw("api/sessions/\(id)/control", body: body)
    }

    /// Questions saved for later (answer when the camera is off).
    public func savedQuestions() async throws -> [SavedQuestion] {
        try await request("api/questions", as: [SavedQuestion].self)
    }

    public func teachSaved(
        _ question: SavedQuestion, text: String, terms: Interpretation, resolution: Resolution? = nil,
        previousId: String? = nil
    ) async throws -> TeachResult {
        struct Body: Encodable {
            var questionId: String; var text: String; var label: String; var categories: [String]; var aliases: [String]
            var resolution: Resolution?; var previousId: String?
        }
        let body = Body(
            questionId: question.id, text: text, label: terms.label, categories: terms.categories,
            aliases: terms.aliases, resolution: resolution, previousId: previousId)
        return try await request("api/questions/\(question.id)/teach", body: body, as: TeachResult.self)
    }

    public func dismissSaved(_ question: SavedQuestion) async throws {
        _ = try await raw("api/questions/\(question.id)/dismiss", body: [String: String]())
    }

    public func asset(_ name: String) async throws -> Data {
        try await perform(URLRequest(url: baseURL.appending(path: "api/assets/\(name)")))
    }

    /// Resumable upload of the phone's original recording; the backend keeps it as a session source.
    public func upload(session id: String, file: URL, progress: @Sendable (Double) -> Void) async throws {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        let size = try handle.seekToEnd()
        var offset: UInt64 = 0
        while offset < size {
            try handle.seek(toOffset: offset)
            let chunk = try handle.read(upToCount: 4 << 20) ?? Data()
            var request = URLRequest(url: baseURL.appending(path: "api/sessions/\(id)/upload"))
            request.url!.append(queryItems: [.init(name: "offset", value: String(offset))])
            request.httpMethod = "PUT"
            request.httpBody = chunk
            struct Next: Decodable { var nextOffset: UInt64 }
            do {
                offset = try JSON.decoder.decode(Next.self, from: try await perform(request)).nextOffset
            } catch let error as APIError where error.status == 409 {
                // An earlier attempt already delivered part of the file: continue where the Mac is.
                struct Conflict: Decodable { var detail: Next }
                offset = try JSON.decoder.decode(Conflict.self, from: error.body).detail.nextOffset
            }
            progress(Double(offset) / Double(max(size, 1)))
        }
        _ = try await raw("api/sessions/\(id)/upload/complete", body: [String: String]())
    }

    public func world(episode: String? = nil) async throws -> WorldSnapshot {
        var url = baseURL.appending(path: "api/v2/world")
        if let episode { url.append(queryItems: [.init(name: "episode_id", value: episode)]) }
        return try JSON.decoder.decode(WorldSnapshot.self, from: try await perform(URLRequest(url: url)))
    }

    public func answer(_ question: WorldQuestion, reply: WorldReply) async throws -> WorldAnswerResult {
        var request = URLRequest(url: baseURL.appending(path: "api/v2/questions/\(question.id)/answers"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSON.encoder.encode(reply)
        let data = try await deliver(request, id: reply.id)
        return try JSON.decoder.decode(WorldAnswerResult.self, from: data)
    }

    public func markOutput(duration: Double) async throws {
        _ = try await raw("api/v2/output", body: ["duration_seconds": min(300, duration)])
    }

    public func touch(_ question: WorldQuestion) async throws {
        _ = try await raw("api/v2/questions/\(question.id)/touch", body: [String: String]())
    }

    public func sendAudio(_ sample: AudioSample, episode: String, device: String) async throws {
        var url = baseURL.appending(path: "api/v2/episodes/\(episode)/audio")
        url.append(queryItems: [
            .init(name: "observation_id", value: sample.id), .init(name: "source_id", value: sample.sourceId),
            .init(name: "clock_id", value: sample.clockId), .init(name: "timestamp", value: String(sample.timestamp)),
            .init(name: "device_id", value: device), .init(name: "playback", value: sample.playback ? "true" : "false")])
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("audio/wav", forHTTPHeaderField: "Content-Type")
        request.httpBody = sample.wav
        request.timeoutInterval = 15
        _ = try await deliver(request, id: sample.id)
    }

    public func retryPending() async throws {
        try await delivery.run { _ = try await self.drain() }
    }

    private func drain(wanted: String? = nil) async throws -> Data? {
        guard let outbox else { throw APIError(status: 0, message: "Cannot open the local observation queue") }
        var result: Data?
        for entry in try await outbox.pending() {
            do {
                let data = try await perform(ObservationOutbox.request(entry))
                if entry.id == wanted { result = data }
                if entry.url.path.hasSuffix("/answers"), entry.id != wanted,
                   let answer = try? JSON.decoder.decode(WorldAnswerResult.self, from: data), answer.needsClarification == true {
                    try await outbox.reject(entry, reason: "A saved answer needs identity clarification. Open Answer later.")
                } else {
                    try await outbox.remove(entry.id)
                }
            } catch let error as APIError where (400..<500).contains(error.status) && error.status != 401 && error.status != 429 {
                // Preserve rejected payloads for review, and let newer observations continue.
                try await outbox.reject(entry, reason: error.message)
                if entry.id == wanted { throw error }
            }
        }
        return result
    }

    private func deliver(_ request: URLRequest, id: String) async throws -> Data {
        guard let outbox else { throw APIError(status: 0, message: "Cannot save observations on this phone") }
        try await outbox.enqueue(id: id, request: request)
        return try await delivery.run {
            if let data = try await self.drain(wanted: id) { return data }
            // Another drain may already have delivered this exact request; the server is idempotent.
            return try await self.perform(request)
        }
    }

    // MARK: transport

    private func request<T: Decodable>(_ path: String, as type: T.Type) async throws -> T {
        try JSON.decoder.decode(T.self, from: try await perform(URLRequest(url: baseURL.appending(path: path))))
    }

    private func request<T: Decodable>(_ path: String, body: some Encodable, as type: T.Type) async throws -> T {
        try JSON.decoder.decode(T.self, from: try await raw(path, body: body))
    }

    private func raw(_ path: String, body: some Encodable) async throws -> Data {
        var request = URLRequest(url: baseURL.appending(path: path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSON.encoder.encode(body)
        return try await perform(request)
    }

    private func perform(_ request: URLRequest) async throws -> Data {
        let (data, response) = try await session.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(status) else {
            throw APIError(status: status, message: Self.message(data, status), body: data)
        }
        return data
    }

    struct Multipart {
        let boundary = "grounded-" + UUID().uuidString
        private var body = Data()

        mutating func add(_ name: String, data: Data, type: String = "application/octet-stream", filename: String? = nil) {
            var header = "--\(boundary)\r\nContent-Disposition: form-data; name=\"\(name)\""
            if let filename { header += "; filename=\"\(filename)\"" }
            header += "\r\nContent-Type: \(type)\r\n\r\n"
            body.append(Data(header.utf8))
            body.append(data)
            body.append(Data("\r\n".utf8))
        }

        func finished() -> Data { body + Data("--\(boundary)--\r\n".utf8) }
    }

    static func message(_ data: Data, _ status: Int) -> String {
        if status == 401 { return "This phone is not paired. Enter the pairing code from the Mac." }
        struct Detail: Decodable { var detail: String }
        if let detail = try? JSONDecoder().decode(Detail.self, from: data) { return detail.detail }
        let text = String(decoding: data, as: UTF8.self)
        return text.isEmpty ? "Server error \(status)" : text
    }
}

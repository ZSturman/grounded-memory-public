import Foundation
import CryptoKit

/// Disk-backed, server-scoped requests. Original IDs survive timeouts, relaunches, and retries.
/// Failed requests remain on disk; originals are separately preserved by the recording pipeline.
public actor ObservationOutbox {
    public struct Entry: Codable, Sendable {
        var id: String
        var url: URL
        var contentType: String
        var body: Data
        var createdAt: Date
        var failure: String?
    }
    private let directory: URL

    public init(server: URL, root: URL? = nil) throws {
        let name = SHA256.hash(data: Data(server.absoluteString.utf8)).map { String(format: "%02x", $0) }.joined()
        let base = try root ?? FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                                      appropriateFor: nil, create: true)
        directory = base.appending(path: "GroundedMemory/Outbox/\(name)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    public func enqueue(id: String, request: URLRequest) throws {
        guard let url = request.url, let body = request.httpBody else { return }
        // Hash the identifier rather than allowing identifiers to become file paths.
        let entry = Entry(id: id, url: url, contentType: request.value(forHTTPHeaderField: "Content-Type") ?? "application/json",
                          body: body, createdAt: Date())
        let path = file(id)
        if FileManager.default.fileExists(atPath: path.path) { return }
        try JSONEncoder().encode(entry).write(to: path, options: .atomic)
    }

    public func pending() throws -> [Entry] {
        try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }
            .map { try JSONDecoder().decode(Entry.self, from: Data(contentsOf: $0)) }
            .sorted { $0.createdAt < $1.createdAt }
    }

    public func reject(_ entry: Entry, reason: String) throws {
        var entry = entry
        entry.failure = reason
        let rejected = directory.appending(path: "NeedsAttention", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: rejected, withIntermediateDirectories: true)
        try JSONEncoder().encode(entry).write(to: rejected.appending(path: file(entry.id).lastPathComponent), options: .atomic)
        try remove(entry.id)
    }

    public func issues() throws -> [String] {
        let folder = directory.appending(path: "NeedsAttention", directoryHint: .isDirectory)
        guard FileManager.default.fileExists(atPath: folder.path) else { return [] }
        return try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)
            .map { try JSONDecoder().decode(Entry.self, from: Data(contentsOf: $0)) }
            .map { $0.failure ?? "A saved request needs review" }
    }

    public func remove(_ id: String) throws { try FileManager.default.removeItem(at: file(id)) }

    public static func request(_ entry: Entry) -> URLRequest {
        var request = URLRequest(url: entry.url)
        request.httpMethod = "POST"
        request.httpBody = entry.body
        request.setValue(entry.contentType, forHTTPHeaderField: "Content-Type")
        return request
    }

    private func file(_ id: String) -> URL {
        let name = SHA256.hash(data: Data(id.utf8)).map { String(format: "%02x", $0) }.joined()
        return directory.appending(path: name + ".json")
    }
}

/// Actor reentrancy must not allow two network drains to race each other.
actor DeliveryGate {
    private var tail: Task<Void, Never>?

    func run<T: Sendable>(_ operation: @escaping @Sendable () async throws -> T) async throws -> T {
        let previous = tail
        let task = Task {
            await previous?.value
            return try await operation()
        }
        tail = Task { _ = try? await task.value }
        return try await task.value
    }
}

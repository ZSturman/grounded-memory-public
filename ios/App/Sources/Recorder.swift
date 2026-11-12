import ARKit
import AVFoundation

/// Writes the original ARKit camera frames and microphone audio to a local movie for later upload.
/// Called only from the capture queue.
final class Recorder {
    let url: URL
    private var writer: AVAssetWriter?
    private var video: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var audio: AVAssetWriterInput?
    private var start: CMTime?
    private(set) var failure: String?

    static let folder: URL = {
        // Application Support, not Caches: originals must survive until they reach the Mac.
        let folder = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appending(path: "Recordings", directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return folder
    }()

    /// The session id is kept beside the movie so an interrupted upload can be resumed later.
    init(session: String) {
        url = Self.folder.appending(path: UUID().uuidString.lowercased() + ".mov")
        try? Data(session.utf8).write(to: url.appendingPathExtension("session"))
    }

    /// Recordings still on this phone, with the session each belongs to.
    static func pending() -> [(url: URL, session: String)] {
        let files = (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
        return files.filter { $0.pathExtension == "mov" }.compactMap { movie in
            guard let data = try? Data(contentsOf: movie.appendingPathExtension("session")) else { return nil }
            return (movie, String(decoding: data, as: UTF8.self))
        }
    }

    static func remove(_ movie: URL) {
        try? FileManager.default.removeItem(at: movie)
        try? FileManager.default.removeItem(at: movie.appendingPathExtension("session"))
    }

    var name: String { url.lastPathComponent }

    func append(_ frame: ARFrame) {
        let time = CMTime(seconds: frame.timestamp, preferredTimescale: 1_000_000)
        if writer == nil && failure == nil { begin(width: CVPixelBufferGetWidth(frame.capturedImage), height: CVPixelBufferGetHeight(frame.capturedImage), at: time) }
        guard let video, video.isReadyForMoreMediaData, writer?.status == .writing else { return }
        adaptor?.append(frame.capturedImage, withPresentationTime: time)
        if writer?.status == .failed {
            // e.g. the hardware encoder is revoked when the app leaves the foreground.
            failure = writer?.error?.localizedDescription ?? "Recording stopped unexpectedly"
        }
    }

    func append(audio buffer: CMSampleBuffer) {
        guard let audio, let start, audio.isReadyForMoreMediaData, writer?.status == .writing,
            CMSampleBufferGetPresentationTimeStamp(buffer) >= start
        else { return }
        audio.append(buffer)
    }

    /// Closes the movie. A failed segment is still returned: its fragments stay readable up to the failure.
    func finish() async -> URL? {
        if let writer, writer.status == .writing {
            video?.markAsFinished()
            audio?.markAsFinished()
            await writer.finishWriting()
            if writer.status != .completed { failure = writer.error?.localizedDescription }
        }
        let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
        guard size > 0 else {
            Self.remove(url)
            return nil
        }
        return url
    }

    private func begin(width: Int, height: Int, at time: CMTime) {
        do {
            let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
            // Fragments keep everything up to the last few seconds playable if the app is killed.
            writer.movieFragmentInterval = CMTime(seconds: 5, preferredTimescale: 600)
            let video = AVAssetWriterInput(
                mediaType: .video,
                outputSettings: [AVVideoCodecKey: AVVideoCodecType.hevc, AVVideoWidthKey: width, AVVideoHeightKey: height])
            video.expectsMediaDataInRealTime = true
            // Sensor frames are landscape; the phone is held upright.
            video.transform = CGAffineTransform(rotationAngle: .pi / 2)
            let audio = AVAssetWriterInput(
                mediaType: .audio,
                outputSettings: [AVFormatIDKey: kAudioFormatMPEG4AAC, AVNumberOfChannelsKey: 1, AVSampleRateKey: 44_100])
            audio.expectsMediaDataInRealTime = true
            writer.add(video)
            if writer.canAdd(audio) { writer.add(audio) }
            adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: video)
            guard writer.startWriting() else { throw writer.error ?? CocoaError(.fileWriteUnknown) }
            writer.startSession(atSourceTime: time)
            (self.writer, self.video, self.audio, start) = (writer, video, audio, time)
        } catch {
            failure = "Recording unavailable: \(error.localizedDescription)"
            writer = nil
        }
    }
}

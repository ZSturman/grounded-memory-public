import ARKit
import CoreImage
import UIKit
import GroundedKit

/// Pose of the camera when a frame was sent, so a later answer can be cast back into the world.
struct SentCamera {
    var transform: simd_float4x4
    var intrinsics: simd_float3x3
    var resolution: SIMD2<Double>
}

/// A label fixed to a point in the room rather than to the screen.
struct Marker {
    enum Kind { case known, question }
    var text: String
    var detail: String
    var kind: Kind
    var world: SIMD3<Float>
}

struct ProjectedMarker: Identifiable, Equatable {
    var id: String
    var text: String
    var detail: String
    var isQuestion: Bool
    var point: CGPoint
}

/// Owns the ARKit session. Every ARKit callback and all mutable state live on `queue`.
final class CaptureEngine: NSObject, ARSessionDelegate, @unchecked Sendable {
    let session = ARSession()
    /// Called on `queue` with an upright JPEG and optional pose/depth; call `frameHandled()` when answered.
    var onFrame: ((Sample, SentCamera) -> Void)?
    var onAudio: ((AudioSample) -> Void)?
    let clockId = UUID().uuidString
    private var trackingFrameId = UUID().uuidString
    private let environmentalAudio = EnvironmentalAudio()
    private var isPlaying = false
    private(set) var sensor: SensorDescriptor?
    var onMarkers: (([ProjectedMarker]) -> Void)?
    var onTracking: ((String?) -> Void)?
    /// Called on `queue` when a recording segment ends early, with the reason.
    var onRecordingNote: ((String) -> Void)?

    private let queue = DispatchQueue(label: "grounded.capture", qos: .userInitiated)
    private let images = CIContext()
    private var recorder: Recorder?
    private var recordingSession: String?
    /// Segments being closed or closed; all are uploaded at the end.
    private var segments: [Task<URL?, Never>] = []
    private var foreground: [NSObjectProtocol] = []
    private var start: TimeInterval?
    private var lastSent: TimeInterval = 0
    private var inFlight = false
    private var markers: [String: Marker] = [:]
    private var viewport = CGSize.zero
    private var frames = 0
    private var trackingNote: String?

    var recordingName: String? { queue.sync { recorder?.name } }

    func run(recordingFor sessionId: String) {
        let configuration = ARWorldTrackingConfiguration()
        configuration.providesAudioData = true
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            configuration.frameSemantics.insert(.sceneDepth)
        }
        sensor = Self.describe(configuration)
        queue.sync { recordingSession = sessionId }
        // iOS revokes the video encoder in the background: close the segment cleanly; a new one starts on return.
        foreground = [
            NotificationCenter.default.addObserver(
                forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: nil
            ) { [weak self] _ in
                guard let self else { return }
                self.queue.async { self.closeSegment(note: "Recording paused while the app was in the background") }
            }
        ]
        session.delegateQueue = queue
        session.delegate = self
        environmentalAudio.onChunk = { [weak self] wav, timestamp, playback in
            guard let self, let recordingSession = self.recordingSession else { return }
            self.onAudio?(AudioSample(wav: wav, timestamp: timestamp, clockId: self.clockId,
                                      sourceId: recordingSession + "-audio", playback: playback))
        }
        session.run(configuration)
    }

    /// Ends capture and returns every recording segment written in this session, in order.
    func stop() async -> [URL] {
        session.pause()
        foreground.forEach(NotificationCenter.default.removeObserver)
        let closing = queue.sync { () -> [Task<URL?, Never>] in
            environmentalAudio.flush()
            closeSegment(note: nil)
            recordingSession = nil
            return segments
        }
        var urls: [URL] = []
        for segment in closing { if let url = await segment.value { urls.append(url) } }
        return urls
    }

    /// On `queue`: finish the current segment; the next frame starts a new one while recording.
    private func closeSegment(note: String?) {
        guard let recorder else { return }
        self.recorder = nil
        segments.append(Task { await recorder.finish() })
        if let note { onRecordingNote?(note) }
    }

    func setViewport(_ size: CGSize) { queue.async { self.viewport = size } }

    func setPlayback(_ active: Bool) { queue.async { self.isPlaying = active } }

    func frameHandled() { queue.async { self.inFlight = false } }

    /// Fix a label at the 3D point under a region's center, as seen from the camera that sent its frame.
    func anchor(_ key: String, bbox: [Double], from camera: SentCamera, text: String, detail: String, kind: Marker.Kind) {
        queue.async {
            let point = Geometry.sensorPoint(fromPortrait: Geometry.center(of: bbox))
            let ray = Geometry.ray(
                through: point, resolution: camera.resolution, intrinsics: camera.intrinsics, transform: camera.transform)
            let query = ARRaycastQuery(origin: ray.origin, direction: ray.direction, allowing: .estimatedPlane, alignment: .any)
            let hit = self.session.raycast(query).first.map {
                SIMD3<Float>($0.worldTransform.columns.3.x, $0.worldTransform.columns.3.y, $0.worldTransform.columns.3.z)
            }
            self.markers[key] = Marker(
                text: text, detail: detail, kind: kind, world: hit ?? ray.origin + ray.direction * 1.2)
        }
    }

    /// Screen point -> portrait-normalized point in the camera image, through ARKit's aspect-fill display.
    func imagePoint(forViewPoint point: CGPoint) -> SIMD2<Double>? {
        queue.sync {
            guard let frame = session.currentFrame, viewport.width > 0, viewport.height > 0 else { return nil }
            let toImage = frame.displayTransform(for: .portrait, viewportSize: viewport).inverted()
            let sensor = CGPoint(x: point.x / viewport.width, y: point.y / viewport.height).applying(toImage)
            return Geometry.portraitPoint(fromSensor: SIMD2(Double(sensor.x), Double(sensor.y)))
        }
    }

    func removeMarker(_ key: String) { queue.async { self.markers[key] = nil } }

    // MARK: ARSessionDelegate (on `queue`)

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        if recorder == nil, let recordingSession { recorder = Recorder(session: recordingSession) }
        recorder?.append(frame)
        if let failure = recorder?.failure {
            closeSegment(note: "Recording restarted after an error: \(failure)")
        }
        if start == nil { start = frame.timestamp }
        frames += 1
        if frames % 2 == 0 { project(frame) }
        guard !inFlight, frame.timestamp - lastSent >= 0.5, let start, let jpeg = upright(frame.capturedImage) else { return }
        inFlight = true
        lastSent = frame.timestamp
        let resolution = SIMD2(Double(frame.camera.imageResolution.width), Double(frame.camera.imageResolution.height))
        let camera = SentCamera(transform: frame.camera.transform, intrinsics: frame.camera.intrinsics, resolution: resolution)
        let t = frame.camera.transform, k = frame.camera.intrinsics
        let pose = CameraPose(
            transform: [t.columns.0, t.columns.1, t.columns.2, t.columns.3].flatMap { [$0.x, $0.y, $0.z, $0.w] },
            intrinsics: [k.columns.0, k.columns.1, k.columns.2].flatMap { [$0.x, $0.y, $0.z] },
            resolution: [resolution.x, resolution.y])
        let depth = frame.sceneDepth.flatMap { Self.millimetres($0.depthMap) }
        let confidence = frame.sceneDepth?.confidenceMap.flatMap(Self.bytes)
        var sample = Sample(
            jpeg: jpeg, mediaTime: frame.timestamp - start, camera: pose, depth: depth?.data, confidence: confidence,
            depthSize: depth.map { [$0.width, $0.height] })
        sample.clockId = clockId
        sample.trackingFrameId = trackingFrameId
        onFrame?(sample, camera)
    }

    func session(_ session: ARSession, didOutputAudioSampleBuffer audioSampleBuffer: CMSampleBuffer) {
        recorder?.append(audio: audioSampleBuffer)
        if let start { environmentalAudio.append(audioSampleBuffer, origin: start, playback: isPlaying) }
    }

    func sessionWasInterrupted(_ session: ARSession) {
        trackingFrameId = UUID().uuidString
        markers.removeAll()
        environmentalAudio.flush()
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        trackingFrameId = UUID().uuidString
        markers.removeAll()
        onTracking?(error.localizedDescription)
    }

    func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        let note: String?
        switch camera.trackingState {
        case .normal: note = nil
        case .notAvailable: note = "Tracking unavailable"
        case .limited(.excessiveMotion): note = "Move a little slower"
        case .limited(.insufficientFeatures): note = "Point at something with more detail or light"
        case .limited: note = "Look around slowly to get started"
        }
        if note != trackingNote {
            trackingNote = note
            onTracking?(note)
        }
    }

    // MARK: helpers

    /// What this capture is: device, camera, lens and depth sensor. Sent with every sample.
    private static func describe(_ configuration: ARWorldTrackingConfiguration) -> SensorDescriptor {
        var system = utsname()
        uname(&system)
        let model = withUnsafeBytes(of: system.machine) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
        let format = configuration.videoFormat
        let lens: String
        switch format.captureDeviceType {
        case .builtInUltraWideCamera: lens = "ultra-wide"
        case .builtInTelephotoCamera: lens = "telephoto"
        case .builtInWideAngleCamera: lens = "wide"
        default: lens = format.captureDeviceType.rawValue
        }
        return SensorDescriptor(
            source: "arkit-world-tracking", projection: "rectilinear", device: model,
            platform: "iOS " + UIDevice.current.systemVersion, camera: format.captureDevicePosition == .front ? "front" : "back",
            lens: lens, resolution: [Int(format.imageResolution.width), Int(format.imageResolution.height)],
            fps: format.framesPerSecond,
            depth: configuration.frameSemantics.contains(.sceneDepth) ? "lidar" : nil)
    }

    /// LiDAR depth (Float32 metres) -> little-endian UInt16 millimetres.
    private static func millimetres(_ buffer: CVPixelBuffer) -> (data: Data, width: Int, height: Int)? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        var values = [UInt16](repeating: 0, count: width * height)
        for row in 0..<height {
            let line = (base + row * stride).assumingMemoryBound(to: Float32.self)
            for column in 0..<width {
                let metres = line[column]
                values[row * width + column] = metres.isFinite ? UInt16(clamping: Int(metres * 1000)) : 0
            }
        }
        return (values.withUnsafeBytes { Data($0) }, width, height)
    }

    private static func bytes(_ buffer: CVPixelBuffer) -> Data? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        var data = Data(capacity: width * height)
        for row in 0..<height { data.append(UnsafeRawPointer(base + row * stride).assumingMemoryBound(to: UInt8.self), count: width) }
        return data
    }

    private func upright(_ buffer: CVPixelBuffer) -> Data? {
        let image = CIImage(cvPixelBuffer: buffer).oriented(.right)
        let scale = 960 / max(image.extent.width, image.extent.height)
        return images.jpegRepresentation(
            of: image.transformed(by: CGAffineTransform(scaleX: scale, y: scale)),
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
            options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.7])
    }

    private func project(_ frame: ARFrame) {
        guard viewport != .zero else { return }
        let view = frame.camera.transform.inverse
        let visible = markers.compactMap { key, marker -> ProjectedMarker? in
            let local = view * SIMD4(marker.world, 1)
            guard local.z < -0.05 else { return nil }  // behind the camera
            let point = frame.camera.projectPoint(marker.world, orientation: .portrait, viewportSize: viewport)
            guard point.x > -40, point.y > -40, point.x < viewport.width + 40, point.y < viewport.height + 40 else { return nil }
            return ProjectedMarker(
                id: key, text: marker.text, detail: marker.detail, isQuestion: marker.kind == .question, point: point)
        }
        onMarkers?(visible.sorted { $0.id < $1.id })
    }
}

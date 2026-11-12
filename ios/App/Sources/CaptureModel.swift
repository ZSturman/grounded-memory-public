import AVFoundation
import GroundedKit
import SwiftUI

/// One live session: frames out, analysis back, and the person's answers.
@MainActor
final class CaptureModel: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {
    enum Completion: Equatable {
        case savedToMac, keptOnPhone, noRecording
        var nextStep: String {
            switch self {
            case .savedToMac: return "Next: inspect beliefs on your Mac in Capture & world learning. To review saved evidence, open Collection → Collect saved inputs now → Labeling."
            case .keptOnPhone: return "Next: tap Done, reconnect to your Mac, then use Upload now on the home screen. Keep the recording on this phone until upload succeeds."
            case .noRecording: return "No original recording was captured. Inspect any saved observations and questions on your Mac, or start another session after checking camera and microphone access."
            }
        }
    }
    enum Phase: Equatable { case starting, running, stopping, uploading(Double), finished(String, Completion), failed(String) }
    /// A name the Mac already knows: is this the same thing/place?
    struct Clarification { var isPlace: Bool; var text: String; var terms: Interpretation; var matches: [Entity] }

    @Published var phase: Phase = .starting
    @Published var state: SessionState?
    @Published var world: WorldSnapshot?
    @Published var worldChoices: [WorldAnswerResult.Choice] = []
    @Published var expectedEvent = ""
    @Published var selectedWorldQuestion: WorldQuestion?
    private var playback: AVAudioPlayer?
    private var lastWorldQuestion: String?
    var worldQuestion: WorldQuestion? { selectedWorldQuestion ?? world?.attention }
    @Published var markers: [ProjectedMarker] = []
    @Published var tracking: String?
    @Published var error: String?
    @Published var answer = ""
    @Published var placeText = ""
    @Published var proposal: Interpretation?
    @Published var clarification: Clarification?
    @Published var guessRejected = false
    @Published var busy = false
    @Published var speak = false
    @Published var analysed = 0
    @Published var latency: Double?
    @Published var crops: [String: UIImage] = [:]
    @Published var recordingNote: String?
    /// Confirmation shown until dismissed or replaced, e.g. "Saved for later".
    @Published var notice: String?
    private var questionShownAt = Date()
    /// Untouched questions are saved for later after this long, so capture never stalls on one thing.
    static let questionPatience: TimeInterval = 20
    @Published var fixing = false
    @Published var includeTaps = true
    @Published var confirmed: Set<String> = []

    let engine = CaptureEngine()
    let client: GroundedClient
    private var sessionId: String?
    private var cameras: [String: SentCamera] = [:]
    private let voice = AVSpeechSynthesizer()
    private var spoken: String?

    init(client: GroundedClient) { self.client = client; super.init(); voice.delegate = self }

    func start() async {
        do {
            sessionId = try await client.createLiveSession()
            engine.onFrame = { [weak self] sample, camera in Task { @MainActor in await self?.send(sample, camera) } }
            engine.onAudio = { [weak self] sample in Task { @MainActor in
                guard let self, let sessionId = self.sessionId else { return }
                do { try await self.client.sendAudio(sample, episode: sessionId, device: self.engine.sensor?.device ?? "iPhone") }
                catch { self.error = "Live audio unavailable; saved for retry. " + AppModel.describe(error) }
            } }
            engine.onMarkers = { [weak self] markers in Task { @MainActor in self?.markers = markers } }
            engine.onTracking = { [weak self] note in Task { @MainActor in self?.tracking = note } }
            engine.onRecordingNote = { [weak self] note in Task { @MainActor in self?.recordingNote = note } }
            engine.run(recordingFor: sessionId!)
            phase = .running
            Task { await self.moveOnWhenIgnored() }
            Task { await self.refreshWorld() }
        } catch {
            phase = .failed(AppModel.describe(error))
        }
    }

    private func send(_ sample: Sample, _ camera: SentCamera) async {
        defer { engine.frameHandled() }
        guard let sessionId, phase == .running, let recording = engine.recordingName, let sensor = engine.sensor else { return }
        let began = Date()
        do {
            let state = try await client.send(sample, session: sessionId, recording: recording, sensor: sensor)
            latency = Date().timeIntervalSince(began)
            analysed += 1
            error = nil
            if let frame = state.latest.frame { cameras[frame] = camera }
            if cameras.count > 200 { cameras = cameras.filter { $0.key == state.question?.frame } }
            apply(state, camera: camera)
        } catch {
            self.error = AppModel.describe(error)
        }
    }

    func apply(_ new: SessionState, camera: SentCamera) {
        if new.status == "stopped", phase == .running {
            // Stopped from the Mac: finish and upload the recording here too.
            Task { await stop() }
            return
        }
        let previousQuestion = state?.question?.id
        state = new
        if let snapshot = new.world { applyWorld(snapshot) }
        for region in new.latest.regions {
            // Weak appearance matches (shelf edge "carpet" at 0.50) are not shown as recognitions.
            guard let match = region.best, match.appearance >= 0.75 else { continue }
            engine.anchor(
                "e:" + match.id, bbox: region.bbox, from: camera, text: match.label,
                detail: "\(Int(match.appearance * 100))%", kind: .known)
        }
        if let preview = new.question?.preview { load(preview) }
        if let question = new.question {
            if question.id != previousQuestion {
                questionShownAt = Date()
                engine.anchor(
                    "q", bbox: question.region.bbox, from: cameras[question.frame] ?? camera, text: "?", detail: "",
                    kind: .question)
                proposal = nil
                answer = ""
                load(question.region.asset)
                say(question.text)
            }
        } else if previousQuestion != nil {
            engine.removeMarker("q")
        }
        if new.world == nil && new.context == nil { say(placePrompt) }
    }

    var placePrompt: String {
        if let guess = state?.placeGuess, !guessRejected { return "Is this \(guess.label)?" }
        return "Where are we?"
    }

    // MARK: place

    func confirmGuess() async {
        guard let guess = state?.placeGuess else { return }
        await setPlace(guess.label, resolution: .same, previous: guess.id)
    }

    func setPlace(_ label: String, resolution: Resolution? = nil, previous: String? = nil) async {
        guard let sessionId, !label.trimmingCharacters(in: .whitespaces).isEmpty else { return }
        await run {
            let result = try await self.client.setPlace(
                session: sessionId, label: label.trimmingCharacters(in: .whitespaces), resolution: resolution,
                previousId: previous)
            if result.needsClarification == true {
                self.clarification = Clarification(
                    isPlace: true, text: label, terms: Interpretation(label: label), matches: result.matches ?? [])
            } else {
                self.clarification = nil
                self.state = try await self.client.state(session: sessionId)
            }
        }
    }

    // MARK: teaching

    /// The person tapped an object: make it the question instead of waiting for the system to pick.
    func tap(_ location: CGPoint) async {
        guard let sessionId, clarification == nil,
            let point = engine.imagePoint(forViewPoint: location)
        else { return }
        await run {
            let new = try await self.client.point(session: sessionId, at: point)
            if let question = new.question, let camera = self.cameras[question.frame] {
                self.apply(new, camera: camera)
            } else {
                self.state = new
            }
        }
    }

    /// Correct the outline on the frozen frame: a tapped point (include/exclude), a size step, or a reset.
    func refine(point: SIMD2<Double>? = nil, granularity: String? = nil, reset: Bool = false) async {
        guard let sessionId, let question = state?.question else { return }
        let include = includeTaps
        await run {
            let new = try await self.client.refine(
                session: sessionId, question: question.id, points: point.map { [($0.x, $0.y, include)] } ?? [],
                granularity: granularity, reset: reset)
            self.state = new
            if let snapshot = new.world { self.applyWorld(snapshot) }
            guard let updated = new.question else { return }
            if let preview = updated.preview { self.load(preview) }
            self.load(updated.region.asset)
            if let camera = self.cameras[updated.frame] {
                self.engine.anchor("q", bbox: updated.region.bbox, from: camera, text: "?", detail: "", kind: .question)
            }
        }
    }

    /// The person agrees with a measured relation; only then is it remembered as a claim.
    func confirm(_ relation: ObservedRelation) async {
        await run {
            try await self.client.confirmRelation(relation.id)
            self.confirmed.insert(relation.id)
        }
    }

    /// Accept what the Mac already thinks this is; the person's confirmation is the evidence.
    func acceptSuggestion(_ label: String) {
        answer = "It's the \(label)."
        proposal = Interpretation(label: label)
    }

    func interpret() async {
        let text = answer.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        await run {
            do {
                self.proposal = try await self.client.interpret(text)
            } catch {
                // The assistant is optional; the person's own words remain valid teaching.
                self.proposal = Interpretation(label: text)
            }
        }
    }

    func teach(resolution: Resolution? = nil, previous: String? = nil) async {
        guard let sessionId, let question = state?.question, let proposal else { return }
        let text = answer.trimmingCharacters(in: .whitespacesAndNewlines)
        await run {
            let result = try await self.client.teach(
                session: sessionId, question: question.id, text: text, terms: proposal, resolution: resolution,
                previousId: previous)
            if result.needsClarification == true {
                self.clarification = Clarification(
                    isPlace: false, text: text, terms: proposal, matches: result.matches ?? [])
                result.matches?.compactMap { $0.examples?.last?.asset }.forEach(self.load)
                return
            }
            if let camera = self.cameras[question.frame] {
                self.engine.anchor(
                    "e:" + (result.entityId ?? question.id), bbox: question.region.bbox, from: camera,
                    text: proposal.label, detail: "taught", kind: .known)
            }
            self.engine.removeMarker("q")
            self.clarification = nil
            self.proposal = nil
            self.answer = ""
            self.state = try await self.client.state(session: sessionId)
        }
    }

    func resolve(_ resolution: Resolution, with entity: Entity) async {
        guard let clarification else { return }
        if clarification.isPlace {
            await setPlace(clarification.text, resolution: resolution, previous: entity.id)
        } else {
            await teach(resolution: resolution, previous: entity.id)
        }
    }

    /// Save the current question for later without waiting, if the person hasn't started on it.
    private func moveOnWhenIgnored() async {
        while phase == .running {
            try? await Task.sleep(for: .seconds(2))
            let untouched = answer.isEmpty && proposal == nil && clarification == nil && !fixing && !busy
            if world == nil, state?.question != nil, untouched, Date().timeIntervalSince(questionShownAt) > Self.questionPatience {
                await control("later", reason: "unanswered")
                show("Saved for later — answer it from the home screen")
            }
        }
    }

    func show(_ text: String) {
        notice = text
    }

    func control(_ action: String, reason: String? = nil) async {
        guard let sessionId else { return }
        await run {
            try await self.client.control(session: sessionId, action, reason: reason)
            self.proposal = nil
            self.answer = ""
            self.clarification = nil
            self.engine.removeMarker("q")
            self.state = try await self.client.state(session: sessionId)
        }
    }

    // MARK: finish

    func stop() async {
        guard let sessionId else { return }
        phase = .stopping
        voice.stopSpeaking(at: .immediate)
        let segments = await engine.stop()
        try? await client.control(session: sessionId, "stop")
        guard !segments.isEmpty else { return phase = .finished(recordingNote ?? "Observation stopped. No original recording was captured.", .noRecording) }
        for (index, url) in segments.enumerated() {
            do {
                try await client.upload(session: sessionId, file: url) { progress in
                    Task { @MainActor in self.phase = .uploading((Double(index) + progress) / Double(segments.count)) }
                }
                Recorder.remove(url)
            } catch {
                return phase = .finished(
                    "The recording is still on this phone. Upload failed: \(AppModel.describe(error))", .keptOnPhone)
            }
        }
        let parts = segments.count == 1 ? "recording" : "recording (\(segments.count) parts)"
        phase = .finished("The \(parts) was saved on your Mac.", .savedToMac)
    }

    // MARK: helpers

    private func run(_ work: @escaping () async throws -> Void) async {
        busy = true
        defer { busy = false }
        do { try await work() } catch { self.error = AppModel.describe(error) }
    }

    func load(_ asset: String) {
        guard crops[asset] == nil else { return }
        Task {
            if let data = try? await client.asset(asset), let image = UIImage(data: data) { crops[asset] = image }
        }
    }

    private func applyWorld(_ snapshot: WorldSnapshot) {
        world = snapshot
        if let question = worldQuestion {
            if let region = question.region { load(region.asset) }
            for ref in question.evidenceRefs where ref.hasSuffix(".jpg") || ref.hasSuffix(".png") { load(ref) }
            if lastWorldQuestion != question.id {
                lastWorldQuestion = question.id
                answer = ""
                expectedEvent = ""
                worldChoices = []
                say(question.text)
            }
        }
    }

    private func refreshWorld() async {
        while phase == .running {
            if let sessionId {
                do {
                    try await client.retryPending()
                    applyWorld(try await client.world(episode: sessionId))
                } catch { self.error = "Live inference unavailable; recording continues. " + AppModel.describe(error) }
            }
            try? await Task.sleep(for: .seconds(3))
        }
    }

    func answerWorld(_ response: String = "answer", entityId: String? = nil) async {
        guard let question = worldQuestion else { return }
        await run {
            let reply = WorldReply(question: question, text: self.answer, response: response, entityId: entityId,
                                   expectedEvent: self.expectedEvent.isEmpty ? nil : self.expectedEvent)
            let result = try await self.client.answer(question, reply: reply)
            if result.needsClarification == true {
                self.worldChoices = result.matches ?? []
                return
            }
            self.selectedWorldQuestion = nil
            self.worldChoices = []
            self.answer = ""
            self.expectedEvent = ""
            if let sessionId = self.sessionId {
                self.state = try await self.client.state(session: sessionId)
                self.applyWorld(try await self.client.world(episode: sessionId))
            }
        }
    }

    func touchWorld() async {
        if let question = worldQuestion { try? await client.touch(question) }
    }

    func playWorldSound(_ asset: String) async {
        do {
            let data = try await client.asset(asset)
            engine.setPlayback(true)
            let player = try AVAudioPlayer(data: data)
            playback = player
            player.play()
            try? await Task.sleep(for: .seconds(player.duration + 0.5))
            engine.setPlayback(voice.isSpeaking)
        } catch { self.error = AppModel.describe(error); engine.setPlayback(voice.isSpeaking) }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in self.engine.setPlayback(self.playback?.isPlaying == true) }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in self.engine.setPlayback(self.playback?.isPlaying == true) }
    }

    private func say(_ text: String) {
        guard speak, spoken != text else { return }
        spoken = text
        engine.setPlayback(true)
        voice.speak(AVSpeechUtterance(string: text))
    }
}

import AVFoundation
import GroundedKit
import SwiftUI

struct WorldSavedAnswerView: View {
    let client: GroundedClient
    let question: WorldQuestion
    var done: () -> Void
    @Environment(\.dismiss) private var dismiss
    @StateObject private var images = ImageCache()
    @State private var text = ""
    @State private var expected = ""
    @State private var choices: [WorldAnswerResult.Choice] = []
    @State private var error: String?
    @State private var busy = false
    @State private var player: AVAudioPlayer?

    var body: some View {
        Form {
            Section {
                Text(question.text).font(.headline)
                Text(question.reason).foregroundStyle(.secondary)
                if let image = question.region?.asset ?? question.evidenceRefs.first(where: { $0.hasSuffix(".jpg") }) {
                    images.view(image, client: client).frame(height: 200)
                }
                if let sound = question.evidenceRefs.first(where: { $0.hasSuffix(".wav") }) {
                    Button("Listen to evidence") { Task {
                        do {
                            let audio = try await client.asset(sound)
                            player = try AVAudioPlayer(data: audio)
                            try await client.markOutput(duration: (player?.duration ?? 30) + 1)
                            player?.play()
                        }
                        catch { self.error = AppModel.describe(error) }
                    } }
                }
            }
            Section("Your answer") {
                TextField("Use your own words", text: $text, axis: .vertical)
                if ["explain", "clarify", "predict"].contains(question.kind) {
                    TextField("What should happen within 30 seconds? (optional)", text: $expected)
                }
                if choices.isEmpty {
                    Button("Remember") { Task { await save() } }.disabled(text.isEmpty)
                } else {
                    ForEach(choices) { choice in Button("Same \(choice.label)") { Task { await save(entity: choice.id) } } }
                    Button("New, with the same name") { Task { await save(entity: "new") } }
                }
                Button("I don't know") { Task { await save(response: "unknown") } }
                Button("Not important") { Task { await save(response: "dismiss") } }
            }
            if let error { Text(error).foregroundStyle(.red) }
        }
        .disabled(busy)
        .navigationTitle("Learning together")
    }

    private func save(response: String = "answer", entity: String? = nil) async {
        busy = true
        defer { busy = false }
        do {
            let reply = WorldReply(question: question, text: text, response: response, entityId: entity,
                                   expectedEvent: expected.isEmpty ? nil : expected)
            let result = try await client.answer(question, reply: reply)
            if result.needsClarification == true { choices = result.matches ?? []; return }
            done(); dismiss()
        } catch { self.error = "Saved for retry if disconnected. " + AppModel.describe(error) }
    }
}

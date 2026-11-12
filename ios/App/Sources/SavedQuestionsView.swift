import GroundedKit
import SwiftUI

/// Things put aside during capture, answered while the camera is off.
struct SavedQuestionsView: View {
    let client: GroundedClient
    @StateObject private var images = ImageCache()
    @State private var items: [SavedQuestion] = []
    @State private var worldItems: [WorldQuestion] = []
    @State private var queueIssues: [String] = []
    @State private var error: String?
    @State private var loading = true

    var body: some View {
        List {
            Section {
                FirstUseGuide(id: "saved", steps: ["Open a saved question and inspect its image or sound.", "Answer in your own words, or choose I don’t know or Not important.", "Return to Start looking whenever you want to observe again."], tip: "Saved requests needing attention are preserved on this phone. Read the reported issue before retrying; do not delete the queue to resolve an error.", compact: true)
            }
            if let error { Text(error).foregroundStyle(.red) }
            if !loading && items.isEmpty && worldItems.isEmpty {
                ContentUnavailableView(
                    "Nothing waiting", systemImage: "checkmark.circle",
                    description: Text("Questions you put aside while looking around appear here."))
            }
            if !queueIssues.isEmpty {
                Section("Saved requests needing attention") {
                    Text("\(queueIssues.count) requests were kept on this phone for review. \(queueIssues.first ?? "")")
                        .font(.footnote)
                }
            }
            ForEach(worldItems) { question in
                NavigationLink {
                    WorldSavedAnswerView(client: client, question: question) { Task { await reload() } }
                } label: { VStack(alignment: .leading) { Text(question.text); Text(question.reason).font(.caption) } }
            }
            ForEach(items) { item in
                NavigationLink {
                    AnswerSavedView(client: client, item: item, images: images) { Task { await reload() } }
                } label: {
                    HStack(spacing: 12) {
                        images.view(item.preview ?? item.region.asset, client: client)
                            .frame(width: 64, height: 64)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        VStack(alignment: .leading) {
                            Text(item.place).font(.headline)
                            Text(item.reason == "unanswered" ? "Not answered in time" : "Saved for later")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
                .swipeActions {
                    Button("Not important") {
                        Task {
                            try? await client.dismissSaved(item)
                            await reload()
                        }
                    }
                    .tint(.gray)
                }
            }
        }
        .navigationTitle("Answer later")
        .toolbar { HelpMenu(stage: "saved") }
        .refreshable { await reload() }
        .task { await reload() }
    }

    private func reload() async {
        do {
            try await client.retryPending()
            worldItems = try await client.world().questions.filter { $0.status == "open" || $0.status == "active" }
            queueIssues = (try? await client.outbox?.issues()) ?? []
            items = try await client.savedQuestions()
            error = nil
        } catch {
            self.error = AppModel.describe(error)
        }
        loading = false
    }
}

private struct AnswerSavedView: View {
    let client: GroundedClient
    let item: SavedQuestion
    @ObservedObject var images: ImageCache
    var done: () -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var answer = ""
    @State private var proposal: Interpretation?
    @State private var matches: [Entity] = []
    @State private var busy = false
    @State private var error: String?

    var body: some View {
        Form {
            Section {
                images.view(item.preview ?? item.region.asset, client: client)
                    .frame(maxWidth: .infinity, minHeight: 240, maxHeight: 360)
                Text("Seen in \(item.place). What is the outlined thing?").font(.subheadline)
            }
            if let proposal {
                Section("I'll remember") {
                    LabeledContent("Name") {
                        TextField("Name", text: Binding(get: { proposal.label }, set: { self.proposal?.label = $0 }))
                            .multilineTextAlignment(.trailing)
                    }
                    if !proposal.categories.isEmpty {
                        LabeledContent("Kind of", value: proposal.categories.joined(separator: ", "))
                    }
                    if matches.isEmpty {
                        Button("Remember") { Task { await save() } }
                        Button("Change answer") { self.proposal = nil }
                    }
                }
                if !matches.isEmpty {
                    Section("I already know a “\(proposal.label)”. Is this the same one?") {
                        ForEach(matches) { entity in
                            if let asset = entity.examples?.last?.asset {
                                images.view(asset, client: client).frame(height: 120)
                            }
                            Button("Same thing") { Task { await save(.same, entity) } }
                            Button("Another one of the same kind") { Task { await save(.sameKind, entity) } }
                            Button("Different, just the same name") { Task { await save(.unrelated, entity) } }
                        }
                    }
                }
            } else {
                Section("Your answer") {
                    TextField("That's my stapler…", text: $answer, axis: .vertical)
                    Button("Next") { Task { await interpret() } }.disabled(answer.isEmpty)
                }
                Section {
                    Button("Not important", role: .destructive) {
                        Task {
                            try? await client.dismissSaved(item)
                            finish()
                        }
                    }
                }
            }
            if let error { Section { Text(error).foregroundStyle(.red) } }
        }
        .disabled(busy)
        .overlay { if busy { ProgressView() } }
        .navigationTitle("What is this?")
    }

    private func interpret() async {
        busy = true
        defer { busy = false }
        let text = answer.trimmingCharacters(in: .whitespacesAndNewlines)
        proposal = (try? await client.interpret(text)) ?? Interpretation(label: text)
    }

    private func save(_ resolution: Resolution? = nil, _ previous: Entity? = nil) async {
        guard let proposal else { return }
        busy = true
        defer { busy = false }
        do {
            let result = try await client.teachSaved(
                item, text: answer.trimmingCharacters(in: .whitespacesAndNewlines), terms: proposal,
                resolution: resolution, previousId: previous?.id)
            if result.needsClarification == true {
                matches = result.matches ?? []
            } else {
                finish()
            }
        } catch {
            self.error = AppModel.describe(error)
        }
    }

    private func finish() {
        done()
        dismiss()
    }
}

/// Loads assets through the pinned client (AsyncImage cannot use the pinned certificate).
@MainActor
final class ImageCache: ObservableObject {
    @Published private var images: [String: UIImage] = [:]

    func view(_ name: String, client: GroundedClient) -> some View {
        Group {
            if let image = images[name] {
                Image(uiImage: image).resizable().scaledToFit()
            } else {
                ProgressView().task {
                    if let data = try? await client.asset(name), let image = UIImage(data: data) { self.images[name] = image }
                }
            }
        }
    }
}

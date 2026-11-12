import ARKit
import GroundedKit
import SceneKit
import SwiftUI

/// Full-screen camera with labels fixed to objects and one question card at a time.
struct CaptureView: View {
    @StateObject var model: CaptureModel
    var onClose: () -> Void

    var body: some View {
        ZStack {
            CameraView(session: model.engine.session)
                .ignoresSafeArea()
                .gesture(SpatialTapGesture().onEnded { tap in Task { await model.tap(tap.location) } })
            GeometryReader { geometry in
                ZStack {
                    ForEach(model.markers) { MarkerLabel(marker: $0).position($0.point) }
                }
                .onAppear { model.engine.setViewport(geometry.size) }
                .onChange(of: geometry.size) { _, size in model.engine.setViewport(size) }
            }
            .ignoresSafeArea()
            .allowsHitTesting(false)
            VStack(spacing: 10) {
                topBar
                FirstUseGuide(id: "capture", title: "Looking and learning", steps: ["Show a scene; an unknown place is fine.", "Tap an object to inspect it. Answer in your own words, or choose Later.", "Tap Stop and wait for saving to finish.", "Check the completion message for the next step on your phone or Mac."], tip: "Later keeps a question for Answer later. Stop finishes this session and saves the recording.", compact: true)
                Spacer()
                if let notice = model.notice {
                    HStack {
                        Text(notice).font(.footnote.bold())
                        Button("Dismiss") { model.notice = nil }.font(.caption)
                    }
                    .padding(10)
                    .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12))
                }
                relationStrip
                card
                    .padding()
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 22))
                    .disabled(model.busy)
                    .animation(.snappy, value: model.state?.question?.id)
            }
            .padding(.horizontal, 12)
            .padding(.bottom, 8)
        }
        .task { await model.start() }
        .fullScreenCover(isPresented: $model.fixing) { OutlineEditor(model: model) }
    }

    /// Relations measured from pose and depth, offered for confirmation.
    @ViewBuilder private var relationStrip: some View {
        let open = (model.state?.relations ?? []).filter { !model.confirmed.contains($0.id) }.prefix(2)
        if case .running = model.phase, !open.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                Text("I measured").font(.caption).foregroundStyle(.secondary)
                ForEach(Array(open)) { relation in
                    HStack {
                        Image(systemName: "ruler")
                        Text(relation.sentence).font(.subheadline)
                        Spacer()
                        Button("Confirm") { Task { await model.confirm(relation) } }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                    }
                }
            }
            .padding(10)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
        }
    }

    // MARK: top

    private var topBar: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Label(model.world?.location.label ?? model.state?.context?.label ?? "Unknown place", systemImage: "mappin.and.ellipse")
                    .font(.subheadline.bold())
                Spacer()
                HelpMenu(stage: "capture").labelStyle(.iconOnly)
                Toggle(isOn: $model.speak) { Image(systemName: model.speak ? "speaker.wave.2" : "speaker.slash") }
                    .toggleStyle(.button)
                    .accessibilityLabel("Speak questions")
                if case .running = model.phase {
                    Button(role: .destructive) { Task { await model.stop() } } label: {
                        Label("Stop", systemImage: "stop.circle.fill")
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
            HStack(spacing: 8) {
                Circle().fill(model.error == nil ? .green : .red).frame(width: 8, height: 8)
                Text(status).font(.caption).lineLimit(2)
                if let certainty = model.world?.location.certainty { Text(certainty).font(.caption) }
            }
            if let request = model.world?.executive?.request {
                Label(request, systemImage: "scope").font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    private var status: String {
        if let error = model.error { return error }
        if let tracking = model.tracking { return tracking }
        if let note = model.recordingNote, model.analysed % 20 < 6 { return note }
        guard model.analysed > 0 else { return "Connecting to your Mac…" }
        let ms = Int((model.latency ?? 0) * 1000)
        return "Recording · \(model.analysed) frames understood · \(ms) ms"
    }

    // MARK: cards

    @ViewBuilder private var card: some View {
        switch model.phase {
        case .starting: Label("Starting camera…", systemImage: "camera")
        case .stopping: ProgressView("Finishing recording…")
        case .uploading(let progress): ProgressView("Saving recording to your Mac", value: progress)
        case .finished(let message, let completion):
            VStack(spacing: 12) {
                Text(message).font(.headline)
                Text(completion.nextStep).font(.footnote)
                Button("Done", action: onClose).buttonStyle(.borderedProminent)
            }
        case .failed(let message):
            VStack(spacing: 12) {
                Text(message)
                Text("Tap Done to check the Mac connection. Any saved recording remains available under Upload now.").font(.footnote)
                Button("Done", action: onClose).buttonStyle(.borderedProminent)
            }
        case .running:
            if let clarification = model.clarification {
                clarificationCard(clarification)
            } else if let question = model.worldQuestion {
                worldQuestionCard(question)
            } else if model.world == nil && model.state?.context == nil {
                placeCard
            } else if let question = model.state?.question {
                questionCard(question)
            } else {
                idleCard
            }
        }
    }

    private func worldQuestionCard(_ question: WorldQuestion) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(question.reason).font(.caption).foregroundStyle(.secondary)
            Text(question.text).font(.title3.bold())
            if let region = question.region {
                HStack { crop(region.asset)
                    if model.state?.question?.id == question.id { Button("Fix outline") { model.fixing = true } }
                }
            }
            else if let image = question.evidenceRefs.first(where: { $0.hasSuffix(".jpg") }) { crop(image) }
            if let sound = question.evidenceRefs.first(where: { $0.hasSuffix(".wav") }) {
                Button("Listen to the sound", systemImage: "waveform") { Task { await model.playWorldSound(sound) } }
                Text("Where it was heard does not establish where it came from.").font(.caption2)
            }
            TextField("Your answer", text: $model.answer, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .onChange(of: model.answer) { _, _ in Task { await model.touchWorld() } }
            if ["explain", "clarify", "predict"].contains(question.kind) {
                TextField("What should happen next? (optional, within 30 seconds)", text: $model.expectedEvent)
                    .textFieldStyle(.roundedBorder)
            }
            if !model.worldChoices.isEmpty {
                ForEach(model.worldChoices) { choice in
                    Button("Same \(choice.label)") { Task { await model.answerWorld(entityId: choice.id) } }
                }
                Button("New, with the same name") { Task { await model.answerWorld(entityId: "new") } }
            }
            HStack {
                if question.kind == "intervene" {
                    Button("I'll do it now") { Task { await model.answerWorld("yes") } }
                    Button("Not now") { Task { await model.answerWorld("no") } }
                }
                if question.proposition.entityId != nil {
                    Button("Yes") { Task { await model.answerWorld("yes") } }
                    Button("No") { Task { await model.answerWorld("no") } }
                }
                Button("Remember") { Task { await model.answerWorld() } }.disabled(model.answer.isEmpty)
                Button("Later") { Task { await model.answerWorld("later") } }
                Button("Don't know") { Task { await model.answerWorld("unknown") } }
            }.buttonStyle(.bordered)
            HStack {
                Button("Not important") { Task { await model.answerWorld("dismiss") } }
                Menu("Saved questions") {
                    ForEach(model.world?.questions.filter { $0.status == "open" } ?? []) { saved in
                        Button(saved.text) { model.selectedWorldQuestion = saved }
                    }
                }
            }.font(.footnote)
        }
    }

    @ViewBuilder private var placeCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(model.placePrompt).font(.title2.bold())
            if let guess = model.state?.placeGuess, !model.guessRejected {
                Text("It looks like a place you showed me before (\(Int(guess.appearance * 100))% similar).")
                    .font(.footnote).foregroundStyle(.secondary)
                HStack {
                    Button("Yes") { Task { await model.confirmGuess() } }.buttonStyle(.borderedProminent)
                    Button("No, somewhere else") { model.guessRejected = true }.buttonStyle(.bordered)
                }
            } else {
                TextField("Kitchen, office, garage…", text: $model.placeText)
                    .textFieldStyle(.roundedBorder)
                    .submitLabel(.done)
                    .onSubmit { Task { await model.setPlace(model.placeText) } }
                HStack {
                    Button("Set place") { Task { await model.setPlace(model.placeText) } }
                        .buttonStyle(.borderedProminent)
                        .disabled(model.placeText.isEmpty)
                    Button("I don't know") { Task { await model.setPlace("Unknown place") } }.buttonStyle(.bordered)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func questionCard(_ question: Question) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 12) {
                Button { model.fixing = true } label: { crop(question.preview ?? question.region.asset) }
                    .buttonStyle(.plain)
                VStack(alignment: .leading, spacing: 4) {
                    Text("What is this?").font(.title2.bold())
                    Button("Fix outline", systemImage: "scribble") { model.fixing = true }
                        .font(.footnote)
                    Text("The yellow ? marks it. Type a short answer, or tap another object instead.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
            }
            if model.proposal == nil, let guess = question.region.best, guess.appearance >= 0.75 {
                Button("Yes, it's the \(guess.label) (\(Int(guess.appearance * 100))% alike)") {
                    model.acceptSuggestion(guess.label)
                }
                .buttonStyle(.bordered)
            }
            if let proposal = model.proposal {
                proposalEditor(proposal)
            } else {
                TextField("That's my Epson printer…", text: $model.answer, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .submitLabel(.done)
                    .onSubmit { Task { await model.interpret() } }
                HStack {
                    Button("Next") { Task { await model.interpret() } }
                        .buttonStyle(.borderedProminent)
                        .disabled(model.answer.isEmpty)
                    Button("Not important") {
                        Task {
                            await model.control("skip")
                            model.show("Noted as not important")
                        }
                    }
                    .buttonStyle(.bordered)
                    Button("Later") {
                        Task {
                            await model.control("later")
                            model.show("Saved for later — answer it from the home screen")
                        }
                    }
                    .buttonStyle(.bordered)
                }
            }
        }
    }

    private func proposalEditor(_ proposal: Interpretation) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("“\(model.answer)”").font(.footnote).foregroundStyle(.secondary)
            LabeledContent("Name") {
                TextField("Name", text: binding(\.label)).multilineTextAlignment(.trailing)
            }
            if !proposal.categories.isEmpty {
                LabeledContent("Kind of", value: proposal.categories.joined(separator: ", "))
            }
            if !proposal.aliases.isEmpty {
                LabeledContent("Also called", value: proposal.aliases.joined(separator: ", "))
            }
            HStack {
                Button("Remember") { Task { await model.teach() } }
                    .buttonStyle(.borderedProminent)
                    .disabled(proposal.label.isEmpty)
                Button("Change answer") { model.proposal = nil }.buttonStyle(.bordered)
            }
            Text("Names must use your own words.").font(.caption2).foregroundStyle(.secondary)
        }
    }

    private func clarificationCard(_ clarification: CaptureModel.Clarification) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("I already know a “\(clarification.terms.label)”.").font(.title3.bold())
            ForEach(clarification.matches) { entity in
                HStack(spacing: 12) {
                    if let asset = entity.examples?.last?.asset { crop(asset) }
                    VStack(alignment: .leading, spacing: 6) {
                        Text(clarification.isPlace ? "Is this the same place?" : "Is this the same one?")
                        Button(clarification.isPlace ? "Same place" : "Same thing") {
                            Task { await model.resolve(.same, with: entity) }
                        }
                        .buttonStyle(.borderedProminent)
                        if !clarification.isPlace {
                            Button("Another one of the same kind") { Task { await model.resolve(.sameKind, with: entity) } }
                        }
                        Button("Different, just the same name") { Task { await model.resolve(.unrelated, with: entity) } }
                    }
                    .buttonStyle(.bordered)
                }
            }
            Button("Cancel") { model.clarification = nil }
        }
    }

    private var idleCard: some View {
        HStack {
            Image(systemName: "eye")
            Text("Observing and learning. Tap a thing to inspect it, or review saved questions.")
            Spacer()
            Button(model.state?.paused == true ? "Resume" : "Pause") {
                Task { await model.control(model.state?.paused == true ? "resume" : "pause") }
            }
        }
        .font(.subheadline)
    }

    // MARK: pieces

    private func crop(_ asset: String) -> some View {
        Group {
            if let image = model.crops[asset] {
                Image(uiImage: image).resizable().scaledToFit()
            } else {
                ProgressView().onAppear { model.load(asset) }
            }
        }
        .frame(width: 84, height: 84)
        .background(Color.black.opacity(0.25), in: RoundedRectangle(cornerRadius: 10))
    }

    private func binding(_ path: WritableKeyPath<Interpretation, String>) -> Binding<String> {
        Binding(
            get: { model.proposal?[keyPath: path] ?? "" },
            set: { model.proposal?[keyPath: path] = $0 })
    }
}

private struct MarkerLabel: View {
    var marker: ProjectedMarker

    var body: some View {
        if marker.isQuestion {
            Text("?")
                .font(.title.bold())
                .frame(width: 44, height: 44)
                .background(.yellow, in: Circle())
                .overlay(Circle().stroke(.white, lineWidth: 2))
                .shadow(radius: 4)
        } else {
            HStack(spacing: 4) {
                Text(marker.text).bold()
                Text(marker.detail).foregroundStyle(.secondary)
            }
            .font(.footnote)
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(.thinMaterial, in: Capsule())
            .shadow(radius: 3)
        }
    }
}

/// ARKit's camera feed, sized to fill the screen; labels use the same viewport for projection.
private struct CameraView: UIViewRepresentable {
    let session: ARSession

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView(frame: .zero)
        view.session = session
        view.automaticallyUpdatesLighting = false
        return view
    }

    func updateUIView(_ view: ARSCNView, context: Context) {}
}

/// Correct what the question covers, on the frozen frame: tap to include or exclude, or step the size.
private struct OutlineEditor: View {
    @ObservedObject var model: CaptureModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            VStack(spacing: 14) {
                if let question = model.state?.question, let name = question.preview, let image = model.crops[name] {
                    GeometryReader { geometry in
                        let shown = fitted(image.size, in: geometry.size)
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .frame(width: geometry.size.width, height: geometry.size.height)
                            .contentShape(Rectangle())
                            .gesture(
                                SpatialTapGesture().onEnded { tap in
                                    let x = (tap.location.x - shown.minX) / shown.width
                                    let y = (tap.location.y - shown.minY) / shown.height
                                    guard (0...1).contains(x), (0...1).contains(y) else { return }
                                    Task { await model.refine(point: SIMD2(x, y)) }
                                })
                    }
                    controls(question.region.prompt)
                } else {
                    ProgressView("Loading the frozen frame…").frame(maxHeight: .infinity)
                }
            }
            .padding()
            .disabled(model.busy)
            .overlay { if model.busy { ProgressView().controlSize(.large) } }
            .navigationTitle("Fix outline")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }
    }

    private func controls(_ prompt: Prompt?) -> some View {
        VStack(spacing: 10) {
            Picker("Taps", selection: $model.includeTaps) {
                Text("Tap adds").tag(true)
                Text("Tap removes").tag(false)
            }
            .pickerStyle(.segmented)
            HStack {
                Button("Smaller", systemImage: "minus.magnifyingglass") { Task { await model.refine(granularity: "smaller") } }
                Button("Larger", systemImage: "plus.magnifyingglass") { Task { await model.refine(granularity: "larger") } }
                Button("Start over", systemImage: "arrow.counterclockwise") { Task { await model.refine(reset: true) } }
            }
            .buttonStyle(.bordered)
            Text("The yellow line is what I'll remember. **Tap the photo** on parts that belong, or switch to removing and tap what doesn't.")
                .font(.footnote)
                .foregroundStyle(.secondary)
            if let error = model.error { Text(error).font(.footnote).foregroundStyle(.red) }
        }
    }

    private func fitted(_ size: CGSize, in container: CGSize) -> CGRect {
        let scale = min(container.width / size.width, container.height / size.height)
        let shown = CGSize(width: size.width * scale, height: size.height * scale)
        return CGRect(
            x: (container.width - shown.width) / 2, y: (container.height - shown.height) / 2, width: shown.width,
            height: shown.height)
    }
}

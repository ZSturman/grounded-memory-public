import GroundedKit
import SwiftUI

@main
struct GroundedMemoryApp: App {
    @StateObject private var app = AppModel()

    var body: some Scene {
        WindowGroup {
            RootView().environmentObject(app)
        }
    }
}

private struct RootView: View {
    @EnvironmentObject var app: AppModel
    @State private var capturing: CaptureModel?
    @State private var pending = Recorder.pending()
    @State private var uploading: String?
    @State private var uploadNotice: String?

    var body: some View {
        if let capturing {
            CaptureView(model: capturing) { self.capturing = nil }
        } else if app.step == .ready, let client = app.client {
            home(client)
        } else {
            SetupView()
        }
    }

    private func home(_ client: GroundedClient) -> some View {
        NavigationStack {
            homeContent(client)
        }
    }

    private func homeContent(_ client: GroundedClient) -> some View {
        ScrollView {
        VStack(spacing: 20) {
            Image(systemName: "eye.circle").font(.system(size: 64))
            Text("Grounded Memory").font(.largeTitle.bold())
            Text("Show me around. I'll ask what things are and remember what you teach me, on your Mac.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
            FirstUseGuide(id: "home", steps: ["Tap Start looking and allow camera and microphone access. Keep the app in the foreground.", "Show a simple scene. Tap an object or answer a question. The place can remain unknown.", "Tap Stop and wait for saving. If a recording stays on this phone, use Upload now.", "On your Mac, inspect beliefs in Capture & world learning. For reviewed memories, open Collection, collect saved inputs, then open Labeling."], tip: "Live world learning happens as you observe. Reviewing and testing saved memories is a separate workflow on your Mac.")
            if let error = app.error {
                VStack(alignment: .leading, spacing: 8) {
                    Label(error, systemImage: "exclamationmark.triangle").foregroundStyle(.red)
                    Text("Keep the recording on this phone. Check the Mac connection and use Upload now again when available.").font(.footnote)
                }.accessibilityElement(children: .combine)
            }
            if let uploadNotice {
                VStack(spacing: 8) {
                    Text(uploadNotice).font(.footnote)
                    Button("Dismiss confirmation") { self.uploadNotice = nil }.font(.caption)
                }
            }
            if !pending.isEmpty {
                VStack(spacing: 8) {
                    Text("\(pending.count) recording\(pending.count == 1 ? "" : "s") not yet saved to your Mac")
                        .font(.subheadline.bold())
                    Button(uploading ?? "Upload now") { Task { await upload(client) } }
                        .buttonStyle(.bordered)
                        .disabled(uploading != nil)
                }
                .padding()
                .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 14))
            }
            Button {
                capturing = CaptureModel(client: client)
            } label: {
                Label("Start looking", systemImage: "camera.viewfinder").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            Text("Later saves a question without stopping observation. Answer later also shows saved requests needing attention.")
                .font(.footnote).foregroundStyle(.secondary)
            NavigationLink {
                SavedQuestionsView(client: client)
            } label: {
                Label("Answer later", systemImage: "tray.full").frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .controlSize(.large)
            Text("Connected to \(client.baseURL.host() ?? "your Mac")").font(.footnote).foregroundStyle(.secondary)
            Button("Forget this Mac", role: .destructive) { app.forget() }.font(.footnote)
        }
        .padding(24)
        }
        .toolbar { HelpMenu(stage: "home") }
        .onAppear { pending = Recorder.pending() }
    }

    private func upload(_ client: GroundedClient) async {
        app.error = nil
        uploadNotice = nil
        for item in pending {
            uploading = "Uploading…"
            do {
                try await client.upload(session: item.session, file: item.url) { progress in
                    Task { @MainActor in uploading = "Uploading \(Int(progress * 100))%" }
                }
                Recorder.remove(item.url)
            } catch {
                uploading = nil
                app.error = AppModel.describe(error)
                break
            }
        }
        uploading = nil
        pending = Recorder.pending()
        if pending.isEmpty && app.error == nil {
            uploadNotice = "Recordings saved on your Mac. Next: on the Mac, open Collection → Collect saved inputs now → Labeling."
        }
    }
}

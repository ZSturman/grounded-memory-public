import SwiftUI

/// Device-local presentation preferences, separate from pairing, evidence and the outbox.
struct FirstUseGuide: View {
    let title: String
    let steps: [String]
    let tip: String
    @AppStorage private var hideSteps: Bool
    @AppStorage private var hideTip: Bool

    init(id: String, title: String = "Your next steps", steps: [String], tip: String, compact: Bool = false) {
        self.title = title
        self.steps = steps
        self.tip = tip
        _hideSteps = AppStorage(wrappedValue: compact, "gm.guidance.v1.\(id).steps")
        _hideTip = AppStorage(wrappedValue: false, "gm.guidance.v1.\(id).tip")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(title).font(.subheadline.bold())
                Spacer()
                Button(hideSteps ? "Help & steps" : "Hide steps") { hideSteps.toggle() }
                    .font(.caption)
                    .accessibilityLabel(hideSteps ? "Show step-by-step help" : "Hide step-by-step help")
            }
            if !hideSteps {
                ScrollView {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(Array(steps.enumerated()), id: \.offset) { index, step in
                            Text("\(index + 1). \(step)").font(.footnote)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }.frame(maxWidth: .infinity, alignment: .leading)
                }.frame(maxHeight: 180)
            }
            if !hideTip {
                Text(tip).font(.footnote).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                Button("Dismiss tip") { hideTip = true }.font(.caption)
            }
        }
        .padding(12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
    }
}

struct HelpMenu: View {
    let stage: String
    var body: some View {
        Menu {
            Button("Help & steps") { UserDefaults.standard.set(false, forKey: "gm.guidance.v1.\(stage).steps") }
            Button("Show tips again") {
                for id in ["setup", "home", "capture", "saved"] {
                    UserDefaults.standard.set(false, forKey: "gm.guidance.v1.\(id).steps")
                    UserDefaults.standard.set(false, forKey: "gm.guidance.v1.\(id).tip")
                }
            }
        } label: { Label("Help & steps", systemImage: "questionmark.circle") }
        .accessibilityLabel("Help and tips")
    }
}

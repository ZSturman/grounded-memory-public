import SwiftUI

/// Connect to the Mac: find it, compare the certificate fingerprint, enter the pairing code.
struct SetupView: View {
    @EnvironmentObject var app: AppModel
    @StateObject private var discovery = Discovery()
    @State private var code = ""

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Text(progress).font(.subheadline.bold())
                    Text(nextStep).font(.footnote).foregroundStyle(.secondary)
                    FirstUseGuide(id: "setup", steps: ["Keep make phone running on your Mac, with both devices on the same Wi-Fi.", "Choose your Mac and compare the certificate check with its terminal.", "Enter the pairing code. Next you’ll see Start looking and Answer later."], tip: "Pairing connects this phone to your Mac. The web browser pairs separately; both use the same saved evidence.", compact: true)
                }
                switch app.step {
                case .findMac, .ready: findMac
                case .compareCertificate(let fingerprint): compare(fingerprint)
                case .enterCode: enterCode
                }
                if let error = app.error {
                    Section { Label(error, systemImage: "exclamationmark.triangle").foregroundStyle(.red) }
                }
                troubleshooting
            }
            .navigationTitle("Connect to your Mac")
            .toolbar { HelpMenu(stage: "setup") }
            .disabled(app.busy)
            .overlay { if app.busy { ProgressView() } }
        }
        .onAppear { discovery.start() }
        .onDisappear { discovery.stop() }
    }

    private var progress: String {
        switch app.step {
        case .findMac, .ready: return "Step 1 of 3 · Find your Mac"
        case .compareCertificate: return "Step 2 of 3 · Check it is your Mac"
        case .enterCode: return "Step 3 of 3 · Pair this iPhone"
        }
    }

    private var nextStep: String {
        switch app.step {
        case .findMac, .ready: return "Choose the Mac running Grounded Memory. Next, compare its certificate check."
        case .compareCertificate: return "Compare with the Mac’s terminal before trusting it. Next, enter the pairing code."
        case .enterCode: return "Use the code from the Mac’s terminal. After pairing, you can start a learning session."
        }
    }

    private var findMac: some View {
        Section {
            Text("On your Mac, run **make phone** in the Grounded Memory folder. It prints an address, a pairing code and a certificate check.")
            if discovery.macs.isEmpty {
                Label("Looking for Macs on this Wi-Fi…", systemImage: "wifi")
                    .foregroundStyle(.secondary)
            }
            ForEach(discovery.macs) { mac in
                Button {
                    app.serverText = mac.url
                    Task { await app.connect() }
                } label: {
                    Label("\(mac.name) — \(mac.url)", systemImage: "desktopcomputer")
                }
            }
            TextField("Or type the Phone address, e.g. 192.0.2.10:3443", text: $app.serverText)
                .keyboardType(.URL)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
            Button("Connect") { Task { await app.connect() } }
                .disabled(app.serverText.isEmpty)
        } header: {
            Text("1 · Find your Mac")
        }
    }

    private func compare(_ fingerprint: String) -> some View {
        Section {
            Text("Check that your Mac's terminal shows this same certificate check:")
            Text(fingerprint)
                .font(.system(.title2, design: .monospaced).bold())
                .frame(maxWidth: .infinity)
            Button("It matches — trust this Mac") { app.trustCertificate() }
            Button("It doesn't match", role: .destructive) { app.forget() }
        } header: {
            Text("2 · Confirm it's your Mac")
        } footer: {
            Text("This keeps the connection private without installing a profile on your iPhone.")
        }
    }

    private var enterCode: some View {
        Section {
            TextField("Six-digit pairing code", text: $code)
                .keyboardType(.numberPad)
                .font(.title2.monospacedDigit())
            Button("Pair this iPhone") { Task { await app.pair(code: code) } }
                .disabled(code.count < 6)
            Button("Forget this Mac", role: .destructive) { app.forget() }
        } header: {
            Text("3 · Enter the pairing code")
        } footer: {
            Text("The code is accepted for 10 minutes after make phone starts. If it expired, restart make phone and use the new code.")
        }
    }

    private var troubleshooting: some View {
        Section("Troubleshooting") {
            Text("• Mac not listed: make sure make phone is running and both devices are on the same Wi-Fi (not a guest network).")
            Text("• Allow Local Network access: Settings › Privacy & Security › Local Network › Grounded Memory.")
            Text("• Mac's address changed: restart make phone and connect again.")
        }
        .font(.footnote)
        .foregroundStyle(.secondary)
    }
}

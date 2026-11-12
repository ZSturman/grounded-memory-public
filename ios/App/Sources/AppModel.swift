import Foundation
import GroundedKit
import Network
import SwiftUI

/// Which Mac this phone talks to, and whether it is trusted and paired.
@MainActor
final class AppModel: ObservableObject {
    enum Step: Equatable { case findMac, compareCertificate(String), enterCode, ready }

    @Published var step: Step = .findMac
    @Published var serverText: String
    @Published var error: String?
    @Published var busy = false
    @Published private(set) var client: GroundedClient?

    private var pendingAuthority: Data?
    private let defaults = UserDefaults.standard

    init() {
        serverText = defaults.string(forKey: "server") ?? ""
        if let url = URL(string: serverText), let authority = defaults.data(forKey: "authority") {
            client = GroundedClient(baseURL: url, authority: authority)
            step = .enterCode
            Task { await checkPairing() }
        }
    }

    var server: URL? {
        let text = serverText.trimmingCharacters(in: .whitespaces)
        return URL(string: text.contains("://") ? text : "https://" + text)
    }

    /// Step 1: download the Mac's public root so the person can compare its fingerprint.
    func connect() async {
        guard let url = server else { return fail("Enter the Phone address printed by make phone.") }
        await run {
            let authority = try await GroundedClient.fetchAuthority(from: url)
            self.pendingAuthority = authority
            self.serverText = url.absoluteString
            self.step = .compareCertificate(Authority.fingerprint(authority))
        }
    }

    /// Step 2: the fingerprint matched the Mac's terminal; pin it from now on.
    func trustCertificate() {
        guard let url = server, let authority = pendingAuthority else { return }
        defaults.set(url.absoluteString, forKey: "server")
        defaults.set(authority, forKey: "authority")
        client = GroundedClient(baseURL: url, authority: authority)
        step = .enterCode
    }

    /// Step 3: the six-digit code, accepted for ten minutes after the Mac starts.
    func pair(code: String) async {
        guard let client else { return }
        await run {
            try await client.pair(code: code.trimmingCharacters(in: .whitespaces))
            self.step = .ready
        }
    }

    func checkPairing() async {
        guard let client else { return }
        do { if try await client.isPaired() { step = .ready } } catch { self.error = Self.describe(error) }
    }

    func forget() {
        defaults.removeObject(forKey: "authority")
        HTTPCookieStorage.shared.cookies?.forEach(HTTPCookieStorage.shared.deleteCookie)
        client = nil
        pendingAuthority = nil
        step = .findMac
    }

    func run(_ work: @escaping () async throws -> Void) async {
        busy = true
        error = nil
        defer { busy = false }
        do { try await work() } catch { self.error = Self.describe(error) }
    }

    private func fail(_ message: String) { error = message }

    static func describe(_ error: Error) -> String {
        if let error = error as? URLError {
            switch error.code {
            case .cannotConnectToHost, .timedOut, .cannotFindHost, .networkConnectionLost:
                return "Can't reach the Mac. Check that make phone is running, both devices share Wi-Fi, and Local Network access is allowed in Settings › Grounded Memory."
            case .cancelled, .serverCertificateUntrusted, .secureConnectionFailed:
                return "The Mac's certificate changed or doesn't match. Use Forget this Mac and connect again."
            default: break
            }
        }
        return error.localizedDescription
    }
}

/// Finds Macs advertised by `make phone` (`_groundedmemory._tcp`, TXT `url=`).
@MainActor
final class Discovery: ObservableObject {
    struct Mac: Identifiable, Hashable { var id: String { url }; var name: String; var url: String }

    @Published var macs: [Mac] = []
    private var browser: NWBrowser?

    func start() {
        guard browser == nil else { return }
        let browser = NWBrowser(for: .bonjourWithTXTRecord(type: "_groundedmemory._tcp", domain: nil), using: .tcp)
        browser.browseResultsChangedHandler = { [weak self] results, _ in
            let macs = results.compactMap { result -> Mac? in
                guard case .bonjour(let txt) = result.metadata, let url = txt["url"],
                    case .service(let name, _, _, _) = result.endpoint
                else { return nil }
                return Mac(name: name, url: url)
            }
            Task { @MainActor in self?.macs = macs }
        }
        browser.start(queue: .main)
        self.browser = browser
    }

    func stop() {
        browser?.cancel()
        browser = nil
    }
}

import CryptoKit
import Foundation
import Security

/// The Mac's local HTTPS proxy uses Caddy's private root. The app pins that root after the person
/// compares its fingerprint with the one `make phone` prints, so no iOS profile install is needed.
public enum Authority {
    public static func der(fromPEM pem: String) -> Data? {
        let body = pem.split(whereSeparator: \.isNewline)
            .filter { !$0.hasPrefix("-----") }
            .joined()
        return Data(base64Encoded: body)
    }

    /// Same short form as scripts/manage.py: first 8 and last 6 hex digits of the SHA-256.
    public static func fingerprint(_ der: Data) -> String {
        let hex = SHA256.hash(data: der).map { String(format: "%02X", $0) }.joined()
        return hex.prefix(8) + "…" + hex.suffix(6)
    }
}

final class TrustDelegate: NSObject, URLSessionDelegate {
    /// nil only while downloading the public root itself (trust on first use, confirmed by the person).
    let anchor: SecCertificate?

    init(anchor: Data?) {
        self.anchor = anchor.flatMap { SecCertificateCreateWithData(nil, $0 as CFData) }
    }

    func urlSession(
        _ session: URLSession, didReceive challenge: URLAuthenticationChallenge,
        completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void
    ) {
        guard challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust,
            let trust = challenge.protectionSpace.serverTrust
        else { return completionHandler(.performDefaultHandling, nil) }
        if let anchor {
            SecTrustSetAnchorCertificates(trust, [anchor] as CFArray)
            SecTrustSetAnchorCertificatesOnly(trust, true)
            guard SecTrustEvaluateWithError(trust, nil) else { return completionHandler(.cancelAuthenticationChallenge, nil) }
        }
        completionHandler(.useCredential, URLCredential(trust: trust))
    }
}

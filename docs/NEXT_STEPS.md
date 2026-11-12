# Validation and next steps

Snapshot prepared November 12, 2026.

## Current state

Research implementation. Supplied perception adapters target Apple Silicon/macOS; the project is not a general-purpose trained robot mind. Physical iPhone acceptance and recognition calibration remain outstanding. Model provisioning downloads optional weights.

## Validation

PASS: 34 world-model and API tests. Model loading and physical sensing were not exercised.

Publication checks: the exact Git tree and its new history are scanned locally with Gitleaks before the repository becomes public. Reviewed scanner exceptions are restricted to synthetic fixture secrets, an invalid test token, a browser preference name and the published NIST AES test vector. Private histories are not included.

## Next action

Follow the README quick start, run the documented focused tests, then use isolated inputs for a manual walkthrough. Real hardware, release signing, background behavior and live-server compatibility are outside this publication check.

# Connect a phone camera

## iPhone app (recommended)

The native app in `ios/` uses ARKit for a full-screen camera, optional depth, independent pose, live environmental audio, and questions about places, sounds, things and events. It finds the Mac through Bonjour and pins the Mac's local certificate, so no profile has to be installed.

The setup screen shows **Step 1 of 3 → Step 2 of 3 → Step 3 of 3** and explains what happens next. After pairing, the home screen has a first-session checklist. **Help & steps** reopens instructions and **Show tips again** restores dismissed help on this phone; these settings do not affect pairing, recordings, saved replies, or browser preferences. Capture help stays inline and does not open a walkthrough over active questions.

1. On the Mac run `make phone`. Note the pairing code and the **iPhone app certificate check** (for example `B5217FC7…1FC64C`).
2. Build and install from the project root:
   ```sh
   cd ios && xcodegen generate
   xcodebuild -project GroundedMemory.xcodeproj -scheme GroundedMemory -destination 'id=<iPhone UDID>' \
     -derivedDataPath build/DerivedData-ios -allowProvisioningUpdates build
   xcrun devicectl device install app --device <iPhone UDID> "build/DerivedData-ios/Build/Products/Debug-iphoneos/Grounded Memory.app"
   ```
   `xcrun devicectl list devices` shows the UDID. Free signing expires after seven days; reinstall then.
3. In the app, allow Local Network access, choose the Mac, check that the certificate check matches the terminal, and enter the pairing code (accepted for ten minutes after startup). Pairing survives backend restarts.
4. **Start looking.** The learner can begin with an unknown place. Answer or defer its questions, confirm a place candidate, or **tap an object** to direct attention. **Stop** uploads the original recording; the app also saves samples and replies before transmission so interruptions can be retried.

The app sends vision samples to `/api/v2/episodes/{id}/samples`, including a stable observation ID, clock ID, tracking-frame ID, upright JPEG, sensor descriptor, pose even without depth, and optional LiDAR. It separately converts microphone audio locally to mono 16-bit PCM at 16 kHz and sends approximately two-second WAV chunks. Audio is environmental evidence; it is not continuously transcribed into statements.

The top bar shows the estimated place and its uncertainty. Question cards can contain an image or sound, short replies, yes/no confirmation, explanations, and an optional next event expected within 30 seconds. Use **Later**, **Don't know**, or **Not important** without stopping observation. The server limits proactive interruptions and saves untouched questions after 20 seconds. **Answer later** on the home screen includes v2 questions and historical visual questions.

After **Stop**, wait for the completion message:

- **Saved on your Mac:** inspect beliefs in the Mac's **Capture & world learning** console. For reviewed memories, open **Collection → Collect saved inputs now → Labeling** on the Mac.
- **Still on this phone:** tap **Done**, check that the Mac is reachable, and use **Upload now** from the home screen. Upload errors stay visible there, with the recording preserved for another attempt.
- **No original recording:** inspect any saved observations/questions and check camera/microphone access before another session. This message does not claim that a recording reached the Mac.

Capture confirmations remain until dismissed or replaced. **Answer later** also identifies saved requests needing attention. Live learning happens while observing; Collection imports saved evidence for a separate human-review workflow. Neither upload nor answering a question approves reviewed recall.

Tap an object to direct attention; **Fix outline** retains the existing include/exclude and Smaller/Larger controls. Correcting an outline changes the proposition version, so an old queued reply cannot teach a different outline. Spoken questions and evidence playback mark their own audio as system output.

Original recordings use five-second fragments. Vision/audio/reply requests are stored in a server-scoped disk queue before transmission and drained serially. A failed network request remains queued. Requests rejected as invalid or needing clarification are preserved under `Application Support/GroundedMemory/Outbox/.../NeedsAttention`; Answer later reports these issues. Do not delete the queue while diagnosing a missing answer. Background capture remains subject to iOS restrictions: keep the app in the foreground, and expect new recording segments after interruptions.

The backend must be restarted and this app rebuilt to activate the world-model protocol. Generic-device compilation and GroundedKit tests pass. Physical audio timing, spoken-output exclusion, room recognition, outline correction and reconnect acceptance remain to be performed using [NEXT_STEPS](NEXT_STEPS.md).

## Connected-device verification — November 11, 2026

Signed build **0.2.0 (4)** is installed on the physical iPhone 16 Pro Max. Pairing, home help collapse/tip dismissal/restoration, Answer later navigation, recovery of both pending uploads, and start/stop/saved completion were verified. A short session delivered 26 vision samples and 11 audio chunks without HTTP errors; the retained original contains 18.149 seconds of 1920×1440 video and audio. Originals remain in normal local storage.

iPhone Mirroring showed a camera-preview restriction and a black preview. Direct phone inspection of visual framing, recognition, teaching, spoken-output exclusion, accessibility, and deliberate live reconnect behavior still needs acceptance. File delivery and passing software tests do not establish recognition accuracy. See [the current acceptance record](NEXT_STEPS.md#connected-iphone-update--2026-11-11).

## Safari (fallback)

Use a disposable test scene first. The phone and Mac must share a network without client isolation.

1. From the project root, run `make doctor`. Caddy and FFmpeg should be present.
2. Run `make phone`. It prints an HTTPS address and pairing code. Keep the Mac awake.
3. Caddy uses its own local certificate authority. Its root certificate is normally under `~/Library/Application Support/Caddy/pki/authorities/local/root.crt`. Locate the actual path from Caddy's startup output if different.
4. Transfer **only `root.crt`**, never any private key, to your phone using AirDrop. Inspect that the certificate came from your own Mac.
5. On iPhone, open Settings → General → VPN & Device Management. The certificate is listed under **Downloaded Profile** but is not yet installed: tap it, then **Install**, enter the passcode, and **Install** again. Then open Settings → General → About → **Certificate Trust Settings** and enable full trust for "Caddy Local Authority". Both steps are required. This is a user-operated one-time trust step. Remove that profile later if no longer using the app.
6. Open the printed HTTPS address in Safari. Enter the pairing code.
7. Choose **Live camera**, then **Connect phone / webcam**. Allow camera and microphone access. Verify observations and questions appear without requiring a named place.
8. Answer with text or **Speak reply**, review the transcript, interpret and confirm teaching.
9. Press **Stop recording**. Verify playback appears, reopen the experience on the Mac, and confirm the learned name persists after a restart.

Without the phone browser, `scripts/camera_stream.py` streams the iPhone through Continuity Camera (AVFoundation device 0) into the same live endpoint. It checks perception, questions, recording and teaching, but not Safari, certificates or the phone interface.

Do not bypass certificate warning screens. Correct certificate trust or the hostname/IP instead. A changed Mac IP requires rerunning `make phone` and opening its new address. Check macOS firewall permission and Wi-Fi client isolation if WebRTC cannot connect. Recording during screen lock/backgrounding is not guaranteed; keep Safari foregrounded.

Physical iPhone camera, microphone, synchronization, reconnection and voice acceptance remain pending; the existence of a live endpoint does not establish those results.

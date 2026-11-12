import XCTest
import simd

@testable import GroundedKit

final class GroundedKitTests: XCTestCase {
    func testWorldQuestionCanBeAudioOnlyAndKeepsPropositionVersion() throws {
        let json = #"{"learner_id":"robot-a","episode_id":"e","location":{"place_id":null,"label":"Unknown place","certainty":"unknown","candidates":[]},"attention":{"id":"q","episode_id":"e","version":3,"kind":"identify","text":"What is that sound?","reason":"unfamiliar","evidence_refs":["sound.wav"],"region":null,"proposition":{"place_id":null,"event_key":"sound:one"},"status":"active"},"questions":[],"beliefs":[],"predictions":[],"recent_events":[],"status":"observing"}"#
        let state = try JSON.decoder.decode(WorldSnapshot.self, from: Data(json.utf8))
        let question = try XCTUnwrap(state.attention)
        XCTAssertNil(question.region)
        XCTAssertEqual(question.evidenceRefs, ["sound.wav"])
        let reply = WorldReply(question: question, text: "Running water")
        let encoded = String(decoding: try JSON.encoder.encode(reply), as: UTF8.self)
        XCTAssertTrue(encoded.contains(#""question_version":3"#))
    }

    func testControlModeDecodesAndOlderServersOmitIt() throws {
        let json = #"{"learner_id":"robot-a","episode_id":"e","location":{"place_id":"p","label":"Kitchen","certainty":"taught","candidates":[],"continuous":true},"executive":{"mode":"VERIFY","reason":"This view partly resembles Kitchen","request":"Turn slowly","working_set":[]},"attention":null,"questions":[],"beliefs":[],"predictions":[],"recent_events":[],"status":"observing"}"#
        let state = try JSON.decoder.decode(WorldSnapshot.self, from: Data(json.utf8))
        XCTAssertEqual(state.executive?.mode, "VERIFY")
        XCTAssertEqual(state.executive?.request, "Turn slowly")
        let older = json.replacingOccurrences(of: #""executive":{"mode":"VERIFY","reason":"This view partly resembles Kitchen","request":"Turn slowly","working_set":[]},"#, with: "")
        XCTAssertNil(try JSON.decoder.decode(WorldSnapshot.self, from: Data(older.utf8)).executive)
    }

    func testOutboxSurvivesReopeningAndPreservesRejectedEvidence() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let server = URL(string: "https://localhost:8443")!
        let queue = try ObservationOutbox(server: server, root: root)
        var request = URLRequest(url: server.appending(path: "api/v2/questions/q/answers"))
        request.httpMethod = "POST"
        request.httpBody = Data(#"{"id":"stable-reply"}"#.utf8)
        try await queue.enqueue(id: "../stable-reply", request: request)
        let reopened = try ObservationOutbox(server: server, root: root)
        let entries = try await reopened.pending()
        XCTAssertEqual(entries.count, 1)
        XCTAssertEqual(ObservationOutbox.request(entries[0]).httpBody, request.httpBody)
        try await reopened.reject(entries[0], reason: "Question version changed")
        let remaining = try await reopened.pending()
        let issues = try await reopened.issues()
        XCTAssertTrue(remaining.isEmpty)
        XCTAssertEqual(issues, ["Question version changed"])
        let otherServer = try ObservationOutbox(server: URL(string: "https://other.local")!, root: root)
        let other = try await otherServer.pending()
        XCTAssertTrue(other.isEmpty)
    }

    func testDecodesBackendLeanState() throws {
        let json = """
            {"id":"s1","mode":"live","status":"live","paused":false,
             "context":{"place_id":"p1","label":"Office","unknown":false,"capture_time":null,"relative_time":true},
             "place_guess":{"id":"p1","label":"Office","appearance":0.93,"score":0.93},
             "question":{"id":"q1","text":"What is this?","media_time":2.5,"frame":"f.jpg",
               "context":{"place_id":"p1","label":"Office"},
               "region":{"id":"r1","bbox":[0.1,0.2,0.3,0.4],"asset":"a.png","salience":0.4,"encoder":"dinov2","mask":true,"matches":[]}},
             "latest":{"media_time":2.5,"frame":"f.jpg","regions":[
               {"id":"r2","bbox":[0,0,1,1],"asset":"b.png","salience":0.2,"matches":[{"id":"e1","label":"printer","appearance":0.98,"score":0.98}]}]}}
            """
        let state = try JSON.decoder.decode(SessionState.self, from: Data(json.utf8))
        XCTAssertEqual(state.context?.placeId, "p1")
        XCTAssertEqual(state.placeGuess?.label, "Office")
        XCTAssertEqual(state.question?.region.bbox, [0.1, 0.2, 0.3, 0.4])
        XCTAssertEqual(state.latest.regions.first?.best?.label, "printer")
    }

    func testDecodesClarificationAndEncodesSnakeCaseTeaching() throws {
        let json = #"{"needs_clarification":true,"matches":[{"id":"e1","kind":"object","label":"carpet","examples":[{"asset":"c.png","features":[1]}],"claims":[]}]}"#
        let result = try JSON.decoder.decode(TeachResult.self, from: Data(json.utf8))
        XCTAssertEqual(result.needsClarification, true)
        XCTAssertEqual(result.matches?.first?.examples?.first?.asset, "c.png")
        struct Body: Encodable { var previousId: String; var resolution: Resolution }
        let body = String(decoding: try JSON.encoder.encode(Body(previousId: "e1", resolution: .sameKind)), as: UTF8.self)
        XCTAssertTrue(body.contains(#""previous_id":"e1""#) && body.contains(#""same_kind""#))
    }

    func testFingerprintMatchesLauncherFormat() throws {
        let pem = "-----BEGIN CERTIFICATE-----\naGVsbG8=\n-----END CERTIFICATE-----\n"
        let der = try XCTUnwrap(Authority.der(fromPEM: pem))
        XCTAssertEqual(der, Data("hello".utf8))
        // sha256("hello") = 2CF24DBA5FB0A30E26E83B2AC5B9E29E1B161E5C1FA7425E73043362938B9824
        XCTAssertEqual(Authority.fingerprint(der), "2CF24DBA…8B9824")
    }

    func testPortraitBoxMapsBackToSensorOrientation() {
        // Top-left of the upright image is the bottom-left of the landscape sensor image.
        XCTAssertEqual(Geometry.sensorPoint(fromPortrait: SIMD2(0, 0)), SIMD2(0, 1))
        XCTAssertEqual(Geometry.sensorPoint(fromPortrait: SIMD2(1, 0)), SIMD2(0, 0))
        let p = SIMD2(0.25, 0.75)
        XCTAssertEqual(Geometry.portraitPoint(fromSensor: Geometry.sensorPoint(fromPortrait: p)), p)
        let center = Geometry.center(of: [0.2, 0.4, 0.2, 0.2])
        XCTAssertEqual(center.x, 0.3, accuracy: 1e-9)
        XCTAssertEqual(center.y, 0.5, accuracy: 1e-9)
    }

    func testRayThroughImageCenterFollowsCameraForward() {
        let intrinsics = simd_float3x3(SIMD3(1000, 0, 0), SIMD3(0, 1000, 0), SIMD3(960, 720, 1))
        let ray = Geometry.ray(
            through: SIMD2(0.5, 0.5), resolution: SIMD2(1920, 1440), intrinsics: intrinsics,
            transform: matrix_identity_float4x4)
        XCTAssertEqual(ray.direction, SIMD3(0, 0, -1))
        let right = Geometry.ray(
            through: SIMD2(1, 0.5), resolution: SIMD2(1920, 1440), intrinsics: intrinsics,
            transform: matrix_identity_float4x4)
        XCTAssertGreaterThan(right.direction.x, 0)
        let top = Geometry.ray(
            through: SIMD2(0.5, 0), resolution: SIMD2(1920, 1440), intrinsics: intrinsics,
            transform: matrix_identity_float4x4)
        XCTAssertGreaterThan(top.direction.y, 0)
    }

    func testSensorAndPoseEncodeForTheSamplesEndpoint() throws {
        let sensor = SensorDescriptor(
            source: "arkit", projection: "rectilinear", device: "iPhone17,2", platform: "iOS 27.0", camera: "back",
            lens: "wide", resolution: [1920, 1440], fps: 60, depth: "lidar")
        let json = String(decoding: try JSON.encoder.encode(sensor), as: UTF8.self)
        XCTAssertTrue(json.contains(#""modality":"video""#) && json.contains(#""projection":"rectilinear""#))
        var form = GroundedClient.Multipart()
        form.add("meta", data: Data("{}".utf8), type: "application/json")
        form.add("depth", data: Data([1, 2]), filename: "depth")
        let body = String(decoding: form.finished(), as: UTF8.self)
        XCTAssertTrue(body.contains(#"name="depth"; filename="depth""#))
        XCTAssertTrue(body.hasSuffix("--\(form.boundary)--\r\n"))
        let relation = ObservedRelation(id: "r", subject: "lamp", predicate: "next_to", object: "desk")
        XCTAssertEqual(relation.sentence, "lamp next to desk")
    }

    func testServerErrorsReadDetailMessages() {
        XCTAssertEqual(GroundedClient.message(Data(#"{"detail":"Question changed"}"#.utf8), 409), "Question changed")
        XCTAssertEqual(GroundedClient.message(Data("Pair this browser first".utf8), 401).prefix(19), "This phone is not p")
    }
}

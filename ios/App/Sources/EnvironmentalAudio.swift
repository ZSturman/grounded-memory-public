import AVFoundation
import GroundedKit

/// Runs on the capture queue. Converts microphone samples locally; never transcribes ambient speech.
final class EnvironmentalAudio {
    private var converter: AVAudioConverter?
    private var inputRate: Double?
    private var samples = Data()
    private var began: Double?
    private var containsPlayback = false
    var onChunk: ((Data, Double, Bool) -> Void)?

    func append(_ sample: CMSampleBuffer, origin: Double, playback: Bool) {
        guard let description = CMSampleBufferGetFormatDescription(sample),
              let format = AVAudioFormat(cmAudioFormatDescription: description) as AVAudioFormat?,
              let input = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(CMSampleBufferGetNumSamples(sample))),
              let outputFormat = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16000, channels: 1, interleaved: true)
        else { return }
        input.frameLength = input.frameCapacity
        guard CMSampleBufferCopyPCMDataIntoAudioBufferList(sample, at: 0, frameCount: Int32(input.frameLength),
                                                         into: input.mutableAudioBufferList) == noErr else { return }
        if inputRate != format.sampleRate {
            flush()
            converter = AVAudioConverter(from: format, to: outputFormat)
            inputRate = format.sampleRate
        }
        guard let converter,
              let output = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: AVAudioFrameCount(Double(input.frameLength) * 16000 / format.sampleRate + 128))
        else { return }
        var supplied = false
        var error: NSError?
        converter.convert(to: output, error: &error) { _, status in
            if supplied { status.pointee = .noDataNow; return nil }
            supplied = true
            status.pointee = .haveData
            return input
        }
        guard error == nil, output.frameLength > 0, let pointer = output.int16ChannelData?[0] else { return }
        let timestamp = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample)) - origin
        guard timestamp.isFinite, timestamp >= 0 else { return }
        if began == nil { began = timestamp }
        containsPlayback = containsPlayback || playback
        samples.append(Data(bytes: pointer, count: Int(output.frameLength) * 2))
        if samples.count >= 64000 { flush() }
    }

    func flush() {
        guard let began, !samples.isEmpty else { return }
        var wav = Data("RIFF".utf8)
        func integer<T: FixedWidthInteger>(_ value: T) { var little = value.littleEndian; withUnsafeBytes(of: &little) { wav.append(contentsOf: $0) } }
        integer(UInt32(samples.count + 36)); wav.append(Data("WAVEfmt ".utf8))
        integer(UInt32(16)); integer(UInt16(1)); integer(UInt16(1)); integer(UInt32(16000))
        integer(UInt32(32000)); integer(UInt16(2)); integer(UInt16(16))
        wav.append(Data("data".utf8)); integer(UInt32(samples.count)); wav.append(samples)
        onChunk?(wav, began, containsPlayback)
        samples.removeAll(keepingCapacity: true)
        self.began = nil
        containsPlayback = false
    }
}

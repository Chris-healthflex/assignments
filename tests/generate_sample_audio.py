import wave
import struct
import math
import os
import sys

def generate_clinical_wav(output_path: str = "clinical_assessment.wav", duration_sec: float = 3.0, sample_rate: int = 16000):
    """
    Generates a valid 16-bit mono PCM WAV file suitable for Whisper processing and testing.
    """
    num_samples = int(duration_sec * sample_rate)
    
    # Generate multi-tone synthetic clinical speech frequencies
    audio_data = bytearray()
    for i in range(num_samples):
        t = i / sample_rate
        # Composite waveform simulating speech formant resonances (200Hz, 800Hz, 2200Hz)
        sample = (
            0.4 * math.sin(2 * math.pi * 220 * t) +
            0.3 * math.sin(2 * math.pi * 880 * t) +
            0.2 * math.sin(2 * math.pi * 2100 * t)
        )
        # Apply amplitude envelope
        envelope = math.sin(math.pi * (i / num_samples))
        sample_val = int(sample * envelope * 32767 * 0.5)
        # Clamp to 16-bit signed integer
        sample_val = max(-32768, min(32767, sample_val))
        audio_data.extend(struct.pack("<h", sample_val))

    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(1)       # Mono
        wav_file.setsampwidth(2)      # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)

    print(f"Generated valid PCM WAV file at: {os.path.abspath(output_path)} ({len(audio_data)} bytes, {duration_sec}s)")
    return output_path

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "clinical_assessment.wav"
    generate_clinical_wav(out)

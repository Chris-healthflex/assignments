#!/usr/bin/env python3
"""
CLI entrypoint to run the Stance Health clinical assessment pipeline directly on a WAV file.
Usage:
    python scripts/run_pipeline.py clinical_assessment.wav -o output_assessment.json
"""
import sys
import json
import argparse
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.services.audio import validate_wav_header
from app.services.transcription import transcribe
from app.services.extraction import build_pipeline_graph


def main():
    parser = argparse.ArgumentParser(description="Transcribe and extract FirstAssessment from a WAV audio file.")
    parser.add_argument("audio_path", help="Path to input .wav audio file")
    parser.add_argument("-o", "--output", help="Optional output path for generated FirstAssessment JSON")
    parser.add_argument("--save-transcript", help="Optional output path for raw transcript")
    parser.add_argument("--threshold", type=float, default=settings.CONFIDENCE_THRESHOLD, help="Confidence threshold")

    args = parser.parse_args()
    audio_path = Path(args.audio_path)

    if not audio_path.exists():
        print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Validating audio header for: {audio_path}")
    with open(audio_path, "rb") as f:
        validate_wav_header(f)

    print(f"[*] Running Whisper transcription ({settings.WHISPER_MODEL})...")
    t0 = time.time()
    transcript_result = transcribe(str(audio_path), model_size=settings.WHISPER_MODEL)
    t_whisper = time.time() - t0
    print(f"[+] Transcribed {len(transcript_result.text)} characters in {t_whisper:.2f}s")

    if args.save_transcript:
        with open(args.save_transcript, "w", encoding="utf-8") as f:
            f.write(transcript_result.text)
        print(f"[+] Saved transcript to: {args.save_transcript}")

    print("[*] Running LangGraph extraction pipeline (extract -> ground -> normalize -> audit)...")
    t1 = time.time()
    pipeline = build_pipeline_graph()
    result_state = pipeline.invoke({
        "transcript": transcript_result.text,
        "segments": transcript_result.segments,
    })
    t_pipeline = time.time() - t1

    assessment = result_state["assessment"]
    report = result_state["confidence_report"]

    print(f"[+] Extraction complete in {t_pipeline:.2f}s")
    print(f"[*] Confidence overall: {report.overall} (Threshold: {report.threshold}) -> Passed: {report.passed}")
    if report.flags:
        print(f"[*] Flags ({len(report.flags)}):")
        for flag in report.flags:
            print(f"    - {flag.field} (conf: {flag.confidence}, grounded: {flag.grounded}): {flag.reason}")

    output_json = json.dumps(assessment.model_dump(), indent=2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"[+] Saved output assessment to: {args.output}")
    else:
        print("\n--- FIRST ASSESSMENT JSON ---")
        print(output_json)

    sys.exit(0 if report.passed else 2)


if __name__ == "__main__":
    main()

# Voice/Note → Structured Clinical Assessment Form Filler

An end-to-end clinical assessment pipeline that converts a clinician's voice recording into a structured, validated, grounded clinical assessment.

The system performs:

**WAV Audio → Local Whisper → Clinical Transcript → LangGraph Extraction → Pydantic Validation → Grounding Verification → Confidence Scoring → Structured JSON → FastAPI + MongoDB**

---

## Overview

This project implements a production-oriented pipeline for transforming unstructured clinical voice notes into a structured `FirstAssessment` object.

The pipeline is designed around four important principles:

1. **The transcript is the source of truth**
2. **The LLM must produce structured data**
3. **Unsupported information must not be invented**
4. **Uncertain or missing information must remain explicitly uncertain**

The system separates transcription, extraction, validation, grounding, persistence, and API concerns so each component can be tested and replaced independently.

---

## Architecture

```text
                    ┌──────────────────────┐
                    │   WAV / Audio Input  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Audio I/O       │
                    │ Validation / Loading │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Local Whisper STT  │
                    │  Speech → Text       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Clinical Transcript │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │           LangGraph             │
              │                                 │
              │  ┌───────────────────────────┐  │
              │  │ Clinical Extraction       │  │
              │  ├───────────────────────────┤  │
              │  │ Subjective Extraction     │  │
              │  ├───────────────────────────┤  │
              │  │ Objective Extraction      │  │
              │  ├───────────────────────────┤  │
              │  │ Goals Extraction           │  │
              │  ├───────────────────────────┤  │
              │  │ Plan Extraction            │  │
              │  └───────────────────────────┘  │
              └───────────────┬─────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │ Pydantic Validation  │
                    │   FirstAssessment    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Grounding Verification│
                    │ Transcript Evidence  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Confidence Scoring   │
                    │ + Uncertainty Flags  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Structured JSON    │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    ▼                      ▼
          ┌──────────────────┐   ┌──────────────────┐
          │     FastAPI      │   │     MongoDB      │
          │    REST API      │   │    Persistence   │
          └──────────────────┘   └──────────────────┘

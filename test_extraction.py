from app.transcribe import transcribe_audio
from app.graph import clinical_graph


# ---------------------------------------------------------
# 1. Transcribe the provided WAV file
# ---------------------------------------------------------

audio_path = "clinical_assessment.wav"

print("\n--- TRANSCRIPTION ---\n")

transcription = transcribe_audio(audio_path)

print(transcription)


# ---------------------------------------------------------
# 2. Run transcription through LangGraph
# ---------------------------------------------------------

result = clinical_graph.invoke({
    "transcription": transcription
})


# ---------------------------------------------------------
# 3. Print structured clinical assessment JSON
# ---------------------------------------------------------

print("\n--- STRUCTURED CLINICAL ASSESSMENT ---\n")

print(
    result["assessment"].model_dump_json(indent=2)
)
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("transcription.txt", "r", encoding="utf-8") as f:
    transcription = f.read()

response = client.responses.create(
    model="gpt-4o-mini",
    input=f"""
You are a clinical documentation assistant.

Extract the following physiotherapy assessment into a clear
clinical summary.

Include ONLY information explicitly supported by the transcript.
Do NOT invent, assume, estimate, or correct clinical values.

Include:
1. Clinical history
2. Chief complaint
3. Duration
4. Subjective assessments
5. Objective measurements
6. Goals
7. Recommendations
8. Patient advice

TRANSCRIPT:
{transcription}
"""
)

print("\n--- CLINICAL EXTRACTION ---\n")
print(response.output_text)
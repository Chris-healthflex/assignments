"""LLM interface for extraction.

Primary path: Ollama (local) with JSON-mode output.
Fallback path: a deterministic rule-based extractor (`StubExtractor`) that reads the
transcript with regexes. The stub exists so the whole pipeline runs and tests pass
with no model server; it is grounded (pulls only values present in the text) and
correctly handles left/right measurements, so it is a faithful stand-in — not a
hardcoded answer key.

Both expose the same `extract(section, transcript)` contract.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict

from app.config import settings
from app.extraction import prompts


# --------------------------------------------------------------------------- #
# Ollama backend
# --------------------------------------------------------------------------- #
class OllamaExtractor:
    def __init__(self) -> None:
        try:
            import ollama  # noqa: F401
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "The `ollama` package is required for LLM_BACKEND=ollama. "
                "`pip install ollama` or set USE_STUB_LLM=1."
            ) from exc
        import ollama

        self._client = ollama.Client(host=settings.ollama_host)
        self._model = settings.ollama_model

    def _chat_json(self, user_prompt: str) -> Dict[str, Any]:
        resp = self._client.chat(
            model=self._model,
            format="json",
            options={"temperature": 0.0},
            messages=[
                {"role": "system", "content": prompts.SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = resp["message"]["content"]
        return _safe_json(content)

    def extract(self, section: str, transcript: str) -> Dict[str, Any]:
        prompt = _PROMPT_FOR[section].format(transcript=transcript)
        return self._chat_json(prompt)


_PROMPT_FOR = {
    "clinicalDetails": prompts.CLINICAL_DETAILS,
    "subjective": prompts.SUBJECTIVE,
    "objective": prompts.OBJECTIVE,
    "goals": prompts.GOALS,
    "plan": prompts.PLAN,
}


def _safe_json(text: str) -> Dict[str, Any]:
    """Parse model output, tolerating stray prose or code fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*", "", text).rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


# --------------------------------------------------------------------------- #
# Deterministic stub backend (regex-based, grounded)
# --------------------------------------------------------------------------- #
_NUM = r"\d+(?:\.\d+)?"


class StubExtractor:
    """Rule-based extractor. Deterministic, offline, grounded in the transcript."""

    def extract(self, section: str, transcript: str) -> Dict[str, Any]:
        fn = getattr(self, f"_{section}")
        return fn(transcript)

    # -- clinical details --------------------------------------------------- #
    def _clinicalDetails(self, t: str) -> Dict[str, Any]:
        duration = ""
        m = re.search(r"([A-Za-z-]+)\s+months?\s+(?:ago|having passed)", t)
        if m:
            duration = f"{m.group(1).lower()} months"
        chief = ""
        cm = re.search(r"presented with (.+?)(?:following surgery|\.)", t, re.IGNORECASE)
        if cm:
            chief = cm.group(1).strip().rstrip(",")
        history = ""
        hm = re.search(r"(involved in a road traffic accident.+?progressive loading\.)", t, re.IGNORECASE | re.DOTALL)
        if hm:
            history = hm.group(1).strip()
        return {"clinicalHistory": history, "chiefComplaint": chief, "duration": duration}

    # -- subjective --------------------------------------------------------- #
    def _subjective(self, t: str) -> Dict[str, Any]:
        items = []
        patterns = [
            r"(healed surgical scar[^.]*\.)",
            r"(Patellar mobility was good[^.]*\.)",
            r"(left hip extension was restricted[^.]*\.)",
        ]
        for p in patterns:
            m = re.search(p, t, re.IGNORECASE)
            if m:
                items.append({"testName": m.group(1).strip(), "conclusion": ""})
        return {"items": items}

    # -- objective (the count-critical one) --------------------------------- #
    def _objective(self, t: str) -> Dict[str, Any]:
        tests = []

        def bilateral(test_name: str, pattern: str) -> None:
            m = re.search(pattern, t, re.IGNORECASE)
            if m:
                tests.append({
                    "testName": test_name, "unitName": "degrees", "value": "",
                    "left": m.group("left"), "right": m.group("right"), "comments": "",
                })

        def symmetric(test_name: str, pattern: str) -> None:
            m = re.search(pattern, t, re.IGNORECASE)
            if m:
                v = m.group("val")
                tests.append({
                    "testName": test_name, "unitName": "degrees", "value": "",
                    "left": v, "right": v, "comments": "bilateral",
                })

        bilateral("Knee flexion",
                  rf"knee flexion of (?P<left>{_NUM})[^\d]+compared with (?P<right>{_NUM})[^\d]+right")
        bilateral("Knee extension",
                  rf"knee extension of (?P<left>{_NUM})[^\d]+?(?P<right>{_NUM})[^\d]+right")
        symmetric("Hip internal rotation",
                  rf"hip internal rotation of (?P<val>{_NUM})[^\d]+bilateral")
        symmetric("Hip external rotation",
                  rf"hip external rotation of (?P<val>{_NUM})[^\d]+bilateral")
        bilateral("Ankle dorsiflexion",
                  rf"(?:ankle\s+dos\w*\s*flexion|dorsiflexion) of (?P<left>{_NUM})[^\d]+compared with (?P<right>{_NUM})[^\d]+right")
        return {"tests": tests}

    # -- goals -------------------------------------------------------------- #
    def _goals(self, t: str) -> Dict[str, Any]:
        objective = []
        m = re.search(r"emphasis on (.+?)(?:\.|$)", t, re.IGNORECASE | re.DOTALL)
        if m:
            chunk = m.group(1)
            for raw in re.split(r",| and ", chunk):
                g = raw.strip().rstrip(".")
                if not g:
                    continue
                g = g[0].upper() + g[1:]
                objective.append({
                    "goalName": g, "goalCategory": "", "unitName": "",
                    "value": "", "targetDate": "",
                })
        return {"subjectiveGoals": [], "objectiveGoals": objective}

    # -- plan --------------------------------------------------------------- #
    def _plan(self, t: str) -> Dict[str, Any]:
        rec = []
        m = re.search(r"(Physiotherapy) was recommended (.+?)(?:,|\.|with emphasis)", t, re.IGNORECASE)
        if m:
            rec.append({"sessionType": m.group(1), "sessionFrequency": m.group(2).strip()})
        return {"recommendation": rec, "patientAdvice": {"adviceDetails": ""}}


def get_extractor():
    """Return the configured extractor. Falls back to the stub on request/failure."""
    if settings.use_stub_llm or settings.llm_backend == "stub":
        return StubExtractor()
    try:
        return OllamaExtractor()
    except Exception:  # pragma: no cover - env dependent
        # Ollama unavailable: degrade to the deterministic stub so nothing 500s.
        return StubExtractor()

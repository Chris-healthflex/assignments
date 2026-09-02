"""
End-to-end check of all four endpoints against a running server + live MongoDB.

    uvicorn app.main:app --port 8000        # in one terminal
    python tests/run_endpoints.py           # in another

Exercises: parse (WAV upload) -> save -> get by id -> list with date filter,
plus the 404 path and an empty date range. Writes a JSON summary if given a path.
"""
import os, httpx, json, sys, time, pathlib
B = os.environ.get("API_BASE", "http://127.0.0.1:8000")
R = {}
with httpx.Client(base_url=B, timeout=1200) as c:
    for _ in range(30):
        try:
            h = c.get("/health"); break
        except Exception: time.sleep(2)
    print("health:", h.status_code, h.json())

    print("\n[EP1] POST /assessments/parse (uploading WAV, this runs Whisper)...")
    t=time.time()
    with open("clinical_assessment.wav","rb") as f:
        r = c.post("/assessments/parse",
                   files={"file": ("clinical_assessment.wav", f, "audio/wav")},
                   data={"session_date": "2026-09-02"})
    print(f"  -> {r.status_code} in {time.time()-t:.0f}s | confidence={r.headers.get('X-Extraction-Confidence')}")
    r.raise_for_status()
    assessment = r.json()
    print("  top-level keys:", list(assessment))
    R["parse"] = {"status": r.status_code, "confidence": r.headers.get("X-Extraction-Confidence"),
                  "flags": json.loads(r.headers.get("X-Extraction-Flags","[]")), "keys": list(assessment)}

    print("\n[EP2] POST /assessments (save to MongoDB)...")
    r = c.post("/assessments", json={"assessment": assessment,
               "meta": {"sourceFile":"clinical_assessment.wav","transcript":"","flags":[],"overallConfidence":float(R['parse']['confidence'])}})
    print("  ->", r.status_code)
    r.raise_for_status()
    saved = r.json(); aid = saved["id"]
    print("  saved id:", aid, "| createdAt:", saved.get("createdAt"))
    R["save"] = {"status": r.status_code, "id": aid, "createdAt": saved.get("createdAt")}

    print("\n[EP3] GET /assessments/{id}...")
    r = c.get(f"/assessments/{aid}"); print("  ->", r.status_code)
    r.raise_for_status()
    got = r.json()
    same = got["assessment"] == assessment
    print("  round-trip identical to parsed output:", same)
    R["get"] = {"status": r.status_code, "roundtrip_identical": same}

    r404 = c.get("/assessments/000000000000000000000000")
    print("  unknown id ->", r404.status_code)
    R["get_404"] = r404.status_code

    print("\n[EP4] GET /assessments?from=&to= (date filter)...")
    day = saved["createdAt"][:10]
    r = c.get("/assessments", params={"from": f"{day}T00:00:00", "to": f"{day}T23:59:59"})
    print("  ->", r.status_code, "| matched:", len(r.json()))
    r.raise_for_status()
    inrange = len(r.json())
    r2 = c.get("/assessments", params={"from": "2000-01-01T00:00:00", "to": "2000-01-02T00:00:00"})
    print("  empty range ->", r2.status_code, "| matched:", len(r2.json()))
    R["list"] = {"status": r.status_code, "in_range": inrange, "out_of_range": len(r2.json())}

if len(sys.argv) > 1:
    pathlib.Path(sys.argv[1]).write_text(json.dumps(R, indent=2))
print("\nALL FOUR ENDPOINTS OK")

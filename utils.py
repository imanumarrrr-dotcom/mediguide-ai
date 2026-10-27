"""Safe parsing, emergency keyword screening, and optional PDF report."""
import json
import re
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

VALID_URGENCY = {"LOW", "MEDIUM", "HIGH", "EMERGENCY"}

def parse_json_safely(raw: str):
    """Parse JSON even if the model accidentally wraps it in a code fence."""
    if not isinstance(raw, str) or not raw.strip():
        return None, "Empty response"
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
        if not isinstance(obj, dict):
            return None, "Top-level JSON must be an object"
        return obj, None
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(cleaned[start:end+1])
                if isinstance(obj, dict):
                    return obj, None
            except json.JSONDecodeError:
                pass
        return None, "Invalid JSON"

def normalize_assessment(obj):
    """Defensively normalize model output into the dashboard schema."""
    obj = obj if isinstance(obj, dict) else {}
    urgency = str(obj.get("urgency_level", "MEDIUM")).upper().strip()
    if urgency not in VALID_URGENCY:
        urgency = "MEDIUM"
    conditions = obj.get("possible_conditions", [])
    if not isinstance(conditions, list):
        conditions = []
    safe_conditions = []
    for item in conditions[:8]:
        if isinstance(item, dict):
            safe_conditions.append({
                "name": str(item.get("name", "Educational possibility"))[:180],
                "reason": str(item.get("reason", "Not enough detail provided."))[:1000],
            })
    def string_list(key):
        value = obj.get(key, [])
        if not isinstance(value, list):
            return []
        return [str(x)[:1000] for x in value[:12] if str(x).strip()]
    return {
        "summary": str(obj.get("summary", "A summary could not be generated."))[:2000],
        "possible_conditions": safe_conditions,
        "urgency_level": urgency,
        "recommended_next_steps": string_list("recommended_next_steps"),
        "questions_for_doctor": string_list("questions_for_doctor"),
        "warning_signs": string_list("warning_signs"),
        "general_information": str(obj.get("general_information", "This is general educational information only."))[:3000],
    }

def detect_emergency_red_flags(patient):
    """Simple keyword screen only; it cannot rule an emergency in or out."""
    text = " ".join([
        " ".join(patient.get("symptoms", [])),
        patient.get("symptom_notes", ""),
        patient.get("notes", ""),
    ]).lower()
    checks = [
        (r"\b(chest pain|chest pressure|heart attack)\b", "chest pain/pressure"),
        (r"\b(shortness of breath|can't breathe|cannot breathe|difficulty breathing|trouble breathing)\b", "breathing difficulty"),
        (r"\b(stroke|face droop|slurred speech|one-sided weakness)\b", "possible stroke sign"),
        (r"\b(severe bleeding|uncontrolled bleeding|bleeding heavily)\b", "severe bleeding"),
        (r"\b(severe allergic reaction|anaphylaxis|throat swelling)\b", "severe allergic reaction"),
        (r"\b(unconscious|not waking|seizure)\b", "loss of consciousness/seizure"),
    ]
    return [label for pattern, label in checks if re.search(pattern, text)]

def make_pdf_report(item):
    """Generate a simple text-based educational report PDF in memory."""
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50
    def line(text):
        nonlocal y
        words = str(text).split()
        current = ""
        for word in words:
            candidate = (current + " " + word).strip()
            if len(candidate) > 92:
                pdf.drawString(48, y, current)
                y -= 15
                current = word
            else:
                current = candidate
        if current:
            pdf.drawString(48, y, current)
            y -= 15
        if y < 55:
            pdf.showPage()
            y = height - 50
    a = item.get("assessment", {})
    p = item.get("patient", {})
    pdf.setTitle("MediGuide AI Educational Guidance")
    pdf.setFont("Helvetica-Bold", 16)
    line("MediGuide AI — Educational Symptom Guidance")
    pdf.setFont("Helvetica", 10)
    line("NOT A DIAGNOSIS. Not a substitute for a licensed healthcare professional.")
    line(f"Created: {item.get('created_at','')}")
    line(f"Age: {p.get('age','')} | Gender: {p.get('gender','')} | Duration: {p.get('duration','')}")
    line(f"Symptoms: {', '.join(p.get('symptoms', []))} {p.get('symptom_notes','')}")
    line(f"Reported severity: {p.get('severity','')}/10")
    line(f"Urgency suggestion (not validated triage): {a.get('urgency_level','')}")
    line("")
    line("SUMMARY")
    line(a.get("summary", ""))
    line("")
    line("POSSIBLE CONDITIONS — EDUCATIONAL ONLY")
    for c in a.get("possible_conditions", []):
        line(f"- {c.get('name','')}: {c.get('reason','')}")
    line("")
    for heading, key in [("NEXT STEPS","recommended_next_steps"),("QUESTIONS FOR A CLINICIAN","questions_for_doctor"),("WARNING SIGNS","warning_signs")]:
        line(heading)
        for entry in a.get(key, []):
            line("- " + entry)
        line("")
    line("If symptoms are severe or life-threatening, seek emergency help immediately.")
    pdf.save()
    return buffer.getvalue()

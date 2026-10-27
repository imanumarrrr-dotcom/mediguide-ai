"""Reusable prompts and output schema for the educational assistant."""
import json
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

ASSESSMENT_SCHEMA = {
    "summary": "Brief patient symptom summary.",
    "possible_conditions": [{"name": "Educational possibility, not diagnosis", "reason": "Why it may be relevant and uncertainty"}],
    "urgency_level": "LOW, MEDIUM, HIGH, or EMERGENCY",
    "recommended_next_steps": ["General, safe next step"],
    "questions_for_doctor": ["Question to ask a qualified healthcare professional"],
    "warning_signs": ["Sign that warrants urgent/emergency help"],
    "general_information": "Plain-language general information, not diagnosis or treatment."
}

SYSTEM_RULES = """You are MediGuide AI, an educational symptom information assistant, not a clinician.
SAFETY RULES:
- Never claim a confirmed diagnosis. Never state the user definitely has or does not have a condition.
- Do not prescribe medication, dosage, or tell the user to stop a prescribed medicine.
- Do not replace a clinician or emergency service. Encourage consultation with a qualified healthcare professional.
- If symptoms suggest possible emergency (for example chest pain, trouble breathing, stroke signs, severe bleeding, fainting, severe allergic reaction, sudden confusion), set urgency_level to EMERGENCY and say to seek emergency help immediately.
- Be cautious, non-alarmist, clear, and age-aware. Do not overstate certainty.
- Provide general next steps only. No unsupported claims.
- Reply in the requested language.
- Return ONLY a valid JSON object matching this exact schema; no markdown fences or text outside JSON:

""" + json.dumps(ASSESSMENT_SCHEMA, ensure_ascii=False).replace("{", "{{").replace("}", "}}")


ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_RULES),
    ("human", """Please provide educational guidance based on these patient-reported details.
Age: {age}
Gender: {gender}
Symptoms selected: {symptoms_text}
Other symptom details: {symptom_notes_text}
Duration: {duration}
Self-rated severity (1-10): {severity_text}
Existing conditions: {conditions_text}
Medications/allergies: {medications_text}
Additional notes: {notes_text}
Output language: {language}

Return exactly the required JSON schema. Possible conditions are educational possibilities only.
""")
])

# Required single-string PromptTemplate demonstrating reusable variables.
SINGLE_STRING_PROMPT = PromptTemplate(
    input_variables=["age", "gender", "symptoms_text", "duration", "severity_text", "language"],
    template=("Educational-only symptom guidance request. Never diagnose or prescribe. "
              "Age={age}; gender={gender}; symptoms={symptoms_text}; duration={duration}; "
              "severity={severity_text}/10; answer language={language}. "
              "Encourage professional care and emergency help for red flags.")
)

NARRATIVE_CHAT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "Explain the following structured educational health guidance in clear, calm {language}. Do not add a diagnosis or new medical claims. Remind the user this is not a substitute for a clinician."),
    ("human", "Structured assessment JSON:\n{assessment_json}\n\nWrite a concise, readable summary and next-step explanation.")
])

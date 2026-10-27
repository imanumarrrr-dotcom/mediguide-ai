"""MediGuide AI — educational symptom guidance prototype.

Run: python -m streamlit run app.py
Medical safety: this app is not a diagnostic device and never replaces a clinician.
"""
from __future__ import annotations

import json
import time
from datetime import datetime

import streamlit as st
from openai import OpenAI, AuthenticationError, APIConnectionError, RateLimitError

from src.config import APP_NAME, MODEL_OPTIONS, LANGUAGES, SYMPTOM_OPTIONS, DISCLAIMER
from src.cache_manager import configure_cache, cache_status
from src.chains import build_assessment_chain, stream_narrative, validate_api_key, message_roles_demo
from src.utils import parse_json_safely, normalize_assessment, detect_emergency_red_flags, make_pdf_report

st.set_page_config(page_title=APP_NAME, page_icon="✦", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #080d19;
    --panel: #111b30;
    --panel2: #17243d;
    --line: rgba(145,170,220,.22);
    --cyan: #56e5f5;
    --violet: #a58bff;
    --pink: #ff8bc8;
    --text: #f4f7ff;
    --muted: #b7c3dc;
    --green: #58e0b2;
}

/* Global canvas and typography */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
html, body {
    margin: 0 !important;
    padding: 0 !important;
    background: var(--bg) !important;
}
.stApp {
    background:
        radial-gradient(ellipse at 12% 0%, rgba(78,104,220,.20), transparent 34%),
        radial-gradient(ellipse at 92% 15%, rgba(0,205,220,.10), transparent 32%),
        var(--bg) !important;
    color: var(--text) !important;
}
h1, h2, h3, h4, .brand {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -.025em;
    color: var(--text);
}
p, li, label, .stCaption {
    color: var(--muted);
}

/* Full-width Streamlit header: themed, not black or white */
[data-testid="stHeader"] {
    width: 100% !important;
    max-width: 100% !important;
    background: linear-gradient(90deg, #10182b, #17243d 50%, #10182b) !important;
    border-bottom: 1px solid var(--line) !important;
    box-shadow: none !important;
}
[data-testid="stHeader"] > div,
[data-testid="stToolbar"] {
    background: transparent !important;
}
[data-testid="stHeader"] button,
[data-testid="stHeader"] a,
[data-testid="stToolbar"] button {
    color: var(--text) !important;
}

/* Use the available browser width */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main {
    width: 100% !important;
    max-width: 100% !important;
    background: transparent !important;
}
.block-container,
[data-testid="stMainBlockContainer"] {
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
    padding: 1.5rem 1.7rem 2.5rem !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10182b, #0b1020) !important;
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] * {
    color: var(--text);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--muted);
}

/* Hero panels */
.hero {
    position: relative;
    overflow: hidden;
    padding: 2.15rem 2.25rem;
    border: 1px solid var(--line);
    border-radius: 24px;
    background: linear-gradient(120deg, rgba(26,43,77,.96), rgba(15,26,46,.94));
    box-shadow: 0 18px 50px rgba(0,0,0,.20);
    animation: rise .55s ease both;
}
.hero:after {
    content: "";
    position: absolute;
    top: -90px;
    right: -70px;
    width: 230px;
    height: 230px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(86,229,245,.13), transparent 70%);
    pointer-events: none;
}
.eyebrow {
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .17em;
    text-transform: uppercase;
    color: var(--cyan);
}
.gradient {
    background: linear-gradient(95deg, #fff, #75e9ff, #b9a0ff, #ff9bd1);
    background-size: 220% auto;
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    animation: shimmer 8s linear infinite;
}

/* Custom cards */
.metric-card {
    background: linear-gradient(145deg, rgba(25,38,64,.96), rgba(13,21,37,.96));
    border: 1px solid var(--line);
    border-radius: 17px;
    padding: 1.05rem 1.15rem;
    min-height: 108px;
    box-shadow: 0 10px 28px rgba(0,0,0,.12);
    animation: rise .45s ease both;
}
.metric-label {
    font-size: .70rem;
    letter-spacing: .10em;
    text-transform: uppercase;
    color: #aab8d5;
}
.metric-value {
    font-size: 1.42rem;
    font-weight: 700;
    line-height: 1.35;
    color: #f5f7ff;
    margin-top: .35rem;
    overflow-wrap: anywhere;
}
.glass {
    padding: 1.2rem 1.25rem;
    background: rgba(20,31,52,.78);
    border: 1px solid var(--line);
    border-radius: 17px;
    box-shadow: 0 12px 30px rgba(0,0,0,.12);
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button,
.stDownloadButton > button {
    border-radius: 12px !important;
    border: 1px solid rgba(86,229,245,.38) !important;
    background: linear-gradient(110deg, #24517a, #55428f) !important;
    color: #fff !important;
    font-weight: 700 !important;
    min-height: 2.7rem;
    transition: transform .2s ease, filter .2s ease, box-shadow .2s ease;
}
.stButton > button:hover,
.stFormSubmitButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-1px);
    filter: brightness(1.1);
    box-shadow: 0 8px 24px rgba(86,229,245,.13);
    border-color: var(--cyan) !important;
}

/* Form controls */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
[data-baseweb="select"] > div,
[data-testid="stChatInput"] textarea {
    background: #111b30 !important;
    color: #f5f7ff !important;
    -webkit-text-fill-color: #f5f7ff !important;
    border: 1px solid rgba(145,170,220,.35) !important;
    border-radius: 11px !important;
    caret-color: var(--cyan) !important;
}
.stTextInput input::placeholder,
.stTextArea textarea::placeholder,
[data-testid="stChatInput"] textarea::placeholder {
    color: #8e9db9 !important;
    -webkit-text-fill-color: #8e9db9 !important;
}
[data-baseweb="select"] * {
    color: #f5f7ff !important;
}
[data-baseweb="popover"],
[role="listbox"],
[role="option"] {
    background: #111b30 !important;
    color: #f5f7ff !important;
}
.stSlider label,
.stMultiSelect label,
.stSelectbox label,
.stTextInput label,
.stTextArea label,
.stNumberInput label {
    color: #d7e1f5 !important;
}

/* Slider and checkbox accents */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--cyan) !important;
}
input[type="checkbox"] {
    accent-color: var(--cyan);
}

/* File uploader */
[data-testid="stFileUploader"] section {
    background: #111b30 !important;
    border: 1px dashed rgba(86,229,245,.45) !important;
    border-radius: 14px;
}
[data-testid="stFileUploader"] * {
    color: #e7eeff !important;
}

/* Alerts, expanders, and separators */
[data-testid="stAlert"] {
    border-radius: 13px;
    border: 1px solid var(--line);
}
[data-testid="stExpander"] details {
    background: rgba(17,27,48,.72);
    border: 1px solid var(--line);
    border-radius: 12px;
}
[data-testid="stExpander"] summary {
    color: var(--text) !important;
}
hr {
    border-color: var(--line) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: .45rem;
    border-bottom: 1px solid var(--line);
}
.stTabs [data-baseweb="tab"] {
    background: #111b30;
    border-radius: 10px 10px 0 0;
    color: #dbe5f8;
    padding: .65rem 1rem;
}
.stTabs [aria-selected="true"] {
    border-bottom: 2px solid var(--cyan) !important;
    color: #fff !important;
}

/* Code and JSON panels */
[data-testid="stCode"],
[data-testid="stJson"] {
    border: 1px solid var(--line);
    border-radius: 12px;
}
.small-note {
    font-size: .82rem;
    color: #aebbd5;
}

/* Motion */
@keyframes rise {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes shimmer {
    0% { background-position: 0% 50%; }
    100% { background-position: 220% 50%; }
}

/* Mobile */
@media (max-width: 768px) {
    .block-container,
    [data-testid="stMainBlockContainer"] {
        padding: 1rem !important;
    }
    .hero {
        padding: 1.4rem;
        border-radius: 18px;
    }
    .metric-card {
        padding: .85rem;
    }
}

/* Respect reduced-motion settings */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation: none !important;
        transition: none !important;
    }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "last_assessment" not in st.session_state:
    st.session_state.last_assessment = None
if "connected" not in st.session_state:
    st.session_state.connected = False

def safety_banner():
    st.warning("**Medical safety notice:** MediGuide AI is an educational prototype, not a doctor or diagnostic tool. It cannot confirm a diagnosis or replace professional medical care. Consult a qualified healthcare professional. If symptoms feel severe or life-threatening, contact local emergency services immediately.")

def render_metric(label, value, caption=""):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="small-note">{caption}</div></div>', unsafe_allow_html=True)

def key_gate():
    st.markdown('<div class="hero"><div class="eyebrow">PRIVATE • SAFETY-FIRST • LANGCHAIN</div><h1 style="font-size:2.65rem;margin:.6rem 0">Welcome to <span class="gradient">MediGuide AI</span></h1><p style="font-size:1.05rem">A thoughtful symptom guidance prototype that helps you prepare for a conversation with a healthcare professional.</p></div>', unsafe_allow_html=True)
    st.write("")
    safety_banner()
    left, right = st.columns([1.15, .85], gap="large")
    with left:
        st.markdown("### Unlock your private console")
        st.write("Enter your OpenAI API key to validate access. Your key is held in this browser session only; this app does not write it to `.env` or a file.")
        key = st.text_input("OpenAI API key", type="password", placeholder="sk-…", help="Get a key from the OpenAI platform. API usage may incur charges.")
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Validate key & unlock", type="primary", use_container_width=True):
                if not key.strip():
                    st.error("Please enter your API key.")
                else:
                    with st.spinner("Validating key securely…"):
                        ok, message = validate_api_key(key.strip())
                    if ok:
                        st.session_state.api_key = key.strip()
                        st.session_state.connected = True
                        st.success("Key validated. Console unlocked.")
                        st.rerun()
                    else:
                        st.error(message)
        with c2:
            st.button("Clear input", use_container_width=True, on_click=lambda: st.session_state.update(api_key=""))
        st.caption("Never share your API key. Revoke it from your provider account if you accidentally expose it.")
    with right:
        st.markdown('<div class="glass"><div class="eyebrow">WHAT YOU CAN DO</div><h3>Guidance, not diagnosis</h3><p>✦ Structured symptom summary</p><p>✦ Educational possibilities, clearly qualified</p><p>✦ Urgency and next-step guidance</p><p>✦ Questions to ask a clinician</p><p>✦ Warning signs and downloadable report</p></div>', unsafe_allow_html=True)
        st.info("If you have severe chest pain, trouble breathing, signs of stroke, severe bleeding, or another possible emergency, do not wait for an AI response—seek emergency help now.")
    return

if not st.session_state.connected or not st.session_state.api_key:
    with st.sidebar:
        st.markdown("## ✦ MediGuide AI")
        st.caption("AI-powered symptom guidance")
        st.warning("Educational prototype only. Not a diagnosis.")
        st.markdown("**Status:** Locked — validate API key to continue.")
    key_gate()
    st.stop()

# Sidebar settings and cache switch
with st.sidebar:
    st.markdown("## ✦ MediGuide AI")
    st.caption("AI symptom guidance workspace")
    st.success("Provider connected")
    st.warning("Not a doctor. Not for emergencies.")
    model = st.selectbox("Model", MODEL_OPTIONS, index=0)
    language = st.selectbox("Answer language", LANGUAGES, index=0)
    cache_choice = st.radio("LLM cache", ["In-memory (session)", "SQLite (persistent)", "Off"], index=1)
    cache_enabled = cache_choice != "Off"
    cache_type = "memory" if cache_choice.startswith("In-memory") else "sqlite"
    configure_cache(cache_type if cache_enabled else "off")
    st.caption(cache_status())
    st.divider()
    st.markdown("**Session**")
    st.caption(f"Assessments this session: {len(st.session_state.history)}")
    if st.button("Clear session history", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_assessment = None
        st.rerun()
    if st.button("Lock console / disconnect key", use_container_width=True):
        st.session_state.api_key = ""
        st.session_state.connected = False
        st.session_state.last_assessment = None
        st.rerun()

st.markdown('<div class="hero"><div class="eyebrow">PATIENT GUIDANCE • EDUCATIONAL PROTOTYPE</div><h1 style="font-size:2.5rem;margin:.5rem 0">Understand symptoms. <span class="gradient">Prepare your next step.</span></h1><p>Share what you are experiencing to receive general information and prepare for a clinician conversation.</p></div>', unsafe_allow_html=True)
safety_banner()

tabs = st.tabs(["✦ New assessment", "▤ Assessment history", "ⓘ About & safety"])
with tabs[0]:
    with st.form("symptom_form", clear_on_submit=False):
        st.markdown("### 01 · About you")
        a, b, c = st.columns(3)
        with a:
            age_raw = st.text_input("Age", placeholder="e.g. 28", help="Enter age in years.")
        with b:
            gender = st.selectbox("Gender", ["Prefer not to say", "Female", "Male", "Non-binary", "Other"])
        with c:
            duration = st.selectbox("How long have symptoms been present?", ["Just started / less than 24 hours", "1–3 days", "4–7 days", "1–2 weeks", "More than 2 weeks", "Comes and goes"])
        st.markdown("### 02 · Symptoms")
        symptoms = st.multiselect("Select all that apply", SYMPTOM_OPTIONS, help="Choose all that match. You can add details below.")
        symptom_notes = st.text_area("Other symptoms / details", placeholder="Where is it? When did it start? Is it getting better or worse?", height=90)
        severity = st.slider("Overall symptom severity (1 = mild, 10 = severe)", 1, 10, 3)
        st.markdown("### 03 · Medical context")
        d, e = st.columns(2)
        with d:
            conditions = st.text_area("Existing medical conditions", placeholder="e.g. asthma, diabetes, allergies, or none known", height=100)
        with e:
            medications = st.text_area("Current medications & allergies", placeholder="List medicines, doses if known, and allergies", height=100)
        notes = st.text_area("Additional notes", placeholder="Recent travel, exposure, pregnancy possibility if relevant, or anything else you want to mention", height=90)
        consent = st.checkbox("I understand this is educational information, not a diagnosis or treatment plan.")
        submitted = st.form_submit_button("✦ Assess symptoms safely", type="primary", use_container_width=True)

    if submitted:
        age_text = age_raw.strip()
        if not age_text or not age_text.isdigit() or not 0 <= int(age_text) <= 120:
            st.error("Please enter a valid age from 0 to 120.")
        elif not symptoms and not symptom_notes.strip():
            st.warning("Please select at least one symptom or add a symptom description. No API call was made.")
        elif not consent:
            st.warning("Please confirm that you understand the educational safety notice.")
        else:
            patient = {
                "age": int(age_text), "gender": gender, "symptoms": symptoms,
                "symptom_notes": symptom_notes.strip(), "duration": duration,
                "severity": severity, "conditions": conditions.strip(),
                "medications": medications.strip(), "notes": notes.strip(),
                "language": language,
            }
            red_flags = detect_emergency_red_flags(patient)
            if red_flags:
                st.error("**Potential emergency warning detected.** Do not wait for an AI assessment. Seek emergency medical help now if these symptoms are happening.")
                st.write("Flagged terms:", ", ".join(red_flags))
                st.write("This safety check is only a basic keyword screen and cannot rule an emergency in or out.")
            client = OpenAI(api_key=st.session_state.api_key)
            try:
                with st.spinner("Preparing a safety-focused assessment…"):
                    chain = build_assessment_chain(st.session_state.api_key, model)
                    inputs = {**patient, "symptoms_text": ", ".join(symptoms) if symptoms else "Not selected",
                              "conditions_text": conditions.strip() or "Not provided",
                              "medications_text": medications.strip() or "Not provided",
                              "notes_text": notes.strip() or "Not provided",
                              "symptom_notes_text": symptom_notes.strip() or "Not provided",
                              "severity_text": str(severity)}
                    # LLMChain is deliberately used to meet the assignment requirement.
                    result = chain.invoke(inputs)
                    raw = result.get("text", "") if isinstance(result, dict) else str(result)
                parsed, parse_error = parse_json_safely(raw)
                if parse_error:
                    st.error("The model returned a response that could not be safely parsed as JSON. Please try again. No diagnosis has been made.")
                    with st.expander("Raw model output (debug)"):
                        st.code(raw[:12000])
                else:
                    assessment = normalize_assessment(parsed)
                    # Show a live narrative as requested by the assignment. This is a separate,
                    # short streaming call so the structured JSON remains safely parseable.
                    st.markdown("### Live guidance summary")
                    narrative_inputs = {**inputs, "assessment_json": json.dumps(assessment, ensure_ascii=False)}
                    with st.spinner("Streaming readable guidance…"):
                        streamed = st.write_stream(stream_narrative(st.session_state.api_key, model, narrative_inputs))
                    st.session_state.last_assessment = {"patient": patient, "assessment": assessment, "raw": raw,
                                                         "narrative": str(streamed), "created_at": datetime.now().isoformat(timespec="seconds"),
                                                         "model": model}
                    st.session_state.history.insert(0, st.session_state.last_assessment)
                    st.success("Assessment generated. Review it with a qualified healthcare professional.")
                    st.rerun()
            except AuthenticationError:
                st.error("The API key was rejected. Lock the console and enter a valid key.")
            except RateLimitError:
                st.error("The provider rate limit or billing quota was reached. Check your provider account and try later.")
            except APIConnectionError:
                st.error("Could not connect to the provider. Check your internet connection and try again.")
            except Exception as exc:
                st.error("The assessment failed safely. Please try again. No diagnosis has been made.")
                with st.expander("Technical details"):
                    st.code(f"{type(exc).__name__}: {exc}")

with tabs[1]:
    st.markdown("### Previous assessments (current browser session)")
    if not st.session_state.history:
        st.info("No assessments yet. Your history will appear here after an assessment.")
    else:
        for i, item in enumerate(st.session_state.history):
            with st.expander(f"{item['created_at']} · {item['assessment'].get('urgency_level','UNKNOWN')} urgency"):
                st.write(item["assessment"].get("summary", ""))
                st.caption(f"Model: {item.get('model','')} · Age: {item['patient'].get('age','')}")
                st.json(item["assessment"])
                st.download_button("Download this assessment JSON", json.dumps(item, ensure_ascii=False, indent=2), file_name=f"mediguide_assessment_{i+1}.json", mime="application/json", key=f"hist_json_{i}")

with tabs[2]:
    st.markdown("### About MediGuide AI")
    st.write("MediGuide AI is an educational prototype for learning LangChain and Streamlit. It is not validated for clinical use.")
    st.markdown("**It cannot:**")
    st.write("- Diagnose a condition, determine that you are safe, or prescribe medication.")
    st.write("- Replace an examination, tests, or advice from a licensed healthcare professional.")
    st.write("- Reliably identify every emergency from text.")
    st.markdown("**Privacy:** The API key is held in Streamlit session state and is not saved to `.env` or disk by this app. Symptom text is sent to the selected model provider when you submit an assessment. Avoid entering identifying details.")
    st.markdown("**Cache note:** LLM caches may store model inputs/outputs locally. Turn caching off if you do not want prompts reused by the local cache.")
    with st.expander("LangChain message roles demo (SystemMessage / HumanMessage / AIMessage)"):
        st.caption("This educational example demonstrates how LangChain represents a short conversation.")
        for message in message_roles_demo("What should I prepare before talking to my clinician?"):
            st.markdown(f"**{message.__class__.__name__}**")
            st.write(message.content)

# Results render after form processing, and remain available across reruns.
item = st.session_state.last_assessment
if item:
    st.divider()
    assessment = item["assessment"]
    urgency = str(assessment.get("urgency_level", "UNKNOWN")).upper()
    colors = {"LOW": "#58e0b2", "MEDIUM": "#f4cb68", "HIGH": "#ff9b68", "EMERGENCY": "#ff647c"}
    color = colors.get(urgency, "#2b5bc3")
    st.markdown(f'<div class="hero"><div class="eyebrow">ASSESSMENT RESULT • {item["created_at"]}</div><h2>Guidance dashboard</h2><p>{assessment.get("summary","")}</p><div style="display:inline-block;padding:.45rem .85rem;border-radius:999px;background:{color}20;border:1px solid {color};color:{color};font-weight:800">URGENCY: {urgency}</div><p class="small-note">This is a model-generated educational urgency suggestion, not a validated clinical triage score.</p></div>', unsafe_allow_html=True)
    if urgency == "EMERGENCY":
        st.error("Seek emergency medical help immediately. Do not delay care to continue using this app.")
    elif urgency == "HIGH":
        st.warning("Prompt professional medical evaluation is recommended. If symptoms are severe or worsening, seek urgent care.")
    elif urgency == "MEDIUM":
        st.warning("Consider contacting a qualified healthcare professional, especially if symptoms persist or worsen.")
    else:
        st.info("Monitor how you feel and consult a professional if symptoms persist, worsen, or concern you.")
    m1, m2, m3 = st.columns(3)
    with m1: render_metric("Severity reported", f"{item['patient']['severity']}/10", "Self-reported")
    with m2: render_metric("Duration", item["patient"]["duration"], "Patient-provided")
    with m3: render_metric("Language", item["patient"]["language"], "Requested output")
    t1, t2, t3, t4 = st.tabs(["Possible conditions", "Next steps", "Ask your clinician", "Warning signs"])
    with t1:
        st.caption("Educational possibilities only — not a diagnosis.")
        for condition in assessment["possible_conditions"]:
            with st.container(border=True):
                st.markdown(f"**{condition.get('name','Unspecified')}**")
                st.write(condition.get("reason", ""))
    with t2:
        for step in assessment["recommended_next_steps"]:
            st.markdown(f"- {step}")
    with t3:
        for question in assessment["questions_for_doctor"]:
            st.markdown(f"- {question}")
    with t4:
        for sign in assessment["warning_signs"]:
            st.markdown(f"- {sign}")
        st.error("If a warning sign is happening now or you believe this is an emergency, contact local emergency services immediately.")
    st.markdown("### General information")
    st.write(assessment.get("general_information", "This response is general information only."))
    with st.expander("View structured JSON"):
        st.json(assessment)
    report = make_pdf_report(item)
    st.download_button("⬇ Download guidance report (PDF)", report, file_name="mediguide_guidance_report.pdf", mime="application/pdf", type="primary")
    safety_banner()

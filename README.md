# MediGuide AI — Symptom Assessment & Patient Guidance

A safety-first educational prototype built with Streamlit, LangChain, and OpenAI. It is **not a medical device**, does not diagnose, and must not be used as a replacement for licensed medical care.

## Features
- API-key gate: validates the key before unlocking the console. Key is held in Streamlit session state and is not written to disk by this app.
- Patient form: age, gender, multiple symptoms, duration, severity, conditions, medications/allergies, notes, language.
- LangChain `ChatOpenAI`, reusable `LLMChain`, `PromptTemplate`, `ChatPromptTemplate`, and system/human messages.
- Exact JSON structure with defensive parsing and schema normalization.
- Streaming readable narrative via `llm.stream()` and `st.write_stream()`.
- InMemoryCache and SQLiteCache selector.
- Results dashboard, urgency labels, clinician questions, warning signs, history, downloadable PDF.
- Animated gradient/glass UI with high-contrast controls and reduced-motion support.
- Basic emergency keyword screen; this is not a clinical triage system.

## Recommended setup (Windows)
Use Python 3.12. Check installed interpreters with `py -0p`.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The app opens at `http://localhost:8501`. Enter your OpenAI API key in the first screen. API usage may incur charges.

## Cache modes
- **InMemoryCache:** stores cached prompts and responses in RAM; fastest and cleared when the process restarts.
- **SQLiteCache:** stores cached prompts and responses in `data/llm_cache.db`; survives restarts. Do not enable if you do not want health-related prompts/results stored locally.
- **Off:** no LangChain response cache.

Cache is configured with LangChain `set_llm_cache`. The provider/model may not return cached responses if request details differ.

## Safety and privacy
- Educational only, not a diagnosis, not treatment advice, and not validated for clinical use.
- Seek emergency help immediately for severe or life-threatening symptoms; do not wait for this app.
- Symptom text is sent to the selected model provider. Avoid entering names, addresses, or other identifying details.
- The API key is entered at runtime and not saved by this app. Revoking the session key locks the console.
- The keyword screen may miss emergencies and may flag non-emergencies. It cannot rule an emergency in or out.

## Tests
```powershell
python -m unittest discover -s tests
```

## Suggested submission improvements
Before presenting, test the six scenarios in the assignment PDF, manually inspect all model outputs, and record a short demo. This prototype has not been clinically validated.

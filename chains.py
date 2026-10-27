"""LangChain chain creation, key validation, and streaming helper."""
from openai import OpenAI, AuthenticationError, APIConnectionError, RateLimitError
from langchain_openai import ChatOpenAI
try:
    from langchain.chains import LLMChain  # older langchain (0.x)
except ImportError:
    from langchain_classic.chains import LLMChain  # newer langchain (1.x)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.prompts import ASSESSMENT_PROMPT, NARRATIVE_CHAT_TEMPLATE

def validate_api_key(api_key: str):
    """Validate a key with a minimal provider request; never persist it."""
    try:
        OpenAI(api_key=api_key, timeout=15.0, max_retries=0).models.list()
        return True, "API key validated."
    except AuthenticationError:
        return False, "This API key was rejected. Check it and try again."
    except RateLimitError:
        # Some valid keys can be blocked by quota; distinguish from invalid auth.
        return False, "Provider quota/rate limit prevented validation. Check billing and try again."
    except APIConnectionError:
        return False, "Could not connect to OpenAI. Check your internet connection."
    except Exception as exc:
        return False, f"Could not validate key ({type(exc).__name__}). Check the key and network."

def build_assessment_chain(api_key: str, model_name: str):
    """Build the reusable LLMChain required by the assignment."""
    llm = ChatOpenAI(
        api_key=api_key,
        model=model_name,
        temperature=0.1,
        timeout=60,
        max_retries=1,
    )
    return LLMChain(llm=llm, prompt=ASSESSMENT_PROMPT, verbose=False)

def stream_narrative(api_key: str, model_name: str, inputs: dict):
    """Yield narrative chunks for st.write_stream; uses ChatOpenAI.stream()."""
    llm = ChatOpenAI(api_key=api_key, model=model_name, temperature=0.1, streaming=True, timeout=60, max_retries=1)
    messages = NARRATIVE_CHAT_TEMPLATE.format_messages(
        language=inputs.get("language", "English"),
        assessment_json=inputs.get("assessment_json", "{}"),
    )
    for chunk in llm.stream(messages):
        content = chunk.content
        if isinstance(content, str) and content:
            yield content
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    yield part.get("text", "")


def message_roles_demo(user_text: str):
    """Return a safe example conversation showing System/Human/AI message roles."""
    return [
        SystemMessage(content="You are an educational health information assistant. Never diagnose."),
        HumanMessage(content=user_text or "Please explain how to prepare for a doctor visit."),
        AIMessage(content="I can share general information, but a qualified clinician must assess your health.")
    ]
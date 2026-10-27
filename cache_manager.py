"""LangChain global cache configuration: memory, SQLite, or off."""
from pathlib import Path
from langchain_core.globals import set_llm_cache, get_llm_cache
from langchain_community.cache import InMemoryCache, SQLiteCache

CACHE_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_ACTIVE = None
_MODE = "off"

def configure_cache(mode: str):
    """Set cache once per selected mode; SQLite persists across app restarts."""
    global _ACTIVE, _MODE
    mode = mode if mode in {"memory", "sqlite", "off"} else "off"
    if mode == _MODE and _ACTIVE is not None:
        return
    if mode == "memory":
        _ACTIVE = InMemoryCache()
    elif mode == "sqlite":
        _ACTIVE = SQLiteCache(database_path=str(CACHE_DIR / "llm_cache.db"))
    else:
        _ACTIVE = None
    set_llm_cache(_ACTIVE)
    _MODE = mode

def cache_status():
    if _MODE == "memory":
        return "Cache: RAM only; clears when process restarts."
    if _MODE == "sqlite":
        return "Cache: SQLite file; persists across restarts. Prompts/outputs may be stored locally."
    return "Cache disabled."

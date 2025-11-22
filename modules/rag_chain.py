"""
rag_chain.py
AI Smart Coach core logic:

* LLM feedback via Gemini API
* Voice/audio cues with gTTS and pyttsx3 fallback
* Dataset integration for exercise guidance
* Async-safe design for smooth real-time video
  """

import os
import json
import base64
import tempfile
from functools import lru_cache
from typing import Optional, Dict

# TTS libraries

from gtts import gTTS
import pyttsx3

# Optional LangChain Gemini import

try:
from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
ChatGoogleGenerativeAI = None

# Load exercise dataset for cues

DATASET_FILE = os.path.join(os.path.dirname(**file**), "..", "dataset.json")
EXERCISE_DATA = []
if os.path.exists(DATASET_FILE):
with open(DATASET_FILE, "r", encoding="utf-8") as f:
EXERCISE_DATA = [json.loads(line) for line in f if line.strip()]

# -----------------------------

# Utility functions

# -----------------------------

def _format_metrics(metrics: Optional[Dict[str, object]]) -> str:
"""Format joint and movement metrics into a readable string."""
if not metrics:
return "Metrics not available."
angle = metrics.get("angle")
tempo_state = metrics.get("tempo_state")
reps = metrics.get("reps")
rom = metrics.get("rom_last") or metrics.get("rom_current")
if isinstance(angle, (int, float)) and isinstance(rom, (int, float)):
return f"angle={angle:.1f}deg, tempo_state={tempo_state}, reps={reps}, ROM={rom:.1f}"
return f"tempo_state={tempo_state}, reps={reps}, ROM={rom}"

# Singleton pyttsx3 engine

_engine = None
def _get_engine():
global _engine
if _engine is None:
_engine = pyttsx3.init()
_engine.setProperty('rate', 150)
_engine.setProperty('volume', 1.0)
return _engine

# -----------------------------

# CoachBrain: LLM feedback

# -----------------------------

class CoachBrain:
"""
Generate concise, encouraging coaching cues using Gemini LLM or fallback.
Reads exercise dataset to provide context-aware guidance.
"""

```
def __init__(self):
    self._llm = None
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY")
    if ChatGoogleGenerativeAI and api_key:
        try:
            self._llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.4,
                max_output_tokens=128
            )
        except Exception:
            self._llm = None

def get_feedback(
    self,
    cue: str,
    movement: str,
    metrics: Optional[Dict[str, object]] = None
) -> str:
    """
    Return a single-sentence coaching cue.
    Falls back to static message if LLM is unavailable.
    """
    # Default message
    base_message = f"{movement}: {cue}. Maintain control and proper form."

    # Format metrics for LLM context
    metrics_text = _format_metrics(metrics)

    # Compose prompt
    exercise_data = next((ex for ex in EXERCISE_DATA if ex["exercise"] == movement), None)
    dataset_text = ""
    if exercise_data:
        dataset_text = (
            f"Instructions: {', '.join(exercise_data['instructions'])}\n"
            f"Common mistakes: {', '.join(exercise_data['common_mistakes'])}\n"
            f"Corrections: {', '.join(exercise_data['corrections'])}"
        )

    prompt = (
        f"You are a supportive biomechanics coach. Use the context below to craft a "
        f"concise, encouraging voice cue for the athlete. Keep it under 25 words.\n\n"
        f"Context:\n{dataset_text}\n\n"
        f"Movement: {movement}\nObserved issue: {cue}\nRecent metrics: {metrics_text}\n\n"
        "Respond with a single sentence in plain English."
    )

    # Invoke LLM
    if self._llm:
        try:
            result = self._llm.invoke(prompt)
            text = getattr(result, "content", None)
            if isinstance(text, list):
                text = " ".join(str(part) for part in text)
            return text.strip() or base_message
        except Exception:
            self._llm = None
    return base_message

# -----------------------------

# CoachVoice: TTS feedback

# -----------------------------

class CoachVoice:
"""
Convert text to audio using gTTS with pyttsx3 fallback.
Provides base64 audio string for embedding in UI.
"""

```
def __init__(self, language: str = "en", slow: bool = False):
    self.language = language
    self.slow = slow

@lru_cache(maxsize=64)
def synthesize(self, text: str) -> Optional[str]:
    """
    Convert text to base64-encoded audio.
    Tries gTTS first; falls back to pyttsx3 offline TTS if needed.
    """
    if not text:
        return None

    # Try gTTS
    try:
        from io import BytesIO
        buffer = BytesIO()
        tts = gTTS(text=text, lang=self.language, slow=self.slow)
        tts.write_to_fp(buffer)
        buffer.seek(0)
        audio_bytes = buffer.read()
        if audio_bytes:
            return base64.b64encode(audio_bytes).decode("ascii")
    except Exception:
        pass

    # Fallback: pyttsx3 offline
    try:
        engine = _get_engine()
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as tmpfile:
            engine.save_to_file(text, tmpfile.name)
            engine.runAndWait()
            tmpfile.seek(0)
            audio_bytes = tmpfile.read()
        if audio_bytes:
            return base64.b64encode(audio_bytes).decode("ascii")
    except Exception:
        return None

def audio_tag(self, audio_b64: Optional[str], autoplay: bool = True) -> Optional[str]:
    """Return HTML audio tag for embedding base64-encoded audio in Streamlit."""
    if not audio_b64:
        return None
    return (
        f"<audio controls {'autoplay' if autoplay else ''}>"
        f"<source src='data:audio/mp3;base64,{audio_b64}' type='audio/mpeg'>"
        "Your browser does not support the audio element."
        "</audio>"
    )

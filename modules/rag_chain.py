
from **future** import annotations
import os
import json
import base64
import tempfile
from io import BytesIO
from functools import lru_cache
from typing import Optional, Dict

from dotenv import load_dotenv
load_dotenv()  # Automatically load API keys from .env

from gtts import gTTS
import pyttsx3  # offline fallback for TTS

# Optional LangChain Gemini import

try:
from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
ChatGoogleGenerativeAI = None

# Load exercise dataset

DATASET_FILE = os.path.join(os.path.dirname(**file**), "dataset.json")
try:
with open(DATASET_FILE, "r", encoding="utf-8") as f:
EXERCISE_DATA = [json.loads(line) for line in f if line.strip()]
except Exception:
EXERCISE_DATA = []

def _format_metrics(metrics: Optional[Dict[str, object]]) -> str:
if not metrics:
return "Metrics not available."
angle = metrics.get("angle")
tempo_state = metrics.get("tempo_state")
reps = metrics.get("reps")
rom = metrics.get("rom_last") or metrics.get("rom_current")
if isinstance(angle, (int, float)) and isinstance(rom, (int, float)):
return f"angle={angle:.1f}deg, tempo_state={tempo_state}, reps={reps}, ROM={rom:.1f}"
return f"tempo_state={tempo_state}, reps={reps}, ROM={rom}"

# pyttsx3 engine singleton

_engine = None
def _get_engine():
global _engine
if _engine is None:
_engine = pyttsx3.init()
_engine.setProperty('rate', 150)
_engine.setProperty('volume', 1.0)
return _engine

class CoachBrain:
#"""Generates concise, encouraging cues using Gemini LLM or static fallback."""

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
    base_message = f"{movement}: {cue}. Maintain control and keep joints stacked."
    if not self._llm:
        return base_message

    metrics_text = _format_metrics(metrics)
    prompt = (
        "You are a supportive biomechanics coach. Use the context below to craft a "
        "concise, encouraging voice cue for the athlete. Keep it under 25 words.\n\n"
        f"Exercise cues:\n{self._knowledge_bank_text()}\n\n"
        f"Movement: {movement}\nObserved issue: {cue}\nRecent metrics: {metrics_text}\n\n"
        "Respond with a single sentence in plain English."
    )
    try:
        result = self._llm.invoke(prompt)
    except Exception:
        self._llm = None
        return base_message

    text = getattr(result, "content", None)
    if not text:
        return base_message
    if isinstance(text, list):
        text = " ".join(str(part) for part in text)
    return text.strip() or base_message

def _knowledge_bank_text(self) -> str:
    #"""Return all instructions/mistakes/corrections as single text block."""
    lines = []
    for item in EXERCISE_DATA:
        instr = item.get("instructions", [])
        mistakes = item.get("common_mistakes", [])
        corrections = item.get("corrections", [])
        lines.append(f"Exercise: {item.get('exercise')}")
        lines.extend([f"Instruction: {x}" for x in instr])
        lines.extend([f"Mistake: {x}" for x in mistakes])
        lines.extend([f"Correction: {x}" for x in corrections])
    return "\n".join(lines)

class CoachVoice:
#"""Converts text cues to audio with gTTS + pyttsx3 fallback."""

def __init__(self, language: str = "en", slow: bool = False):
    self.language = language
    self.slow = slow

@lru_cache(maxsize=64)
def synthesize(self, text: str) -> Optional[str]:
    if not text:
        return None

    # gTTS attempt
    try:
        buffer = BytesIO()
        tts = gTTS(text=text, lang=self.language, slow=self.slow)
        tts.write_to_fp(buffer)
        buffer.seek(0)
        audio_bytes = buffer.read()
        if audio_bytes:
            return base64.b64encode(audio_bytes).decode("ascii")
    except Exception:
        pass

    # Fallback to pyttsx3
    try:
        engine = _get_engine()
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as tmpfile:
            engine.save_to_file(text, tmpfile.name)
            engine.runAndWait()
            tmpfile.seek(0)
            audio_bytes = tmpfile.read()
        return base64.b64encode(audio_bytes).decode("ascii") if audio_bytes else None
    except Exception:
        return None

def audio_tag(self, audio_b64: Optional[str], autoplay: bool = True) -> Optional[str]:
    if not audio_b64:
        return None
    return (
        f"<audio controls {'autoplay' if autoplay else ''}>"
        f"<source src='data:audio/mp3;base64,{audio_b64}' type='audio/mpeg'>"
        "Your browser does not support the audio element."
        "</audio>"
    )

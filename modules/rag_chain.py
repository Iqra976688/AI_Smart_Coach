# rag_chain.py
"""
AI Smart Coach: Coaching brain and audio utilities.

Provides:
- CoachBrain: produces concise coaching cues (uses optional LLM if configured)
- CoachVoice: converts text -> base64 MP3 audio (gTTS with pyttsx3 fallback)

Design goals:
- Defensive: missing optional libraries won't crash the app.
- Small surface API so app.py can call get_feedback() and synthesize().
"""

from __future__ import annotations
import os
import base64
import tempfile
from io import BytesIO
from functools import lru_cache
from typing import Optional, Dict

# Optional TTS backends
try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

# Optional LangChain / Gemini adapter (kept optional)
try:
    # Note: many setups will not have this. We handle absence gracefully.
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
except Exception:
    ChatGoogleGenerativeAI = None

# -----------------------
# Knowledge bank (static cues)
# -----------------------
KNOWLEDGE_BANK = [
    "Keep joints stacked to protect ligaments during compound lifts.",
    "Maintain a neutral spine by bracing the core before every rep.",
    "Drive from the hips and keep knees tracking over toes in lower-body work.",
    "Balance tempo: controlled eccentric, powerful concentric for strength movements.",
    "Use steady nasal breathing during physiotherapy drills to avoid bracing.",
    "Explosive sports motions need a stable base - focus on balance before speed.",
]

# -----------------------
# Helper: format metrics to short string
# -----------------------
def _format_metrics(metrics: Optional[Dict[str, object]]) -> str:
    if not metrics:
        return "no metrics"
    angle = metrics.get("angle")
    tempo = metrics.get("tempo_last") or metrics.get("tempo_avg")
    reps = metrics.get("reps")
    rom = metrics.get("rom_last") or metrics.get("rom_current")
    parts = []
    if isinstance(angle, (int, float)):
        parts.append(f"angle={angle:.1f}°")
    if isinstance(tempo, (int, float)):
        parts.append(f"tempo={tempo:.2f}s")
    if isinstance(reps, (int, float)):
        parts.append(f"reps={int(reps)}")
    if isinstance(rom, (int, float)):
        parts.append(f"ROM={rom:.1f}°")
    return ", ".join(parts) if parts else "metrics present"

# -----------------------
# CoachBrain
# -----------------------
class CoachBrain:
    """
    Generate concise, encouraging coaching cues. If a Gemini/LangChain LLM is
    configured via environment variables and langchain_google_genai is installed,
    use it. Otherwise fall back to static, rule-based messages.
    """

    def __init__(self):
        self._llm = None
        # Look for typical env var names for Google GenAI
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY")
        if ChatGoogleGenerativeAI and api_key:
            try:
                # Create a lightweight LangChain wrapper (if available)
                self._llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0.35,
                    max_output_tokens=120,
                )
            except Exception:
                # If anything fails, disable LLM use silently
                self._llm = None

    def _fallback_message(self, cue: str, movement: str, metrics: Optional[Dict[str, object]] = None) -> str:
        """Short static fallback that is always available."""
        metrics_text = _format_metrics(metrics)
        base = f"{movement}: {cue}. {metrics_text}."
        # Trim to one short sentence if too long
        if len(base) > 200:
            return base.split(".")[0] + "."
        return base

    def _build_prompt(self, cue: str, movement: str, metrics: Optional[Dict[str, object]]) -> str:
        metrics_text = _format_metrics(metrics)
        prompt = (
            "You are a concise, supportive biomechanics coach. Using the context below, "
            "write a single-sentence voice cue (under 25 words) that helps the user fix the issue.\n\n"
            "Context:\n" + "\n".join(KNOWLEDGE_BANK) + "\n\n"
            f"Movement: {movement}\nObserved issue: {cue}\nRecent metrics: {metrics_text}\n\n"
            "Return a single, plain English sentence."
        )
        return prompt

    def get_feedback(self, cue: str, movement: str, metrics: Optional[Dict[str, object]] = None) -> str:
        """
        Return a short (single-sentence) coaching cue.
        - If LLM is available it will be used; otherwise fallback is returned.
        """
        if not cue:
            cue = "Maintain good form"
        # If no LLM just return fallback
        if not self._llm:
            try:
                return self._fallback_message(cue, movement, metrics)
            except Exception:
                return f"{movement}: {cue}."

        prompt = self._build_prompt(cue, movement, metrics)
        try:
            # Different langchain adapters may return different structures.
            # Use a tolerant approach:
            result = self._llm.invoke(prompt)
            # result may have .content or be a string
            text = None
            if isinstance(result, str):
                text = result
            else:
                text = getattr(result, "content", None) or getattr(result, "text", None)
            if not text:
                # Some adapters return a list of message parts
                if isinstance(result, (list, tuple)):
                    text = " ".join(str(x) for x in result)
            if not text:
                return self._fallback_message(cue, movement, metrics)
            # Ensure short and single-sentence if possible
            text = str(text).strip()
            if "." in text:
                # pick first sentence
                text = text.split(".")[0].strip() + "."
            if len(text.split()) > 25:
                # truncate politely
                text = " ".join(text.split()[:25]) + "..."
            return text
        except Exception:
            # On any error, disable LLM and return fallback
            self._llm = None
            return self._fallback_message(cue, movement, metrics)

# -----------------------
# CoachVoice
# -----------------------
class CoachVoice:
    """
    Convert cue text to base64-encoded MP3 audio suitable for embedding in UI.
    Tries gTTS (online) first, then pyttsx3 (offline) fallback.
    """

    def __init__(self, language: str = "en", slow: bool = False):
        self.language = language
        self.slow = slow

    @lru_cache(maxsize=128)
    def synthesize(self, text: str) -> Optional[str]:
        """
        Convert text to base64-encoded MP3 bytes and return as ASCII string.
        Returns None on failure.
        """
        if not text:
            return None

        # 1) Try gTTS if available
        if gTTS:
            try:
                buf = BytesIO()
                tts = gTTS(text=text, lang=self.language, slow=self.slow)
                tts.write_to_fp(buf)
                buf.seek(0)
                audio_bytes = buf.read()
                if audio_bytes:
                    return base64.b64encode(audio_bytes).decode("ascii")
            except Exception:
                # fall through to pyttsx3
                pass

        # 2) Try pyttsx3 offline (writes to temp file)
        if pyttsx3:
            try:
                engine = pyttsx3.init()
                # Slightly lower rate for clarity
                try:
                    engine.setProperty("rate", 150)
                except Exception:
                    pass
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tf:
                    tmp_name = tf.name
                try:
                    engine.save_to_file(text, tmp_name)
                    engine.runAndWait()
                    with open(tmp_name, "rb") as fh:
                        audio_bytes = fh.read()
                    try:
                        os.remove(tmp_name)
                    except Exception:
                        pass
                    if audio_bytes:
                        return base64.b64encode(audio_bytes).decode("ascii")
                finally:
                    # ensure removal if anything left
                    if os.path.exists(tmp_name):
                        try:
                            os.remove(tmp_name)
                        except Exception:
                            pass
            except Exception:
                pass

        # If both backends unavailable / failed
        return None

    def audio_tag(self, audio_b64: Optional[str], autoplay: bool = False) -> Optional[str]:
        """
        Return an HTML audio tag with the base64 MP3 embedded.
        Set autoplay=True to auto-play in browsers that allow it.
        """
        if not audio_b64:
            return None
        autoplay_attr = "autoplay" if autoplay else ""
        # Use controls for safety so user can play/pause
        return (
            f"<audio controls {autoplay_attr}>"
            f"<source src='data:audio/mpeg;base64,{audio_b64}' type='audio/mpeg'>"
            "Your browser does not support the audio element."
            "</audio>"
        )


# -----------------------
# Simple demo / quick test when running module directly
# -----------------------
if __name__ == "__main__":
    cb = CoachBrain()
    cv = CoachVoice()
    sample = cb.get_feedback("Knees drifting inward on descent", "Squat", {"reps": 3, "angle": 95})
    print("Feedback:", sample)
    audio_b64 = cv.synthesize(sample or "Good job")
    print("Audio ok:", bool(audio_b64))

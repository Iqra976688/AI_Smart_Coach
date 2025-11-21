"""AI coaching brain and audio utilities for the Smart Coach prototype."""
from __future__ import annotations

import base64
import os
from functools import lru_cache
from io import BytesIO
from typing import Dict, Optional

from gtts import gTTS

try:  # Optional LangChain + Gemini stack
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover - graceful degradation when LangChain is unavailable
    ChatGoogleGenerativeAI = None  # type: ignore[assignment]


KNOWLEDGE_BANK = [
    "Keep joints stacked to protect ligaments during compound lifts.",
    "Maintain a neutral spine by bracing the core before every rep.",
    "Drive from the hips and keep knees tracking over toes in lower-body work.",
    "Balance tempo: controlled eccentric, powerful concentric for strength movements.",
    "Use steady nasal breathing during physiotherapy drills to avoid bracing.",
    "Explosive sports motions need a stable base - focus on balance before speed.",
]


def _format_metrics(metrics: Optional[Dict[str, object]]) -> str:
    if not metrics:
        return "Metrics not available."
    angle = metrics.get("angle")
    tempo_state = metrics.get("tempo_state")
    reps = metrics.get("reps")
    rom = metrics.get("rom_last") or metrics.get("rom_current")
    if isinstance(angle, (int, float)) and isinstance(rom, (int, float)):
        return (
            "angle={:.1f}deg, tempo_state={}, reps={}, range_of_motion={:.1f}".format(
                angle, tempo_state, reps, rom
            )
        )
    return "tempo_state={}, reps={}, range_of_motion={}".format(
        tempo_state, reps, rom
    )


class CoachBrain:
    """Generates encouraging cues using Gemini when available, with fallbacks."""

    def __init__(self) -> None:
        self._llm = None
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY")
        if ChatGoogleGenerativeAI and api_key:
            try:
                self._llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0.4,
                    max_output_tokens=128,
                )
            except Exception:
                self._llm = None

    def get_feedback(
        self,
        cue: str,
        movement: str,
        metrics: Optional[Dict[str, object]] = None,
    ) -> str:
        """Return a single-sentence coaching cue synthesised from context."""
        base_message = (
            f"{movement}: {cue}. Maintain control and keep stacking joints."
        )
        context = "\n".join(KNOWLEDGE_BANK)
        metrics_text = _format_metrics(metrics)

        if not self._llm:
            return base_message

        prompt = (
            "You are a supportive biomechanics coach. Use the context below to craft a "
            "concise, encouraging voice cue for the athlete. Keep it under 25 words.\n\n"
            f"Context:\n{context}\n\n"
            f"Movement: {movement}\n"
            f"Observed issue: {cue}\n"
            f"Recent metrics: {metrics_text}\n\n"
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
        if isinstance(text, list):  # Gemini may return a list of parts
            text = " ".join(str(part) for part in text)
        text = str(text).strip()
        return text if text else base_message


class CoachVoice:
    """Turns coaching text into an embeddable audio payload."""

    def __init__(self, language: str = "en", slow: bool = False) -> None:
        self.language = language
        self.slow = slow

    @lru_cache(maxsize=64)
    def synthesize(self, text: str) -> Optional[str]:
        if not text:
            return None
        try:
            buffer = BytesIO()
            tts = gTTS(text=text, lang=self.language, slow=self.slow)
            tts.write_to_fp(buffer)
            buffer.seek(0)
        except Exception:
            return None
        audio_bytes = buffer.read()
        return base64.b64encode(audio_bytes).decode("ascii") if audio_bytes else None

    def audio_tag(self, audio_b64: Optional[str], autoplay: bool = True) -> Optional[str]:
        if not audio_b64:
            return None
        autoplay_attr = "autoplay" if autoplay else ""
        return (
            f"<audio controls {autoplay_attr}>"
            f"<source src='data:audio/mp3;base64,{audio_b64}' type='audio/mpeg'>"
            "Your browser does not support the audio element."
            "</audio>"
        )

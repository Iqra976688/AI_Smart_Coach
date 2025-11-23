"""
RAG chain for AI Smart Coach

Features:
- Semantic search over JSON dataset using Sentence Transformers + FAISS
- Feedback generation via LLM (Google Gemini)
- Audio synthesis (gTTS / pyttsx3)
- Structured CoachReport
"""

import os
import json
import base64
from typing import Optional, Dict, List

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from gtts import gTTS
import pyttsx3

# Load LLM
import openai
import os


# -------------------------------
# CoachReport
# -------------------------------
class CoachReport:
    def __init__(self, cue: str, metrics: Optional[Dict] = None):
        self.cue = cue
        self.metrics = metrics or {}

    def to_dict(self):
        return {"cue": self.cue, "metrics": self.metrics}

# -------------------------------
# CoachVoice
# -------------------------------
class CoachVoice:
    def __init__(self, tts_engine: str = "pyttsx3"):
        self.tts_engine = tts_engine
        if tts_engine == "pyttsx3":
            self.engine = pyttsx3.init()
        else:
            self.engine = None

    def synthesize(self, text: str) -> Optional[str]:
        if not text:
            return None
        try:
            if self.tts_engine == "gTTS":
                tts = gTTS(text)
                tmp_file = "temp_audio.mp3"
                tts.save(tmp_file)
                with open(tmp_file, "rb") as f:
                    audio_bytes = f.read()
                os.remove(tmp_file)
            else:
                # pyttsx3 fallback
                tmp_file = "temp_audio.wav"
                self.engine.save_to_file(text, tmp_file)
                self.engine.runAndWait()
                with open(tmp_file, "rb") as f:
                    audio_bytes = f.read()
                os.remove(tmp_file)
            return base64.b64encode(audio_bytes).decode("utf-8")
        except Exception:
            return None

    def audio_tag(self, b64_audio: str):
        if not b64_audio:
            return ""
        return f"<audio autoplay><source src='data:audio/mp3;base64,{b64_audio}' type='audio/mpeg'></audio>"

# -------------------------------
# CoachBrain (RAG)
# -------------------------------
class CoachBrain:
    def __init__(self, dataset_path: str = "dataset.json", top_k: int = 3, model_name: str = "gpt-4.1"):
        self.dataset_path = dataset_path
        self.top_k = top_k
        self.entries: List[Dict] = []
        self.texts: List[str] = []
        self.embeddings: np.ndarray = None
        self.index: Optional[faiss.IndexFlatL2] = None
        self.model_name = model_name

        # Load LLM API key from env
        openai.api_key = os.getenv("OPENAI_API_KEY")


        self._load_dataset()
        self._build_faiss_index()

    # -------------------------------
    # Load JSON
    # -------------------------------
    def _load_dataset(self):
        if not os.path.exists(self.dataset_path):
            print(f"Dataset file {self.dataset_path} not found")
            return
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            self.entries = json.load(f)
        self.texts = []
        for entry in self.entries:
            if isinstance(entry, dict):
                self.texts.append(entry.get("correction", "") + " " + entry.get("mistake", ""))
            else:
                self.texts.append(str(entry))


    # -------------------------------
    # Build embeddings & FAISS
    # -------------------------------
    def _build_faiss_index(self):
        if not self.texts:
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        except NotImplementedError:
            # Some CPU builds lack support for bf16/half; fall back to full precision on CPU
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self.embeddings = self.embedder.encode(self.texts, convert_to_numpy=True, normalize_embeddings=True)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    # -------------------------------
    # Retrieve top-k context
    # -------------------------------
    def _retrieve_context(self, query: str) -> List[str]:
        if self.index is None:
            return []
        
        # Encode the query
        q_emb = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        
        # Search in FAISS index
        D, I = self.index.search(q_emb, k=min(self.top_k, len(self.entries)))
        
        results = []
        for idx in I[0]:
            if idx < len(self.entries):
                entry = self.entries[idx]
                # Check if entry is a dict or string
                if isinstance(entry, dict):
                    mistake = entry.get("mistake", "")
                    correction = entry.get("correction", "")
                else:
                    mistake = str(entry)
                    correction = ""
                results.append(f"Mistake: {mistake} | Correction: {correction}")
        return results


    # -------------------------------
    # Generate feedback via LLM
    # -------------------------------
    def get_feedback(self, cue: str, exercise_name: str, metrics: Optional[Dict] = None) -> str:
        context_list = self._retrieve_context(cue)
        context_text = "\n".join(context_list)
        prompt = f"""
You are a real-time AI exercise coach.
Exercise: {exercise_name}
Metrics: {metrics or {}}
User cue: {cue}

Use the following context for corrections and guidance:
{context_text}

Provide actionable feedback in a concise and friendly tone.
"""
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
                )
            return resp['choices'][0]['message']['content']
        except Exception as e:
            print("LLM error:", e)
            return cue + (" | " + context_text if context_text else "")

# -------------------------------
# Usage example
# -------------------------------
if __name__ == "__main__":
    brain = CoachBrain(dataset_path="dataset.json")
    voice = CoachVoice()
    cue = "Knees drifting inward on squat descent"
    metrics = {"reps": 5, "angle": 95, "tempo_last": 1.2}
    feedback = brain.get_feedback(cue, "Squat", metrics)
    print("Feedback:", feedback)
    audio_b64 = voice.synthesize(feedback)
    if audio_b64:
        print("Audio length (bytes):", len(base64.b64decode(audio_b64)))

import os
import json
import asyncio
import base64
from datetime import datetime, timedelta
from threading import Thread

import cv2
import numpy as np
import streamlit as st

from modules.pose_logic import PoseEstimator
from modules.rag_chain import CoachBrain, CoachVoice
from modules.agentic_ai import AgenticCoach  # Optional AI agent

# -----------------------------

# Load exercise dataset

# -----------------------------

DATASET_FILE = os.path.join(os.path.dirname(**file**), "dataset.json")
with open(DATASET_FILE, "r", encoding="utf-8") as f:
EXERCISE_DATA = [json.loads(line) for line in f if line.strip()]

# -----------------------------

# Initialize components

# -----------------------------

brain = CoachBrain()      # LLM feedback
voice = CoachVoice()      # TTS feedback
pose_estimator = PoseEstimator()  # Pose detection and landmarks overlay
agentic = AgenticCoach()  # Optional Agentic AI

# -----------------------------

# Streamlit session state

# -----------------------------

if "streak" not in st.session_state:
st.session_state.streak = 0
if "last_reminder" not in st.session_state:
st.session_state.last_reminder = datetime.now() - timedelta(days=1)

# -----------------------------

# Video capture caching

# -----------------------------

@st.cache_resource
def get_video_capture():
return cv2.VideoCapture(0)

# -----------------------------

# Process a single video frame

# -----------------------------

def process_frame(frame):
"""Detect pose and overlay bone landmarks."""
landmarks, annotated_frame = pose_estimator.detect(frame)
return landmarks, annotated_frame

# -----------------------------

# Feedback generation

# -----------------------------

async def feedback_loop(landmarks, current_exercise="squat"):
"""
Generate coaching feedback based on pose landmarks and exercise data.
Returns feedback text and base64 audio.
"""
# Default feedback
cue = "Good form"

```
# Check dataset for current exercise
exercise_data = next((ex for ex in EXERCISE_DATA if ex["exercise"] == current_exercise), None)
if exercise_data:
    # Example: randomly select a mistake for demonstration
    if np.random.rand() < 0.3:
        cue = exercise_data["common_mistakes"][0]

# LLM feedback
feedback_text = brain.get_feedback(cue, current_exercise)
audio_b64 = voice.synthesize(feedback_text)
return feedback_text, audio_b64
```

# -----------------------------

# Video processing thread

# -----------------------------

def video_thread():
cap = get_video_capture()
placeholder = st.empty()
current_exercise = "squat"

```
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    landmarks, annotated_frame = process_frame(frame)

    # Async feedback
    feedback_text, audio_b64 = asyncio.run(feedback_loop(landmarks, current_exercise))

    # Show video with bone overlay
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)

    # Show textual feedback
    st.text(f"Feedback: {feedback_text}")

    # Show audio feedback
    audio_tag = voice.audio_tag(audio_b64)
    if audio_tag:
        st.markdown(audio_tag, unsafe_allow_html=True)

    # Update streak
    st.session_state.streak += 1

    # Reminders every 5 minutes
    if datetime.now() - st.session_state.last_reminder > timedelta(minutes=5):
        st.warning("Take a short break or drink water!")
        st.session_state.last_reminder = datetime.now()
```

# -----------------------------

# Main Streamlit app

# -----------------------------

def main():
st.title("AI Smart Coach")
st.text("Real-time coaching with pose detection, LLM feedback, and voice cues.")
st.text(f"Current streak: {st.session_state.streak}")

```
# Start video in separate thread for smooth async
thread = Thread(target=video_thread, daemon=True)
thread.start()

# Optional: Agentic AI interface
st.subheader("Agentic Coach")
user_input = st.text_input("Ask your coach anything:")
if user_input:
    try:
        response = agentic.query(user_input)
        st.text_area("Agentic Response:", value=response, height=150)
    except Exception:
        st.text_area("Agentic Response:", value="AI agent is unavailable.", height=150)
```

if **name** == "**main**":
main()

#!/usr/bin/env python3
# app.py — Full AI Smart Coach (Streamlit)  

import os
import queue
import textwrap
import time
from typing import Dict, Optional, Tuple

import av
import cv2
import numpy as np
import mediapipe as mp
import streamlit as st

# streamlit-webrtc import
try:
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
except Exception:
    VideoProcessorBase = object
    WebRtcMode = None
    def webrtc_streamer(*args, **kwargs):
        raise RuntimeError("streamlit-webrtc is required. Install with `pip install streamlit-webrtc`.")

# Project modules
from pose_logic import get_profile, list_categories, list_profiles_by_category, FOCUS_AREAS, ExerciseCounter, calculate_angle, generate_coaching_report, CoachReport
from rag_chain import CoachBrain, CoachVoice

# Optional Agentic AI
try:
    from agentic_ai import AgenticCoach
except Exception:
    class AgenticCoach:
        def __init__(self, *a, **k): pass
        def enhance_feedback(self, cue, movement, metrics=None, user_id=None): return ""

# ---- Configuration ----
ICE_SERVERS = [{"urls": ["stun:stun.l.google.com:19302"]}]
FEEDBACK_DEBOUNCE_SEC = 5.0

# ---- Services ----
coach_brain = CoachBrain()
coach_voice = CoachVoice()
agentic_coach = AgenticCoach()

# ---- Streamlit helpers ----
def _trigger_rerun():
    rerun_fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun_fn: rerun_fn()

# ---- Helpers ----
from mediapipe.framework.formats import landmark_pb2

def _landmark_to_point(landmarks: Optional[landmark_pb2.NormalizedLandmarkList], index: int):
    if not landmarks: return None
    try: return (landmarks.landmark[index].x, landmarks.landmark[index].y)
    except Exception: return None

def _world_landmark_to_point(landmarks: Optional[landmark_pb2.LandmarkList], index: int):
    if not landmarks: return None
    try: return (landmarks.landmark[index].x, landmarks.landmark[index].y)
    except Exception: return None

# ---- Video Processor ----
class PoseProcessor(VideoProcessorBase):
    def __init__(self, feedback_queue: queue.Queue, profile_key: str, focus_area: str):
        self._pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.6, model_complexity=1)
        self._drawing = mp.solutions.drawing_utils
        self._drawing_styles = mp.solutions.drawing_styles
        self.feedback_queue = feedback_queue
        self.profile = get_profile(profile_key)
        self.focus_area = focus_area
        self.counter = ExerciseCounter(self.profile)
        self.latest_metrics: Optional[Dict] = None
        self.latest_report: Optional[CoachReport] = None
        self.latest_hud: list[str] = []
        self._last_feedback_time = 0.0
        self._last_headline: Optional[str] = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        overlay_image = image.copy()
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = None
        try: results = self._pose.process(rgb_image)
        except Exception: results = None

        normalized_landmarks = getattr(results, "pose_landmarks", None)
        world_landmarks = getattr(results, "pose_world_landmarks", None)

        if normalized_landmarks:
            try:
                self._drawing.draw_landmarks(overlay_image, normalized_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                                             landmark_drawing_spec=self._drawing_styles.get_default_pose_landmarks_style())
            except Exception: pass

        # ---- Compute metrics ----
        angle_val = None
        rom_val = None
        reps_val = 0
        stage_val = "—"
        tempo_val = None
        active_side = None

        # Example: use first three landmarks for demo; integrate your triplets logic here
        points = []
        if normalized_landmarks:
            for i in range(3):
                pt = _landmark_to_point(normalized_landmarks, i)
                if pt: points.append(pt)
        if len(points) == 3:
            angle_val = calculate_angle(points[0], points[1], points[2])
            metrics = self.counter.update(angle_val)
            reps_val = int(metrics.get("reps", 0))
            stage_val = metrics.get("stage", "—")
            rom_val = metrics.get("rom_current", None)
            tempo_val = metrics.get("tempo_last", None)
            active_side = metrics.get("active_side", None)
            self.latest_metrics = metrics
            try:
                report = generate_coaching_report(self.profile, metrics, normalized_landmarks, world_landmarks, self.focus_area)
                self.latest_report = report
            except Exception:
                report = CoachReport()
                self.latest_report = report

            headline = getattr(report, "headline", lambda: None)()
            if headline and (headline != self._last_headline or time.time() - self._last_feedback_time > FEEDBACK_DEBOUNCE_SEC):
                payload = {"cue": headline, "profile_name": getattr(self.profile, "name", ""), "metrics": metrics, "timestamp": time.time()}
                try: self.feedback_queue.put_nowait(payload)
                except queue.Full: pass
                self._last_headline = headline
                self._last_feedback_time = time.time()

        # ---- Build HUD ----
        lines = [f"Movement: {getattr(self.profile, 'name', '—')}",
                 f"Reps: {reps_val} | Stage: {stage_val}"]
        lines.append(f"Angle: {angle_val:.1f}°" if angle_val else "Angle: —")
        lines.append(f"ROM: {rom_val:.1f}°" if rom_val else "ROM: —")
        lines.append(f"Tempo: {tempo_val:.2f}s" if tempo_val else "Tempo: —")
        if active_side:
            lines.append(f"Side: {active_side}")
        self.latest_hud = lines

        base_y = 40
        for line in self.latest_hud:
            cv2.putText(overlay_image, line, (21, base_y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(overlay_image, line, (20, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            base_y += 32

        return av.VideoFrame.from_ndarray(overlay_image, format="bgr24")

# ---- Main Streamlit App ----
def main():
    st.set_page_config(page_title="AI Smart Coach", layout="wide")
    st.title("AI Smart Coach — Real-Time Form Assistant")

    categories = list_categories() or ["Gym Coaching"]
    selected_category = st.sidebar.selectbox("Mode", categories)
    profiles = list_profiles_by_category(selected_category) or [get_profile("bicep_curl")]
    profile_map = {p.name: p for p in profiles}
    profile_name = st.sidebar.selectbox("Movement", list(profile_map.keys()))
    selected_profile = profile_map[profile_name]
    focus_area = st.sidebar.selectbox("Focus area", FOCUS_AREAS)

    # Session state
    if "feedback_queue" not in st.session_state: st.session_state["feedback_queue"] = queue.Queue(maxsize=8)
    if "streak_days" not in st.session_state: st.session_state["streak_days"] = 0
    if "reminders_done" not in st.session_state: st.session_state["reminders_done"] = 0
    if "user_id" not in st.session_state: st.session_state["user_id"] = "user_local_1"

    feedback_queue = st.session_state["feedback_queue"]

    video_col, stats_col = st.columns([3,2])
    with video_col:
        st.subheader("Live Camera Feed")
        try:
            webrtc_ctx = webrtc_streamer(
                key="coach",
                mode=WebRtcMode.SENDRECV if WebRtcMode else None,
                rtc_configuration={"iceServers": ICE_SERVERS},
                media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
                video_processor_factory=lambda: PoseProcessor(feedback_queue, selected_profile.key, focus_area),
            )
        except Exception as e:
            st.error(f"Unable to start webcam: {e}")
            webrtc_ctx = None

    with stats_col:
        st.subheader("Session Metrics / Feedback")
        # Collect feedback payloads
        feedback_payload = None
        try:
            while feedback_queue and not feedback_queue.empty():
                n = feedback_queue.get_nowait()
                if isinstance(n, dict):
                    feedback_payload = n
        except Exception: pass

        metrics = feedback_payload.get("metrics") if feedback_payload else {}
        report = getattr(feedback_payload, "report", None) or CoachReport()

        st.metric("Repetitions", int(metrics.get("reps", 0)))
        st.metric("Stage", metrics.get("stage", "—"))
        st.metric("Angle", f"{metrics.get('angle', '—'):.1f}" if isinstance(metrics.get("angle"), (int,float)) else "—")
        st.metric("ROM", f"{metrics.get('rom_current', '—'):.1f}" if isinstance(metrics.get("rom_current"), (int,float)) else "—")
        st.metric("Tempo (last)", f"{metrics.get('tempo_last', '—'):.2f}" if isinstance(metrics.get("tempo_last"), (int,float)) else "—")

        # RAG/Agentic feedback
        guidance_block = ""
        audio_tag_html = None
        if feedback_payload:
            cue_text = feedback_payload.get("cue", "Keep your form tight")
            try:
                cue_body = coach_brain.get_feedback(cue_text, selected_profile.name, metrics)
            except Exception:
                cue_body = cue_text
            try:
                agentic_feedback = agentic_coach.enhance_feedback(cue_text, selected_profile.name, metrics, user_id=st.session_state["user_id"])
            except Exception:
                agentic_feedback = ""
            combined_feedback = cue_body
            if agentic_feedback:
                combined_feedback += f"\n\nAgentic: {agentic_feedback}"
            guidance_block = combined_feedback
            try:
                audio_b64 = coach_voice.synthesize(combined_feedback)
                audio_tag_html = coach_voice.audio_tag(audio_b64) if audio_b64 else None
            except Exception: audio_tag_html = None

        if guidance_block:
            st.markdown("**Coach Feedback**")
            st.markdown(f"> {guidance_block}")
            if audio_tag_html:
                st.markdown(audio_tag_html, unsafe_allow_html=True)

        # Streak / reminders
        st.session_state["reminders_done"] += 1
        if st.session_state["reminders_done"] % 10 == 0:
            st.session_state["streak_days"] += 1
        st.markdown(f"**Streak Days:** {st.session_state['streak_days']}  |  Reminders Completed: {st.session_state['reminders_done']}")

    # Auto-refresh
    if webrtc_ctx and getattr(webrtc_ctx, "state", None) and getattr(webrtc_ctx.state, "playing", False):
        now_ts = time.time()
        last_refresh = st.session_state.get("_last_metrics_refresh", 0.0)
        if now_ts - last_refresh > 0.6:
            st.session_state["_last_metrics_refresh"] = now_ts
            _trigger_rerun()

if __name__ == "__main__":
    main()

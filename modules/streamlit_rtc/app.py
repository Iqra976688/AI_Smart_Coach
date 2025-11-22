
#"""Streamlit interface for the Real-Time AI Smart Coach prototype with reminders, streak tracking, and Agentic AI."""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import queue
import textwrap
import time
from typing import Dict, Optional, Tuple

import av
import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from pose_logic import (
    CoachReport,
    ExerciseCounter,
    ExerciseProfile,
    FOCUS_AREAS,
    calculate_angle,
    generate_coaching_report,
    get_profile,
    list_categories,
    list_profiles_by_category,
)
from mediapipe.framework.formats import landmark_pb2
from rag_chain import CoachBrain, CoachVoice
from agentic_ai import AgenticCoach  # <-- Agentic AI module

# ---- WebRTC Configuration ----
ICE_SERVERS = [{"urls": ["stun:stun.l.google.com:19302"]}]
FEEDBACK_DEBOUNCE_SEC = 5.0
TRIPLET_SWITCH_MARGIN = 6.0

coach_brain = CoachBrain()
coach_voice = CoachVoice()
agentic_coach = AgenticCoach()  # Initialize Agentic AI

# ---- Streamlit rerun helper ----
def _trigger_rerun() -> None:
    rerun_fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun_fn:
        rerun_fn()
        return
    try:
        from streamlit.runtime.scriptrunner import RerunException, RerunData  # type: ignore
    except Exception:
        try:
            from streamlit.script_runner import RerunException, RerunData  # type: ignore
        except Exception:
            return
    raise RerunException(RerunData())

# ---- Helpers to extract landmarks ----
def _landmark_to_point(landmarks: Optional[landmark_pb2.NormalizedLandmarkList], index: int) -> Optional[tuple[float, float]]:
    if not landmarks:
        return None
    try:
        landmark = landmarks.landmark[index]
    except IndexError:
        return None
    return (landmark.x, landmark.y)

def _world_landmark_to_point(landmarks: Optional[landmark_pb2.LandmarkList], index: int) -> Optional[tuple[float, float]]:
    if not landmarks:
        return None
    try:
        landmark = landmarks.landmark[index]
    except IndexError:
        return None
    return (landmark.x, landmark.y)

# ---- Video processor ----
class PoseProcessor(VideoProcessorBase):
    """Runs MediaPipe pose tracking and overlays coaching data."""
    def __init__(self, feedback_queue: queue.Queue[object], profile_key: str, focus_area: str):
        self._pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.6, model_complexity=1)
        self._drawing = mp.solutions.drawing_utils
        self._drawing_styles = mp.solutions.drawing_styles
        self.feedback_queue = feedback_queue
        self.profile_key: str = ""
        self.profile: ExerciseProfile = get_profile(profile_key)
        self.active_triplet: Tuple[int, int, int] = self.profile.primary_triplet
        self.focus_area = focus_area
        self.counter = ExerciseCounter(self.profile)
        self.latest_metrics: Optional[Dict[str, object]] = None
        self.latest_report: Optional[CoachReport] = None
        self.latest_hud: list[str] = []
        self.latest_overlay_frame: Optional[np.ndarray] = None
        self._last_feedback_time = 0.0
        self._last_headline: Optional[str] = None
        self.set_profile(profile_key, focus_area)

    def __del__(self):
        self._pose.close()

    def set_profile(self, profile_key: str, focus_area: str) -> None:
        if profile_key != self.profile_key:
            self.profile_key = profile_key
            self.profile = get_profile(profile_key)
            self.active_triplet = self.profile.primary_triplet
            self.counter = ExerciseCounter(self.profile)
            self.latest_metrics = None
            self.latest_report = None
        self.focus_area = focus_area

    def _extract_triplet_points(self, triplet: Tuple[int, int, int], normalized_landmarks: Optional[landmark_pb2.NormalizedLandmarkList], world_landmarks: Optional[landmark_pb2.LandmarkList]) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        for idx in triplet:
            point = _world_landmark_to_point(world_landmarks, idx)
            if point is None:
                point = _landmark_to_point(normalized_landmarks, idx)
            if point is None:
                return []
            points.append(point)
        return points

    def _active_side_label(self) -> Optional[str]:
        if not self.profile.alternate_triplet:
            return None
        return "Right" if self.active_triplet == self.profile.alternate_triplet else "Left"

    def _draw_overlay_text(self, overlay_image: np.ndarray) -> None:
        lines = self.latest_hud or ["Waiting for movement data…"]
        base_y = 40
        for line in lines:
            cv2.putText(overlay_image, line, (21, base_y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(overlay_image, line, (20, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            base_y += 32

    def _build_stats_panel(self, metrics: Optional[Dict[str, object]], report: Optional[CoachReport], height: int, width: int) -> np.ndarray:
        panel = np.full((height, width, 3), (24, 24, 24), dtype=np.uint8)
        data = metrics or {}

        def put_line(text: str, y: int, *, color=(255, 255, 255), scale: float = 0.7) -> int:
            cv2.putText(panel, text, (28, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), max(1, int(4 * scale)), cv2.LINE_AA)
            cv2.putText(panel, text, (26, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, max(1, int(2 * scale)), cv2.LINE_AA)
            return y + int(34 * scale)

        angle_val = data.get("angle")
        rom_val = data.get("rom_last") or data.get("rom_current")
        tempo_val = data.get("tempo_last")
        tempo_state = data.get("tempo_state", "steady")
        active_side = data.get("active_side")
        reps = int(data.get("reps", 0))
        stage = data.get("stage", "—")

        y_pos = 60
        y_pos = put_line("AI Smart Coach", y_pos, color=(0, 200, 255), scale=0.9)
        y_pos = put_line(f"Movement: {self.profile.name}", y_pos)
        if active_side:
            y_pos = put_line(f"Side: {active_side}", y_pos)
        y_pos = put_line(f"Reps: {reps}  |  Stage: {stage}", y_pos)
        y_pos = put_line(f"Angle: {angle_val:.1f}° | ROM: {rom_val:.1f}°" if angle_val and rom_val else "Angle: — | ROM: —", y_pos)
        y_pos = put_line(f"Tempo: {tempo_val:.2f}s ({tempo_state})" if tempo_val else "Tempo: —", y_pos)

        if report and report.primary_cues:
            y_pos = put_line("Cues:", y_pos, color=(50, 205, 50))
            for cue in report.primary_cues[:4]:
                for wrapped in textwrap.wrap(cue, 32):
                    y_pos = put_line(f"- {wrapped}", y_pos, color=(220, 220, 220), scale=0.6)
        return panel

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        overlay_image = image.copy()
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb_image)
        normalized_landmarks = results.pose_landmarks
        world_landmarks = results.pose_world_landmarks

        if normalized_landmarks:
            self._drawing.draw_landmarks(overlay_image, normalized_landmarks, mp.solutions.pose.POSE_CONNECTIONS, landmark_drawing_spec=self._drawing_styles.get_default_pose_landmarks_style())

        metrics: Optional[Dict[str, object]] = None
        report: Optional[CoachReport] = None
        observations: list[tuple[Tuple[int, int, int], float]] = []

        triplets = [self.active_triplet, self.profile.primary_triplet, self.profile.alternate_triplet]
        seen_triplets = set()
        for triplet in triplets:
            if triplet and triplet not in seen_triplets:
                seen_triplets.add(triplet)
                points = self._extract_triplet_points(triplet, normalized_landmarks, world_landmarks)
                if len(points) == 3:
                    angle_val = calculate_angle(points[0], points[1], points[2])
                    observations.append((triplet, angle_val))

        if observations:
            chosen_triplet, chosen_angle = observations[0]
            self.active_triplet = chosen_triplet
            metrics = self.counter.update(chosen_angle)
            metrics["profile"] = self.profile.key
            metrics["profile_name"] = self.profile.name
            active_side = self._active_side_label()
            if active_side:
                metrics["active_side"] = active_side
            self.latest_metrics = metrics
            report = generate_coaching_report(self.profile, metrics, normalized_landmarks, world_landmarks, self.focus_area)
            self.latest_report = report

            headline = report.headline() if report else None
            if headline and (headline != self._last_headline or time.time() - self._last_feedback_time > FEEDBACK_DEBOUNCE_SEC):
                payload = {"cue": headline, "profile_key": self.profile.key, "profile_name": self.profile.name, "metrics": metrics, "report": report.to_dict() if report else {}, "timestamp": time.time()}
                if active_side:
                    payload["active_side"] = active_side
                try:
                    self.feedback_queue.put_nowait(payload)
                except queue.Full:
                    pass
                self._last_headline = headline
                self._last_feedback_time = time.time()

            self.latest_hud = [f"Movement: {self.profile.name}", f"Cue: {headline}"] if headline else []

        self._draw_overlay_text(overlay_image)
        height, width, _ = overlay_image.shape
        stats_panel = self._build_stats_panel(metrics, report, height, width)
        combined = np.zeros((height, width * 2, 3), dtype=np.uint8)
        combined[:, :width] = overlay_image
        combined[:, width:] = stats_panel
        self.latest_overlay_frame = combined
        return av.VideoFrame.from_ndarray(combined, format="bgr24")

# ---- Main Streamlit app ----
def main() -> None:
    st.set_page_config(page_title="AI Smart Coach", page_icon="🏋", layout="wide")
    st.title("AI Smart Coach — Real-Time Form Assistant")

    st.sidebar.header("Session Controls")
    categories = list_categories()
    default_category = "Gym Coaching" if "Gym Coaching" in categories else categories[0]
    selected_category = st.sidebar.selectbox("Mode", categories, index=categories.index(default_category))
    profiles = list_profiles_by_category(selected_category)
    if not profiles:
        profiles = [get_profile("bicep_curl")]
    profile_map = {p.name: p for p in profiles}
    profile_name = st.sidebar.selectbox("Movement", list(profile_map.keys()))
    selected_profile = profile_map[profile_name]
    focus_area = st.sidebar.selectbox("Focus area (optional)", FOCUS_AREAS)
    st.sidebar.markdown(f"**Description:** {selected_profile.description}")

    if "feedback_queue" not in st.session_state:
        st.session_state["feedback_queue"] = queue.Queue(maxsize=8)
    if "reminders_done" not in st.session_state:
        st.session_state["reminders_done"] = 0
    if "streak_days" not in st.session_state:
        st.session_state["streak_days"] = 0

    feedback_queue: queue.Queue[object] = st.session_state["feedback_queue"]

    video_col, stats_col = st.columns([3, 2])
    with video_col:
        st.subheader("Live Camera Feed")
        webrtc_ctx = webrtc_streamer(
            key="coach",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": ICE_SERVERS},
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            async_processing=True,
            video_processor_factory=lambda: PoseProcessor(feedback_queue, selected_profile.key, focus_area),
        )

    metrics: Dict[str, object] = {}
    report = CoachReport()
    if webrtc_ctx.video_processor:
        processor: PoseProcessor = webrtc_ctx.video_processor
        processor.set_profile(selected_profile.key, focus_area)
        metrics = processor.latest_metrics or {}
        report = processor.latest_report or CoachReport()

    feedback_payload: Optional[dict[str, object]] = None
    while not feedback_queue.empty():
        next_item = feedback_queue.get_nowait()
        if isinstance(next_item, dict):
            feedback_payload = next_item

    guidance_block = None
    audio_tag_html = None
    if feedback_payload:
        cue_text = str(feedback_payload.get("cue", "")).strip()
        metrics_for_ai = feedback_payload.get("metrics")
        cue_body = coach_brain.get_feedback(cue_text or "Keep your form tight", selected_profile.name, metrics_for_ai)
        agentic_feedback = agentic_coach.enhance_feedback(cue_text or "Keep your form tight", selected_profile.name, metrics_for_ai)
        cue_body += f"\n\nAgentic AI: {agentic_feedback}"
        audio_b64 = coach_voice.synthesize(cue_body)
        audio_tag_html = coach_voice.audio_tag(audio_b64)
        guidance_block = cue_body
        st.session_state["reminders_done"] += 1
        if st.session_state["reminders_done"] % 10 == 0:
            st.session_state["streak_days"] += 1

    with stats_col:
        st.subheader("Session Metrics")
        st.metric("Repetitions", int(metrics.get("reps", 0)))
        stage_value = metrics.get("stage", "—")
        active_side = metrics.get("active_side")
        if active_side:
            stage_value = f"{stage_value} ({active_side})"
        st.metric("Stage", stage_value)
        st.metric("Joint Angle", f"{metrics.get('angle', '—'):.1f}" if metrics.get('angle') else "—")
        rom_value = metrics.get("rom_last") or metrics.get("rom_current")
        st.metric("Range of Motion", f"{rom_value:.1f}" if rom_value else "—")
        if guidance_block:
            st.markdown("**Coach Feedback**")
            st.markdown(f"> {guidance_block}")
            if audio_tag_html:
                st.markdown(audio_tag_html, unsafe_allow_html=True)
        st.markdown(f"**Streak Days:** {st.session_state['streak_days']} | Reminders Completed: {st.session_state['reminders_done']}")

    if webrtc_ctx and webrtc_ctx.state.playing:
        now_ts = time.time()
        last_refresh = st.session_state.get("_last_metrics_refresh", 0.0)
        if now_ts - last_refresh > 0.6:
            st.session_state["_last_metrics_refresh"] = now_ts
            _trigger_rerun()
    else:
        st.session_state.pop("_last_metrics_refresh", None)

if __name__ == "__main__":
    main()
```

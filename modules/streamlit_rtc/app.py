"""app.py — Streamlit interface for the Real-Time AI Smart Coach prototype.

This updated file:
- Integrates MediaPipe pose tracking via streamlit-webrtc
- Uses the project's pose_logic for metrics & reports
- Uses rag_chain.CoachBrain / CoachVoice for RAG + TTS (safe fallbacks)
- Uses agentic_ai.AgenticCoach for agentic decisions/enhancements (safe fallbacks)
- Adds reminders & streak tracking in session_state
- Includes defensive checks to avoid runtime crashes during demos
- Sets async_processing=False to reduce webcam glitching in VS Code
"""

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

# streamlit-webrtc is optional — handle import error gracefully
try:
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
except Exception:
    VideoProcessorBase = object  # fallback type so file imports cleanly
    WebRtcMode = None
    def webrtc_streamer(*args, **kwargs):  # type: ignore
        raise RuntimeError("streamlit-webrtc not available. Install streamlit-webrtc to enable live webcam.")

# project modules (must exist in repo)
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

# rag + agentic modules (they exist in your repo)
from rag_chain import CoachBrain, CoachVoice
from agentic_ai import AgenticCoach

# ---- Config ----
ICE_SERVERS = [{"urls": ["stun:stun.l.google.com:19302"]}]
FEEDBACK_DEBOUNCE_SEC = 5.0
TRIPLET_SWITCH_MARGIN = 6.0

# ---- Services ----
coach_brain = CoachBrain()    # RAG / LLM fallback internal logic
coach_voice = CoachVoice()    # TTS (gTTS) wrapper
agentic_coach = AgenticCoach()  # Agentic decision-maker

# ---- Utilities ----
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


def _landmark_to_point(landmarks: Optional[landmark_pb2.NormalizedLandmarkList], index: int) -> Optional[tuple[float, float]]:
    if not landmarks:
        return None
    try:
        landmark = landmarks.landmark[index]
    except Exception:
        return None
    return (landmark.x, landmark.y)


def _world_landmark_to_point(landmarks: Optional[landmark_pb2.LandmarkList], index: int) -> Optional[tuple[float, float]]:
    if not landmarks:
        return None
    try:
        landmark = landmarks.landmark[index]
    except Exception:
        return None
    return (landmark.x, landmark.y)


def _format_angle(value: Optional[object]) -> str:
    return f"{value:.1f}°" if isinstance(value, (int, float)) else "—"


def _format_seconds(value: Optional[object]) -> str:
    return f"{value:.2f}s" if isinstance(value, (int, float)) else "—"


def _format_ratio(value: Optional[object]) -> str:
    try:
        return f"{value * 100:.0f}%" if isinstance(value, (int, float)) else "—"
    except Exception:
        return "—"


# ---- Video processor ----
class PoseProcessor(VideoProcessorBase):
    """Runs MediaPipe pose tracking and overlays coaching data."""

    def __init__(self, feedback_queue: queue.Queue[object], profile_key: str, focus_area: str):
        # model_complexity=1 gives a reasonable tradeoff for demo
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
        try:
            self._pose.close()
        except Exception:
            pass

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
        if not getattr(self.profile, "alternate_triplet", None):
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
        reps = int(data.get("reps", 0)) if isinstance(data.get("reps", 0), (int, float)) else 0
        stage = data.get("stage", "—")

        y_pos = 60
        y_pos = put_line("AI Smart Coach", y_pos, color=(0, 200, 255), scale=0.9)
        y_pos = put_line(f"Movement: {getattr(self.profile, 'name', '—')}", y_pos)
        if active_side:
            y_pos = put_line(f"Side: {active_side}", y_pos)
        y_pos = put_line(f"Reps: {reps}  |  Stage: {stage}", y_pos)
        y_pos = put_line(f"Angle: {_format_angle(angle_val)}  |  ROM: {_format_angle(rom_val)}", y_pos)
        y_pos = put_line(f"Tempo: {_format_seconds(tempo_val)} ({tempo_state})", y_pos)

        if report:
            balance = _format_ratio(getattr(report, "balance_score", None))
            speed = _format_ratio(getattr(report, "speed_score", None))
            foot = _format_ratio(getattr(report, "foot_placement_score", None))
            energy = _format_ratio(getattr(report, "energy_efficiency", None))
            y_pos = put_line(f"Balance: {balance}  |  Speed: {speed}", y_pos)
            y_pos = put_line(f"Footwork: {foot}  |  Energy: {energy}", y_pos)

            headline = report.headline() if hasattr(report, "headline") else None
            if headline:
                for wrapped in textwrap.wrap(headline, 32):
                    y_pos = put_line(wrapped, y_pos, color=(0, 215, 255))
                y_pos += 10

            primary_cues = getattr(report, "primary_cues", None) or []
            if primary_cues:
                y_pos = put_line("Cues:", y_pos, color=(50, 205, 50))
                for cue in primary_cues[:4]:
                    for wrapped in textwrap.wrap(cue, 32):
                        y_pos = put_line(f"- {wrapped}", y_pos, color=(220, 220, 220), scale=0.6)
                y_pos += 6

            safety_warnings = getattr(report, "safety_warnings", None) or []
            if safety_warnings:
                y_pos = put_line("Safety Alerts:", y_pos, color=(0, 140, 255))
                for warning in safety_warnings[:3]:
                    for wrapped in textwrap.wrap(warning, 32):
                        y_pos = put_line(f"! {wrapped}", y_pos, color=(0, 140, 255), scale=0.6)
                y_pos += 6

        else:
            y_pos = put_line("Waiting for pose data…", y_pos, color=(200, 200, 200))

        return panel

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        overlay_image = image.copy()
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # safely process frame
        results = None
        try:
            results = self._pose.process(rgb_image)
        except Exception:
            results = None

        normalized_landmarks = getattr(results, "pose_landmarks", None) if results else None
        world_landmarks = getattr(results, "pose_world_landmarks", None) if results else None

        if normalized_landmarks:
            try:
                self._drawing.draw_landmarks(
                    overlay_image,
                    normalized_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self._drawing_styles.get_default_pose_landmarks_style(),
                )
            except Exception:
                pass

        metrics: Optional[Dict[str, object]] = None
        report: Optional[CoachReport] = None
        observations: list[tuple[Tuple[int, int, int], float]] = []

        triplets = [self.active_triplet, getattr(self.profile, "primary_triplet", None), getattr(self.profile, "alternate_triplet", None)]
        seen_triplets = set()
        for triplet in triplets:
            if triplet and triplet not in seen_triplets:
                seen_triplets.add(triplet)
                points = self._extract_triplet_points(triplet, normalized_landmarks, world_landmarks)
                if len(points) == 3:
                    angle_val = calculate_angle(points[0], points[1], points[2])
                    observations.append((triplet, angle_val))

        if observations:
            # prefer current active_triplet if present
            chosen_triplet, chosen_angle = observations[0]
            for t, a in observations:
                if t == self.active_triplet:
                    chosen_triplet, chosen_angle = t, a
                    break

            self.active_triplet = chosen_triplet
            metrics = self.counter.update(chosen_angle)
            metrics["profile"] = getattr(self.profile, "key", "unknown")
            metrics["profile_name"] = getattr(self.profile, "name", "unknown")
            active_side = self._active_side_label()
            if active_side:
                metrics["active_side"] = active_side
            self.latest_metrics = metrics

            try:
                report = generate_coaching_report(self.profile, metrics, normalized_landmarks, world_landmarks, self.focus_area)
            except Exception:
                report = None
            self.latest_report = report

            headline = report.headline() if report and hasattr(report, "headline") else None
            if headline and (headline != self._last_headline or time.time() - self._last_feedback_time > FEEDBACK_DEBOUNCE_SEC):
                payload = {
                    "cue": headline,
                    "profile_key": getattr(self.profile, "key", None),
                    "profile_name": getattr(self.profile, "name", None),
                    "metrics": metrics,
                    "report": report.to_dict() if (report and hasattr(report, "to_dict")) else {},
                    "timestamp": time.time(),
                }
                if active_side:
                    payload["active_side"] = active_side
                try:
                    self.feedback_queue.put_nowait(payload)
                except queue.Full:
                    pass
                self._last_headline = headline
                self._last_feedback_time = time.time()

            self._update_hud_cache(metrics, report)
        else:
            if not self.latest_hud:
                self.latest_hud = ["Waiting for movement data…"]
            metrics = self.latest_metrics
            report = self.latest_report

        self._draw_overlay_text(overlay_image)

        height, width, _ = overlay_image.shape
        stats_panel = self._build_stats_panel(metrics, report, height, width)
        combined = np.zeros((height, width * 2, 3), dtype=np.uint8)
        combined[:, :width] = overlay_image
        combined[:, width:] = stats_panel
        self.latest_overlay_frame = combined

        return av.VideoFrame.from_ndarray(combined, format="bgr24")

    def _update_hud_cache(self, metrics: Dict[str, object], report: Optional[CoachReport]) -> None:
        rom_display = metrics.get("rom_current")
        rom_text = f"ROM: {rom_display:.1f}°" if isinstance(rom_display, (int, float)) else "ROM: —"
        angle_value = metrics.get("angle")
        angle_text = f"Angle: {angle_value:.1f}°" if isinstance(angle_value, (int, float)) else "Angle: —"
        tempo_value = metrics.get("tempo_last")
        tempo_state = metrics.get("tempo_state") or "steady"
        tempo_text = f"Tempo: {tempo_value:.2f}s ({tempo_state})" if isinstance(tempo_value, (int, float)) else "Tempo: —"
        side_label = metrics.get("active_side")
        stage_line = f"Stage: {metrics.get('stage', '—')} | Reps: {metrics.get('reps', 0)}"
        if side_label:
            stage_line += f" | Side: {side_label}"
        lines = [f"Movement: {getattr(self.profile, 'name', '—')}", stage_line, f"{angle_text} | {rom_text}", tempo_text]
        headline = report.headline() if report and hasattr(report, "headline") else None
        if headline:
            lines.append(f"Cue: {headline}")
        self.latest_hud = lines


# ---- Main Streamlit app ----
def main() -> None:
    st.set_page_config(page_title="AI Smart Coach", page_icon="🏋", layout="wide")
    st.markdown(
        """
        <style>
        video { width: 100% !important; max-width: 1280px !important; height: auto !important; border-radius: 12px; box-shadow: 0 0 24px rgba(0,0,0,0.25); }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("AI Smart Coach — Real-Time Form Assistant")

    st.sidebar.header("Session Controls")
    categories = list_categories() or []
    default_category = "Gym Coaching" if "Gym Coaching" in categories else (categories[0] if categories else "Gym Coaching")
    category_index = categories.index(default_category) if default_category in categories else 0
    selected_category = st.sidebar.selectbox("Mode", categories or ["Gym Coaching"], index=category_index, help="Choose the coaching mode.")

    profiles = list_profiles_by_category(selected_category) or []
    if not profiles:
        profiles = [get_profile("bicep_curl")]
    profile_map = {p.name: p for p in profiles}
    profile_name = st.sidebar.selectbox("Movement", list(profile_map.keys()))
    selected_profile = profile_map[profile_name]

    focus_area = st.sidebar.selectbox("Focus area (optional)", FOCUS_AREAS)
    st.sidebar.markdown(f"**Description:** {getattr(selected_profile, 'description', '')}")
    if getattr(selected_profile, "notes", None):
        st.sidebar.markdown("**Coaching Notes**")
        for note in selected_profile.notes:
            st.sidebar.markdown(f"- {note}")

    # session state init
    if "feedback_queue" not in st.session_state:
        st.session_state["feedback_queue"] = queue.Queue(maxsize=8)
    if "reminders_done" not in st.session_state:
        st.session_state["reminders_done"] = 0
    if "streak_days" not in st.session_state:
        st.session_state["streak_days"] = 0
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = "user_local_1"

    feedback_queue: queue.Queue[object] = st.session_state["feedback_queue"]

    video_col, stats_col = st.columns([3, 2])

    with video_col:
        st.subheader("Live Camera Feed")
        try:
            webrtc_ctx = webrtc_streamer(
                key="coach",
                mode=WebRtcMode.SENDRECV if WebRtcMode else None,
                rtc_configuration={"iceServers": ICE_SERVERS},
                media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
                async_processing=False,  # reduce glitching for VS Code demos
                video_processor_factory=lambda: PoseProcessor(feedback_queue, selected_profile.key, focus_area),
            )
        except Exception as e:
            st.error(f"Unable to start webcam stream: {e}")
            webrtc_ctx = None

    metrics: Dict[str, object] = {}
    report = CoachReport() if "CoachReport" in globals() else None
    if webrtc_ctx and getattr(webrtc_ctx, "video_processor", None):
        processor: PoseProcessor = webrtc_ctx.video_processor
        processor.set_profile(selected_profile.key, focus_area)
        metrics = processor.latest_metrics or {}
        report = processor.latest_report or (CoachReport() if "CoachReport" in globals() else None)

    # collect feedback payloads (safe)
    feedback_payload: Optional[dict[str, object]] = None
    try:
        while feedback_queue and not feedback_queue.empty():
            n = feedback_queue.get_nowait()
            if isinstance(n, dict):
                feedback_payload = n
    except Exception:
        feedback_payload = feedback_payload

    guidance_block = None
    audio_tag_html = None
    if feedback_payload:
        cue_text = str(feedback_payload.get("cue", "")).strip()
        metrics_for_ai = feedback_payload.get("metrics", {}) or metrics

        # RAG / LLM feedback (safe)
        try:
            cue_body = coach_brain.get_feedback(cue_text or "Keep your form tight", feedback_payload.get("profile_name", selected_profile.name), metrics_for_ai)
        except Exception:
            cue_body = cue_text or "Keep your form tight"

        # Agentic enhancement (safe) — pass user_id so agent stores progress
        try:
            agentic_feedback = agentic_coach.enhance_feedback(cue_text or "Keep your form tight", feedback_payload.get("profile_name", selected_profile.name), metrics_for_ai, user_id=st.session_state["user_id"])
        except Exception:
            agentic_feedback = ""

        # Combine (short) for UI/TTS
        combined_feedback = cue_body
        if agentic_feedback:
            combined_feedback = f"{combined_feedback}\n\nAgentic: {agentic_feedback}"

        # TTS (may fail if gTTS / network unavailable)
        try:
            audio_b64 = coach_voice.synthesize(combined_feedback)
            audio_tag_html = coach_voice.audio_tag(audio_b64) if audio_b64 else None
        except Exception:
            audio_tag_html = None

        guidance_block = combined_feedback

        # reminders & streaks (demo policy)
        st.session_state["reminders_done"] += 1
        if st.session_state["reminders_done"] % 10 == 0:
            st.session_state["streak_days"] += 1

    # Render stats / feedback
    with stats_col:
        st.subheader("Session Metrics")
        try:
            rep_count = int(metrics.get("reps", 0)) if isinstance(metrics.get("reps", 0), (int, float)) else 0
        except Exception:
            rep_count = 0
        st.metric("Repetitions", rep_count)

        stage_value = str(metrics.get("stage", "—"))
        active_side = metrics.get("active_side")
        if active_side:
            stage_value = f"{stage_value} ({active_side})"
        st.metric("Stage", stage_value)

        st.metric("Joint Angle", _format_angle(metrics.get("angle")))
        rom_value = metrics.get("rom_last") or metrics.get("rom_current")
        st.metric("Range of Motion", _format_angle(rom_value))

        st.metric("Tempo (last)", _format_seconds(metrics.get("tempo_last")))

        if isinstance(report, CoachReport):
            st.metric("Balance", _format_ratio(getattr(report, "balance_score", None)))
            st.metric("Speed", _format_ratio(getattr(report, "speed_score", None)))
            primary_cues = getattr(report, "primary_cues", None) or []
            if primary_cues:
                st.markdown("**Primary Cues**")
                for cue in primary_cues:
                    st.markdown(f"- {cue}")
            safety_warnings = getattr(report, "safety_warnings", None) or []
            if safety_warnings:
                st.markdown("**Safety Alerts**")
                for w in safety_warnings:
                    st.warning(w)

        if guidance_block:
            st.markdown("**Coach Feedback**")
            st.markdown(f"> {guidance_block}")
            if audio_tag_html:
                st.markdown(audio_tag_html, unsafe_allow_html=True)

        st.markdown(f"**Streak Days:** {st.session_state['streak_days']}  |  Reminders Completed: {st.session_state['reminders_done']}")

    # Auto-refresh but slower to reduce glitching in demos
    if webrtc_ctx and getattr(webrtc_ctx, "state", None) and getattr(webrtc_ctx.state, "playing", False):
        now_ts = time.time()
        last_refresh = st.session_state.get("_last_metrics_refresh", 0.0)
        if now_ts - last_refresh > 0.6:
            st.session_state["_last_metrics_refresh"] = now_ts
            _trigger_rerun()
    else:
        st.session_state.pop("_last_metrics_refresh", None)


if __name__ == "__main__":
    main()

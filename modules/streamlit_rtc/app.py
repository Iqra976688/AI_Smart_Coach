"""Streamlit interface for the Real-Time AI Smart Coach prototype."""
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


ICE_SERVERS = [{"urls": ["stun:stun.l.google.com:19302"]}]
FEEDBACK_DEBOUNCE_SEC = 5.0
TRIPLET_SWITCH_MARGIN = 6.0


coach_brain = CoachBrain()
coach_voice = CoachVoice()


def _trigger_rerun() -> None:
    rerun_fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun_fn:
        rerun_fn()
        return
    try:
        from streamlit.runtime.scriptrunner import RerunException, RerunData  # type: ignore
    except Exception:
        try:  # pragma: no cover - legacy fallback
            from streamlit.script_runner import RerunException, RerunData  # type: ignore
        except Exception:
            return

    raise RerunException(RerunData())


def _landmark_to_point(
    landmarks: Optional[landmark_pb2.NormalizedLandmarkList], index: int
) -> Optional[tuple[float, float]]:
    if not landmarks:
        return None
    try:
        landmark = landmarks.landmark[index]
    except IndexError:
        return None
    return (landmark.x, landmark.y)


def _world_landmark_to_point(
    landmarks: Optional[landmark_pb2.LandmarkList], index: int
) -> Optional[tuple[float, float]]:
    if not landmarks:
        return None
    try:
        landmark = landmarks.landmark[index]
    except IndexError:
        return None
    return (landmark.x, landmark.y)


class PoseProcessor(VideoProcessorBase):
    """Video transformer that runs MediaPipe pose and overlays coaching data."""

    def __init__(
        self,
        feedback_queue: queue.Queue[object],
        profile_key: str,
        focus_area: str,
    ):
        self._pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )
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

    def _update_hud_cache(
        self, metrics: Dict[str, object], report: Optional[CoachReport]
    ) -> None:
        rom_display = metrics.get("rom_current")
        rom_text = (
            f"ROM: {rom_display:.1f}°" if isinstance(rom_display, (int, float)) else "ROM: —"
        )
        angle_value = metrics.get("angle")
        angle_text = (
            f"Angle: {angle_value:.1f}°" if isinstance(angle_value, (int, float)) else "Angle: —"
        )
        tempo_value = metrics.get("tempo_last")
        tempo_state = metrics.get("tempo_state") or "steady"
        if isinstance(tempo_value, (int, float)):
            tempo_text = f"Tempo: {tempo_value:.2f}s ({tempo_state})"
        else:
            tempo_text = "Tempo: —"

        side_label = metrics.get("active_side")
        stage_line = f"Stage: {metrics.get('stage', '—')} | Reps: {metrics.get('reps', 0)}"
        if side_label:
            stage_line += f" | Side: {side_label}"

        lines = [
            f"Movement: {self.profile.name}",
            stage_line,
            f"{angle_text} | {rom_text}",
            tempo_text,
        ]

        headline = report.headline() if report else None
        if headline:
            lines.append(f"Cue: {headline}")

        self.latest_hud = lines

    def _extract_triplet_points(
        self,
        triplet: Tuple[int, int, int],
        normalized_landmarks: Optional[landmark_pb2.NormalizedLandmarkList],
        world_landmarks: Optional[landmark_pb2.LandmarkList],
    ) -> list[tuple[float, float]]:
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
            cv2.putText(
                overlay_image,
                line,
                (21, base_y + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay_image,
                line,
                (20, base_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            base_y += 32

    def _build_stats_panel(
        self,
        metrics: Optional[Dict[str, object]],
        report: Optional[CoachReport],
        height: int,
        width: int,
    ) -> np.ndarray:
        panel = np.full((height, width, 3), (24, 24, 24), dtype=np.uint8)
        data = metrics or {}

        def put_line(text: str, y: int, *, color=(255, 255, 255), scale: float = 0.7) -> int:
            cv2.putText(
                panel,
                text,
                (28, y + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (0, 0, 0),
                max(1, int(4 * scale)),
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                text,
                (26, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                max(1, int(2 * scale)),
                cv2.LINE_AA,
            )
            return y + int(34 * scale)

        def fmt_angle(value: Optional[object]) -> str:
            return f"{value:.1f}°" if isinstance(value, (int, float)) else "—"

        def fmt_seconds(value: Optional[object]) -> str:
            return f"{value:.2f}s" if isinstance(value, (int, float)) else "—"

        def fmt_ratio(value: Optional[object]) -> str:
            return f"{value * 100:.0f}%" if isinstance(value, (int, float)) else "—"

        rep_count = int(data.get("reps", 0))
        stage = data.get("stage", "—")
        angle = fmt_angle(data.get("angle"))
        rom_val = data.get("rom_last")
        if not isinstance(rom_val, (int, float)):
            rom_val = data.get("rom_current")
        rom = fmt_angle(rom_val)
        tempo_last = fmt_seconds(data.get("tempo_last"))
        tempo_avg = fmt_seconds(data.get("tempo_avg"))
        tempo_state = data.get("tempo_state", "steady")
        active_side = data.get("active_side")

        y_pos = 60
        y_pos = put_line("AI Smart Coach", y_pos, color=(0, 200, 255), scale=0.9)
        y_pos = put_line(f"Movement: {self.profile.name}", y_pos)
        if active_side:
            y_pos = put_line(f"Side: {active_side}", y_pos)
        y_pos = put_line(f"Reps: {rep_count}  |  Stage: {stage}", y_pos)
        y_pos = put_line(f"Angle: {angle}  |  ROM: {rom}", y_pos)
        y_pos = put_line(f"Tempo: {tempo_last} ({tempo_state})", y_pos)
        y_pos = put_line(f"Tempo avg: {tempo_avg}", y_pos)

        if report:
            balance = fmt_ratio(report.balance_score)
            speed = fmt_ratio(report.speed_score)
            foot = fmt_ratio(report.foot_placement_score)
            energy = fmt_ratio(report.energy_efficiency)
            y_pos = put_line(f"Balance: {balance}  |  Speed: {speed}", y_pos)
            y_pos = put_line(f"Footwork: {foot}  |  Energy: {energy}", y_pos)

            headline = report.headline()
            if headline:
                for wrapped in textwrap.wrap(headline, 32):
                    y_pos = put_line(wrapped, y_pos, color=(0, 215, 255))
                y_pos += 10

            if report.primary_cues:
                y_pos = put_line("Cues:", y_pos, color=(50, 205, 50))
                for cue in report.primary_cues[:4]:
                    for wrapped in textwrap.wrap(cue, 32):
                        y_pos = put_line(f"- {wrapped}", y_pos, color=(220, 220, 220), scale=0.6)
                y_pos += 6

            if report.safety_warnings:
                y_pos = put_line("Safety Alerts:", y_pos, color=(0, 140, 255))
                for warning in report.safety_warnings[:3]:
                    for wrapped in textwrap.wrap(warning, 32):
                        y_pos = put_line(f"! {wrapped}", y_pos, color=(0, 140, 255), scale=0.6)
                y_pos += 6

            if report.tempo_feedback:
                for wrapped in textwrap.wrap(report.tempo_feedback, 32):
                    y_pos = put_line(wrapped, y_pos, color=(180, 180, 255), scale=0.6)
            if report.rom_feedback:
                for wrapped in textwrap.wrap(report.rom_feedback, 32):
                    y_pos = put_line(wrapped, y_pos, color=(180, 180, 255), scale=0.6)
        else:
            y_pos = put_line("Waiting for pose data…", y_pos, color=(200, 200, 200))

        return panel

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        overlay_image = image.copy()
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self._pose.process(rgb_image)
        normalized_landmarks = results.pose_landmarks
        world_landmarks = results.pose_world_landmarks

        if normalized_landmarks:
            self._drawing.draw_landmarks(
                overlay_image,
                normalized_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self._drawing_styles.get_default_pose_landmarks_style(),
            )

        report: Optional[CoachReport] = None
        metrics: Optional[Dict[str, object]] = None
        candidate_triplets: list[Tuple[int, int, int]] = []
        seen_triplets: set[Tuple[int, int, int]] = set()
        for triplet in (
            self.active_triplet,
            self.profile.primary_triplet,
            self.profile.alternate_triplet,
        ):
            if triplet and triplet not in seen_triplets:
                seen_triplets.add(triplet)
                candidate_triplets.append(triplet)

        observations: list[tuple[Tuple[int, int, int], float]] = []
        for triplet in candidate_triplets:
            points = self._extract_triplet_points(triplet, normalized_landmarks, world_landmarks)
            if len(points) == 3:
                angle_value = calculate_angle(points[0], points[1], points[2])
                observations.append((triplet, angle_value))

        if observations:
            chosen_triplet: Optional[Tuple[int, int, int]] = None
            chosen_angle: Optional[float] = None

            for triplet, angle_value in observations:
                if triplet == self.active_triplet:
                    chosen_triplet = triplet
                    chosen_angle = angle_value
                    break

            if chosen_triplet is None:
                chosen_triplet, chosen_angle = observations[0]

            if (
                self.profile.adaptive_side
                and len(observations) > 1
                and chosen_angle is not None
            ):
                best_triplet, best_angle = min(observations, key=lambda item: item[1])
                if (
                    best_triplet != chosen_triplet
                    and best_angle + TRIPLET_SWITCH_MARGIN < chosen_angle
                ):
                    chosen_triplet, chosen_angle = best_triplet, best_angle

            if chosen_triplet is not None and chosen_angle is not None:
                self.active_triplet = chosen_triplet
                metrics = self.counter.update(chosen_angle)
                metrics["profile"] = self.profile.key
                metrics["profile_name"] = self.profile.name
                active_side = self._active_side_label()
                if active_side:
                    metrics["active_side"] = active_side
                self.latest_metrics = metrics

                report = generate_coaching_report(
                    self.profile,
                    metrics,
                    normalized_landmarks,
                    world_landmarks,
                    self.focus_area,
                )
                self.latest_report = report

                headline = report.headline() if report else None
                if headline and (
                    headline != self._last_headline
                    or time.time() - self._last_feedback_time > FEEDBACK_DEBOUNCE_SEC
                ):
                    payload = {
                        "cue": headline,
                        "profile_key": self.profile.key,
                        "profile_name": self.profile.name,
                        "metrics": metrics,
                        "report": report.to_dict() if report else {},
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


def main() -> None:
    st.set_page_config(page_title="AI Smart Coach", page_icon="🏋", layout="wide")
    st.markdown(
        """
        <style>
        video {
            width: 100% !important;
            max-width: 1280px !important;
            height: auto !important;
            border-radius: 12px;
            box-shadow: 0 0 24px rgba(0, 0, 0, 0.25);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("AI Smart Coach — Real-Time Form Assistant")

    st.sidebar.header("Session Controls")
    categories = list_categories()
    default_category = "Gym Coaching" if "Gym Coaching" in categories else categories[0]
    category_index = categories.index(default_category) if categories else 0
    selected_category = st.sidebar.selectbox(
        "Mode", categories, index=category_index, help="Choose the coaching mode."
    )

    profiles = list_profiles_by_category(selected_category)
    if not profiles:
        profiles = [get_profile("bicep_curl")]
    profile_map = {profile.name: profile for profile in profiles}
    profile_name = st.sidebar.selectbox("Movement", list(profile_map.keys()))
    selected_profile = profile_map[profile_name]

    focus_area = st.sidebar.selectbox(
        "Focus area (optional)",
        FOCUS_AREAS,
        help="Highlight an area to receive additional safeguarding cues.",
    )

    st.sidebar.markdown(f"**Description:** {selected_profile.description}")
    if selected_profile.notes:
        st.sidebar.markdown("**Coaching Notes**")
        for note in selected_profile.notes:
            st.sidebar.markdown(f"- {note}")

    if "feedback_queue" not in st.session_state:
        st.session_state["feedback_queue"] = queue.Queue(maxsize=8)
    feedback_queue: queue.Queue[object] = st.session_state["feedback_queue"]

    video_col, stats_col = st.columns([3, 2])

    with video_col:
        st.subheader("Live Camera Feed")
        webrtc_ctx = webrtc_streamer(
            key="coach",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": ICE_SERVERS},
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            video_processor_factory=lambda: PoseProcessor(
                feedback_queue, selected_profile.key, focus_area
            ),
        )

    metrics: Dict[str, object] = {}
    report = CoachReport()
    if webrtc_ctx.video_processor:
        processor: PoseProcessor = webrtc_ctx.video_processor
        processor.set_profile(selected_profile.key, focus_area)
        metrics = processor.latest_metrics or {}
        report = processor.latest_report or CoachReport()

    feedback_payload: Optional[dict[str, object]] = None
    fallback_headline: Optional[str] = None
    while not feedback_queue.empty():
        next_item = feedback_queue.get_nowait()
        if isinstance(next_item, dict):
            feedback_payload = next_item
        elif isinstance(next_item, str):
            fallback_headline = next_item

    headline = report.headline()
    guidance_block = None
    audio_tag_html = None

    if feedback_payload:
        cue_text = str(feedback_payload.get("cue", "")).strip()
        headline = cue_text or headline
        metrics_for_ai = feedback_payload.get("metrics")
        cue_body = coach_brain.get_feedback(
            cue_text or headline or "Keep your form tight",
            feedback_payload.get("profile_name", selected_profile.name),
            metrics_for_ai if isinstance(metrics_for_ai, dict) else metrics,
        )
        audio_b64 = coach_voice.synthesize(cue_body)
        audio_tag_html = coach_voice.audio_tag(audio_b64)
        guidance_block = cue_body
    elif fallback_headline:
        headline = fallback_headline

    def _format_angle(value: Optional[object]) -> str:
        return f"{value:.1f}°" if isinstance(value, (int, float)) else "—"

    def _format_seconds(value: Optional[object]) -> str:
        return f"{value:.2f}s" if isinstance(value, (int, float)) else "—"

    def _format_ratio(value: Optional[object]) -> str:
        return f"{value * 100:.0f}%" if isinstance(value, (int, float)) else "—"

    with stats_col:
        st.subheader("Session Metrics")
        rep_col, stage_col = st.columns(2)
        rep_col.metric("Repetitions", int(metrics.get("reps", 0)))
        stage_value = str(metrics.get("stage", "—"))
        active_side = metrics.get("active_side")
        if active_side:
            stage_value = f"{stage_value} ({active_side})"
        stage_col.metric("Stage", stage_value)

        angle_col, rom_col = st.columns(2)
        angle_col.metric("Joint Angle", _format_angle(metrics.get("angle")))
        rom_value = metrics.get("rom_last") or metrics.get("rom_current")
        rom_col.metric("Range of Motion", _format_angle(rom_value))

        tempo_col, tempo_avg_col = st.columns(2)
        tempo_col.metric("Tempo (last)", _format_seconds(metrics.get("tempo_last")))
        tempo_avg_col.metric("Tempo (avg)", _format_seconds(metrics.get("tempo_avg")))

        perf_col1, perf_col2 = st.columns(2)
        perf_col1.metric("Balance", _format_ratio(report.balance_score))
        perf_col2.metric("Speed", _format_ratio(report.speed_score))

        perf_col3, perf_col4 = st.columns(2)
        perf_col3.metric("Foot Placement", _format_ratio(report.foot_placement_score))
        perf_col4.metric("Energy Efficiency", _format_ratio(report.energy_efficiency))

        if headline:
            if report.safety_warnings:
                st.error(headline)
            else:
                st.success(headline)

        if guidance_block:
            st.markdown("**Coach Feedback**")
            st.markdown(f"> {guidance_block}")
            if audio_tag_html:
                st.markdown(audio_tag_html, unsafe_allow_html=True)

        if report.tempo_feedback:
            st.info(report.tempo_feedback)
        if report.rom_feedback:
            st.info(report.rom_feedback)

        st.markdown("**Primary Cues**")
        if report.primary_cues:
            for cue in report.primary_cues:
                st.markdown(f"- {cue}")
        else:
            st.markdown("- Waiting for movement data…")

        if report.safety_warnings:
            st.markdown("**Safety Alerts**")
            for warning in report.safety_warnings:
                st.warning(warning)

        if report.angle_of_movement is not None:
            st.caption(f"Angle of movement: {report.angle_of_movement:.1f}°")

    st.caption(
        "Real-time coaching powered by MediaPipe pose tracking, multi-mode biomechanics, and Streamlit WebRTC. Adjust the column divider to resize the live feed."
    )

    if webrtc_ctx and webrtc_ctx.state.playing:
        now_ts = time.time()
        last_refresh = st.session_state.get("_last_metrics_refresh", 0.0)
        if now_ts - last_refresh > 0.4:
            st.session_state["_last_metrics_refresh"] = now_ts
            _trigger_rerun()
    else:
        st.session_state.pop("_last_metrics_refresh", None)


if __name__ == "__main__":
    main()

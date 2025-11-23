import os
import time
import queue
import random
import streamlit as st
import av
import cv2
import tempfile
import numpy as np

from pose_logic import PoseProcessor, get_profile, list_profiles_by_category, list_categories, FOCUS_AREAS
from rag_chain import CoachBrain, CoachVoice, CoachReport

# -------------------------------
# Initialize AI Coaches
# -------------------------------
brain = CoachBrain(dataset_path="./dataset.json", model_name="gpt-4.1-mini")
voice = CoachVoice(tts_engine="pyttsx3")

# Optional Agentic AI fallback
try:
    from agentic_ai import AgenticCoach
except Exception:
    class AgenticCoach:
        def __init__(self, *a, **k): pass
        def enhance_feedback(self, cue, movement, metrics=None, user_id=None): return ""

agentic_coach = AgenticCoach()

# -------------------------------
# Streamlit-webrtc
# -------------------------------
try:
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
except Exception:
    VideoProcessorBase = object
    WebRtcMode = None
    def webrtc_streamer(*args, **kwargs):
        raise RuntimeError("streamlit-webrtc required")

ICE_SERVERS = [{"urls": ["stun:stun.l.google.com:19302"]}]

# -------------------------------
# Streamlit helpers
# -------------------------------
def _trigger_rerun():
    rerun_fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun_fn:
        rerun_fn()

# -------------------------------
# Video / Image Processor
# -------------------------------
class PoseVideoProcessor(VideoProcessorBase):
    def __init__(self, feedback_queue, selected_profile, show_skeleton: bool = True):
        self.processor = PoseProcessor(feedback_queue, selected_profile.key, focus_area="Form")
        self.latest_metrics = {}
        self.latest_hud = []
        self.show_skeleton = show_skeleton

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        out_img = self.processor.process(img)

        if self.show_skeleton and self.processor.last_landmarks:
            self.processor.mp_draw.draw_landmarks(
                out_img,
                self.processor.last_landmarks,
                self.processor.mp_pose.POSE_CONNECTIONS
            )

        self.latest_metrics = self.processor.latest_metrics
        self.latest_hud = ["{}: {:.1f}".format(k, v) for k, v in self.latest_metrics.items() if isinstance(v, (int, float))]
        return av.VideoFrame.from_ndarray(out_img, format="bgr24")

    # For uploaded images/videos
    def process(self, frame):
        out_img = self.processor.process(frame)
        if self.show_skeleton and self.processor.last_landmarks:
            self.processor.mp_draw.draw_landmarks(
                out_img,
                self.processor.last_landmarks,
                self.processor.mp_pose.POSE_CONNECTIONS
            )
        self.latest_metrics = self.processor.latest_metrics
        return out_img

    def set_show_skeleton(self, value: bool):
        self.show_skeleton = value

# -------------------------------
# Main App
# -------------------------------
def main():
    st.set_page_config(page_title="AI Smart Coach", layout="wide")
    st.title("AI Smart Coach — Real-Time Form Assistant")

    # -------------------------------
    # Sidebar: Select Category & Exercise
    # -------------------------------
    category = st.sidebar.selectbox("Select Mode", list_categories())
    exercises = list_profiles_by_category(category)
    exercise_keys = [p.key for p in exercises]
    selected_key = st.sidebar.selectbox("Select Exercise", exercise_keys)
    selected_profile = get_profile(selected_key)
    if not selected_profile:
        st.error("Invalid exercise selected.")
        return

    # -------------------------------
    # Session State
    # -------------------------------
    if "feedback_queue" not in st.session_state:
        st.session_state["feedback_queue"] = queue.Queue(maxsize=8)
    feedback_queue = st.session_state["feedback_queue"]

    if "user_id" not in st.session_state:
        st.session_state["user_id"] = "user_1"
    if "reminders_done" not in st.session_state:
        st.session_state["reminders_done"] = 0
    if "streak_days" not in st.session_state:
        st.session_state["streak_days"] = 0

    # -------------------------------
    # Media input: Webcam / Video / Image
    # -------------------------------
    show_skeleton = st.sidebar.checkbox("Show Pose Skeleton", value=True)
    input_option = st.sidebar.radio("Input Source", ["Webcam (Live Demo)", "Upload Video", "Upload Image"])
    video_processor_factory = lambda: PoseVideoProcessor(feedback_queue, selected_profile, show_skeleton)

    stframe = st.empty()

    if input_option == "Webcam (Live Demo)":
        webrtc_ctx = webrtc_streamer(
            key="coach",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=video_processor_factory,
            rtc_configuration={"iceServers": ICE_SERVERS},
        )
        if webrtc_ctx and webrtc_ctx.video_processor:
            processor = webrtc_ctx.video_processor
            processor.set_show_skeleton(show_skeleton)
            metrics = processor.latest_metrics

    else:
        upload_types = ["mp4", "mov", "avi"] if input_option == "Upload Video" else ["jpg", "png", "jpeg"]
        uploaded_file = st.file_uploader(f"Upload {input_option.split()[1]}", type=upload_types)
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            processor = PoseVideoProcessor(feedback_queue, selected_profile, show_skeleton)

            if input_option == "Upload Video":
                cap = cv2.VideoCapture(tfile.name)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out_frame = processor.process(frame)
                    stframe.image(cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB), channels="RGB")
                cap.release()
            else:  # Single image
                img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
                out_frame = processor.process(img)
                stframe.image(cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB), channels="RGB")

    # -------------------------------
    # Metrics Dashboard
    # -------------------------------
    st.header("Session Metrics")
    video_col, stats_col = st.columns([2.5, 1.5], gap="large")

    with stats_col:
        feedback_payload = None
        try:
            while feedback_queue and not feedback_queue.empty():
                n = feedback_queue.get_nowait()
                if isinstance(n, dict):
                    feedback_payload = n
        except Exception:
            pass

        metrics = feedback_payload.get("metrics") if feedback_payload else {}
        report_obj = feedback_payload.get("report") if feedback_payload else None
        if isinstance(report_obj, CoachReport):
            report = report_obj.to_dict()
        elif isinstance(report_obj, dict):
            report = report_obj
        else:
            report = {}

        def metric_card(title, value, color="#1FB41F"):
            st.markdown(
                f"<div style='background-color:{color}; color:white; padding:12px; border-radius:8px; text-align:center; margin-bottom:5px;'>"
                f"<h4 style='margin:0'>{title}</h4>"
                f"<p style='margin:0'>{value}</p></div>", unsafe_allow_html=True
            )

        metric_card("Reps", int(metrics.get("reps", 0)), "#365472")
        metric_card("Stage", metrics.get("stage", "—"), "#3B3350")
        angle_val = metrics.get("angle")
        angle_text = f"{angle_val:.1f}°" if isinstance(angle_val, (int, float)) else "—"
        metric_card("Joint Angle", angle_text, "#5B644E")
        rom_val = metrics.get("rom_current")
        rom_text = f"{rom_val:.1f}°" if isinstance(rom_val, (int, float)) else "—"
        metric_card("Range of Motion", rom_text, "#8C600F")
        tempo_val = metrics.get("tempo_last")
        tempo_text = f"{tempo_val:.2f}s" if isinstance(tempo_val, (int, float)) else "—"
        metric_card("Tempo (last)", tempo_text, "#2F2023")

    # -------------------------------
    # Coach Feedback
    # -------------------------------
    guidance_block = ""
    audio_tag_html = None
    if feedback_payload:
        cue_text = feedback_payload.get("cue", "Maintain good form")
        try:
            cue_body = brain.get_feedback(cue_text, selected_profile.name, metrics)
        except Exception:
            cue_body = cue_text
        try:
            agentic_feedback = agentic_coach.enhance_feedback(
                cue_text, selected_profile.name, metrics, user_id=st.session_state["user_id"]
            )
        except Exception:
            agentic_feedback = ""
        guidance_block = cue_body
        if agentic_feedback:
            guidance_block += f"\n\nAgentic: {agentic_feedback}"
        try:
            audio_b64 = voice.synthesize(guidance_block)
            audio_tag_html = voice.audio_tag(audio_b64) if audio_b64 else None
        except Exception:
            audio_tag_html = None

    if guidance_block:
        st.markdown(
            f"<div style='background-color:#E0F7FA; padding:10px; border-radius:10px; color:#00796B;'>"
            f"<h4>Coach Feedback</h4>"
            f"<p>{guidance_block}</p></div>", unsafe_allow_html=True
        )
        if audio_tag_html:
            st.markdown(audio_tag_html, unsafe_allow_html=True)

    # -------------------------------
    # Streak & Reminder
    # -------------------------------
    today = time.strftime("%Y-%m-%d")
    if "last_reminder_day" not in st.session_state:
        st.session_state["last_reminder_day"] = None
    if st.session_state["last_reminder_day"] != today:
        st.session_state["last_reminder_day"] = today
        reminder_text = random.choice([
            "🏋️ Time for your workout! Push through!",
            "💦 Stay consistent! Every rep counts!",
            "🔥 Keep form tight for best results!",
            "💪 Focus on quality over quantity!"
        ])
        st.session_state["reminders_done"] += 1
        st.session_state["streak_days"] += 1
    else:
        reminder_text = None

    if reminder_text:
        st.markdown(
            f"<div style='background-color:#F9ED69; padding:10px; border-radius:8px; text-align:center; font-weight:bold;'>{reminder_text}</div>",
            unsafe_allow_html=True
        )

    streak_days = st.session_state.get("streak_days", 0)
    streak_msg = f"🔥 {streak_days}-day streak!" if streak_days > 0 else "Let's start your streak!"
    st.markdown(
        f"<div style='background-color:#FFD700; padding:10px; border-radius:8px; text-align:center; font-weight:bold;'>{streak_msg} | Reminders: {st.session_state['reminders_done']}</div>",
        unsafe_allow_html=True
    )

    # -------------------------------
    # Auto-refresh for webcam
    # -------------------------------
    if input_option == "Webcam (Live Demo)":
        if getattr(st.session_state, "_last_metrics_refresh", None) is None:
            st.session_state["_last_metrics_refresh"] = 0.0
        now_ts = time.time()
        last_refresh = st.session_state.get("_last_metrics_refresh", 0.0)
        if now_ts - last_refresh > 0.6:
            st.session_state["_last_metrics_refresh"] = now_ts
            _trigger_rerun()

if __name__ == "__main__":
    main()

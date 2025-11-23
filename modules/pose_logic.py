import time
import math
import queue
from typing import Dict, Optional, List
import cv2
import mediapipe as mp

# -------------------------------
# Helper: calculate angle between 3 points
# -------------------------------
def calculate_angle(a, b, c):
    """
    Returns angle at point b (in degrees) formed by points a-b-c
    a, b, c = (x, y)
    """
    try:
        ba = (a[0]-b[0], a[1]-b[1])
        bc = (c[0]-b[0], c[1]-b[1])
        cos_angle = (ba[0]*bc[0] + ba[1]*bc[1]) / (math.hypot(*ba)*math.hypot(*bc)+1e-6)
        angle = math.degrees(math.acos(max(-1, min(1, cos_angle))))
        return angle
    except Exception:
        return 0.0

# -------------------------------
# Exercise-specific joint mappings
# -------------------------------
EXERCISE_KEYPOINTS = {
    "squat": [("left_hip","left_knee","left_ankle"), ("right_hip","right_knee","right_ankle")],
    "push_up": [("left_shoulder","left_elbow","left_wrist"), ("right_shoulder","right_elbow","right_wrist")],
    "plank": [("left_shoulder","left_elbow","left_wrist"), ("right_shoulder","right_elbow","right_wrist")],
    "lunge": [("left_hip","left_knee","left_ankle"), ("right_hip","right_knee","right_ankle")],
    "shoulder_press": [("left_shoulder","left_elbow","left_wrist"), ("right_shoulder","right_elbow","right_wrist")],
    "deadlift": [("left_hip","left_knee","left_ankle"), ("right_hip","right_knee","right_ankle")],
    "bicep_curl": [("left_shoulder","left_elbow","left_wrist"), ("right_shoulder","right_elbow","right_wrist")],
    "running": [("left_hip","left_knee","left_ankle"), ("right_hip","right_knee","right_ankle")],
    "cricket_batting": [("left_shoulder","left_elbow","left_wrist"), ("right_shoulder","right_elbow","right_wrist")],
    "cricket_bowling": [("left_shoulder","left_elbow","left_wrist"), ("right_shoulder","right_elbow","right_wrist")],
    "tennis_swing": [("left_shoulder","left_elbow","left_wrist"), ("right_shoulder","right_elbow","right_wrist")],
    "football_kick": [("left_hip","left_knee","left_ankle"), ("right_hip","right_knee","right_ankle")],
    "shoulder_circles": [("left_shoulder","left_elbow","left_wrist"), ("right_shoulder","right_elbow","right_wrist")],
    "ankle_pumps": [("left_ankle","left_knee","left_hip"), ("right_ankle","right_knee","right_hip")],
    "wrist_flexion": [("left_wrist","left_elbow","left_shoulder"), ("right_wrist","right_elbow","right_shoulder")],
    "wrist_extension": [("left_wrist","left_elbow","left_shoulder"), ("right_wrist","right_elbow","right_shoulder")],
}

# -------------------------------
# Landmark name mapping
# -------------------------------
LANDMARK_NAMES = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

# -------------------------------
# PoseProcessor Class
# -------------------------------
class PoseProcessor:
    """
    Multi-exercise pose processor for AI Smart Coach.
    Gym: reps/stage/angles/ROM/tempo
    Sports: angles/ROM/tempo
    Physiotherapy: angles/ROM/controlled movement
    """
    REPS_BASED = ["squat","push_up","plank","lunge","bicep_curl","shoulder_press","deadlift"]

    def __init__(self, feedback_queue: queue.Queue, exercise_key: str, focus_area: str = "Form"):
        self.feedback_queue = feedback_queue
        self.exercise_key = exercise_key
        self.focus_area = focus_area

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

        self.latest_metrics = {}
        self.latest_hud = []

        self.reps = 0
        self.stage = None
        self.angles = {}
        self.rom_min = {}
        self.rom_max = {}
        self.last_rep_time = None
        self.tempo_last = None
        self.last_landmarks = None

        self.joint_sets = EXERCISE_KEYPOINTS.get(exercise_key, [])

    # -------------------------------
    # Process a single frame
    # -------------------------------
    def process(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        landmarks = results.pose_landmarks
        self.last_landmarks = landmarks

        if landmarks:
            angles = {}
            for joint_triplets in self.joint_sets:
                a_pt = self._landmark_point(landmarks, joint_triplets[0])
                b_pt = self._landmark_point(landmarks, joint_triplets[1])
                c_pt = self._landmark_point(landmarks, joint_triplets[2])
                if a_pt and b_pt and c_pt:
                    angle = calculate_angle(a_pt, b_pt, c_pt)
                    angles[joint_triplets[1]] = angle
                    self.rom_min[joint_triplets[1]] = min(self.rom_min.get(joint_triplets[1], angle), angle)
                    self.rom_max[joint_triplets[1]] = max(self.rom_max.get(joint_triplets[1], angle), angle)

                    h, w, _ = frame.shape
                    a_px = tuple(int(x*w) for x in a_pt)
                    b_px = tuple(int(x*w) for x in b_pt)
                    c_px = tuple(int(x*w) for x in c_pt)
                    cv2.line(frame, a_px, b_px, (0,255,0), 2)
                    cv2.line(frame, b_px, c_px, (0,255,0), 2)
                    cv2.putText(frame, f"{int(angle)}", b_px, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            self.angles = angles

            if self.exercise_key in self.REPS_BASED:
                self._update_reps_and_stage()

            self.latest_metrics = {
                "reps": self.reps,
                "stage": self.stage or "—",
                "angle": list(angles.values())[0] if angles else 0.0,
                "rom_current": sum(angles.values())/len(angles) if angles else 0.0,
                "tempo_last": self.tempo_last or 0.0
            }

            payload = {
                "metrics": self.latest_metrics,
                "report": None,
                "cue": f"Focus on {self.focus_area}"
            }
            try:
                if not self.feedback_queue.full():
                    self.feedback_queue.put_nowait(payload)
            except:
                pass

        return frame

    # -------------------------------
    # Convert Mediapipe landmark to (x,y)
    # -------------------------------
    def _landmark_point(self, landmarks, name):
        idx = LANDMARK_NAMES.get(name)
        if idx is None:
            return None
        lm = landmarks.landmark[idx]
        return (lm.x, lm.y)

    # -------------------------------
    # Update reps & stage
    # -------------------------------
    def _update_reps_and_stage(self):
        if not self.angles:
            return

        primary_angle = list(self.angles.values())[0]
        prev_stage = self.stage

        down_thresh, up_thresh = 160, 50
        if self.exercise_key in ["bicep_curl","push_up","plank"]:
            down_thresh, up_thresh = 160, 40
        elif self.exercise_key in ["squat","lunge","deadlift"]:
            down_thresh, up_thresh = 170, 90
        elif self.exercise_key in ["shoulder_press","tennis_swing","cricket_batting","cricket_bowling"]:
            down_thresh, up_thresh = 160, 50
        elif self.exercise_key in ["running","football_kick"]:
            down_thresh, up_thresh = 160, 70
        elif self.exercise_key in ["shoulder_circles","ankle_pumps","wrist_flexion","wrist_extension"]:
            down_thresh, up_thresh = 160, 0

        if primary_angle > down_thresh:
            self.stage = "down"
        elif primary_angle < up_thresh:
            self.stage = "up"

        if prev_stage == "down" and self.stage == "up":
            self.reps += 1
            now = time.time()
            if self.last_rep_time:
                self.tempo_last = now - self.last_rep_time
            self.last_rep_time = now

# -------------------------------
# Exercise Profiles
# -------------------------------
class ExerciseProfile:
    def __init__(self, key: str, name: str, category: str):
        self.key = key
        self.name = name
        self.category = category

# Dictionary of all exercises
PROFILES = {
    # Gym
    "squat": ExerciseProfile("squat","Squat","Gym"),
    "push_up": ExerciseProfile("push_up","Push Up","Gym"),
    "plank": ExerciseProfile("plank","Plank","Gym"),
    "lunge": ExerciseProfile("lunge","Lunge","Gym"),
    "shoulder_press": ExerciseProfile("shoulder_press","Shoulder Press","Gym"),
    "deadlift": ExerciseProfile("deadlift","Deadlift","Gym"),
    "bicep_curl": ExerciseProfile("bicep_curl","Bicep Curl","Gym"),

    # Sports
    "running": ExerciseProfile("running","Running","Sports"),
    "cricket_batting": ExerciseProfile("cricket_batting","Cricket Batting","Sports"),
    "cricket_bowling": ExerciseProfile("cricket_bowling","Cricket Bowling","Sports"),
    "tennis_swing": ExerciseProfile("tennis_swing","Tennis Swing","Sports"),
    "football_kick": ExerciseProfile("football_kick","Football Kick","Sports"),

    # Physiotherapy
    "shoulder_circles": ExerciseProfile("shoulder_circles","Shoulder Circles","Physiotherapy"),
    "ankle_pumps": ExerciseProfile("ankle_pumps","Ankle Pumps","Physiotherapy"),
    "wrist_flexion": ExerciseProfile("wrist_flexion","Wrist Flexion","Physiotherapy"),
    "wrist_extension": ExerciseProfile("wrist_extension","Wrist Extension","Physiotherapy"),
}

def get_profile(key: str) -> ExerciseProfile:
    return PROFILES.get(key)

def list_profiles_by_category(category: str) -> List[ExerciseProfile]:
    return [p for p in PROFILES.values() if p.category == category]

def list_categories() -> List[str]:
    return ["Gym","Sports","Physiotherapy"]

FOCUS_AREAS = ["Form", "Tempo", "Range of Motion"]

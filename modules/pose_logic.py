"""Pose analysis utilities for the AI Smart Coach application."""
from __future__ import annotations

import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


Point2D = Tuple[float, float]


@dataclass(frozen=True)
class ExerciseProfile:
    key: str
    name: str
    category: str
    description: str
    primary_triplet: Tuple[int, int, int]
    down_threshold: float
    up_threshold: float
    tempo_range: Tuple[float, float]
    rom_target: float
    safety_min_angle: float
    safety_max_angle: float
    cues: Dict[str, str]
    alternate_triplet: Optional[Tuple[int, int, int]] = None
    adaptive_side: bool = False
    min_rom_ratio: float = 0.55
    tempo_tolerance: float = 0.6
    notes: List[str] = field(default_factory=list)


@dataclass
class CoachReport:
    primary_cues: List[str] = field(default_factory=list)
    tempo_feedback: Optional[str] = None
    rom_feedback: Optional[str] = None
    safety_warnings: List[str] = field(default_factory=list)
    balance_score: Optional[float] = None
    speed_score: Optional[float] = None
    angle_of_movement: Optional[float] = None
    foot_placement_score: Optional[float] = None
    energy_efficiency: Optional[float] = None

    def headline(self) -> Optional[str]:
        if self.safety_warnings:
            return self.safety_warnings[0]
        if self.primary_cues:
            return self.primary_cues[0]
        if self.tempo_feedback:
            return self.tempo_feedback
        if self.rom_feedback:
            return self.rom_feedback
        return None

    def to_dict(self) -> Dict[str, object]:
        return {
            "primary_cues": list(self.primary_cues),
            "tempo_feedback": self.tempo_feedback,
            "rom_feedback": self.rom_feedback,
            "safety_warnings": list(self.safety_warnings),
            "balance_score": self.balance_score,
            "speed_score": self.speed_score,
            "angle_of_movement": self.angle_of_movement,
            "foot_placement_score": self.foot_placement_score,
            "energy_efficiency": self.energy_efficiency,
        }


def calculate_angle(a: Point2D, b: Point2D, c: Point2D) -> float:
    a_x, a_y = a
    b_x, b_y = b
    c_x, c_y = c
    angle = np.degrees(
        np.arctan2(c_y - b_y, c_x - b_x) - np.arctan2(a_y - b_y, a_x - b_x)
    )
    angle = (angle + 360) % 360
    if angle > 180:
        angle = 360 - angle
    return angle


@dataclass
class ExerciseCounter:
    profile: ExerciseProfile
    stage: str = "UP"
    reps: int = 0
    last_transition_time: Optional[float] = None
    rep_durations: deque = field(default_factory=lambda: deque(maxlen=50))
    min_angle: float = 1e6
    max_angle: float = 0.0
    last_rom: float = 0.0
    last_tempo: Optional[float] = None
    last_angle: Optional[float] = None

    def reset_rom_window(self, angle: float) -> None:
        self.min_angle = angle
        self.max_angle = angle

    def update(self, angle: float, now: Optional[float] = None) -> Dict[str, object]:
        if now is None:
            now = time.time()

        if self.last_angle is None:
            self.last_angle = angle
        angle_delta = abs(angle - self.last_angle)
        self.last_angle = angle

        self.min_angle = min(self.min_angle, angle)
        self.max_angle = max(self.max_angle, angle)

        rep_completed = False
        rep_duration: Optional[float] = None
        quality_cues: List[str] = []

        down_trigger = max(0.0, self.profile.down_threshold - 4.0)
        up_trigger = min(180.0, self.profile.up_threshold + 4.0)

        if angle <= down_trigger and self.stage == "UP":
            self.stage = "DOWN"
            self.reset_rom_window(angle)
            self.last_transition_time = now
        elif angle >= up_trigger and self.stage == "DOWN":
            self.stage = "UP"
            rep_duration = (
                now - self.last_transition_time if self.last_transition_time else None
            )
            rep_range = self.max_angle - self.min_angle
            required_rom = max(12.0, self.profile.rom_target * self.profile.min_rom_ratio)
            rom_valid = rep_range >= required_rom

            tempo_valid = True
            if rep_duration is not None:
                self.last_tempo = rep_duration
                min_tempo = max(
                    0.4, self.profile.tempo_range[0] * self.profile.tempo_tolerance
                )
                if rep_duration < min_tempo:
                    tempo_valid = False

            rep_completed = rom_valid and tempo_valid
            if rep_completed:
                if rep_duration is not None:
                    self.rep_durations.append(rep_duration)
                self.reps += 1
                self.last_rom = rep_range
            else:
                if not rom_valid:
                    quality_cues.append(
                        "Complete the full range before finishing the rep."
                    )
                if rep_duration is not None and not tempo_valid:
                    quality_cues.append("Slow the tempo for better control.")
                self.last_rom = rep_range
            self.reset_rom_window(angle)

        avg_tempo = statistics.mean(self.rep_durations) if self.rep_durations else None
        tempo_state: Optional[str] = None
        if rep_duration is not None:
            low, high = self.profile.tempo_range
            if rep_duration < low:
                tempo_state = "fast"
            elif rep_duration > high:
                tempo_state = "slow"
            else:
                tempo_state = "on_target"

        safety_flags: List[str] = []
        if angle < self.profile.safety_min_angle:
            safety_flags.append("Avoid collapsing the joint  protect the hinge angle.")
        if angle > self.profile.safety_max_angle:
            safety_flags.append("Do not hyperextend past the safe range.")

        rom_current = self.max_angle - self.min_angle
        rom_ratio = (
            (self.last_rom / self.profile.rom_target)
            if self.profile.rom_target > 0 and self.last_rom
            else None
        )

        return {
            "stage": self.stage,
            "reps": self.reps,
            "angle": angle,
            "angle_delta": angle_delta,
            "rom_current": rom_current,
            "rom_last": self.last_rom if self.last_rom else rom_current,
            "rom_ratio": rom_ratio,
            "tempo_last": rep_duration or self.last_tempo,
            "tempo_avg": avg_tempo,
            "tempo_state": tempo_state,
            "rep_completed": rep_completed,
            "safety_flags": safety_flags,
            "quality_cues": quality_cues,
            "timestamp": now,
        }


def _get_point(
    landmarks: Optional[object], index: int
) -> Optional[Tuple[float, float, float]]:
    if not landmarks:
        return None
    try:
        landmark = landmarks.landmark[index]
    except (AttributeError, IndexError):
        return None
    return (landmark.x, landmark.y, getattr(landmark, "z", 0.0))


def _calc_slope(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    dy = b[1] - a[1]
    dx = b[0] - a[0]
    if abs(dx) < 1e-6:
        return float("inf")
    return dy / dx


def _calc_verticality(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    vector = np.array([b[0] - a[0], b[1] - a[1]])
    vertical = np.array([0.0, 1.0])
    cos_angle = np.dot(vector, vertical) / (
        np.linalg.norm(vector) * np.linalg.norm(vertical) + 1e-6
    )
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _calc_knee_valgus(
    hip: Tuple[float, float, float],
    knee: Tuple[float, float, float],
    ankle: Tuple[float, float, float],
) -> float:
    knee_over_foot = knee[0] - ankle[0]
    hip_over_foot = hip[0] - ankle[0]
    return float(knee_over_foot - hip_over_foot)


def _calc_symmetry(
    left: Optional[Tuple[float, float, float]],
    right: Optional[Tuple[float, float, float]],
) -> Optional[float]:
    if not left or not right:
        return None
    return round(1.0 - min(1.0, abs(left[1] - right[1]) / 0.15), 2)


EXERCISE_LIBRARY: Dict[str, ExerciseProfile] = {
    "squat": ExerciseProfile(
        key="squat",
        name="Squat",
        category="Gym Coaching",
        description="Tracks hip-knee-ankle stacking for depth and posture.",
        primary_triplet=(24, 26, 28),
        down_threshold=100.0,
        up_threshold=155.0,
        tempo_range=(1.3, 3.5),
        rom_target=65.0,
        safety_min_angle=40.0,
        safety_max_angle=175.0,
        cues={
            "form": "Drive knees out and keep chest proud.",
            "tempo": "Smooth three-count squat: 2s down, 1s up.",
        },
        notes=["Track hip depth relative to knee for safe range."],
    ),
    "push_up": ExerciseProfile(
        key="push_up",
        name="Push-up",
        category="Gym Coaching",
        description="Monitors elbow flexion and core alignment for push-ups.",
        primary_triplet=(12, 14, 16),
        down_threshold=70.0,
        up_threshold=165.0,
        tempo_range=(1.0, 2.5),
        rom_target=90.0,
        safety_min_angle=45.0,
        safety_max_angle=175.0,
        cues={
            "form": "Keep your hips level with shoulders; avoid sagging.",
            "tempo": "Lower under control and press explosively without bouncing.",
        },
    ),
    "plank": ExerciseProfile(
        key="plank",
        name="Plank",
        category="Gym Coaching",
        description="Evaluates neutral spine alignment during static planks.",
        primary_triplet=(12, 24, 28),
        down_threshold=150.0,
        up_threshold=175.0,
        tempo_range=(10.0, 90.0),
        rom_target=15.0,
        safety_min_angle=120.0,
        safety_max_angle=190.0,
        cues={
            "form": "Brace the core  avoid arching or rounding the lower back.",
            "tempo": "Hold steady breathing; maintain calm cadence.",
        },
        min_rom_ratio=0.3,
    ),
    "deadlift": ExerciseProfile(
        key="deadlift",
        name="Deadlift",
        category="Gym Coaching",
        description="Monitors hip hinge mechanics and spinal neutrality.",
        primary_triplet=(24, 26, 28),
        down_threshold=95.0,
        up_threshold=165.0,
        tempo_range=(1.5, 3.5),
        rom_target=70.0,
        safety_min_angle=55.0,
        safety_max_angle=175.0,
        cues={
            "form": "Push the floor away and keep the bar close  hinge from hips.",
            "tempo": "Controlled pull with deliberate lockout.",
        },
    ),
    "lunge": ExerciseProfile(
        key="lunge",
        name="Lunge",
        category="Gym Coaching",
        description="Evaluates front knee tracking and torso stability.",
        primary_triplet=(24, 26, 28),
        down_threshold=105.0,
        up_threshold=165.0,
        tempo_range=(1.5, 3.0),
        rom_target=60.0,
        safety_min_angle=50.0,
        safety_max_angle=175.0,
        cues={
            "form": "Keep front knee stacked over the ankle and torso upright.",
            "tempo": "Dip smoothly  no bouncing at the bottom.",
        },
    ),
    "shoulder_press": ExerciseProfile(
        key="shoulder_press",
        name="Shoulder Press",
        category="Gym Coaching",
        description="Tracks overhead press mechanics for scapular rhythm.",
        primary_triplet=(12, 14, 16),
        down_threshold=70.0,
        up_threshold=160.0,
        tempo_range=(1.2, 2.8),
        rom_target=95.0,
        safety_min_angle=50.0,
        safety_max_angle=185.0,
        cues={
            "form": "Lock ribs down, press overhead, and finish with biceps by ears.",
            "tempo": "Drive up for 1s, control the descent for 2s.",
        },
    ),
    "bicep_curl": ExerciseProfile(
        key="bicep_curl",
        name="Bicep Curl",
        category="Gym Coaching",
        description="Measures elbow flexion for curling variations.",
        primary_triplet=(11, 13, 15),
        down_threshold=60.0,
        up_threshold=155.0,
        tempo_range=(1.0, 2.5),
        rom_target=110.0,
        safety_min_angle=20.0,
        safety_max_angle=185.0,
        cues={
            "form": "Pin elbows to your sides; avoid swinging the torso.",
            "tempo": "Lift smooth and resist on the way down.",
        },
        alternate_triplet=(12, 14, 16),
        adaptive_side=True,
        min_rom_ratio=0.6,
        tempo_tolerance=0.7,
    ),
    "custom_dataset": ExerciseProfile(
        key="custom_dataset",
        name="Custom Dataset Movement",
        category="Gym Coaching",
        description="Generic template for newly added dataset movements.",
        primary_triplet=(12, 14, 16),
        down_threshold=120.0,
        up_threshold=160.0,
        tempo_range=(1.0, 4.0),
        rom_target=60.0,
        safety_min_angle=30.0,
        safety_max_angle=185.0,
        cues={
            "form": "Maintain control and follow your custom protocol cues.",
            "tempo": "Match the intended cadence for this dataset clip.",
        },
    ),
    "jump": ExerciseProfile(
        key="jump",
        name="Jumping Mechanics",
        category="Sports Analysis",
        description="Tracks countermovement depth, arm swing, and landing symmetry.",
        primary_triplet=(24, 26, 28),
        down_threshold=105.0,
        up_threshold=165.0,
        tempo_range=(0.8, 2.0),
        rom_target=70.0,
        safety_min_angle=60.0,
        safety_max_angle=175.0,
        cues={
            "form": "Load hips back and land softly through the mid-foot.",
            "tempo": "Snap quickly but absorb under control.",
        },
    ),
    "running": ExerciseProfile(
        key="running",
        name="Running Stride",
        category="Sports Analysis",
        description="Assesses stride cadence, hip extension, and foot strike.",
        primary_triplet=(24, 26, 28),
        down_threshold=115.0,
        up_threshold=170.0,
        tempo_range=(0.4, 1.0),
        rom_target=55.0,
        safety_min_angle=50.0,
        safety_max_angle=180.0,
        cues={
            "form": "Drive knees forward and keep foot strike beneath the hip.",
            "tempo": "Cadence steady  aim for 170-180 steps/min.",
        },
    ),
    "throw": ExerciseProfile(
        key="throw",
        name="Throwing Mechanics",
        category="Sports Analysis",
        description="Examines shoulder external rotation and trunk sequencing.",
        primary_triplet=(12, 14, 16),
        down_threshold=80.0,
        up_threshold=150.0,
        tempo_range=(0.6, 1.5),
        rom_target=90.0,
        safety_min_angle=40.0,
        safety_max_angle=195.0,
        cues={
            "form": "Lead with the elbow and finish with full follow-through.",
            "tempo": "Explode through release while keeping control.",
        },
    ),
    "cricket": ExerciseProfile(
        key="cricket",
        name="Cricket Bat/Bowl",
        category="Sports Analysis",
        description="Monitors front-shoulder alignment and hip-shoulder separation.",
        primary_triplet=(12, 14, 16),
        down_threshold=90.0,
        up_threshold=160.0,
        tempo_range=(0.7, 1.6),
        rom_target=85.0,
        safety_min_angle=45.0,
        safety_max_angle=190.0,
        cues={
            "form": "Stay tall through delivery and rotate through the hips.",
            "tempo": "Rhythmic run-up with crisp release.",
        },
    ),
    "football_kick": ExerciseProfile(
        key="football_kick",
        name="Football Kick",
        category="Sports Analysis",
        description="Tracks plant foot stability and swing-leg extension.",
        primary_triplet=(24, 26, 28),
        down_threshold=105.0,
        up_threshold=170.0,
        tempo_range=(0.6, 1.4),
        rom_target=75.0,
        safety_min_angle=50.0,
        safety_max_angle=185.0,
        cues={
            "form": "Lock plant foot and snap through the ball.",
            "tempo": "Smooth approach, explosive strike.",
        },
    ),
    "tennis_swing": ExerciseProfile(
        key="tennis_swing",
        name="Tennis Swing",
        category="Sports Analysis",
        description="Evaluates racket drop, hip rotation, and follow-through.",
        primary_triplet=(12, 14, 16),
        down_threshold=85.0,
        up_threshold=150.0,
        tempo_range=(0.5, 1.5),
        rom_target=90.0,
        safety_min_angle=40.0,
        safety_max_angle=190.0,
        cues={
            "form": "Load hips, rotate shoulders, and finish across the body.",
            "tempo": "Fluid acceleration through contact.",
        },
    ),
    "physio_posture": ExerciseProfile(
        key="physio_posture",
        name="Physio Posture Drill",
        category="Physiotherapy",
        description="Guides neutral posture and micro-movements for rehabilitation.",
        primary_triplet=(12, 24, 28),
        down_threshold=150.0,
        up_threshold=175.0,
        tempo_range=(5.0, 60.0),
        rom_target=20.0,
        safety_min_angle=130.0,
        safety_max_angle=190.0,
        cues={
            "form": "Stack ears over shoulders and hips.",
            "tempo": "Slow, mindful breathing to reduce tension.",
        },
        min_rom_ratio=0.25,
        tempo_tolerance=0.8,
    ),
    "physio_compensation": ExerciseProfile(
        key="physio_compensation",
        name="Compensation Detection",
        category="Physiotherapy",
        description="Detects uneven loading and compensatory shifts.",
        primary_triplet=(23, 25, 27),
        down_threshold=120.0,
        up_threshold=170.0,
        tempo_range=(2.0, 6.0),
        rom_target=45.0,
        safety_min_angle=60.0,
        safety_max_angle=185.0,
        cues={
            "form": "Distribute weight evenly left-to-right.",
            "tempo": "Smooth transitions without jerks.",
        },
        min_rom_ratio=0.4,
    ),
    "physio_control": ExerciseProfile(
        key="physio_control",
        name="Controlled Rehab Motion",
        category="Physiotherapy",
        description="Ensures gentle, pain-free range with tempo monitoring.",
        primary_triplet=(11, 13, 15),
        down_threshold=135.0,
        up_threshold=170.0,
        tempo_range=(3.0, 8.0),
        rom_target=35.0,
        safety_min_angle=70.0,
        safety_max_angle=185.0,
        cues={
            "form": "Glide through range without sudden acceleration.",
            "tempo": "Count 3 seconds each direction.",
        },
        min_rom_ratio=0.35,
        tempo_tolerance=0.75,
    ),
}


DEFAULT_PROFILE_KEY = "bicep_curl"

FOCUS_AREAS = [
    "None",
    "Lower back",
    "Knees",
    "Shoulders",
    "Ankles",
    "Elbows",
    "Neck",
]


def get_profile(key: str) -> ExerciseProfile:
    return EXERCISE_LIBRARY.get(key, EXERCISE_LIBRARY[DEFAULT_PROFILE_KEY])


def list_profiles_by_category(category: str) -> List[ExerciseProfile]:
    return sorted(
        (profile for profile in EXERCISE_LIBRARY.values() if profile.category == category),
        key=lambda profile: profile.name,
    )


def list_categories() -> List[str]:
    return sorted({profile.category for profile in EXERCISE_LIBRARY.values()})


def generate_coaching_report(
    profile: ExerciseProfile,
    metrics: Dict[str, object],
    normalized_landmarks: Optional[object],
    world_landmarks: Optional[object],
    focus_area: Optional[str] = None,
) -> CoachReport:
    _ = world_landmarks
    report = CoachReport()

    angle = float(metrics.get("angle", 0.0))
    rom_ratio = metrics.get("rom_ratio")
    tempo_state = metrics.get("tempo_state")
    tempo_last = metrics.get("tempo_last")
    angle_delta = float(metrics.get("angle_delta", 0.0))
    rep_completed = bool(metrics.get("rep_completed"))

    if tempo_state == "fast":
        report.tempo_feedback = "Tempo too fast  slow the eccentric phase."
    elif tempo_state == "slow":
        report.tempo_feedback = "Tempo too slow  add controlled speed."
    elif tempo_last:
        report.tempo_feedback = f"Tempo steady at {tempo_last:.2f}s per rep."

    if rom_ratio is not None:
        if rom_ratio < 0.75:
            report.rom_feedback = "Increase range of motion to hit full depth."
        elif rom_ratio > 1.1:
            report.rom_feedback = "Range looks aggressive  ensure control."
        else:
            report.rom_feedback = "Solid range of motion maintained."

    report.angle_of_movement = round(angle, 1)

    left_shoulder = _get_point(normalized_landmarks, 11)
    right_shoulder = _get_point(normalized_landmarks, 12)
    left_hip = _get_point(normalized_landmarks, 23)
    right_hip = _get_point(normalized_landmarks, 24)
    left_knee = _get_point(normalized_landmarks, 25)
    right_knee = _get_point(normalized_landmarks, 26)
    left_ankle = _get_point(normalized_landmarks, 27)
    right_ankle = _get_point(normalized_landmarks, 28)
    left_elbow = _get_point(normalized_landmarks, 13)
    left_wrist = _get_point(normalized_landmarks, 15)

    report.balance_score = _calc_symmetry(left_hip, right_hip)
    if report.balance_score is None and left_shoulder and right_shoulder:
        report.balance_score = _calc_symmetry(left_shoulder, right_shoulder)

    if tempo_last:
        optimal = sum(profile.tempo_range) / 2
        report.speed_score = round(max(0.0, 1.0 - abs(tempo_last - optimal) / optimal), 2)
        report.energy_efficiency = round(
            min(1.0, max(0.0, tempo_last / profile.tempo_range[1])), 2
        )

    if left_ankle and right_ankle:
        stance_width = abs(left_ankle[0] - right_ankle[0])
        report.foot_placement_score = round(max(0.0, min(1.0, stance_width / 0.6)), 2)

    if focus_area and focus_area != "None":
        report.safety_warnings.append(
            f"Protect your {focus_area.lower()}  move with control."
        )

    def add_primary(message: str) -> None:
        if message and message not in report.primary_cues:
            report.primary_cues.append(message)

    def add_warning(message: str) -> None:
        if message and message not in report.safety_warnings:
            report.safety_warnings.append(message)

    for flag in metrics.get("safety_flags", []):
        add_warning(flag)

    for cue in metrics.get("quality_cues", []):
        add_primary(cue)

    if profile.key == "squat":
        if left_hip and left_knee and left_ankle:
            valgus = _calc_knee_valgus(left_hip, left_knee, left_ankle)
            if abs(valgus) > 0.05:
                add_primary("Drive knees outward to track over toes.")
        if left_shoulder and left_hip:
            torso_angle = _calc_verticality(left_shoulder, left_hip)
            if torso_angle > 20:
                add_primary("Keep chest up  hinge from hips without folding forward.")
    elif profile.key == "push_up":
        if left_hip and left_shoulder:
            hip_drop = left_hip[1] - left_shoulder[1]
            if hip_drop < -0.05:
                add_primary("Lift hips  avoid sagging through the midsection.")
        if left_elbow and left_shoulder and left_elbow[0] - left_shoulder[0] > 0.2:
            add_primary("Tuck elbows closer to your ribs for shoulder safety.")
    elif profile.key == "plank":
        if left_shoulder and left_hip and left_ankle:
            shoulder_hip = abs(left_shoulder[1] - left_hip[1])
            hip_ankle = abs(left_hip[1] - left_ankle[1])
            if abs(shoulder_hip - hip_ankle) > 0.08:
                add_primary("Create a straight line from shoulders to heels.")
    elif profile.key == "deadlift":
        if left_shoulder and left_hip:
            back_angle = _calc_verticality(left_shoulder, left_hip)
            if back_angle > 25:
                add_warning("Maintain a neutral spine  engage lats and brace core.")
        if left_knee and left_ankle and left_hip:
            shin_slope = _calc_slope(left_ankle, left_knee)
            if abs(shin_slope) > 0.5:
                add_primary("Push hips back to load hamstrings instead of knees.")
    elif profile.key == "lunge":
        if left_knee and left_ankle:
            knee_over_toe = left_knee[0] - left_ankle[0]
            if knee_over_toe > 0.12:
                add_warning("Keep front knee stacked over ankle to protect joints.")
    elif profile.key == "shoulder_press":
        if left_elbow and left_wrist and left_elbow[0] - left_wrist[0] > 0.05:
            add_primary("Press straight overhead  avoid drifting bar forward.")
    elif profile.key == "bicep_curl":
        hip = _get_point(normalized_landmarks, 23)
        if left_elbow and hip and abs(left_elbow[0] - hip[0]) > 0.1:
            add_primary("Pin elbows to your sides for strict curls.")
    elif profile.key == "jump":
        if left_ankle and right_ankle:
            landing_symmetry = abs(left_ankle[1] - right_ankle[1])
            if landing_symmetry > 0.04:
                add_warning("Land evenly on both feet to absorb impact safely.")
        if rep_completed and tempo_state == "fast":
            add_primary("Load deeper before take-off to generate more force.")
    elif profile.key == "running":
        if left_ankle and left_knee:
            overstride = left_ankle[0] - left_knee[0]
            if overstride > 0.15:
                add_primary("Strike closer under hips to reduce braking forces.")
    elif profile.key == "throw":
        left_wrist_extra = _get_point(normalized_landmarks, 15)
        if left_wrist_extra and left_elbow and left_wrist_extra[1] > left_elbow[1] + 0.05:
            add_primary("Lead with the elbow  keep wrist stacked until release.")
    elif profile.key == "cricket":
        if left_shoulder and right_shoulder:
            shoulder_drop = right_shoulder[1] - left_shoulder[1]
            if shoulder_drop > 0.12:
                add_primary("Stay tall through delivery  avoid excessive shoulder dip.")
    elif profile.key == "football_kick":
        if right_knee and right_ankle and right_knee[0] - right_ankle[0] < -0.05:
            add_primary("Snap through with full extension and follow-through.")
    elif profile.key == "tennis_swing":
        if left_shoulder and left_hip:
            rotation = abs(left_shoulder[0] - left_hip[0])
            if rotation < 0.08:
                add_primary("Rotate hips and shoulders together for more power.")
    elif profile.category == "Physiotherapy":
        if angle_delta > 25:
            add_warning("Slow down  keep rehab movements smooth and pain-free.")
        if report.balance_score is not None and report.balance_score < 0.7:
            add_primary("Shift weight evenly to avoid compensation.")

    if not report.primary_cues:
        add_primary(profile.cues.get("form", "Maintain good alignment."))

    return report

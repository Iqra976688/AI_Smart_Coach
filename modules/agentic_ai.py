"""
Agentic coaching module for AI Smart Coach.

Provides:
- AgenticCoach class (import as AgenticCoach)
- Methods:
    - update_progress(user_id, exercise_name, performance_score)
    - choose_next_exercise(user_id)
    - get_feedback(user_id)
    - enhance_feedback(cue, movement, metrics, user_id=None)

Self-contained and defensive so missing pieces in the host app won't break imports.
"""

from __future__ import annotations
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class AgenticCoach:
    def __init__(
        self,
        exercise_list: Optional[List[Dict]] = None,
        feedback_messages: Optional[Dict[str, List[str]]] = None,
        motivational_messages: Optional[List[str]] = None,
    ):
        # Default comprehensive exercise list (Gym / Sports / Physiotherapy)
        self.exercise_list = exercise_list or [
            {"name": "Squat", "key": "squat", "difficulty": 2, "duration": 30},
            {"name": "Push-Up", "key": "push_up", "difficulty": 2, "duration": 20},
            {"name": "Plank", "key": "plank", "difficulty": 2, "duration": 40},
            {"name": "Lunge", "key": "lunge", "difficulty": 2, "duration": 25},
            {"name": "Deadlift", "key": "deadlift", "difficulty": 3, "duration": 30},
            {"name": "Shoulder Press", "key": "shoulder_press", "difficulty": 3, "duration": 20},
            {"name": "Bicep Curl", "key": "bicep_curl", "difficulty": 1, "duration": 20},
            {"name": "Running (stride)", "key": "running_stride", "difficulty": 2, "duration": 60},
            {"name": "Cricket Bat", "key": "cricket_bat", "difficulty": 3, "duration": 20},
            {"name": "Cricket Bowl", "key": "cricket_bowl", "difficulty": 3, "duration": 20},
            {"name": "Tennis Swing", "key": "tennis_swing", "difficulty": 3, "duration": 20},
            {"name": "Football Kick", "key": "football_kick", "difficulty": 2, "duration": 15},
            {"name": "Shoulder Circles", "key": "shoulder_circles", "difficulty": 1, "duration": 30},
            {"name": "Ankle Pumps", "key": "ankle_pumps", "difficulty": 1, "duration": 30},
            {"name": "Wrist Flexion/Extension", "key": "wrist_flexion_extension", "difficulty": 1, "duration": 20},
        ]

        # Default feedback messages
        default_feedback = {
            "Squat": [
                "Drive hips back, keep chest up and knees tracking over toes.",
                "Pause briefly at depth and drive through heels for a strong finish.",
            ],
            "Push-Up": [
                "Keep a straight plank line and drive through the chest without elbow flare.",
                "Lower with control, then press explosively while keeping core engaged.",
            ],
            "Plank": [
                "Create a straight line from head to heels and brace the core.",
                "Avoid sagging — imagine stacking shoulders over hips.",
            ],
            "Lunge": [
                "Keep front knee stacked over ankle and torso upright.",
                "Step with control and push from the front heel to return.",
            ],
            "Deadlift": [
                "Hinge at the hips, keep the bar close and maintain a neutral spine.",
                "Drive hips forward at the top rather than pulling with the back.",
            ],
            "Shoulder Press": [
                "Brace your core, press overhead in a straight line and avoid leaning back.",
                "Control the descent and avoid locking the elbows aggressively.",
            ],
            "Bicep Curl": [
                "Pin elbows to sides and move only the forearm—no torso swing.",
                "Squeeze at the top and lower under control for full tension.",
            ],
            "Running (stride)": [
                "Shorten ground contact, increase cadence slightly and keep torso upright.",
                "Drive the knee and extend the ankle for efficient push-off.",
            ],
            "Cricket Bat": [
                "Rotate the hips then the shoulders; keep eyes on the ball and follow through.",
                "Transfer weight from back to front foot on contact for power.",
            ],
            "Cricket Bowl": [
                "Run-up rhythm and arm path matter—keep a strong front arm for balance.",
                "Land softly and drive through the hips for pace.",
            ],
            "Tennis Swing": [
                "Rotate the hips first, then bring the racket through with relaxed wrists.",
                "Use your legs and core for power rather than only the arm.",
            ],
            "Football Kick": [
                "Plant supporting foot beside the ball and strike with the instep.",
                "Follow through toward target for accuracy and power.",
            ],
            "Shoulder Circles": [
                "Move slowly through full comfortable range and avoid pain.",
                "Keep movements controlled and breathe to relax the shoulder girdle.",
            ],
            "Ankle Pumps": [
                "Point and flex the foot slowly to promote circulation and mobility.",
                "Keep motion within comfort and avoid forcing range.",
            ],
            "Wrist Flexion/Extension": [
                "Move smoothly through flexion and extension, maintain a relaxed grip.",
                "Control the tempo to reduce tendon strain and build controlled strength.",
            ],
        }
        self.feedback_messages = default_feedback if feedback_messages is None else {**default_feedback, **feedback_messages}

        # Motivational messages
        self.motivational_messages = motivational_messages or [
            "Great job — keep it up!",
            "Nice form — a little more control now.",
            "You're improving every rep!",
            "Consistency beats intensity — keep showing up.",
            "Small wins add up — good work!",
        ]

        # User tracking
        self.user_progress: Dict[str, Dict] = {}
        self.daily_streaks: Dict[str, int] = {}
        self.last_session_time: Dict[str, datetime] = {}

    # ---------------------------
    # Core Methods
    # ---------------------------
    def update_progress(self, user_id: str, exercise_name: str, performance_score: float) -> None:
        now = datetime.now()
        self.user_progress[user_id] = {
            "exercise": exercise_name,
            "score": max(0.0, min(100.0, float(performance_score))),
            "timestamp": now,
        }
        last_time = self.last_session_time.get(user_id)
        if last_time:
            delta_days = (now.date() - last_time.date()).days
            if delta_days == 1:
                self.daily_streaks[user_id] = self.daily_streaks.get(user_id, 0) + 1
            elif delta_days > 1:
                self.daily_streaks[user_id] = 1
        else:
            self.daily_streaks[user_id] = 1
        self.last_session_time[user_id] = now

    def choose_next_exercise(self, user_id: str) -> Dict:
        # If no history: pick easy
        if user_id not in self.user_progress:
            easy = [ex for ex in self.exercise_list if ex.get("difficulty", 1) == 1]
            return random.choice(easy) if easy else random.choice(self.exercise_list)

        last = self.user_progress[user_id]
        last_name = last.get("exercise")
        last_score = float(last.get("score", 0.0))
        last_difficulty = next((ex.get("difficulty", 1) for ex in self.exercise_list if ex["name"] == last_name or ex.get("key") == last_name), 1)

        if last_score >= 85:
            desired = last_difficulty + 1
            candidates = [ex for ex in self.exercise_list if ex.get("difficulty", 1) == desired] or \
                         [ex for ex in self.exercise_list if ex.get("difficulty", 1) == last_difficulty]
        elif last_score < 55:
            candidates = [ex for ex in self.exercise_list if ex.get("difficulty", 1) <= last_difficulty]
        else:
            candidates = [ex for ex in self.exercise_list if ex.get("difficulty", 1) == last_difficulty]

        return random.choice(candidates) if candidates else random.choice(self.exercise_list)

    def get_feedback(self, user_id: str) -> List[str]:
        messages: List[str] = []
        if user_id in self.user_progress:
            last_ex = self.user_progress[user_id].get("exercise")
            msgs = self.feedback_messages.get(last_ex, [])
            if msgs:
                messages.append(random.choice(msgs))
        messages.append(random.choice(self.motivational_messages))
        return messages

    def check_rest_needed(self, user_id: str, threshold_minutes: int = 5) -> bool:
        last_time = self.last_session_time.get(user_id)
        if not last_time:
            return False
        return (datetime.now() - last_time) < timedelta(minutes=threshold_minutes)

    def _heuristic_score_from_metrics(self, metrics: Optional[Dict]) -> float:
        if not metrics:
            return 60.0
        numeric_vals = []
        explicit = metrics.get("score")
        if isinstance(explicit, (int, float)):
            numeric_vals.append(max(0.0, min(100.0, explicit)))
        for key in ("balance_score", "speed_score", "foot_placement_score", "energy_efficiency"):
            v = metrics.get(key)
            if isinstance(v, (int, float)):
                numeric_vals.append(float(v)*100 if 0.0 <= v <= 1.0 else float(v))
        reps = metrics.get("reps")
        if isinstance(reps, (int, float)) and reps > 0:
            numeric_vals.append(min(95.0, 50.0 + float(reps)*4.0))
        angle = metrics.get("angle")
        if isinstance(angle, (int, float)):
            numeric_vals.append(70.0)
        tempo_state = metrics.get("tempo_state")
        if tempo_state == "on_target":
            numeric_vals.append(80.0)
        elif tempo_state == "fast":
            numeric_vals.append(55.0)
        elif tempo_state == "slow":
            numeric_vals.append(60.0)
        return float(sum(numeric_vals)/len(numeric_vals)) if numeric_vals else 60.0

    def enhance_feedback(
        self,
        cue: str,
        movement: str,
        metrics: Optional[Dict] = None,
        user_id: Optional[str] = None,
    ) -> str:
        score = self._heuristic_score_from_metrics(metrics)
        uid = user_id or "anon"
        try:
            self.update_progress(uid, movement, score)
        except Exception:
            pass
        next_ex = self.choose_next_exercise(uid)
        ex_msgs = self.feedback_messages.get(movement, [])
        specific = random.choice(ex_msgs) if ex_msgs else f"Focus on controlled movement for {movement}."
        motivate = random.choice(self.motivational_messages)
        score_text = f"Performance score: {int(round(score))}/100."
        suggestion = f"Next: {next_ex.get('name', next_ex.get('key','Unknown'))} ({next_ex.get('duration','')}s)."
        combined = f"{specific} {score_text} {suggestion} {motivate}"
        tokens = combined.split()
        if len(tokens) > 40:
            combined = " ".join(tokens[:40]) + "..."
        return combined

if __name__ == "__main__":
    agent = AgenticCoach()
    demo_metrics = {"reps": 5, "angle": 95, "tempo_last": 1.2, "balance_score": 0.9}
    print(agent.enhance_feedback("Knees drifting inward on descent", "Squat", demo_metrics, user_id="demo_user"))

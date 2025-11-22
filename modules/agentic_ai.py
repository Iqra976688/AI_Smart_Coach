# agentic_ai.py
"""
Agentic coaching module for AI Smart Coach.

Provides:
- AgenticCoach class (import as AgenticCoach)
- Methods:
    - update_progress(user_id, exercise_name, performance_score)
    - choose_next_exercise(user_id)
    - get_feedback(user_id)
    - enhance_feedback(cue, movement, metrics, user_id=None)

This is self-contained and has sensible defaults so it won't raise import errors.
"""

from __future__ import annotations
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class AgenticCoach:
    """
    A lightweight agent that decides what exercise to suggest next
    and synthesizes short guidance based on past performance.

    Usage:
        from agentic_ai import AgenticCoach
        agent = AgenticCoach()
        agent.update_progress("user1", "Squat", 82)
        next_ex = agent.choose_next_exercise("user1")
        feedback = agent.enhance_feedback("Knees drifting", "Squat", metrics, user_id="user1")
    """

    def __init__(
        self,
        exercise_list: Optional[List[Dict]] = None,
        feedback_messages: Optional[Dict[str, List[str]]] = None,
        motivational_messages: Optional[List[str]] = None,
    ):
        # Default exercises (you can replace from your main app)
        self.exercise_list = exercise_list or [
            {"name": "Squat", "difficulty": 1, "duration": 30},
            {"name": "Push-Up", "difficulty": 1, "duration": 20},
            {"name": "Plank", "difficulty": 2, "duration": 40},
            {"name": "Lunge", "difficulty": 2, "duration": 25},
            {"name": "Deadlift (light)", "difficulty": 3, "duration": 20},
        ]

        # Default feedback messages keyed by exercise
        self.feedback_messages = feedback_messages or {
            "Squat": [
                "Keep your knees aligned and push hips back.",
                "Engage core to maintain posture.",
                "Lower until thighs are parallel, then drive up from the heels.",
            ],
            "Push-Up": [
                "Keep your hips straight and core tight.",
                "Lower smoothly and avoid elbow flare.",
            ],
            "Plank": [
                "Keep a straight line from head to heels.",
                "Avoid sagging in the lower back.",
            ],
        }

        self.motivational_messages = motivational_messages or [
            "Great job — keep it up!",
            "Nice form — a little more control now.",
            "You're improving every rep!",
        ]

        # user_id -> last progress dict
        self.user_progress: Dict[str, Dict] = {}
        self.daily_streaks: Dict[str, int] = {}
        self.last_session_time: Dict[str, datetime] = {}

    def update_progress(self, user_id: str, exercise_name: str, performance_score: float) -> None:
        """
        Store the latest performance score (0-100) for a user and update streaks.
        """
        now = datetime.now()
        self.user_progress[user_id] = {
            "exercise": exercise_name,
            "score": float(max(0.0, min(100.0, performance_score))),
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
        """
        Simple policy:
        - If no history: pick easiest (difficulty == 1)
        - If last score >= 80: try slightly harder
        - If last score < 50: suggest same or easier
        - Otherwise: keep same difficulty
        Returns a dict describing the chosen exercise.
        """
        if user_id not in self.user_progress:
            candidates = [ex for ex in self.exercise_list if ex.get("difficulty", 1) == 1]
            if not candidates:
                return random.choice(self.exercise_list)
            return random.choice(candidates)

        last = self.user_progress[user_id]
        last_ex_name = last.get("exercise")
        last_score = float(last.get("score", 0.0))
        last_difficulty = next((ex["difficulty"] for ex in self.exercise_list if ex["name"] == last_ex_name), 1)

        if last_score >= 80:
            desired = last_difficulty + 1
            candidates = [ex for ex in self.exercise_list if ex.get("difficulty", 1) == desired]
            if not candidates:
                candidates = [ex for ex in self.exercise_list if ex.get("difficulty", 1) == last_difficulty]
        elif last_score < 50:
            candidates = [ex for ex in self.exercise_list if ex.get("difficulty", 1) <= last_difficulty]
        else:
            candidates = [ex for ex in self.exercise_list if ex.get("difficulty", 1) == last_difficulty]

        if not candidates:
            candidates = self.exercise_list
        return random.choice(candidates)

    def get_feedback(self, user_id: str) -> List[str]:
        """
        Return a short list: [exercise_specific_feedback, motivational_message]
        """
        messages = []
        if user_id in self.user_progress:
            last_ex = self.user_progress[user_id].get("exercise")
            msgs = self.feedback_messages.get(last_ex, [])
            if msgs:
                messages.append(random.choice(msgs))
        messages.append(random.choice(self.motivational_messages))
        return messages

    def check_rest_needed(self, user_id: str, threshold_minutes: int = 5) -> bool:
        """
        Returns True if the user should rest (demo threshold).
        """
        last_time = self.last_session_time.get(user_id)
        if not last_time:
            return False
        return (datetime.now() - last_time) < timedelta(minutes=threshold_minutes)

    def _heuristic_score_from_metrics(self, metrics: Optional[Dict]) -> float:
        """
        Convert observed metrics (angle, reps, balance_score, speed_score, etc.)
        to a 0-100 heuristic performance score. This is approximate and conservative.
        """
        if not metrics:
            return 60.0

        # If explicit score provided, prefer it
        explicit = metrics.get("score")
        if isinstance(explicit, (int, float)):
            return float(max(0.0, min(100.0, explicit)))

        numeric_vals = []
        # Common fields that indicate quality between 0..1
        for key in ("balance_score", "speed_score", "foot_placement_score", "energy_efficiency"):
            v = metrics.get(key)
            if isinstance(v, (int, float)):
                # many of these are 0..1 in our system — convert to 0..100
                if 0.0 <= v <= 1.0:
                    numeric_vals.append(v * 100.0)
                else:
                    numeric_vals.append(float(v))

        # Use reps and tempo to nudge score
        reps = metrics.get("reps")
        if isinstance(reps, (int, float)) and reps > 0:
            numeric_vals.append(min(90.0, 50.0 + float(reps) * 4.0))

        # Angle / ROM heuristics (if ideal target angle provided, we could compare; here we simply reward presence)
        angle = metrics.get("angle")
        if isinstance(angle, (int, float)):
            numeric_vals.append(70.0)  # generic positive signal

        if numeric_vals:
            return float(sum(numeric_vals) / len(numeric_vals))

        # Fallback
        return 60.0

    def enhance_feedback(
        self,
        cue: str,
        movement: str,
        metrics: Optional[Dict] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Produce an enhanced, agentic feedback string.
        - Computes a heuristic performance score from metrics
        - Updates internal user progress (if user_id provided)
        - Chooses a next exercise suggestion
        - Returns a short combined feedback string suitable for UI/TTS
        """
        # Heuristic performance score
        score = self._heuristic_score_from_metrics(metrics)

        uid = user_id or "anon"
        # Update progress so agent learns (simple memory)
        try:
            self.update_progress(uid, movement, score)
        except Exception:
            # Never raise to host app; keep fault-tolerant
            pass

        # Choose next exercise to recommend
        next_ex = self.choose_next_exercise(uid)

        # Compose feedback pieces
        # Prefer exercise-specific messages if available
        ex_msgs = self.feedback_messages.get(movement, [])
        specific = random.choice(ex_msgs) if ex_msgs else f"Focus on controlled movement for {movement}."

        # Motivational
        motivate = random.choice(self.motivational_messages)

        # Short summary of computed score
        score_text = f"Performance score: {int(round(score))}/100."

        # Next exercise suggestion
        suggestion = f"Next: {next_ex['name']} ({next_ex.get('duration', '')}s)."

        # Final combined message (short)
        combined = f"{specific} {score_text} {suggestion} {motivate}"
        # Keep it concise
        return " ".join(str(combined).split())

# If module run directly, simple demo:
if __name__ == "__main__":
    agent = AgenticCoach()
    print(agent.enhance_feedback("Knees drifting", "Squat", {"reps": 5, "angle": 95}))

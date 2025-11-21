# ---------------- Complete Agentic AI Module ----------------
# Handles adaptive exercise decisions, feedback, motivation, streaks, rest, reminders, and safety alerts

import random
from datetime import datetime, timedelta

class AgenticAI:
    def __init__(self, exercise_list, feedback_messages, motivational_messages):
        """
        exercise_list: list of dicts, each dict: {
            "name": str,
            "difficulty": int (1-5),
            "duration": int (seconds or reps)
        }
        feedback_messages: dict with exercise_name -> list of corrections
        motivational_messages: list of strings
        """
        self.exercise_list = exercise_list
        self.feedback_messages = feedback_messages
        self.motivational_messages = motivational_messages
        self.user_progress = {}        # user_id -> dict of last exercise data
        self.daily_streaks = {}        # user_id -> consecutive active days
        self.last_session_time = {}    # user_id -> datetime of last session

    # ---------------- Update User Progress ----------------
    def update_progress(self, user_id, exercise_name, performance_score, unsafe=False):
        """
        performance_score: 0-100
        unsafe: True if pose detection detects unsafe movement
        """
        self.user_progress[user_id] = {
            "exercise": exercise_name,
            "score": performance_score,
            "timestamp": datetime.now(),
            "unsafe": unsafe
        }

        # Update streak
        last_time = self.last_session_time.get(user_id)
        if last_time:
            days_since_last = (datetime.now() - last_time).days
            if days_since_last == 1:
                self.daily_streaks[user_id] += 1
            elif days_since_last > 1:
                self.daily_streaks[user_id] = 1
        else:
            self.daily_streaks[user_id] = 1

        self.last_session_time[user_id] = datetime.now()

    # ---------------- Decide Next Exercise ----------------
    def choose_next_exercise(self, user_id):
        """
        Adaptive exercise selection based on last performance
        """
        if user_id not in self.user_progress:
            # First exercise: choose easiest
            candidates = [ex for ex in self.exercise_list if ex["difficulty"] == 1]
        else:
            last_score = self.user_progress[user_id]["score"]
            last_exercise = self.user_progress[user_id]["exercise"]
            last_difficulty = next(
                ex["difficulty"] for ex in self.exercise_list if ex["name"] == last_exercise
            )

            if last_score >= 80:
                # Increase difficulty
                candidates = [ex for ex in self.exercise_list if ex["difficulty"] == last_difficulty + 1]
                if not candidates:
                    candidates = [ex for ex in self.exercise_list if ex["difficulty"] == last_difficulty]
            elif last_score < 50:
                # Easier or same exercise
                candidates = [ex for ex in self.exercise_list if ex["difficulty"] <= last_difficulty]
            else:
                # Same difficulty
                candidates = [ex for ex in self.exercise_list if ex["difficulty"] == last_difficulty]

        next_exercise = random.choice(candidates)
        return next_exercise

    # ---------------- Provide Feedback and Motivation ----------------
    def get_feedback(self, user_id):
        messages = []
        # Last exercise feedback
        if user_id in self.user_progress:
            last_exercise = self.user_progress[user_id]["exercise"]
            exercise_feedback = self.feedback_messages.get(last_exercise, [])
            if exercise_feedback:
                messages.append(random.choice(exercise_feedback))

        # Motivational message
        messages.append(random.choice(self.motivational_messages))
        return messages

    # ---------------- Check if Rest Needed ----------------
    def check_rest_needed(self, user_id):
        if user_id in self.last_session_time:
            elapsed = datetime.now() - self.last_session_time[user_id]
            if elapsed < timedelta(minutes=5):  # demo threshold
                return True
        return False

    # ---------------- Missed-Day Reminder ----------------
    def get_reminder(self, user_id):
        last_time = self.last_session_time.get(user_id)
        if last_time and (datetime.now() - last_time).days > 1:
            return f"Hey {user_id}, you missed a day! Let's get back to it!"
        return None

    # ---------------- Safety Alert ----------------
    def check_safety(self, user_id):
        if user_id in self.user_progress:
            if self.user_progress[user_id].get("unsafe"):
                return "Unsafe movement detected! Please correct your form or stop."
        return None

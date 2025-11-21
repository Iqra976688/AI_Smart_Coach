# ---------------- Agentic AI Module ----------------
# This module decides the next exercise/action for the user
# Compatible with external pose detection / RAG feedback modules

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
        self.user_progress = {}  # user_id -> dict of last exercise data
        self.daily_streaks = {}  # user_id -> number of days exercised
        self.last_session_time = {}  # user_id -> datetime of last session

    def update_progress(self, user_id, exercise_name, performance_score):
        """
        performance_score: 0-100 (from pose+RAG feedback)
        """
        self.user_progress[user_id] = {
            "exercise": exercise_name,
            "score": performance_score,
            "timestamp": datetime.now()
        }

        # Update streak
        last_time = self.last_session_time.get(user_id)
        if last_time:
            if (datetime.now() - last_time).days == 1:
                self.daily_streaks[user_id] += 1
            elif (datetime.now() - last_time).days > 1:
                self.daily_streaks[user_id] = 1
        else:
            self.daily_streaks[user_id] = 1
        self.last_session_time[user_id] = datetime.now()

    def choose_next_exercise(self, user_id):
        """
        Decide next exercise based on last performance and streak
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
                # Improve difficulty slightly
                candidates = [ex for ex in self.exercise_list if ex["difficulty"] == last_difficulty + 1]
                if not candidates:
                    candidates = [ex for ex in self.exercise_list if ex["difficulty"] == last_difficulty]
            elif last_score < 50:
                # Suggest same or easier exercise
                candidates = [ex for ex in self.exercise_list if ex["difficulty"] <= last_difficulty]
            else:
                candidates = [ex for ex in self.exercise_list if ex["difficulty"] == last_difficulty]

        next_exercise = random.choice(candidates)
        return next_exercise

    def get_feedback(self, user_id):
        """
        Return a random motivational message and last exercise feedback
        """
        messages = []
        # Feedback from last exercise
        if user_id in self.user_progress:
            last_exercise = self.user_progress[user_id]["exercise"]
            exercise_feedback = self.feedback_messages.get(last_exercise, [])
            if exercise_feedback:
                messages.append(random.choice(exercise_feedback))

        # Motivational message
        messages.append(random.choice(self.motivational_messages))
        return messages

    def check_rest_needed(self, user_id):
        """
        Suggest rest if last session too short (demo threshold)
        """
        if user_id in self.last_session_time:
            elapsed = datetime.now() - self.last_session_time[user_id]
            if elapsed < timedelta(minutes=5):  # demo threshold
                return True
        return False


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    exercises = [
        {"name": "Squat", "difficulty": 1, "duration": 30},
        {"name": "Push-Up", "difficulty": 1, "duration": 20},
        {"name": "Plank", "difficulty": 2, "duration": 40},
        {"name": "Lunge", "difficulty": 2, "duration": 25},
    ]

    feedback_messages = {
        "Squat": ["Keep your knees aligned and push hips back.", "Engage core to maintain posture."],
        "Push-Up": ["Keep your hips straight.", "Elbows should not flare out."],
    }

    motivational_messages = ["Great job! Keep it up!", "You're improving every session!"]

    agent = AgenticAI(exercises, feedback_messages, motivational_messages)

    user_id = "user123"
    agent.update_progress(user_id, "Squat", performance_score=85)
    next_ex = agent.choose_next_exercise(user_id)
    feedback = agent.get_feedback(user_id)
    rest_needed = agent.check_rest_needed(user_id)

    print("Next Exercise:", next_ex)
    print("Feedback:", feedback)
    print("Rest Needed:", rest_needed)

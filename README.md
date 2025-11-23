# 🌟 **AI Smart Coach – README**

# 🏋️‍♂️ **AI Smart Coach**

A smart, real-time AI-powered personal trainer that guides **gym exercises**, **sports movements**, and **physiotherapy routines** using **Computer Vision**, **LLMs**, **RAG**, and **Agentic AI**.

---

# 🔥 **Key Highlights**

## ✅ **Real-Time Posture Analysis**

* Detects joint positions (knees, elbows, shoulders, spine)
* Shows skeleton overlay
* Calculates angles
* Highlights mistakes visually

---

## 🎙️ **Instant Voice + Text Coaching**

* Corrects your form in real time
* Motivates you
* Helps prevent injury

**Examples:**

* "Straighten your back!"
* "Slow down your reps."
* "Excellent control, keep going!"

---

## 🤖 **Agentic AI (Self-Adaptive Coach)**

The system *thinks* and makes decisions like a real coach.

### **What It Does:**

* Chooses the next exercise
* Decides set duration & reps
* Adds rest when you’re tired
* Stops instantly if unsafe movement is detected
* Adjusts difficulty based on performance
* Builds daily + weekly plans

---

## 📚 **RAG Knowledge System**

Includes a small knowledge base for:

* Correct exercise form
* Physiotherapy guidelines
* Sports techniques
* Injury prevention
* Common mistakes

This ensures the coach gives **safe**, **accurate**, and **trustworthy** feedback.

---

# 🧠 **Core Features**

## 🏋️ **Gym Exercises Supported**

* Squats
* Push-ups
* Planks
* Deadlifts
* Lunges
* Shoulder press
* Bicep curls
* And more...

---

## ⚽ **Sports Movement Analysis**

Supports:

* Cricket batting & bowling
* Football kicks
* Running stride
* Jump form
* Tennis swings

Checks for:

* Balance
* Foot placement
* Speed
* Angles
* Stability

---

## 🩺 **Physiotherapy-Safe Mode**

* Slow controlled movement detection
* Compensation movement detection
* Safety-first feedback
* Posture correction
* Gentle reminders

> ⚠️ The system **does not replace medical professionals**—it only assists between sessions.

---

# 🔔 **Habits, Reminders & Reports**

## 🕒 **Smart Reminders**

* "Try to keep your breath steady."
* "Stand straight with good posture."
* "Stretch before sleeping!"

## 📅 **Streak Tracking**

* Daily streaks
* Consistency score
* Improvement history

## 📊 **Session Reports**

* Accuracy score
* Reps completed
* Total time
* Mistakes made
* Suggested next exercises

---

# 🛠️ **Tech Stack**

### **Computer Vision**

* MediaPipe / YOLO-Pose / MoveNet
* OpenCV

### **LLMs & AI**

* OpenAI GPT models
* RAG (Retrieval Augmented Generation)
* Agentic decision-making

### **Frontend / UI**

* Streamlit 

### **Training Tools**

* Google Colab
* Hugging Face Spaces
* VS Code

---

# ⚙️ **How the System Is Built**

## 🔸 **Phase 1: Build in Google Colab (Using Videos or Pictures)**

Because webcam doesn’t work well in Colab, the system is trained and tested using **uploaded videos**, like:

* squat.mp4
* pushup.mp4
* physio_knee.mp4

This allows:

* Smooth debugging
* Clean pose extraction
* Reliable frame-by-frame testing

## 🔸 **Phase 2: Deploy on Huggin Face Space (Using Videos or Pictures)**

* bicep_curl.mp4
* tennis_swing.png
* physio_knee.png
  
## 🔸 **Phase 3: Locally run (Webcam Enabled)**

### ✔ Run on Local Laptop

* Latest Gradio
* Webcam works smoothly



---

# 🧩 **Architecture Overview**

## 🧠 **Agents in the System**

* **Pose Analysis Agent** – reads angles, posture
* **Performance Agent** – understands fatigue & accuracy
* **Planning Agent** – creates workout plans
* **Decision Agent** – picks the next move
* **Safety Agent** – prevents injury
* **LLM Coaching Agent** – explains mistakes
* **RAG Agent** – ensures knowledge accuracy

All agents communicate through a shared state.

---


## Prerequisites

- Windows, macOS, or Linux with Python 3.10+ installed.
- (Optional) Google Gemini API key for AI feedback (set `GOOGLE_API_KEY`).
- Webcam for live tracking.

## Configuration & secrets

Store runtime secrets (API keys, service credentials) in a `.env` file at the project root. This project uses `python-dotenv` (already loaded in `app.py`) so environment values will be available at runtime.

Example `.env` (PowerShell):

```powershell
$env:GOOGLE_API_KEY = "your-gemini-key-here"
```

Security notes:

- Do not commit `.env` to version control. A `.gitignore` entry is provided to exclude `.env` files.
- Prefer per-developer environment variables or secret stores (GitHub Actions secrets, Azure Key Vault, etc.) for CI/CD.
- Rotate keys regularly and avoid pasting secrets into shared documents or issue trackers.

## Setup Instructions

```powershell
# Clone or download this repository, then navigate into it
cd "AI Smart Coach"

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# (Optional) Configure environment variables
# Example using PowerShell:
$env:GOOGLE_API_KEY = "your-gemini-key-here"

# Run the Streamlit app
streamlit run app.py
```

When the Streamlit dashboard opens in your browser, grant camera permissions. Use the sidebar to select movement modes. The combined video pane shows the live feed and real-time coaching overlay; the right column lists metrics, cues, and AI audio playback controls.


## Troubleshooting

- If `streamlit` is not recognized, ensure the virtual environment is activated before running commands.
- For MediaPipe GPU errors on Windows, install the latest Visual C++ redistributables.
- Missing AI cues? Confirm `GOOGLE_API_KEY` is set and the `langchain-google-genai` package installed.

## Clone the Repository

```powershell
git clone https://github.com/Iqra976688/AI_Smart_Coach.git
```


## License

This prototype is provided for Aspire GenAI Hackathon exploration. Adapt as needed for your project.


Model Structure 
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/d7662b54-0728-4415-b78d-4dcfb910d053" />


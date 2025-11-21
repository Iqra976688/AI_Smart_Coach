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

* "Time for your daily workout!"
* "Don’t forget your knee rehab session."
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

* Gradio
* Streamlit (optional)

### **Training Tools**

* Google Colab
* Hugging Face Spaces

---

# ⚙️ **How the System Is Built**

## 🔸 **Phase 1: Build in Google Colab (Using Videos)**

Because webcam doesn’t work well in Colab, the system is trained and tested using **uploaded videos**, like:

* squat.mp4
* pushup.mp4
* physio_knee.mp4

This allows:

* Smooth debugging
* Clean pose extraction
* Reliable frame-by-frame testing

## 🔸 **Phase 2: Local & Online Deployment (Webcam Enabled)**

Once the model works:

### ✔ Run on Local Laptop

* Latest Gradio
* Webcam works smoothly

### ✔ Deploy on Hugging Face Spaces

* HTTPS support ensures stable webcam

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

# 👥 **Team Roles**

### **Abdul Basit**

* Full code development
* Feature implementation

### **Areeba**

* RAG system
* Knowledge base and requirements

### **Sami**

* Dataset collection
* Movement library creation

---

# 🚀 **Project Summary**

AI Smart Coach is a next-generation personal training system that watches your movement, corrects your form, sends reminders, builds habits, analyzes posture, and uses Agentic AI to plan your next steps—all while being safe for gym, sports, and physiotherapy.

This project gives users a complete personal trainer experience from their own camera.

---



Model Structure 
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/d7662b54-0728-4415-b78d-4dcfb910d053" />


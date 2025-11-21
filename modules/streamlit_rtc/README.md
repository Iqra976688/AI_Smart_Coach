# AI Smart Coach

Real-time fitness coaching prototype built with Streamlit, MediaPipe pose tracking, and multimodal AI feedback.

## Features

- Live WebRTC feed with on-frame pose skeleton and coaching overlay.
- Adaptive exercise detection (e.g., bicep curls on either arm).
- Rep counting, tempo/ROM validation, and performance scoring.
- AI-generated coaching cues with optional text-to-speech playback.
- Exercise library spanning gym, sports, and physio movements.

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

When the Streamlit dashboard opens in your browser, grant camera permissions. Use the sidebar to select movement modes and focus areas. The combined video pane shows the live feed and real-time coaching overlay; the right column lists metrics, cues, and AI audio playback controls.

## Troubleshooting

- If `streamlit` is not recognized, ensure the virtual environment is activated before running commands.
- For MediaPipe GPU errors on Windows, install the latest Visual C++ redistributables.
- Missing AI cues? Confirm `GOOGLE_API_KEY` is set and the `langchain-google-genai` package installed.

## License

This prototype is provided for Aspire GenAI Hackathon exploration. Adapt as needed for your project.

# VisionAssist

VisionAssist is a real-time assistive vision app designed to help blind and visually impaired users understand their surroundings, ask questions naturally, and receive calm, step-by-step guidance through voice.

## Team

- Drumil Kotecha

## Project Description

VisionAssist uses live camera input, computer vision, OCR, object localization, and generative AI to provide a voice-first experience for accessibility and navigation. The app is designed around three core goals:

1. Describe the environment in short, useful, low-latency spoken updates.
2. Answer user questions and follow natural voice commands.
3. Give safe, step-by-step guidance toward important objects or locations.

The system is built to prioritize safety and clarity. It warns about obstacles, reports nearby objects, reads visible text, and avoids overwhelming the user with unnecessary information.

## Demo Presentation, PPT and Research paper

- Folder: https://drive.google.com/drive/folders/1KnQU7wo3lh0yIYt-IwTGL6j0KYGiFvY5?usp=sharing

- Demo Presentation: https://youtu.be/QbfTVtVooHc

- PPT link: https://drive.google.com/file/d/1XnB0hwRCF5RNAeV-BM9YIU8rglQxhNam/view?usp=sharing

- PPT in PDF since not sure if you will be able to open .key file: https://drive.google.com/file/d/1fr5Fw4JxYUzohSzeF6j4xRE4mZNHTFJY/view?usp=sharing

- Research paper: https://docs.google.com/document/d/1pMmo6VMmVzkQVLv70e8A_TJIh2MyRMD8/edit?usp=sharing&ouid=114921771539241515547&rtpof=true&sd=true


- Github repo link: https://github.com/Drumil23/Vision-Assist


## Key Features

- Real-time scene analysis from a live camera feed
- Concise audio descriptions of nearby surroundings
- Voice interaction for questions and navigation requests
- Object detection and spatial reasoning for common indoor items
- OCR for reading visible text
- Safety warnings for obstacles and hazards
- Accessibility-focused interface with speech and haptic feedback support

## System Overview

The application is organized into a small set of cooperating modules:

- `app.py` handles the Streamlit user interface, camera input, voice controls, and orchestration.
- `utils/inference.py` runs the background perception loops for scene descriptions, OCR, and object search.
- `models/scene.py` connects to Gemini for scene understanding and conversational responses.
- `models/reader.py` performs OCR using EasyOCR.
- `models/finder.py` estimates object location and relative depth using OpenCLIP and depth estimation.
- `utils/audio.py` handles text-to-speech output and playback.
- `utils/voice.py` captures and transcribes spoken commands.

### Data Flow

1. The camera provides a frame to the app.
2. The inference engine stores the latest frame and sends it to the scene, OCR, and finder loops.
3. The scene model generates short spoken descriptions when the environment changes.
4. The OCR pipeline extracts text when visible.
5. The finder pipeline searches for user-specified objects and returns approximate location and distance.
6. Spoken responses are queued and played through the audio layer.

## Technical Stack

- Python 3.10+
- Streamlit for the application UI
- OpenCV for camera and image processing
- Google Gemini via `google-genai` for conversational scene understanding
- EasyOCR for text recognition
- OpenCLIP for object matching
- Depth Anything V2 for approximate depth estimation
- SpeechRecognition for voice input
- gTTS or platform speech output for text-to-speech

## Repository Structure

```text
visionassist/
├── app.py
├── requirements.txt
├── packages.txt
├── models/
│   ├── __init__.py
│   ├── finder.py
│   ├── reader.py
│   └── scene.py
└── utils/
    ├── __init__.py
    ├── audio.py
    ├── inference.py
    └── voice.py
```

## Installation

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install system packages if needed

On Debian/Ubuntu-based systems, install the packages listed in `packages.txt`:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1
```

If you need microphone support locally, install any additional audio packages required by your OS.

## Configuration

VisionAssist uses environment variables for some services.

Create a `.env` file or export the variables in your shell:

```bash
export GEMINI_API_KEY="your_api_key_here"
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

If you are deploying to Streamlit Community Cloud, store secrets in `.streamlit/secrets.toml` or the platform secret manager instead of committing them to the repository.

## Running the App

```bash
streamlit run app.py
```

When the app starts, it opens the camera view and background perception loops. You can speak to the app, type a question, or ask it to locate objects and describe the scene.

## Usage Examples

### Ask about the environment

- “What is in front of me?”
- “Describe the room.”
- “Is there anything blocking my way?”

### Find an object

- “Where is the fridge?”
- “Find my keys.”
- “Locate the door.”

### Navigation requests

- “Help me go to the kitchen.”
- “Take me to the table.”
- “Guide me to the sink.”

### Safety commands

- “Stop guidance.”
- “Quiet mode.”
- “Repeat that.”

## Accessibility and Safety Design

VisionAssist is designed to be calm, direct, and safe:

- Short instructions instead of long narration
- Immediate interruption for urgent hazards
- Step-by-step guidance for navigation tasks
- No reliance on visual-only cues for core interaction
- Speech output that can be muted when needed

## Performance Notes

- The app is intended to be low-latency and responsive.
- Perception runs in background threads so camera handling and guidance remain interactive.
- Object search and OCR are separated from scene narration to reduce overload.
- For deployment, lighter models and fewer background updates are recommended to keep startup time and memory usage manageable.

## Deployment Notes

- Ensure the `GEMINI_API_KEY` secret is configured in the deployment environment.
- Verify that all Python dependencies in `requirements.txt` are installed.
- If microphone or camera features are unavailable on the platform, provide a fallback path or disable only the affected feature.

## Development Notes

- `app.py` is the main entry point.
- `utils/inference.py` starts the background perception engine automatically.
- `models/scene.py` stores conversation history and handles Gemini calls.
- `models/finder.py` and `models/reader.py` are the main computer vision modules.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

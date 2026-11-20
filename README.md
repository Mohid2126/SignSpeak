# SignSpeak

A beginner-friendly hand-sign recognition lab. Collect your own webcam examples, train a small model locally, and test signs with camera snapshots.

## What it does

1. A camera snapshot shows one hand sign.
2. MediaPipe finds 21 hand landmarks (wrist, joints, fingertips).
3. The app saves the 63 normalized landmark coordinates with your chosen label.
4. A Random Forest model learns patterns from your examples.
5. A new snapshot is classified against the signs you taught it.

The project starts with still hand shapes rather than moving gestures. It does not upload the camera image; examples are written to `data/landmarks.csv` on your Mac.

## Mac setup

Install Python 3.11 or 3.12 if you don't already have one. MediaPipe wheels may not support every Python release, so use Python 3.12 for this project.

```bash
cd ~/Documents/Codex/2026-11-19/ok/SignSpeak
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501). Safari will ask permission to use the camera; allow it for the local app.

## Deploy online with Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository and set the main file to `app.py`.
3. Choose Python 3.12 in the app's advanced settings if a Python version option is shown.
4. Deploy, open the HTTPS app URL, and allow camera access in the browser.

The generic hand-landmark model in `models/hand_landmarker.task` is part of the app. Training examples and your trained classifier are intentionally ignored by Git, so each deployment starts with an empty training set. Camera snapshots and resulting examples are handled by the Streamlit server; when hosted online, the server is the cloud app, not your Mac. Do not capture sensitive images or personal information in a public demo.

## How to teach it

- Start with three distinct still signs, for example Open Palm, Fist, and Peace.
- Capture about 20 examples per sign, in decent light, with small changes in angle and distance.
- Keep the full hand in the frame. Use the same camera and similar lighting during training and prediction.
- Train after you have at least 3 examples for every sign. More varied examples usually work better.
- Try a fresh snapshot in the live section.

## Important limits

This is a learning demo, not a sign-language translator. A still-image model only recognizes the hand shapes you train it on. Real sign languages also use movement, orientation, facial expression, and context. Do not use it for accessibility-critical communication.

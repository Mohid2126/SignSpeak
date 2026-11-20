from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_FILE = DATA_DIR / "landmarks.csv"
MODEL_FILE = MODEL_DIR / "gesture_model.joblib"
META_FILE = MODEL_DIR / "model_info.json"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="SignSpeak · Gesture Lab", page_icon="✋", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root{--ink:#f2f4ff;--cyan:#74f4ff;--violet:#b690ff;--lime:#d6ff75}
.stApp{background:radial-gradient(ellipse at 82% 0%,#22234a 0%,#101522 48%,#090d16 100%);color:var(--ink)}
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif}[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1200px;padding-top:2rem;padding-bottom:4rem}
.hero{padding:38px;border:1px solid #74f4ff55;border-radius:28px;background:linear-gradient(115deg,#101f32ee,#211b3dee);box-shadow:0 20px 80px #0008,0 0 50px #74f4ff14;position:relative;overflow:hidden;margin-bottom:24px}
.hero:after{content:'✋';position:absolute;right:55px;top:15px;font-size:140px;opacity:.14;filter:drop-shadow(0 0 30px #74f4ff)}
.eyebrow,.kicker{font:500 11px 'DM Mono',monospace;letter-spacing:.17em;text-transform:uppercase;color:var(--cyan)}
.hero h1{font-size:clamp(44px,7vw,78px);letter-spacing:-.07em;line-height:.92;margin:17px 0;color:var(--ink)}.hero p{font-size:17px;color:#c4c8dc;max-width:720px}
[data-testid="stMetric"]{background:#141a28;border:1px solid #74f4ff33;padding:16px;border-radius:16px}[data-testid="stMetricValue"]{color:var(--cyan)}
.stButton>button{border-radius:12px;min-height:44px;border:1px solid #74f4ff66;background:#172433;color:white;font-weight:600}.stButton>button:hover{border-color:var(--lime);box-shadow:0 0 20px #74f4ff44}.stButton>button[kind="primary"]{background:linear-gradient(100deg,#74f4ff,#b690ff);color:#07121c;border:0}
[data-testid="stExpander"],[data-testid="stCameraInput"]{background:#121927;border:1px solid #39445a;border-radius:15px} [data-testid="stProgress"]>div>div{background:linear-gradient(90deg,#74f4ff,#b690ff)}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

LANDMARKER_PATH = MODEL_DIR / "hand_landmarker.task"


@st.cache_resource
def get_hand_landmarker():
    if not LANDMARKER_PATH.exists():
        raise FileNotFoundError("The bundled hand_landmarker.task model is missing from models/.")
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_landmarks(image: np.ndarray):
    """Return normalized 63-value hand landmark vector, or None if no hand."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = get_hand_landmarker().detect(mp_image)
    if not result.hand_landmarks:
        return None
    points = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)
    points -= points[0]
    scale = np.max(np.linalg.norm(points[:, :2], axis=1))
    if scale > 1e-6:
        points /= scale
    return points.flatten()


def load_data():
    if not DATA_FILE.exists():
        return pd.DataFrame(columns=[f"f{i}" for i in range(63)] + ["label", "created_at"])
    try:
        return pd.read_csv(DATA_FILE)
    except Exception:
        return pd.DataFrame(columns=[f"f{i}" for i in range(63)] + ["label", "created_at"])


def save_example(features, label):
    row = {f"f{i}": float(value) for i, value in enumerate(features)}
    row["label"] = label
    row["created_at"] = int(time.time())
    old = load_data()
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(DATA_FILE, index=False)


st.markdown("""<div class="hero"><div class="eyebrow">YOUR CAMERA · YOUR GESTURES · YOUR MODEL</div>
<h1>SignSpeak<span style="color:#74f4ff">.</span></h1>
<p>Teach your Mac to recognize hand signs. Capture examples with your camera, train a gesture model, then try it live.</p></div>""", unsafe_allow_html=True)

st.markdown('<div class="kicker">01 / Collect gesture examples</div>', unsafe_allow_html=True)
st.write("Choose a sign name, show that sign clearly to the camera, and save examples from slightly different angles and distances.")
camera_col, control_col = st.columns([1.4, 1], gap="large")
with camera_col:
    camera_image = st.camera_input("Take a picture of one hand sign", key="collection_camera")
with control_col:
    label = st.text_input("What sign are you showing?", value="Open Palm", max_chars=30)
    st.caption("Begin with 3 signs, such as Open Palm, Fist, and Peace. Take about 20 pictures of each, changing the angle and distance a little.")
    if camera_image:
        digest = str(hash(camera_image.getvalue()))
        feature_key = f"collection_features_{digest}"
        if feature_key not in st.session_state:
            raw = np.frombuffer(camera_image.getvalue(), dtype=np.uint8)
            frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            with st.spinner("Finding your hand landmarks…"):
                st.session_state[feature_key] = extract_landmarks(frame)
        features = st.session_state[feature_key]
        if features is None:
            st.warning("I couldn’t find a hand. Try brighter light and keep your whole hand in frame.")
        elif st.button(f"Save this {label or 'sign'} example", type="primary", use_container_width=True):
            if not label.strip():
                st.error("Enter a name for this sign first.")
            else:
                save_example(features, label.strip())
                st.success(f"Saved one {label.strip()} example.")

st.divider()
st.markdown('<div class="kicker">02 / Review your training set</div>', unsafe_allow_html=True)
data = load_data()
if data.empty:
    st.info("Your training set is empty. Start by saving several examples of your first sign.")
else:
    counts = data.groupby("label").size().sort_values(ascending=False)
    metric_cols = st.columns(max(1, min(len(counts), 4)))
    for col, (gesture, count) in zip(metric_cols, counts.items()):
        with col:
            st.metric(gesture, int(count), help="Number of camera examples saved")
    st.dataframe(counts.rename("Examples").to_frame(), use_container_width=True)

st.markdown('<div class="kicker">03 / Train your gesture model</div>', unsafe_allow_html=True)
if st.button("✦ Train my hand-sign model", type="primary", use_container_width=True):
    data = load_data()
    counts = data.groupby("label").size() if not data.empty else pd.Series(dtype=int)
    if len(counts) < 2:
        st.error("Save examples for at least two different signs before training.")
    elif counts.min() < 3:
        st.warning("Save at least 3 examples of every sign. Around 20 each is a much better start.")
    else:
        feature_columns = [f"f{i}" for i in range(63)]
        X = data[feature_columns].to_numpy(dtype=np.float32)
        y = data["label"].astype(str).to_numpy()
        model = RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42)
        if counts.min() >= 5:
            X_train, X_valid, y_train, y_valid = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )
            model.fit(X_train, y_train)
            validation = accuracy_score(y_valid, model.predict(X_valid))
            model.fit(X, y)
            st.info(f"Practice check on held-out examples: {validation:.0%}. This is a small local dataset, so test with new camera pictures too.")
        else:
            model.fit(X, y)
        joblib.dump(model, MODEL_FILE)
        META_FILE.write_text(json.dumps({"labels": sorted(counts.index.tolist()), "examples": int(len(data))}, indent=2))
        st.session_state["model_ready"] = True
        st.success(f"Model trained on {len(data)} pictures across {len(counts)} signs. Try it in the live panel below.")

st.divider()
st.markdown('<div class="kicker">04 / Try live recognition</div>', unsafe_allow_html=True)
st.write("Streamlit’s camera takes snapshots rather than continuous video, so this live lab recognizes one camera frame at a time. Press **Take picture** to classify the sign you’re holding.")
live_image = st.camera_input("Take a live test picture", key="live_camera")
if live_image:
    digest = str(hash(live_image.getvalue()))
    feature_key = f"live_features_{digest}"
    if feature_key not in st.session_state:
        raw = np.frombuffer(live_image.getvalue(), dtype=np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        with st.spinner("Reading the hand shape…"):
            st.session_state[feature_key] = extract_landmarks(frame)
    features = st.session_state[feature_key]
    if features is None:
        st.warning("No hand detected in this picture. Try again with your full hand visible.")
    elif not MODEL_FILE.exists():
        st.warning("Train your model first using the button above.")
    else:
        model = joblib.load(MODEL_FILE)
        probabilities = model.predict_proba([features])[0]
        order = np.argsort(probabilities)[::-1]
        best = int(order[0])
        st.subheader(f"I think this is: {model.classes_[best]}")
        st.progress(float(probabilities[best]))
        st.caption(f"Model confidence: {probabilities[best]:.0%}. This is a practice estimate, not a guarantee.")
        st.write("### Other possibilities")
        for i in order[1:3]:
            st.write(f"{model.classes_[i]} · {probabilities[i]:.0%}")

with st.expander("How the AI learns"):
    st.write("MediaPipe finds 21 key points on your hand, like your wrist and fingertips. We turn those points into numbers relative to your wrist. A Random Forest classifier learns which patterns tend to belong to each sign. The examples are saved in this project’s data folder so you can keep adding to them.")

st.caption("SIGNSPEAK · Camera pictures are processed on your Mac. Landmark examples are saved locally in data/landmarks.csv. This starter recognizes still hand shapes; later we can add movement-based signs and speech output.")

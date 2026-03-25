from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import joblib

from utils.config import OUTCOME_MODEL_PATH, INJURY_MODEL_PATH
from utils.feature_extraction import (
    extract_landmarks_from_image,
    extract_landmarks_from_video,
    compute_pitching_features,
    get_feature_vector_from_landmarks_sequence,
)
from utils.injury_risk import rule_based_injury_assessment

app = FastAPI(title="Baseball Pitching Intelligence API")

outcome_model = joblib.load(OUTCOME_MODEL_PATH) if OUTCOME_MODEL_PATH.exists() else None
injury_model = joblib.load(INJURY_MODEL_PATH) if INJURY_MODEL_PATH.exists() else None


@app.get("/")
def root():
    return {"status": "ok", "mode": "pitching"}


@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    landmarks, _ = extract_landmarks_from_image(image)
    if landmarks is None:
        return {"error": "No pitcher pose detected."}

    features = compute_pitching_features(landmarks)
    injury = rule_based_injury_assessment(features)

    prediction = {}
    if outcome_model:
        X = np.array([list(features.values())], dtype=float)
        probs = outcome_model.predict_proba(X)[0]
        cls = outcome_model.predict(X)[0]
        prediction = {
            "label": cls,
            "confidence": float(np.max(probs)),
            "class_probabilities": dict(zip(outcome_model.classes_, probs.tolist()))
        }

    return {"features": features, "injury": injury, "prediction": prediction}


@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    landmarks_seq, _ = extract_landmarks_from_video(temp_path)
    if not landmarks_seq:
        return {"error": "No pitcher pose detected in video."}

    agg_features, per_frame = get_feature_vector_from_landmarks_sequence(landmarks_seq)
    mean_features = {k.replace("_mean", ""): v for k, v in agg_features.items() if k.endswith("_mean")}
    injury = rule_based_injury_assessment(mean_features)

    prediction = {}
    if outcome_model:
        X = np.array([list(mean_features.values())], dtype=float)
        probs = outcome_model.predict_proba(X)[0]
        cls = outcome_model.predict(X)[0]
        prediction = {
            "label": cls,
            "confidence": float(np.max(probs)),
            "class_probabilities": dict(zip(outcome_model.classes_, probs.tolist()))
        }

    return {
        "features": mean_features,
        "per_frame_features": per_frame,
        "injury": injury,
        "prediction": prediction,
    }
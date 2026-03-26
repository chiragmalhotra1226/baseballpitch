import tempfile
import os
import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

from utils.config import OUTCOME_MODEL_PATH, INJURY_MODEL_PATH
from utils.feature_extraction import (
    extract_landmarks_from_image,
    extract_landmarks_from_video,
    compute_pitching_features,
    get_feature_vector_from_landmarks_sequence,
)
from utils.injury_risk import rule_based_injury_assessment, OPTIMAL_RANGES
from utils.visualization import (
    draw_pose_on_image,
    create_risk_gauge,
    create_body_part_risk_chart,
    create_time_series_chart,
    create_feature_radar,
    create_per_frame_risk_trend,
    MLB_BENCHMARKS,
)
from utils.gemini_coach import generate_coaching_plan

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PitchGuard AI",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Dark theme CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Base */
  html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d0d1a !important;
    color: #e0e0e0 !important;
    font-family: 'Inter', sans-serif;
  }
  [data-testid="stSidebar"] {
    background-color: #13131f !important;
    border-right: 1px solid #2a2a3e;
  }
  [data-testid="stSidebar"] * { color: #c0c0d0 !important; }

  /* Cards */
  .pg-card {
    background: #13131f;
    border: 1px solid #2a2a3e;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }
  .pg-metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2a2a3e;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
  }
  .pg-metric-value { font-size: 2rem; font-weight: 700; margin: 6px 0; }
  .pg-metric-label { font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }

  /* Hero */
  .pg-hero {
    background: linear-gradient(135deg, #0d0d1a 0%, #1a0a2e 50%, #0a1628 100%);
    border-bottom: 1px solid #2a2a3e;
    padding: 32px 24px 24px;
    margin-bottom: 24px;
  }
  .pg-hero h1 { font-size: 2.4rem; font-weight: 800; background: linear-gradient(90deg, #7c4dff, #00e5ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; }
  .pg-hero p { color: #888; font-size: 1rem; margin: 8px 0 0; }

  /* Section headers */
  .pg-section-header {
    font-size: 1.1rem; font-weight: 600; color: #e0e0e0;
    border-left: 3px solid #7c4dff;
    padding-left: 12px; margin: 24px 0 12px;
  }

  /* Risk badges */
  .badge-low    { background: #0d2b1e; color: #00c853; border: 1px solid #00c853; border-radius: 6px; padding: 2px 10px; font-size: 0.85rem; font-weight: 600; }
  .badge-medium { background: #2b2000; color: #ffab00; border: 1px solid #ffab00; border-radius: 6px; padding: 2px 10px; font-size: 0.85rem; font-weight: 600; }
  .badge-high   { background: #2b0000; color: #d50000; border: 1px solid #d50000; border-radius: 6px; padding: 2px 10px; font-size: 0.85rem; font-weight: 600; }

  /* Buttons */
  [data-testid="stButton"] button {
    background: linear-gradient(135deg, #7c4dff, #4527a0) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
  }
  [data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #651fff, #311b92) !important;
  }

  /* Inputs */
  [data-testid="stTextInput"] input,
  [data-testid="stSelectbox"] select {
    background: #1e1e2e !important;
    border: 1px solid #2a2a3e !important;
    color: #e0e0e0 !important;
    border-radius: 8px !important;
  }

  /* Expander */
  [data-testid="stExpander"] {
    background: #13131f !important;
    border: 1px solid #2a2a3e !important;
    border-radius: 8px !important;
  }

  /* Dataframe */
  [data-testid="stDataFrame"] { background: #13131f !important; }

  /* Divider */
  hr { border-color: #2a2a3e !important; }

  /* Coach report */
  .coach-report {
    background: #13131f;
    border: 1px solid #7c4dff44;
    border-radius: 12px;
    padding: 28px;
    line-height: 1.7;
  }
  .coach-report h3 { color: #7c4dff; border-bottom: 1px solid #2a2a3e; padding-bottom: 6px; }

  /* Frame grid */
  .frame-caption { font-size: 0.75rem; color: #888; text-align: center; margin-top: 4px; }

  /* Tooltip */
  .tooltip-text { font-size: 0.75rem; color: #666; font-style: italic; }
</style>
""", unsafe_allow_html=True)


# ─── Load models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    out, inj = None, None
    if OUTCOME_MODEL_PATH.exists():
        out = joblib.load(OUTCOME_MODEL_PATH)
    if INJURY_MODEL_PATH.exists():
        inj = joblib.load(INJURY_MODEL_PATH)
    return out, inj

outcome_bundle, injury_bundle = load_models()


# ─── Prediction helpers ────────────────────────────────────────────────────────
def predict_outcome(feature_dict):
    if outcome_bundle is None:
        return {"label": "Model not trained", "confidence": 0.0, "class_probabilities": {}}
    model = outcome_bundle["model"]
    encoder = outcome_bundle["encoder"]
    feat_cols = outcome_bundle["features"]
    X = np.array([[feature_dict.get(f, 0.0) for f in feat_cols]], dtype=float)
    probs = model.predict_proba(X)[0]
    idx = np.argmax(probs)
    return {
        "label": encoder.classes_[idx],
        "confidence": float(probs[idx]),
        "class_probabilities": dict(zip(encoder.classes_, probs.tolist())),
    }


def predict_injury_ml(feature_dict):
    if injury_bundle is None:
        return {"label": "Model not trained", "confidence": 0.0, "class_probabilities": {}}
    model = injury_bundle["model"]
    encoder = injury_bundle["encoder"]
    feat_cols = injury_bundle["features"]
    X = np.array([[feature_dict.get(f, 0.0) for f in feat_cols]], dtype=float)
    probs = model.predict_proba(X)[0]
    idx = np.argmax(probs)
    return {
        "label": encoder.classes_[idx],
        "confidence": float(probs[idx]),
        "class_probabilities": dict(zip(encoder.classes_, probs.tolist())),
    }


def get_feature_importance(feature_dict):
    if outcome_bundle is None:
        return {}
    model = outcome_bundle["model"]
    feat_cols = outcome_bundle["features"]
    try:
        importance = model.feature_importances_
        return dict(sorted(zip(feat_cols, importance), key=lambda x: x[1], reverse=True)[:8])
    except Exception:
        return {}


# ─── UI helpers ────────────────────────────────────────────────────────────────
def risk_badge(level: str) -> str:
    cls = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high"}.get(level, "badge-low")
    return f'<span class="{cls}">{level}</span>'


def delivery_color(label: str) -> str:
    return {"Efficient": "#00c853", "Mechanical_Leak": "#ffab00", "High_Risk_Mechanics": "#d50000"}.get(label, "#aaa")


def render_metrics_row(outcome, injury_rule):
    label = outcome["label"]
    conf = outcome["confidence"]
    risk = injury_rule["overall_risk"]
    risk_index = injury_rule.get("risk_index", 0)
    risk_color = {"Low": "#00c853", "Medium": "#ffab00", "High": "#d50000"}.get(risk, "#aaa")
    label_color = delivery_color(label)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="pg-metric-card">
            <div class="pg-metric-label">Delivery Score</div>
            <div class="pg-metric-value" style="color:{label_color}">{label.replace('_',' ')}</div>
            <div class="tooltip-text">Confidence: {conf:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="pg-metric-card">
            <div class="pg-metric-label">Injury Risk</div>
            <div class="pg-metric-value" style="color:{risk_color}">{risk}</div>
            <div class="tooltip-text">Risk Index: {risk_index}/100</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="pg-metric-card">
            <div class="pg-metric-label">Model Confidence</div>
            <div class="pg-metric-value" style="color:#7c4dff">{conf:.0%}</div>
            <div class="tooltip-text">XGBoost + Rule-based fusion</div>
        </div>""", unsafe_allow_html=True)


def render_feature_table(features):
    rows = []
    for k, v in features.items():
        if k not in OPTIMAL_RANGES:
            continue
        low, high = OPTIMAL_RANGES[k]
        mlb = MLB_BENCHMARKS.get(k, "—")
        if low <= v <= high:
            status = "🟢 Good"
        elif (low - (high-low)*0.5) <= v <= (high + (high-low)*0.5):
            status = "🟡 Warning"
        else:
            status = "🔴 Risk"
        rows.append({
            "Feature": k.replace("_", " ").title(),
            "Value": round(v, 2),
            "Target Range": f"{low} – {high}",
            "MLB Avg": mlb,
            "Status": status,
        })
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_warnings(injury_rule):
    if injury_rule["warnings"]:
        for w in injury_rule["warnings"]:
            st.warning(w)
    else:
        st.success("✅ No major injury flags detected for this frame/sequence.")


def render_feature_importance(feature_dict):
    imp = get_feature_importance(feature_dict)
    if not imp:
        return
    keys = list(imp.keys())
    vals = list(imp.values())
    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(
        x=vals[::-1], y=[k.replace("_", " ").title() for k in keys[::-1]],
        orientation="h",
        marker_color="#7c4dff",
        text=[f"{v:.3f}" for v in vals[::-1]],
        textposition="outside",
        textfont={"color": "#aaa"},
    ))
    fig.update_layout(
        title={"text": "Feature Importance (Why this prediction?)", "font": {"color": "#e0e0e0", "size": 13}},
        paper_bgcolor="#13131f", plot_bgcolor="#1e1e2e",
        xaxis={"tickfont": {"color": "#aaa"}, "gridcolor": "#2a2a3e"},
        yaxis={"tickfont": {"color": "#ccc"}},
        height=300, margin=dict(t=50, b=20, l=10, r=60),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_frame_grid(frames, landmarks_list, per_frame_injuries, max_frames=12):
    st.markdown('<div class="pg-section-header">🎞️ Analyzed Frames</div>', unsafe_allow_html=True)
    show_frames = frames[:max_frames]
    cols_per_row = 4
    for row_start in range(0, len(show_frames), cols_per_row):
        cols = st.columns(cols_per_row)
        for ci, fi in enumerate(range(row_start, min(row_start + cols_per_row, len(show_frames)))):
            with cols[ci]:
                frame = show_frames[fi]
                lm = landmarks_list[fi] if fi < len(landmarks_list) else None
                joint_colors = per_frame_injuries[fi].get("joint_colors", {}) if fi < len(per_frame_injuries) else {}
                overlay = draw_pose_on_image(frame, lm, joint_colors)
                risk_idx = per_frame_injuries[fi].get("risk_index", 0) if fi < len(per_frame_injuries) else 0
                risk_color = "#00c853" if risk_idx < 30 else "#ffab00" if risk_idx < 60 else "#d50000"
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.markdown(f'<div class="frame-caption" style="color:{risk_color}">Frame {fi+1} · Risk {risk_idx}</div>', unsafe_allow_html=True)


def render_coach_report(features, injury_rule, outcome, api_key):
    st.markdown('<div class="pg-section-header">🤖 AI Coaching Report</div>', unsafe_allow_html=True)
    if st.button("⚡ Generate AI Coaching Plan", key="gen_coach"):
        with st.spinner("Analyzing mechanics with Gemini AI..."):
            plan = generate_coaching_plan(features, injury_rule, outcome, api_key=api_key)
        st.markdown(f'<div class="coach-report">{plan}</div>', unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚾ PitchGuard AI")
    st.markdown("---")

    api_key_input = st.text_input(
        "Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Get your key at https://aistudio.google.com/app/apikey",
    )

    st.markdown("---")
    mode = st.selectbox("Analysis Mode", ["📷 Image Upload", "🎬 Video Upload", "📹 Live Webcam"])

    sample_rate = st.slider("Frame Sampling Rate", 1, 10, 3, help="1 = every frame (slow), higher = faster but less detail")

    st.markdown("---")
    with st.expander("⚙️ Advanced Options"):
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
        max_frames_display = st.slider("Max Frames to Display", 4, 24, 12)
        show_radar = st.checkbox("Show Radar Chart", value=True)
        show_importance = st.checkbox("Show Feature Importance", value=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem;color:#555;line-height:1.6">
    <b style="color:#777">How risk is calculated</b><br>
    Rule-based engine checks 14 biomechanical features against phase-aware optimal ranges (ASMI/Driveline research).
    XGBoost model trained on 4,000 synthetic pitcher profiles.
    Single-image analysis is inherently limited — video gives better accuracy.
    </div>
    """, unsafe_allow_html=True)

    if outcome_bundle is None:
        st.warning("⚠️ Models not trained. Run `python train_models.py` first.")


# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="pg-hero">
  <h1>⚾ PitchGuard AI</h1>
  <p>Your Personal MLB Pitching Coach — Biomechanics Analysis & Injury Prevention</p>
</div>
""", unsafe_allow_html=True)


# ─── Session history init ─────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ─── IMAGE MODE ───────────────────────────────────────────────────────────────
if mode == "📷 Image Upload":
    st.markdown('<div class="pg-section-header">📷 Upload Pitcher Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a pitcher image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Detecting pose..."):
            landmarks, pose_landmarks = extract_landmarks_from_image(image)

        if landmarks is None:
            st.error("❌ No pitcher pose detected. Try a clearer image with the full body visible.")
        else:
            features = compute_pitching_features(landmarks)
            injury_rule = rule_based_injury_assessment(features)
            outcome = predict_outcome(features)

            overlay = draw_pose_on_image(image, landmarks, injury_rule.get("joint_colors", {}))

            col_img, col_gauge = st.columns([1.2, 1])
            with col_img:
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col_gauge:
                st.plotly_chart(create_risk_gauge(injury_rule["risk_index"]), use_container_width=True)
                st.plotly_chart(create_body_part_risk_chart(injury_rule["body_part_risks"]), use_container_width=True)

            st.markdown("---")
            render_metrics_row(outcome, injury_rule)
            render_warnings(injury_rule)

            if show_radar:
                st.markdown('<div class="pg-section-header">📊 Mechanics Radar</div>', unsafe_allow_html=True)
                st.plotly_chart(create_feature_radar(features, MLB_BENCHMARKS), use_container_width=True)

            with st.expander("📋 Detailed Biomechanics Table"):
                render_feature_table(features)

            if show_importance:
                with st.expander("🔍 Feature Importance — Why this prediction?"):
                    render_feature_importance(features)

            with st.expander("🧪 Raw Model Outputs"):
                st.json({"outcome": outcome, "injury_rule": {k: v for k, v in injury_rule.items() if k != "joint_colors"}})

            render_coach_report(features, injury_rule, outcome, api_key=api_key_input)

            # Save to history
            st.session_state.history = st.session_state.history[-4:] + [{
                "type": "image", "thumb": cv2.resize(overlay, (80, 60)),
                "outcome": outcome["label"], "risk": injury_rule["overall_risk"],
            }]


# ─── VIDEO MODE ───────────────────────────────────────────────────────────────
elif mode == "🎬 Video Upload":
    st.markdown('<div class="pg-section-header">🎬 Upload Pitcher Video</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a pitcher video", type=["mp4", "mov", "avi"], label_visibility="collapsed")

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            temp_path = tmp.name

        progress = st.progress(0, text="Extracting frames and detecting pose...")

        with st.spinner("Running pose estimation on video frames..."):
            landmarks_seq, frames = extract_landmarks_from_video(temp_path, sample_rate=sample_rate)

        progress.progress(50, text="Computing biomechanical features...")

        if not landmarks_seq:
            st.error("❌ No pitcher pose detected in video. Try a clearer clip.")
        else:
            agg_features, per_frame = get_feature_vector_from_landmarks_sequence(landmarks_seq)
            mean_features = {k.replace("_mean", ""): v for k, v in agg_features.items() if k.endswith("_mean")}

            per_frame_injuries = [rule_based_injury_assessment(f) for f in per_frame]
            injury_rule = rule_based_injury_assessment(mean_features)
            outcome = predict_outcome(mean_features)

            progress.progress(100, text="Analysis complete!")
            progress.empty()

            col_vid, col_gauge = st.columns([1.2, 1])
            with col_vid:
                st.video(temp_path)
            with col_gauge:
                st.plotly_chart(create_risk_gauge(injury_rule["risk_index"]), use_container_width=True)
                st.plotly_chart(create_body_part_risk_chart(injury_rule["body_part_risks"]), use_container_width=True)

            st.markdown("---")
            render_metrics_row(outcome, injury_rule)
            render_warnings(injury_rule)

            render_frame_grid(frames, landmarks_seq, per_frame_injuries, max_frames=max_frames_display)

            st.markdown('<div class="pg-section-header">📈 Temporal Analysis</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                keys_to_plot = st.multiselect(
                    "Select metrics to plot",
                    options=list(per_frame[0].keys()),
                    default=["right_elbow_flexion", "hip_shoulder_separation", "lumbar_spine_angle"],
                )
                if keys_to_plot:
                    st.plotly_chart(create_time_series_chart(per_frame, keys_to_plot), use_container_width=True)
            with c2:
                st.plotly_chart(create_per_frame_risk_trend(per_frame_injuries), use_container_width=True)

            if show_radar:
                st.plotly_chart(create_feature_radar(mean_features, MLB_BENCHMARKS), use_container_width=True)

            with st.expander("📋 Detailed Biomechanics Table (Mean Values)"):
                render_feature_table(mean_features)

            if show_importance:
                with st.expander("🔍 Feature Importance — Why this prediction?"):
                    render_feature_importance(mean_features)

            with st.expander("🧪 Raw Model Outputs"):
                st.json({"outcome": outcome, "injury_rule": {k: v for k, v in injury_rule.items() if k != "joint_colors"}})

            render_coach_report(mean_features, injury_rule, outcome, api_key=api_key_input)

            try:
                os.unlink(temp_path)
            except Exception:
                pass


# ─── WEBCAM MODE ──────────────────────────────────────────────────────────────
else:
    st.markdown('<div class="pg-section-header">📹 Live Webcam Analysis</div>', unsafe_allow_html=True)
    st.info("Capture a frame from your webcam for instant pitching mechanics analysis.")

    webcam = st.camera_input("Point camera at pitcher and capture")
    if webcam:
        file_bytes = np.asarray(bytearray(webcam.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Analyzing..."):
            landmarks, _ = extract_landmarks_from_image(image)

        if landmarks is None:
            st.error("❌ No pitcher pose detected.")
        else:
            features = compute_pitching_features(landmarks)
            injury_rule = rule_based_injury_assessment(features)
            outcome = predict_outcome(features)

            overlay = draw_pose_on_image(image, landmarks, injury_rule.get("joint_colors", {}))

            col_img, col_gauge = st.columns([1.2, 1])
            with col_img:
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col_gauge:
                st.plotly_chart(create_risk_gauge(injury_rule["risk_index"]), use_container_width=True)

            render_metrics_row(outcome, injury_rule)
            render_warnings(injury_rule)

            with st.expander("📋 Biomechanics Table"):
                render_feature_table(features)

            render_coach_report(features, injury_rule, outcome, api_key=api_key_input)


# ─── Session History ──────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown('<div class="pg-section-header">🕐 Session History</div>', unsafe_allow_html=True)
    cols = st.columns(len(st.session_state.history))
    for i, entry in enumerate(st.session_state.history):
        with cols[i]:
            st.image(cv2.cvtColor(entry["thumb"], cv2.COLOR_BGR2RGB), use_container_width=True)
            risk_color = {"Low": "#00c853", "Medium": "#ffab00", "High": "#d50000"}.get(entry["risk"], "#aaa")
            st.markdown(f'<div style="font-size:0.7rem;color:{risk_color};text-align:center">{entry["outcome"].replace("_"," ")} · {entry["risk"]}</div>', unsafe_allow_html=True)

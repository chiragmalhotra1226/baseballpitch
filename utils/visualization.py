import cv2
import mediapipe as mp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

mp_pose = mp.solutions.pose

JOINT_LANDMARKS = {
    "head":             [mp_pose.PoseLandmark.NOSE.value],
    "left_shoulder":    [mp_pose.PoseLandmark.LEFT_SHOULDER.value],
    "right_shoulder":   [mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
    "left_elbow":       [mp_pose.PoseLandmark.LEFT_ELBOW.value],
    "right_elbow":      [mp_pose.PoseLandmark.RIGHT_ELBOW.value],
    "left_wrist":       [mp_pose.PoseLandmark.LEFT_WRIST.value],
    "right_wrist":      [mp_pose.PoseLandmark.RIGHT_WRIST.value],
    "left_knee":        [mp_pose.PoseLandmark.LEFT_KNEE.value],
    "right_knee":       [mp_pose.PoseLandmark.RIGHT_KNEE.value],
    "spine":            [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                         mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value],
    "torso":            [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value],
    "lead_leg":         [mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.RIGHT_KNEE.value],
    "legs":             [mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value],
}

COLOR_MAP = {"green": (0, 220, 100), "yellow": (0, 200, 255), "red": (50, 50, 255)}
MLB_BENCHMARKS = {
    "right_elbow_flexion": 105,
    "hip_shoulder_separation": 35,
    "lumbar_spine_angle": 160,
    "right_shoulder_abduction": 92,
    "lead_leg_block": 163,
    "trunk_tilt": 158,
}


def draw_pose_on_image(image_bgr, landmarks, joint_colors=None):
    output = image_bgr.copy()
    h, w = output.shape[:2]
    if landmarks is None:
        return output

    for connection in mp_pose.POSE_CONNECTIONS:
        i, j = connection
        x1, y1 = int(landmarks[i][0] * w), int(landmarks[i][1] * h)
        x2, y2 = int(landmarks[j][0] * w), int(landmarks[j][1] * h)
        cv2.line(output, (x1, y1), (x2, y2), (100, 220, 100), 2)

    for i in range(len(landmarks)):
        x, y = int(landmarks[i][0] * w), int(landmarks[i][1] * h)
        cv2.circle(output, (x, y), 4, (100, 220, 100), -1)

    if joint_colors:
        for joint_name, severity in joint_colors.items():
            color = COLOR_MAP.get(severity, (100, 220, 100))
            for idx in JOINT_LANDMARKS.get(joint_name, []):
                x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
                cv2.circle(output, (x, y), 9, color, -1)
                cv2.circle(output, (x, y), 9, (255, 255, 255), 1)
    return output


def create_risk_gauge(risk_index: int) -> go.Figure:
    if risk_index < 30:
        color = "#00c853"
        label = "Low Risk"
    elif risk_index < 60:
        color = "#ffab00"
        label = "Moderate Risk"
    else:
        color = "#d50000"
        label = "High Risk"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_index,
        title={"text": f"<b>Injury Risk Index</b><br><span style='font-size:0.9em;color:{color}'>{label}</span>",
               "font": {"size": 16, "color": "#e0e0e0"}},
        number={"font": {"size": 36, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#555", "tickfont": {"color": "#aaa"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#1e1e2e",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30],  "color": "#0d2b1e"},
                {"range": [30, 60], "color": "#2b2000"},
                {"range": [60, 100],"color": "#2b0000"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.8, "value": risk_index},
        }
    ))
    fig.update_layout(
        height=220, margin=dict(t=60, b=10, l=20, r=20),
        paper_bgcolor="#13131f", font={"color": "#e0e0e0"},
    )
    return fig


def create_body_part_risk_chart(body_part_risks: dict) -> go.Figure:
    parts = list(body_part_risks.keys())
    color_map = {"green": "#00c853", "yellow": "#ffab00", "red": "#d50000"}
    colors = [color_map.get(body_part_risks[p], "#555") for p in parts]
    values = [1] * len(parts)

    fig = go.Figure(go.Bar(
        x=parts, y=values,
        marker_color=colors,
        text=[body_part_risks[p].upper() for p in parts],
        textposition="inside",
        textfont={"size": 11, "color": "white"},
        hovertemplate="%{x}: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title={"text": "Risk by Body Part", "font": {"size": 14, "color": "#e0e0e0"}},
        height=180, showlegend=False,
        paper_bgcolor="#13131f", plot_bgcolor="#1e1e2e",
        xaxis={"tickfont": {"color": "#aaa"}, "showgrid": False},
        yaxis={"visible": False},
        margin=dict(t=40, b=10, l=10, r=10),
    )
    return fig


def create_time_series_chart(per_frame_features: list, keys: list) -> go.Figure:
    fig = go.Figure()
    colors = ["#7c4dff", "#00e5ff", "#ff6d00", "#00c853", "#f50057"]

    for i, key in enumerate(keys):
        vals = [f.get(key, None) for f in per_frame_features]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            y=vals, mode="lines+markers", name=key.replace("_", " ").title(),
            line={"color": color, "width": 2},
            marker={"size": 4, "color": color},
        ))
        if key in MLB_BENCHMARKS:
            fig.add_hline(
                y=MLB_BENCHMARKS[key], line_dash="dash",
                line_color=color, opacity=0.4,
                annotation_text=f"MLB avg {key.split('_')[0]}",
                annotation_font_color=color,
            )

    fig.update_layout(
        title={"text": "Angle Evolution Across Frames", "font": {"color": "#e0e0e0", "size": 14}},
        paper_bgcolor="#13131f", plot_bgcolor="#1e1e2e",
        legend={"font": {"color": "#aaa"}, "bgcolor": "#1e1e2e"},
        xaxis={"title": "Frame", "tickfont": {"color": "#aaa"}, "gridcolor": "#2a2a3e"},
        yaxis={"title": "Degrees / Value", "tickfont": {"color": "#aaa"}, "gridcolor": "#2a2a3e"},
        height=320, margin=dict(t=50, b=40, l=50, r=20),
    )
    return fig


def create_feature_radar(features: dict, mlb_avg: dict = None) -> go.Figure:
    keys = ["right_elbow_flexion", "right_shoulder_abduction", "hip_shoulder_separation",
            "lumbar_spine_angle", "lead_leg_block", "trunk_tilt"]
    labels = ["Elbow Flex", "Shoulder Abd", "Hip-Shoulder Sep", "Lumbar Angle", "Lead Leg Block", "Trunk Tilt"]

    def normalize(key, val):
        ranges = {
            "right_elbow_flexion": (60, 170),
            "right_shoulder_abduction": (45, 140),
            "hip_shoulder_separation": (5, 65),
            "lumbar_spine_angle": (115, 180),
            "lead_leg_block": (120, 180),
            "trunk_tilt": (110, 180),
        }
        lo, hi = ranges.get(key, (0, 180))
        return round(max(0, min(100, (val - lo) / (hi - lo) * 100)), 1)

    vals = [normalize(k, features.get(k, 0)) for k in keys]
    vals_closed = vals + [vals[0]]
    labels_closed = labels + [labels[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_closed, theta=labels_closed,
        fill="toself", name="Pitcher",
        line_color="#7c4dff", fillcolor="rgba(124,77,255,0.2)",
    ))

    if mlb_avg:
        mlb_vals = [normalize(k, mlb_avg.get(k, 0)) for k in keys]
        mlb_vals_closed = mlb_vals + [mlb_vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=mlb_vals_closed, theta=labels_closed,
            fill="toself", name="MLB Average",
            line_color="#00e5ff", fillcolor="rgba(0,229,255,0.1)",
            line_dash="dash",
        ))

    fig.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 100], "tickfont": {"color": "#aaa"}},
               "angularaxis": {"tickfont": {"color": "#ccc"}},
               "bgcolor": "#1e1e2e"},
        paper_bgcolor="#13131f",
        legend={"font": {"color": "#aaa"}, "bgcolor": "#1e1e2e"},
        title={"text": "Mechanics Radar vs MLB Average", "font": {"color": "#e0e0e0", "size": 14}},
        height=360, margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig


def create_per_frame_risk_trend(per_frame_risks: list) -> go.Figure:
    indices = [p["risk_index"] for p in per_frame_risks]
    colors_list = ["#00c853" if v < 30 else "#ffab00" if v < 60 else "#d50000" for v in indices]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=indices, mode="lines+markers",
        line={"color": "#7c4dff", "width": 2},
        marker={"color": colors_list, "size": 8},
        name="Risk Index",
    ))
    fig.add_hrect(y0=0,  y1=30,  fillcolor="#00c853", opacity=0.05, line_width=0)
    fig.add_hrect(y0=30, y1=60,  fillcolor="#ffab00", opacity=0.05, line_width=0)
    fig.add_hrect(y0=60, y1=100, fillcolor="#d50000", opacity=0.05, line_width=0)

    fig.update_layout(
        title={"text": "Per-Frame Risk Trend", "font": {"color": "#e0e0e0", "size": 14}},
        paper_bgcolor="#13131f", plot_bgcolor="#1e1e2e",
        xaxis={"title": "Frame", "tickfont": {"color": "#aaa"}, "gridcolor": "#2a2a3e"},
        yaxis={"title": "Risk Index", "range": [0, 100], "tickfont": {"color": "#aaa"}, "gridcolor": "#2a2a3e"},
        height=260, margin=dict(t=50, b=40, l=50, r=20),
    )
    return fig
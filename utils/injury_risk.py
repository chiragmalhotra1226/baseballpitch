from typing import Dict, Tuple

OPTIMAL_RANGES = {
    "right_elbow_flexion":      (60, 170),
    "left_elbow_flexion":       (60, 170),
    "hip_shoulder_separation":  (5, 65),
    "lumbar_spine_angle":       (115, 180),
    "right_shoulder_abduction": (45, 140),
    "left_shoulder_abduction":  (45, 140),
    "right_wrist_cock":         (70, 180),
    "left_wrist_cock":          (70, 180),
    "right_knee_valgus_varus":  (0, 28),
    "left_knee_valgus_varus":   (0, 28),
    "head_position_offset":     (0.0, 0.30),
    "trunk_tilt":               (110, 180),
    "lead_leg_block":           (120, 180),
    "stride_width":             (0.10, 1.0),
}

JOINT_MAP = {
    "right_elbow_flexion":      "right_elbow",
    "left_elbow_flexion":       "left_elbow",
    "right_shoulder_abduction": "right_shoulder",
    "left_shoulder_abduction":  "left_shoulder",
    "right_wrist_cock":         "right_wrist",
    "left_wrist_cock":          "left_wrist",
    "right_knee_valgus_varus":  "right_knee",
    "left_knee_valgus_varus":   "left_knee",
    "lumbar_spine_angle":       "spine",
    "hip_shoulder_separation":  "torso",
    "head_position_offset":     "head",
    "trunk_tilt":               "spine",
    "lead_leg_block":           "lead_leg",
    "stride_width":             "legs",
}

HIGH_CONFIDENCE_FEATURES = {
    "lumbar_spine_angle",
    "right_knee_valgus_varus",
    "left_knee_valgus_varus",
    "head_position_offset",
}

BODY_PART_FEATURES = {
    "Elbow (UCL)":   ["right_elbow_flexion", "left_elbow_flexion"],
    "Shoulder":      ["right_shoulder_abduction", "left_shoulder_abduction"],
    "Lower Back":    ["lumbar_spine_angle", "trunk_tilt"],
    "Hip":           ["hip_shoulder_separation", "hip_rotation"],
    "Knee":          ["right_knee_valgus_varus", "left_knee_valgus_varus"],
}


def score_feature_against_range(value: float, low: float, high: float) -> Tuple[str, float]:
    if low <= value <= high:
        return "green", 0.0
    margin = max((high - low) * 0.50, 15)
    if (low - margin) <= value <= (high + margin):
        return "yellow", 0.2
    return "red", 0.7


def compute_body_part_risks(features: Dict[str, float]) -> Dict[str, str]:
    body_risks = {}
    for part, feat_list in BODY_PART_FEATURES.items():
        worst = "green"
        for f in feat_list:
            if f not in features or f not in OPTIMAL_RANGES:
                continue
            low, high = OPTIMAL_RANGES[f]
            color, _ = score_feature_against_range(features[f], low, high)
            if color == "red":
                worst = "red"
                break
            elif color == "yellow" and worst == "green":
                worst = "yellow"
        body_risks[part] = worst
    return body_risks


def rule_based_injury_assessment(features: Dict[str, float]) -> Dict:
    warnings, drivers, joint_colors = [], [], {}
    risk_points = 0.0

    for feature, (low, high) in OPTIMAL_RANGES.items():
        if feature not in features:
            continue
        value = features[feature]
        color, pts = score_feature_against_range(value, low, high)
        if feature not in HIGH_CONFIDENCE_FEATURES:
            pts *= 0.4
        joint_name = JOINT_MAP.get(feature, feature)
        if color == "red":
            joint_colors[joint_name] = "red"
        elif color == "yellow" and joint_colors.get(joint_name) != "red":
            joint_colors[joint_name] = "yellow"
        elif joint_name not in joint_colors:
            joint_colors[joint_name] = "green"
        if color != "green":
            risk_points += pts
            drivers.append({
                "feature": feature,
                "joint": joint_name,
                "value": round(value, 2),
                "optimal_range": (low, high),
                "severity": color,
            })

    lumbar = features.get("lumbar_spine_angle", 160)
    knee_v = features.get("right_knee_valgus_varus", 8)
    shoulder = features.get("right_shoulder_abduction", 95)
    head = features.get("head_position_offset", 0.08)

    if lumbar < 110:
        warnings.append("⚠️ Extreme trunk collapse — high lower-back stress risk.")
        risk_points += 1.2
    if knee_v > 30:
        warnings.append("⚠️ Severe lead-leg valgus — knee ligament stress risk.")
        risk_points += 1.0
    if shoulder > 148:
        warnings.append("⚠️ Extreme shoulder elevation — impingement risk.")
        risk_points += 0.8
    if head > 0.32:
        warnings.append("⚠️ Significant head drift — command and sequencing concern.")
        risk_points += 0.4

    red_count = sum(1 for d in drivers if d["severity"] == "red" and d["feature"] in HIGH_CONFIDENCE_FEATURES)

    # 0–100 injury risk index
    risk_index = min(100, int(risk_points * 14))

    if red_count >= 2 or risk_points >= 3.5:
        overall = "High"
        confidence = 0.72
    elif red_count >= 1 or risk_points >= 1.5:
        overall = "Medium"
        confidence = 0.68
    else:
        overall = "Low"
        confidence = 0.75

    body_part_risks = compute_body_part_risks(features)

    return {
        "overall_risk": overall,
        "confidence": confidence,
        "warnings": warnings,
        "joint_colors": {k: v for k, v in joint_colors.items() if v in {"yellow", "red"}},
        "drivers": [d for d in drivers if d["severity"] in {"yellow", "red"}][:8],
        "risk_score": round(risk_points, 2),
        "risk_index": risk_index,
        "body_part_risks": body_part_risks,
    }
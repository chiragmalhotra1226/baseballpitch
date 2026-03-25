import random
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    USE_XGB = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    USE_XGB = False
    print("XGBoost not found, falling back to RandomForest.")

from utils.config import MODELS_DIR, OUTCOME_MODEL_PATH, INJURY_MODEL_PATH


def synthesize_training_data(n=4000):
    rows = []
    for _ in range(n):
        archetype = random.choices(
            ["elite", "good", "average", "mechanical_leak", "high_risk"],
            weights=[0.18, 0.27, 0.25, 0.18, 0.12],
        )[0]

        if archetype == "elite":
            row = {
                "right_elbow_flexion":      random.gauss(105, 8),
                "left_elbow_flexion":       random.gauss(110, 10),
                "right_arm_extension":      random.gauss(0.28, 0.03),
                "left_arm_extension":       random.gauss(0.20, 0.03),
                "right_shoulder_abduction": random.gauss(92, 8),
                "left_shoulder_abduction":  random.gauss(75, 8),
                "right_wrist_cock":         random.gauss(145, 10),
                "left_wrist_cock":          random.gauss(130, 10),
                "right_knee_flexion":       random.gauss(158, 6),
                "left_knee_flexion":        random.gauss(155, 6),
                "right_knee_valgus_varus":  abs(random.gauss(8, 4)),
                "left_knee_valgus_varus":   abs(random.gauss(8, 4)),
                "lumbar_spine_angle":       random.gauss(162, 7),
                "head_position_offset":     abs(random.gauss(0.09, 0.02)),
                "shoulder_rotation":        abs(random.gauss(18, 5)),
                "hip_rotation":             abs(random.gauss(22, 5)),
                "hip_shoulder_separation":  abs(random.gauss(32, 6)),
                "stride_width":             random.gauss(0.65, 0.06),
                "lead_leg_block":           random.gauss(163, 6),
                "trunk_tilt":               random.gauss(158, 7),
            }
            outcome, injury_base = "Efficient", 0

        elif archetype == "good":
            row = {
                "right_elbow_flexion":      random.gauss(112, 10),
                "left_elbow_flexion":       random.gauss(115, 10),
                "right_arm_extension":      random.gauss(0.26, 0.04),
                "left_arm_extension":       random.gauss(0.21, 0.04),
                "right_shoulder_abduction": random.gauss(98, 10),
                "left_shoulder_abduction":  random.gauss(82, 10),
                "right_wrist_cock":         random.gauss(138, 12),
                "left_wrist_cock":          random.gauss(128, 12),
                "right_knee_flexion":       random.gauss(152, 8),
                "left_knee_flexion":        random.gauss(150, 8),
                "right_knee_valgus_varus":  abs(random.gauss(11, 5)),
                "left_knee_valgus_varus":   abs(random.gauss(11, 5)),
                "lumbar_spine_angle":       random.gauss(155, 9),
                "head_position_offset":     abs(random.gauss(0.12, 0.03)),
                "shoulder_rotation":        abs(random.gauss(20, 6)),
                "hip_rotation":             abs(random.gauss(20, 6)),
                "hip_shoulder_separation":  abs(random.gauss(28, 8)),
                "stride_width":             random.gauss(0.60, 0.07),
                "lead_leg_block":           random.gauss(155, 8),
                "trunk_tilt":               random.gauss(152, 8),
            }
            outcome, injury_base = "Efficient", 1

        elif archetype == "average":
            row = {
                "right_elbow_flexion":      random.gauss(122, 14),
                "left_elbow_flexion":       random.gauss(124, 14),
                "right_arm_extension":      random.gauss(0.23, 0.05),
                "left_arm_extension":       random.gauss(0.20, 0.05),
                "right_shoulder_abduction": random.gauss(108, 13),
                "left_shoulder_abduction":  random.gauss(90, 12),
                "right_wrist_cock":         random.gauss(128, 14),
                "left_wrist_cock":          random.gauss(120, 14),
                "right_knee_flexion":       random.gauss(145, 10),
                "left_knee_flexion":        random.gauss(143, 10),
                "right_knee_valgus_varus":  abs(random.gauss(15, 6)),
                "left_knee_valgus_varus":   abs(random.gauss(15, 6)),
                "lumbar_spine_angle":       random.gauss(148, 11),
                "head_position_offset":     abs(random.gauss(0.16, 0.04)),
                "shoulder_rotation":        abs(random.gauss(22, 7)),
                "hip_rotation":             abs(random.gauss(18, 7)),
                "hip_shoulder_separation":  abs(random.gauss(24, 10)),
                "stride_width":             random.gauss(0.54, 0.09),
                "lead_leg_block":           random.gauss(147, 10),
                "trunk_tilt":               random.gauss(146, 10),
            }
            outcome, injury_base = "Mechanical_Leak", 2

        elif archetype == "mechanical_leak":
            row = {
                "right_elbow_flexion":      random.gauss(132, 16),
                "left_elbow_flexion":       random.gauss(130, 16),
                "right_arm_extension":      random.gauss(0.21, 0.06),
                "left_arm_extension":       random.gauss(0.19, 0.06),
                "right_shoulder_abduction": random.gauss(118, 15),
                "left_shoulder_abduction":  random.gauss(98, 14),
                "right_wrist_cock":         random.gauss(116, 16),
                "left_wrist_cock":          random.gauss(110, 16),
                "right_knee_flexion":       random.gauss(138, 12),
                "left_knee_flexion":        random.gauss(136, 12),
                "right_knee_valgus_varus":  abs(random.gauss(20, 7)),
                "left_knee_valgus_varus":   abs(random.gauss(20, 7)),
                "lumbar_spine_angle":       random.gauss(142, 12),
                "head_position_offset":     abs(random.gauss(0.20, 0.05)),
                "shoulder_rotation":        abs(random.gauss(26, 8)),
                "hip_rotation":             abs(random.gauss(16, 7)),
                "hip_shoulder_separation":  abs(random.gauss(20, 12)),
                "stride_width":             random.gauss(0.48, 0.10),
                "lead_leg_block":           random.gauss(140, 12),
                "trunk_tilt":               random.gauss(140, 11),
            }
            outcome, injury_base = "Mechanical_Leak", 3

        else:  # high_risk
            row = {
                "right_elbow_flexion":      random.gauss(150, 18),
                "left_elbow_flexion":       random.gauss(148, 18),
                "right_arm_extension":      random.gauss(0.17, 0.06),
                "left_arm_extension":       random.gauss(0.16, 0.06),
                "right_shoulder_abduction": random.gauss(132, 16),
                "left_shoulder_abduction":  random.gauss(112, 15),
                "right_wrist_cock":         random.gauss(102, 18),
                "left_wrist_cock":          random.gauss(98, 18),
                "right_knee_flexion":       random.gauss(130, 14),
                "left_knee_flexion":        random.gauss(128, 14),
                "right_knee_valgus_varus":  abs(random.gauss(26, 8)),
                "left_knee_valgus_varus":   abs(random.gauss(26, 8)),
                "lumbar_spine_angle":       random.gauss(130, 13),
                "head_position_offset":     abs(random.gauss(0.26, 0.06)),
                "shoulder_rotation":        abs(random.gauss(30, 9)),
                "hip_rotation":             abs(random.gauss(14, 7)),
                "hip_shoulder_separation":  abs(random.gauss(12, 10)),
                "stride_width":             random.gauss(0.38, 0.11),
                "lead_leg_block":           random.gauss(128, 14),
                "trunk_tilt":               random.gauss(130, 13),
            }
            outcome, injury_base = "High_Risk_Mechanics", 4

        # Clip to physical bounds
        clip_min = {"right_elbow_flexion": 40, "left_elbow_flexion": 40,
                    "right_arm_extension": 0.04, "left_arm_extension": 0.04,
                    "right_shoulder_abduction": 30, "left_shoulder_abduction": 30,
                    "right_wrist_cock": 40, "left_wrist_cock": 40,
                    "right_knee_flexion": 90, "left_knee_flexion": 90,
                    "right_knee_valgus_varus": 0, "left_knee_valgus_varus": 0,
                    "lumbar_spine_angle": 90, "head_position_offset": 0.01,
                    "shoulder_rotation": 0, "hip_rotation": 0,
                    "hip_shoulder_separation": 0, "stride_width": 0.08,
                    "lead_leg_block": 90, "trunk_tilt": 90}
        clip_max = {k: 180 for k in row}
        clip_max.update({"right_arm_extension": 0.65, "left_arm_extension": 0.65,
                          "head_position_offset": 0.45, "stride_width": 1.1})
        row = {k: float(np.clip(v, clip_min.get(k, 0), clip_max.get(k, 180))) for k, v in row.items()}

        # Injury label
        risk_score = injury_base
        if row["right_elbow_flexion"] > 148 or row["right_elbow_flexion"] < 75:
            risk_score += 1
        if row["lumbar_spine_angle"] < 132:
            risk_score += 1
        if row["right_shoulder_abduction"] > 132:
            risk_score += 1
        if row["right_knee_valgus_varus"] > 24:
            risk_score += 1
        if row["hip_shoulder_separation"] < 10:
            risk_score += 1

        injury = "Low" if risk_score <= 1 else "Medium" if risk_score <= 3 else "High"
        row["outcome_label"] = outcome
        row["injury_label"] = injury
        rows.append(row)

    return pd.DataFrame(rows)


def build_model(use_xgb=True):
    if use_xgb and USE_XGB:
        return XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, verbosity=0,
        )
    return RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=4, random_state=42)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using {'XGBoost' if USE_XGB else 'RandomForest'}")
    print("Generating training data...")
    df = synthesize_training_data(n=4000)
    print(f"Outcome dist:\n{df['outcome_label'].value_counts()}")
    print(f"Injury dist:\n{df['injury_label'].value_counts()}")

    feature_cols = [c for c in df.columns if c not in ["outcome_label", "injury_label"]]
    X = df[feature_cols]

    # Outcome model
    le_out = LabelEncoder()
    y_outcome = le_out.fit_transform(df["outcome_label"])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_outcome, test_size=0.2, random_state=42, stratify=y_outcome)
    outcome_model = build_model()
    outcome_model.fit(X_tr, y_tr)
    print("\nOutcome model:")
    print(classification_report(y_te, outcome_model.predict(X_te), target_names=le_out.classes_))

    # Injury model
    le_inj = LabelEncoder()
    y_injury = le_inj.fit_transform(df["injury_label"])
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y_injury, test_size=0.2, random_state=42, stratify=y_injury)
    injury_model = build_model()
    injury_model.fit(X_tr2, y_tr2)
    print("\nInjury model:")
    print(classification_report(y_te2, injury_model.predict(X_te2), target_names=le_inj.classes_))

    # Save models + encoders together
    joblib.dump({"model": outcome_model, "encoder": le_out, "features": feature_cols}, OUTCOME_MODEL_PATH)
    joblib.dump({"model": injury_model, "encoder": le_inj, "features": feature_cols}, INJURY_MODEL_PATH)
    print("\n✅ Saved models to", MODELS_DIR)


if __name__ == "__main__":
    main()
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")


def generate_coaching_plan(metrics: dict, injury_report: dict, outcome_prediction: dict, api_key: str = None) -> str:
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        return "❌ Gemini API key not found. Enter it in the sidebar."

    try:
        from google import genai
        client = genai.Client(api_key=key)
    except Exception as e:
        return f"❌ Failed to initialize Gemini client: {e}"

    risk = injury_report.get("overall_risk", "Unknown")
    score = outcome_prediction.get("label", "Unknown")
    confidence = outcome_prediction.get("confidence", 0)
    drivers = injury_report.get("drivers", [])
    body_risks = injury_report.get("body_part_risks", {})
    warnings = injury_report.get("warnings", [])

    driver_text = "\n".join([f"  - {d['feature']}: {d['value']} (optimal: {d['optimal_range']}, severity: {d['severity']})" for d in drivers[:6]])
    body_text = "\n".join([f"  - {part}: {status}" for part, status in body_risks.items()])
    warning_text = "\n".join(warnings) if warnings else "None"

    key_metrics = {k: round(v, 1) for k, v in metrics.items() if not any(s in k for s in ["_mean", "_max", "_min", "_std"])}

    prompt = f"""You are an elite MLB pitching biomechanics coach and sports medicine expert.

Analyze the following pitcher data and produce a professional, structured coaching report.

## PITCHER DATA
Delivery Score: {score} (confidence: {confidence:.0%})
Injury Risk: {risk}
Risk Index: {injury_report.get('risk_index', 0)}/100

Key Metrics:
{key_metrics}

Risk Drivers:
{driver_text if driver_text else 'No significant risk drivers detected.'}

Body Part Risk Breakdown:
{body_text}

Warnings:
{warning_text}

## REPORT REQUIREMENTS

Write a complete, professional coaching report with these exact sections:

### 🎯 Delivery Summary
One paragraph summarizing the overall mechanics quality and risk profile.

### 🔬 Top 3 Mechanical Issues
For each issue: name it, explain what the metric shows, why it matters biomechanically, and what it looks like visually.

### 🦴 Anatomy & Injury Risk
Explain which specific structures (UCL, rotator cuff, obliques, etc.) are at risk based on the metrics, and why.

### ✅ 5 Coaching Cues
Specific, actionable verbal cues a pitching coach would give. Each cue should be tied to a specific metric.

### 🏋️ 3 Corrective Drills
Named drills with instructions, sets/reps, and which mechanical fault they address.

### 💪 Strength & Mobility Work
3–4 specific exercises targeting the weak links identified in this analysis.

### 🔥 Warm-Up Routine
A 5-minute pre-outing warm-up tailored to this pitcher's risk profile.

### 📈 30-Day Progression Plan
Week-by-week focus areas to fix the identified issues.

Keep the tone professional but clear. Use bullet points inside sections. Include relevant anatomical terms but explain them briefly.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            return "⚠️ Gemini rate limit reached. Wait 60 seconds and try again, or upgrade to a paid API plan."
        if "404" in err or "NOT_FOUND" in err:
            try:
                response = client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=prompt,
                )
                return response.text
            except Exception as e2:
                return f"❌ Gemini error: {e2}"
        return f"❌ Gemini error: {e}"
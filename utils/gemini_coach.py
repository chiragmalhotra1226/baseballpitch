# OLD (broken — EOL Nov 2025):
# import google.generativeai as genai
# genai.configure(api_key=api_key)
# model = genai.GenerativeModel("gemini-pro")

# NEW (correct):
from google import genai

def generate_coaching_plan(features, injury_rule, outcome, api_key: str) -> str:
    if not api_key:
        return "<p style='color:#888'>⚠️ Enter your Gemini API key in the sidebar to get AI coaching.</p>"
    
    try:
        client = genai.Client(api_key=api_key)
        
        prompt = f"""You are an elite baseball pitching biomechanics coach.
Metrics: {features}
Injury report: {injury_rule}
Outcome prediction: {outcome}

Rules:
- Be concise and direct.
- Use short bullet points only.
- Do NOT write long paragraphs.
- Only mention what is wrong, what injury risk it creates, and how to fix it.

Required format:
SUMMARY
- 1 bullet on overall delivery quality
MECHANICS TO FIX
- 3 bullets max
INJURY RISKS
- 3 bullets max
HOW TO FIX IT
- 3 bullets max with direct cues
DRILLS
- 3 short bullet points
MOBILITY/STRENGTH
- 2 bullet points max"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        # Convert markdown to HTML for rendering
        text = response.text
        lines = text.split('\n')
        html_lines = []
        for line in lines:
            if line.startswith('- ') or line.startswith('• '):
                html_lines.append(f"<li>{line[2:]}</li>")
            elif line.isupper() and line.strip():
                html_lines.append(f"<h3>{line}</h3>")
            else:
                html_lines.append(f"<p>{line}</p>")
        return '\n'.join(html_lines)

    except Exception as e:
        return f"<p style='color:#d50000'>Error generating coaching plan: {str(e)}</p>"
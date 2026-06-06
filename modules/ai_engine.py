"""
ai_engine.py
------------
Central AI engine for AlgebraicVision AI.
Uses Groq (LLaMA-3.3-70B) for all natural language explanations.

All responses are strictly in English.
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

_SYSTEM_PROMPT = """You are AlgebraicVision AI — an advanced mathematics and physics
pattern analysis system. You explain mathematical patterns, equations, and natural
phenomena with scientific precision and engaging clarity.

STRICT RULES:
1. Respond ONLY in English. Never use Urdu, Hindi, Roman Urdu, or any other language.
2. Be scientifically accurate and concise.
3. Use relevant emojis to make explanations engaging.
4. Structure responses clearly with numbered points when asked.
5. Connect mathematics to real-world physics, nature, and engineering.
"""


def _chat(prompt: str, max_tokens: int = 500, temperature: float = 0.6) -> str:
    """Internal helper — sends a prompt and returns the response text."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


# ── Public API ────────────────────────────────────────────────────────────────

def explain_pattern(pattern_name: str, details: str = "") -> str:
    """
    Explain a detected mathematical pattern in clear English.
    """
    prompt = f"""Explain this mathematical pattern detected in an image or expression:

Pattern: {pattern_name}
Details: {details}

Provide exactly:
1. What is this pattern? (2 sentences)
2. Mathematical formula or equation that governs it
3. Where it appears in nature (3 specific examples)
4. Where it appears in physics / engineering (2 examples)
5. One surprising fun fact

Keep total response under 180 words. Use emojis."""

    return _chat(prompt, max_tokens=350)


def analyze_equation(equation_str: str) -> str:
    """
    Deep analysis of any mathematical equation or formula.
    """
    prompt = f"""Analyze this mathematical equation or expression:

Equation: {equation_str}

Provide:
1. What this equation represents (1-2 sentences)
2. The type of curve or surface it produces
3. Key mathematical properties (symmetry, periodicity, asymptotes, etc.)
4. Real-world applications in physics, engineering, or nature (3 examples)
5. Historical context — who discovered/formulated it?

Keep total response under 200 words. Use emojis."""

    return _chat(prompt, max_tokens=400)


def analyze_image_patterns(detected_objects: list,
                            symmetry_score: float,
                            golden_ratio_info: str) -> str:
    """
    AI explanation of patterns found in an uploaded image.
    """
    objects_str = ", ".join(detected_objects) if detected_objects else "unidentified objects"

    prompt = f"""Analyze the mathematical patterns found in this image:

Detected objects: {objects_str}
Symmetry score: {symmetry_score:.1f}%
Golden ratio analysis: {golden_ratio_info}

In 3-4 concise sentences explain:
- Which mathematical patterns are dominant in this image
- What the symmetry score reveals about the structure
- How the golden ratio manifests here
- The overall mathematical beauty and complexity

Maximum 80 words. Be scientific and precise."""

    return _chat(prompt, max_tokens=160, temperature=0.4)


def explain_eda_results(stats: dict) -> str:
    """
    Explain the EDA statistical results in plain English.
    """
    sym   = stats.get("symmetry", {})
    pat   = stats.get("pattern_scores", {})
    frac  = stats.get("fractal_dim", 1.0)
    ar    = stats.get("aspect_ratio", 1.0)
    phi   = 1.6180339887

    top_pattern = max(pat, key=pat.get) if pat else "Unknown"
    top_score   = pat.get(top_pattern, 0)

    prompt = f"""Interpret these image analysis results:

- Horizontal symmetry: {sym.get('horizontal', 0):.1f}%
- Vertical symmetry: {sym.get('vertical', 0):.1f}%
- Fractal dimension: {frac:.3f}
- Aspect ratio: {ar:.4f} (golden ratio = {phi:.4f})
- Dominant pattern: {top_pattern} ({top_score:.1f}% confidence)
- Edge density: {stats.get('edge_density', 0)*100:.2f}%

In 4-5 sentences, explain what these numbers reveal about the mathematical
structure of this image. Mention what type of natural or engineered object
this could be based on these metrics. Use emojis. Max 120 words."""

    return _chat(prompt, max_tokens=240, temperature=0.5)


def explain_reallife(expression: str, expr_type: str, reallife_str: str) -> str:
    """
    Explain the real-life implementations of a mathematical expression.
    """
    prompt = f"""Explain the real-world applications of this mathematical expression:

Expression: {expression}
Type: {expr_type}
Known applications: {reallife_str}

Provide a rich explanation covering:
1. Physics applications (with specific formulas if relevant)
2. Engineering uses
3. Natural phenomena it describes
4. Technology that relies on this mathematics

Keep it engaging, accurate, and under 200 words. Use emojis."""

    return _chat(prompt, max_tokens=400, temperature=0.6)


def explain_symmetry_group(sym_type: str, score: float) -> str:
    """
    Explain what a detected symmetry type means mathematically.
    """
    prompt = f"""Explain this symmetry detection result:

Symmetry type: {sym_type}
Score: {score:.1f}%

Explain:
1. What this symmetry type means in group theory
2. Its mathematical notation (e.g., C2v, D6h)
3. Where this exact symmetry appears in nature and physics
4. Why this score ({score:.1f}%) is significant

Max 150 words. Use emojis."""

    return _chat(prompt, max_tokens=300, temperature=0.5)


def chat_with_ai(user_message: str, context: str = "") -> str:
    """
    General-purpose chat for the AI assistant tab.
    Handles any math/physics/pattern question.
    """
    prompt = f"""Context from current session: {context}

User question: {user_message}

Answer as AlgebraicVision AI. Be helpful, accurate, and engaging.
If the question involves an equation or pattern, explain its visual
representation and real-world significance.
Max 250 words."""

    return _chat(prompt, max_tokens=500, temperature=0.7)

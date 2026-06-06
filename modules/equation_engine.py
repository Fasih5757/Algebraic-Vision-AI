import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def explain_pattern(pattern_name, details=""):
    prompt = f"""You are AlgebraicVision AI — a mathematics visualization assistant.

The user has detected this pattern: {pattern_name}
Details: {details}

Respond ONLY in English. Do NOT use Urdu, Hindi, or Roman Urdu.

Explain in exactly this format:
1. What is this pattern? (2 lines max)
2. Mathematical formula: (one line)
3. Found in nature: (3 examples, comma separated)
4. Fun fact: (1 line)

Keep it under 120 words. Use emojis."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.5
    )
    return response.choices[0].message.content

def analyze_image_patterns(detected_objects, symmetry_score, golden_ratio_info):
    prompt = f"""You are AlgebraicVision AI — a mathematics pattern analyzer.

Image analysis results:
- Detected objects: {detected_objects}
- Symmetry score: {symmetry_score:.1f}%
- Golden ratio info: {golden_ratio_info}

STRICT RULES:
- Respond ONLY in English
- NO Urdu, NO Hindi, NO Roman Urdu whatsoever
- Maximum 60 words total
- No bullet points — plain sentences only
- Be concise and scientific

Describe the mathematical patterns found in this image."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.3
    )
    return response.choices[0].message.content

def analyze_equation(equation_str):
    prompt = f"""You are AlgebraicVision AI.

Equation: {equation_str}

Respond ONLY in English. NO Urdu or Hindi.
In under 80 words explain:
1. What this equation represents
2. Its real world application
3. One interesting property

Use emojis. Be concise."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.4
    )
    return response.choices[0].message.content
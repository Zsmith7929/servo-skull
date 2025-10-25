#!/usr/bin/env python3
import os, sys, re, time, argparse
from pathlib import Path

# Google GenAI SDK (AI Studio & Vertex AI compatible)
from google import genai
from google.genai.types import Content, Part, RequestOptions

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")  # or gemini-2.5-pro
API_KEY = os.environ.get("GEMINI_API_KEY", "")

INSTRUCTION = """You are a precise die-reading assistant.
Look at the image of a single polyhedral die inside a dice tower.
Decide which face is the TOP face (the one facing upward, the result of the roll).
Return ONLY the Arabic numeral printed on the TOP face as an integer (no words, no extra text).
If the top is ambiguous or fully occluded, return ONLY the string 'unknown'.
"""

def parse_int(s: str, max_face: int):
    s = s.strip()
    if s.lower() == "unknown":
        return None
    m = re.search(r"\b(\d{1,3})\b", s)
    if not m:
        return None
    n = int(m.group(1))
    # common dice bounds; allow 100 for percentile
    if n in (1,2,3,4,5,6,8,10,12,20,30,50,60,70,80,90,100) or (1 <= n <= max_face) or max_face==100 and n in range(1,101):
        return n
    return None

def read_die(image_path: str, max_face: int = 20, retries: int = 2, timeout: float = 20.0, verbose=False):
    if not API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set in environment")

    client = genai.Client(api_key=API_KEY)
    img_bytes = Path(image_path).read_bytes()

    # Build content: system-style instruction + image + short user nudge
    content = [
        Content(role="user", parts=[
            Part.from_text(INSTRUCTION),
            Part.from_image(bytes=img_bytes, mime_type="image/jpeg"),
            Part.from_text(f"Die type: d{max_face}. Return only the top face number.")
        ])
    ]

    # Slightly nudge for determinism and short answers
    options = RequestOptions(
        timeout=timeout,
        temperature=0.2,
        max_output_tokens=8,
    )

    last_err = None
    for attempt in range(retries+1):
        try:
            if verbose: print(f"[gemini] request attempt {attempt+1} …", flush=True)
            resp = client.responses.generate(model=MODEL_NAME, contents=content, config=options)
            text = (resp.output_text or "").strip()
            if verbose: print(f"[gemini] raw: {text}")
            n = parse_int(text, max_face)
            if n is not None:
                return n, text
            # second chance with stricter follow-up if it gave prose
            if attempt < retries:
                content.append(Content(role="user", parts=[Part.from_text("Only the integer on the TOP face. If unclear, return 'unknown'.")]))
        except Exception as e:
            last_err = e
            if attempt >= retries:
                raise
            time.sleep(0.8 * (attempt+1))
    raise RuntimeError(f"Gemini did not return a valid number. Last error: {last_err}")

def main():
    ap = argparse.ArgumentParser(description="Read top-face die number using Gemini Vision.")
    ap.add_argument("image", help="Path to the frame (e.g., dice_..._last.jpg)")
    ap.add_argument("--max-face", type=int, default=20, help="Die size (6,8,10,12,20,100).")
    ap.add_argument("-v","--verbose", action="store_true")
    args = ap.parse_args()

    n, raw = read_die(args.image, max_face=args.max_face, verbose=args.verbose)
    print(f"🎲 {n}")
    if args.verbose:
        print(f"[model] {raw}")

if __name__ == "__main__":
    main()

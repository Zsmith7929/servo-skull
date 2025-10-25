#!/usr/bin/env python3
import os, re, argparse
from pathlib import Path

from google import genai
from google.genai.types import Content, Part  # <- keep only these

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
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
    # Allow typical dice outputs; permit 100 for percentile
    if (1 <= n <= max_face) or (max_face == 100 and 1 <= n <= 100):
        return n
    return None

def read_die(image_path: str, max_face: int = 20, retries: int = 1, verbose=False):
    if not API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set in environment")

    client = genai.Client(api_key=API_KEY)
    img_bytes = Path(image_path).read_bytes()

    # Compose content: instruction + image + short nudge
    contents = [
        Content(role="user", parts=[
            Part.from_text(INSTRUCTION),
            Part.from_image(bytes=img_bytes, mime_type="image/jpeg"),
            Part.from_text(f"Die type: d{max_face}. Return only the top face number.")
        ])
    ]

    last_text = ""
    for attempt in range(retries + 1):
        if verbose:
            print(f"[gemini] request attempt {attempt+1} …", flush=True)
        resp = client.responses.generate(
            model=MODEL_NAME,
            contents=contents,
            # Most builds accept generation settings in 'config', but we can omit for maximum compatibility:
            # config={"temperature": 0.2, "max_output_tokens": 8}
        )
        text = (resp.output_text or "").strip()
        last_text = text
        if verbose:
            print(f"[gemini] raw: {text}")

        n = parse_int(text, max_face)
        if n is not None:
            return n, text

        # If it talked too much, add a stricter follow-up and retry once
        if attempt < retries:
            contents.append(Content(role="user", parts=[Part.from_text(
                "Only the integer on the TOP face. If unclear, return 'unknown'. Do not add any extra words."
            )]))

    raise RuntimeError(f"Gemini did not return a valid number. Last response: {last_text!r}")

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

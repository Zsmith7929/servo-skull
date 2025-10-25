#!/usr/bin/env python3
import os, re, argparse
from pathlib import Path
from google import genai
from google.genai import types  # official typed helpers

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")  # or gemini-2.5-pro
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

INSTRUCTION = (
    "You are a precise die-reading assistant.\n"
    "Look at the image of a single polyhedral die.\n"
    "Decide which face is the TOP face (the result of the roll).\n"
    "Return ONLY the Arabic numeral on the TOP face as an integer.\n"
    "If unclear, return exactly: unknown"
)

def parse_int(s: str, max_face: int):
    s = (s or "").strip().lower()
    if s == "unknown":
        return None
    m = re.search(r"\b(\d{1,3})\b", s)
    if not m: return None
    n = int(m.group(1))
    if (1 <= n <= max_face) or (max_face == 100 and 1 <= n <= 100):
        return n
    return None

def read_die(image_path: str, max_face: int = 20, verbose=False):
    if not API_KEY:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY in your shell (bashrc).")

    client = genai.Client(api_key=API_KEY)  # explicit key avoids env-name ambiguity

    # 1) Upload the image (per docs)
    file_ref = client.files.upload(file=image_path)  # returns a File object usable in contents
    if verbose:
        print(f"[upload] name={file_ref.name} uri={getattr(file_ref, 'uri', None)}")

    # 2) Build contents and call generate_content (per docs)
    contents = [
        types.Part.from_text(text=INSTRUCTION),
        file_ref,  # pass the uploaded file directly
        types.Part.from_text(text=f"Die type: d{max_face}. Return only the top face integer."),
    ]
    resp = client.models.generate_content(model=MODEL_NAME, contents=contents)
    text = (getattr(resp, "text", None) or getattr(resp, "output_text", "") or "").strip()
    if verbose:
        print(f"[raw] {text}")

    n = parse_int(text, max_face)
    if n is None:
        raise RuntimeError(f"Model didn’t return a valid integer. Raw: {text!r}")
    return n, text

def main():
    ap = argparse.ArgumentParser(description="Read top-face die number using Gemini (Google GenAI SDK).")
    ap.add_argument("image", help="Path to JPEG/PNG frame (e.g., ..._last.jpg).")
    ap.add_argument("--max-face", type=int, default=20)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    n, raw = read_die(args.image, max_face=args.max_face, verbose=args.verbose)
    print(f"🎲 {n}")
    if args.verbose:
        print(f"[model] {raw}")

if __name__ == "__main__":
    main()
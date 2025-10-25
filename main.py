#!/usr/bin/env python3
import os, sys, subprocess, time, signal, argparse
from datetime import datetime
from pathlib import Path

import requests

# --- Gemini (Google GenAI) ---
from google import genai
from google.genai import types  # typed helpers

# ================== CONFIG ==================
OUTDIR = Path.home() / "dice_captures"
OUTDIR.mkdir(parents=True, exist_ok=True)

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

INSTRUCTION = (
    "You are a precise die-reading assistant.\n"
    "Look at the image of a single polyhedral die.\n"
    "Decide which face is the TOP face (the result of the roll).\n"
    "Return ONLY the Arabic numeral on the TOP face as an integer.\n"
    "If unclear, return exactly: unknown"
)
# ============================================

def nowtag():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def vprint(verbose, *a):
    if verbose: print(*a, flush=True)

def run(cmd, verbose=False, timeout=None):
    vprint(verbose, f"→ RUN: {' '.join(cmd)}")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if verbose:
        if res.stdout.strip(): print("  STDOUT:", res.stdout.strip(), flush=True)
        if res.stderr.strip(): print("  STDERR:", res.stderr.strip(), flush=True)
    return res

def countdown(verbose=False):
    vprint(verbose, "\nGet ready…")
    for n in [3, 2, 1]:
        print(f"{n}…", flush=True); time.sleep(1)
    print("Roll the dice!", flush=True)

# ---------- Camera capture (defaults: AUTO exposure/AWB) ----------
def capture_still(ts, w=1920, h=1080, quality=90, verbose=False):
    """
    Captures a single JPEG using rpicam-still with default (auto) exposure/awb.
    No manual shutter/gain—so images are bright and clean again.
    """
    jpg_path = OUTDIR / f"dice_{ts}_last.jpg"

    if not shutil_which("rpicam-still"):
        print("Error: rpicam-still not found. Install rpicam-apps.", file=sys.stderr)
        sys.exit(1)

    # Keep it minimal: autofocus on, auto exposure, auto white balance.
    # Avoid passing --shutter/--gain/--awb so the camera chooses good brightness.
    cmd = [
        "rpicam-still",
        "--timeout", "1200",              # ~1.2s to settle AF/exposure
        "--autofocus-mode", "auto",
        "--autofocus-range", "macro",
        "--width", str(w), "--height", str(h),
        "--quality", str(quality),
        "-o", str(jpg_path),
        "--nopreview",
    ]
    res = run(cmd, verbose=verbose, timeout=10)
    if res.returncode != 0 or not jpg_path.exists() or jpg_path.stat().st_size == 0:
        raise RuntimeError(f"Still capture failed.\n{res.stderr}")
    vprint(verbose, f"Captured: {jpg_path}")
    return str(jpg_path)

def shutil_which(name):
    from shutil import which
    return which(name) is not None

# ---------- Gemini ----------
def parse_int(text, max_face):
    import re
    s = (text or "").strip().lower()
    if s == "unknown":
        return None
    m = re.search(r"\b(\d{1,3})\b", s)
    if not m: return None
    n = int(m.group(1))
    if (1 <= n <= max_face) or (max_face == 100 and 1 <= n <= 100):
        return n
    return None

def read_with_gemini(image_path, max_face=20, verbose=False):
    if not GEMINI_API_KEY:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in your shell (e.g., ~/.bashrc)")

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Upload file (per SDK docs) and generate
    file_ref = client.files.upload(file=image_path)
    if verbose:
        print(f"[upload] name={file_ref.name} uri={getattr(file_ref, 'uri', None)}")

    contents = [
        types.Part.from_text(text=INSTRUCTION),
        file_ref,
        types.Part.from_text(text=f"Die type: d{max_face}. Return only the top face integer."),
    ]
    resp = client.models.generate_content(model=GEMINI_MODEL, contents=contents)
    text = (getattr(resp, "text", None) or getattr(resp, "output_text", "") or "").strip()
    if verbose:
        print(f"[model] {text}")
    n = parse_int(text, max_face)
    if n is None:
        raise RuntimeError(f"Model did not return a valid integer. Raw: {text!r}")
    return n, text

# ---------- Discord ----------
def post_image_to_discord(webhook_url, image_path, roll, verbose=False):
    if not webhook_url:
        vprint(verbose, "No DISCORD_WEBHOOK_URL set; skipping Discord post.")
        return False
    content = f"🎲 **Roll result:** {roll}"
    files = []
    try:
        files.append(("files[0]", (os.path.basename(image_path), open(image_path, "rb"), "image/jpeg")))
    except Exception as e:
        vprint(verbose, f"Could not attach image: {e}")

    try:
        r = requests.post(webhook_url, data={"content": content}, files=files, timeout=30)
        vprint(verbose, f"Discord response: {r.status_code} {r.text[:200]}")
        return 200 <= r.status_code < 300
    except Exception as e:
        vprint(verbose, f"Discord post error: {e}")
        return False

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Capture a single bright still, read with Gemini, and post to Discord.")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--no-discord", action="store_true", help="Skip Discord post.")
    parser.add_argument("--max-face", type=int, default=20, help="Die size (6/8/10/12/20/100).")

    # Camera options (defaults chosen to restore auto-exposure behavior)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality 1..100")
    parser.add_argument("--countdown", action="store_true", help="Show 3..2..1 before capture.")

    args = parser.parse_args()
    verbose = args.verbose

    # ctrl+c graceful
    signal.signal(signal.SIGINT, lambda *a: sys.exit(130))

    ts = nowtag()

    if args.countdown:
        countdown(verbose=verbose)

    try:
        jpg_path = capture_still(ts, w=args.width, h=args.height, quality=args.quality, verbose=verbose)
    except Exception as e:
        print(f"Capture error: {e}", file=sys.stderr, flush=True); sys.exit(3)

    print(f"Classifying exactly: {jpg_path}", flush=True)

    try:
        roll, raw = read_with_gemini(jpg_path, max_face=args.max_face, verbose=verbose)
        print(f"Detected roll: {roll}", flush=True)
    except Exception as e:
        print(f"Gemini error: {e}", file=sys.stderr, flush=True)
        roll = "unknown"

    if not args.no_discord:
        ok = post_image_to_discord(DISCORD_WEBHOOK_URL, jpg_path, roll, verbose=verbose)
        print("Posted to Discord." if ok else "Failed to post to Discord.", flush=True)

    print("Image:", jpg_path, flush=True)

if __name__ == "__main__":
    import shutil as _shutil
    main()
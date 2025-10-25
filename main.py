#!/usr/bin/env python3
import os, sys, shutil, subprocess, time, signal, argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests

# OCR (required for this numbers-only build)
try:
    import pytesseract
    HAVE_TESS = True
except Exception:
    HAVE_TESS = False

# ================== DEFAULT CONFIG ==================
DURATION_MS = 7000           # 7 seconds
FPS         = 30
RES_W, RES_H = 1920, 1080
BITRATE     = 8_000_000      # ~8 Mbps

# Autofocus tuned for close-ish work (Module 3)
RPICAM_AF_OPTS = [
    "--autofocus-mode", "auto",
    "--autofocus-range", "macro",
    # "--autofocus-speed", "fast",
]

OUTDIR = Path.home() / "dice_captures"
OUTDIR.mkdir(parents=True, exist_ok=True)

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
# =====================================================

# ----------------- Utility helpers -------------------
def nowtag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def vprint(verbose, *a, **kw):
    if verbose:
        print(*a, **kw, flush=True)

def run(cmd, verbose=False, timeout=None):
    vprint(verbose, f"→ RUN: {' '.join(cmd)} (timeout={timeout})")
    try:
        res = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Command timed out: {' '.join(cmd)}")
    if verbose:
        if res.stdout.strip(): print("  STDOUT:", res.stdout.strip(), flush=True)
        if res.stderr.strip(): print("  STDERR:", res.stderr.strip(), flush=True)
    return res

def ms(): return int(time.time() * 1000)
def elapsed_ms(t0): return ms() - t0
def within_budget(t0, budget_ms): return elapsed_ms(t0) < budget_ms

def save_debug(image, path, verbose=False):
    try:
        cv2.imwrite(str(path), image)
        vprint(verbose, f"Saved debug: {path}")
    except Exception as e:
        vprint(verbose, f"Failed to save debug {path}: {e}")

# --------------- Recording / muxing ------------------
def record_video(ts, verbose=False, prefer_libav=True, fps=FPS, w=RES_W, h=RES_H, duration_ms=DURATION_MS):
    """
    Record duration_ms and return path to finalized MP4.
    Falls back to raw .h264 + ffmpeg mux if libav isn't available.
    """
    mp4_path  = OUTDIR / f"dice_{ts}.mp4"
    h264_path = OUTDIR / f"dice_{ts}.h264"

    record_timeout = max(10, int(duration_ms/1000) + 8)

    if prefer_libav:
        vprint(verbose, "Trying direct MP4 with libav muxer…")
        cmd = [
            "rpicam-vid",
            "-t", str(duration_ms),
            "--framerate", str(fps),
            "--width", str(w), "--height", str(h),
            "--bitrate", str(BITRATE),
            *RPICAM_AF_OPTS,
            "--codec", "libav", "--libav-format", "mp4",
            "-o", str(mp4_path),
        ]
        res = run(cmd, verbose=verbose, timeout=record_timeout)
        if res.returncode == 0 and mp4_path.exists() and mp4_path.stat().st_size > 0:
            vprint(verbose, f"Direct MP4 recorded: {mp4_path}")
            return str(mp4_path)
        else:
            vprint(verbose, "Direct MP4 failed or empty; falling back to h264+mux…")

    vprint(verbose, "Recording raw H.264 stream…")
    rec = run([
        "rpicam-vid",
        "-t", str(duration_ms),
        "--framerate", str(fps),
        "--width", str(w), "--height", str(h),
        "--bitrate", str(BITRATE),
        *RPICAM_AF_OPTS,
        "-o", str(h264_path),
    ], verbose=verbose, timeout=record_timeout)
    if rec.returncode != 0 or not h264_path.exists():
        raise RuntimeError(f"Recording failed.\n{rec.stderr}")

    vprint(verbose, "Muxing H.264 → MP4 (ffmpeg, stream copy)…")
    mux = run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(h264_path),
        "-c", "copy",
        str(mp4_path)
    ], verbose=verbose, timeout=60)
    if mux.returncode != 0 or not mp4_path.exists():
        raise RuntimeError(f"MP4 mux failed.\n{mux.stderr}")

    try: h264_path.unlink()
    except Exception: pass

    vprint(verbose, f"Final MP4: {mp4_path}")
    return str(mp4_path)

def extract_last_frame(mp4_path, ts, verbose=False):
    jpg_path = OUTDIR / f"dice_{ts}_last.jpg"
    vprint(verbose, "Extracting last frame (~0.1s before end) as JPEG…")
    res = run([
        "ffmpeg", "-y",
        "-sseof", "-0.1",   # slightly before end avoids boundary issues
        "-i", mp4_path,
        "-vframes", "1",
        "-q:v", "2",
        str(jpg_path)
    ], verbose=verbose, timeout=30)
    if res.returncode != 0 or not jpg_path.exists():
        raise RuntimeError(f"Extract last frame failed:\n{res.stderr}")
    vprint(verbose, f"Last frame: {jpg_path}")
    return str(jpg_path)

# ---------------- OCR-only Classifier ----------------
def resize_max_side(img, max_side):
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img, 1.0
    scale = max_side / float(max(h, w))
    out = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return out, scale

def center_crop(img, frac):
    if not (0.3 <= frac <= 1.0):
        return img
    h, w = img.shape[:2]
    cx, cy = w//2, h//2
    rx, ry = int(w*frac/2), int(h*frac/2)
    x1, y1, x2, y2 = max(0,cx-rx), max(0,cy-ry), min(w,cx+rx), min(h,cy+ry)
    return img[y1:y2, x1:x2]

def ocr_digits_single(img_bgr, psm=6, inv=False, verbose=False):
    """Run Tesseract OCR on a single preprocessed image."""
    if not HAVE_TESS:
        vprint(verbose, "OCR skipped: pytesseract/tesseract not installed.")
        return None, ""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if inv:
        gray = 255 - gray
    config = f"--oem 1 --psm {psm} -c tessedit_char_whitelist=0123456789"
    txt = pytesseract.image_to_string(gray, config=config).strip()
    if verbose: print(f"  OCR(psm={psm}, inv={inv}) -> [{txt}]", flush=True)
    return txt, config

def parse_numeric(txt, allow_zero=False, max_face=20, percentile=False):
    """
    Return int if 1..max_face (or 0 if allow_zero).
    Special handling:
      - percentile=True: map '00' to 100 (or 0 if allow_zero=True)
    """
    if not txt:
        return None
    # Keep only digits
    digits = "".join(ch for ch in txt if ch.isdigit())
    if digits == "" and not allow_zero:
        return None
    # Handle percentile '00'
    if percentile and digits in ("00", "000"):
        return 100 if not allow_zero else 0
    try:
        n = int(digits) if digits != "" else 0
    except Exception:
        return None
    # Range filter
    if allow_zero and n == 0:
        return 0
    if 1 <= n <= max_face or (percentile and n in (10,20,30,40,50,60,70,80,90,100)):
        return n
    return None

def enhance_for_ocr(img_bgr, mode, verbose=False):
    """
    Create OCR-friendly views.
    mode:
      'bw'       -> Otsu
      'clahe'    -> CLAHE + Otsu
      'tophat'   -> white tophat + Otsu
      'morph'    -> opening to remove noise
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if mode == "bw":
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    if mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        g2 = clahe.apply(gray)
        _, bw = cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    if mode == "tophat":
        kernel = np.ones((9,9), np.uint8)
        top = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        _, bw = cv2.threshold(top, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    if mode == "morph":
        g2 = cv2.medianBlur(gray, 3)
        opn = cv2.morphologyEx(g2, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
        _, bw = cv2.threshold(opn, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    return img_bgr

def progressive_ocr(jpg_path, verbose=False,
                    roi_step_sequence=(0.6, 0.8, 1.0),
                    max_side_sequence=(480, 640, 960, 1280),
                    total_budget_ms=1500,
                    allow_zero=False,
                    max_face=20,
                    percentile=False,
                    save_debug_images=True):
    """
    Progressive numbers-only classifier:
      - Verifies exact _last.jpg path
      - For each (ROI frac, max side):
          * Crop center, downscale
          * Generate multiple preprocess variants
          * OCR with various PSMs and inversion
      - Accept first numeric in allowed range
    """
    t0 = ms()
    vprint(verbose, f"[OCR] Start classification on: {jpg_path}")
    img_full = cv2.imread(jpg_path)
    if img_full is None:
        vprint(verbose, "[OCR] Could not load image.")
        return None

    dbg_dir = Path(jpg_path).with_suffix("")
    if save_debug_images:
        Path(str(dbg_dir) + "_dbg").mkdir(parents=True, exist_ok=True)

    preprocess_modes = ["bw", "clahe", "tophat", "morph"]
    psms = [6, 7, 8, 10, 11, 13]  # single block, sparse, etc.

    stage_idx = 0
    for roi_frac in roi_step_sequence:
        for max_side in max_side_sequence:
            if not within_budget(t0, total_budget_ms):
                vprint(verbose, f"[OCR] Time budget exceeded at stage {stage_idx}.")
                return None
            stage_idx += 1
            vprint(verbose, f"[OCR] Stage {stage_idx}: roi={roi_frac}, max_side={max_side}")

            img_roi = center_crop(img_full, roi_frac)
            img_small, scale = resize_max_side(img_roi, max_side)

            if save_debug_images:
                save_debug(img_roi, Path(str(dbg_dir) + f"_dbg/roi_{stage_idx}.jpg"), verbose)
                save_debug(img_small, Path(str(dbg_dir) + f"_dbg/roi_small_{stage_idx}.jpg"), verbose)

            # Try preprocess variants (normal & inverted)
            for mode in preprocess_modes:
                if not within_budget(t0, total_budget_ms): break
                proc = enhance_for_ocr(img_small, mode, verbose)
                if save_debug_images:
                    save_debug(proc, Path(str(dbg_dir) + f"_dbg/{mode}_{stage_idx}.png"), verbose)

                for inv in (False, True):
                    if not within_budget(t0, total_budget_ms): break
                    for psm in psms:
                        if not within_budget(t0, total_budget_ms): break
                        txt, cfg = ocr_digits_single(proc, psm=psm, inv=inv, verbose=verbose)
                        n = parse_numeric(txt, allow_zero=allow_zero, max_face=max_face, percentile=percentile)
                        if n is not None:
                            vprint(verbose, f"[OCR] ✓ RESULT: {n} (mode={mode}, inv={inv}, psm={psm})")
                            return n

            vprint(verbose, f"[OCR] Stage {stage_idx} no result; expanding… (elapsed {elapsed_ms(t0)} ms)")

    vprint(verbose, f"[OCR] No classification result (total {elapsed_ms(t0)} ms).")
    return None

# ---------------- Discord upload ----------------
def post_to_discord(webhook_url, mp4_path, jpg_path, roll_result, verbose=False):
    if not webhook_url:
        vprint(verbose, "No DISCORD_WEBHOOK_URL set; skipping Discord post.")
        return False

    content = f"🎲 **Roll result:** {roll_result if roll_result is not None else 'Unknown'}"
    files = []
    try:
        files.append(("files[0]", (os.path.basename(mp4_path), open(mp4_path, "rb"), "video/mp4")))
    except Exception as e:
        vprint(verbose, f"Could not attach video: {e}")
    try:
        files.append(("files[1]", (os.path.basename(jpg_path), open(jpg_path, "rb"), "image/jpeg")))
    except Exception as e:
        vprint(verbose, f"Could not attach image: {e}")

    vprint(verbose, "Posting to Discord…")
    try:
        r = requests.post(webhook_url, data={"content": content}, files=files, timeout=30)
        vprint(verbose, f"Discord response: {r.status_code} {r.text[:200]}")
        return 200 <= r.status_code < 300
    except Exception as e:
        vprint(verbose, f"Discord post error: {e}")
        return False

# ---------------- Countdown ----------------
def countdown(verbose=False):
    vprint(verbose, "\nGet ready…")
    for n in [3, 2, 1]:
        print(f"{n}…", flush=True)
        time.sleep(1)
    print("Roll the dice!", flush=True)  # recording starts immediately after this returns

# ---------------- Main ----------------
def main():
    global FPS, RES_W, RES_H, DURATION_MS

    parser = argparse.ArgumentParser(
        description="Record Pi Camera, make MP4, grab last frame, OCR-only classify numbers, post to Discord."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--no-discord", action="store_true", help="Skip Discord upload even if webhook is set.")
    parser.add_argument("--no-classify", action="store_true", help="Skip classification step.")

    # capture params (literal defaults to avoid global scoping issues)
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default 30).")
    parser.add_argument("--width", type=int, default=1920, help="Video width (default 1920).")
    parser.add_argument("--height", type=int, default=1080, help="Video height (default 1080).")
    parser.add_argument("--duration-ms", type=int, default=7000, help="Duration in ms (default 7000).")

    # OCR tuning
    parser.add_argument("--ocr-budget", type=int, default=1500, help="Total OCR time budget (ms).")
    parser.add_argument("--max-face", type=int, default=20, help="Max face value (e.g., 20 for d20, 12 for d12, 100 for percentile).")
    parser.add_argument("--allow-zero", action="store_true", help="Allow 0 as valid result (e.g., some rules for percentile).")
    parser.add_argument("--percentile", action="store_true", help="Treat '00' as 100 (or 0 if --allow-zero).")
    parser.add_argument("--roi-seq", type=str, default="0.6,0.8,1.0",
                        help="ROI center crop fractions in order (comma).")
    parser.add_argument("--maxside-seq", type=str, default="480,640,960,1280",
                        help="Max-side sizes in order (comma).")

    args = parser.parse_args()
    verbose = args.verbose

    # apply CLI capture overrides
    FPS, RES_W, RES_H, DURATION_MS = args.fps, args.width, args.height, args.duration_ms

    # deps
    if shutil.which("rpicam-vid") is None:
        print("Error: rpicam-vid not found. Install rpicam-apps.", file=sys.stderr); sys.exit(1)
    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg not found. Install ffmpeg.", file=sys.stderr); sys.exit(1)
    if not HAVE_TESS and not args.no_classify:
        print("Warning: pytesseract/tesseract-ocr not found; OCR will be skipped.", file=sys.stderr)

    # Ctrl+C graceful
    signal.signal(signal.SIGINT, lambda *a: sys.exit(130))

    # parse sequences
    try:
        roi_seq = [float(x.strip()) for x in args.roi_seq.split(",") if x.strip()]
        maxside_seq = [int(x.strip()) for x in args.maxside_seq.split(",") if x.strip()]
    except Exception:
        print("Bad --roi-seq or --maxside-seq format.", file=sys.stderr); sys.exit(2)

    ts = nowtag()

    # countdown → record immediately
    countdown(verbose=verbose)

    try:
        mp4_path = record_video(ts, verbose=verbose, prefer_libav=True, fps=FPS, w=RES_W, h=RES_H, duration_ms=DURATION_MS)
    except Exception as e:
        print(f"Capture error: {e}", file=sys.stderr, flush=True); sys.exit(3)

    try:
        jpg_path = extract_last_frame(mp4_path, ts, verbose=verbose)
    except Exception as e:
        print(f"Last-frame extract error: {e}", file=sys.stderr, flush=True); sys.exit(4)

    # explicitly confirm we’re classifying the exact _last.jpg we saved
    print(f"Classifying exactly: {jpg_path}", flush=True)

    roll = None
    if not args.no_classify:
        roll = progressive_ocr(
            jpg_path,
            verbose=verbose,
            roi_step_sequence=roi_seq,
            max_side_sequence=maxside_seq,
            total_budget_ms=args.ocr_budget,
            allow_zero=args.allow_zero,
            max_face=args.max_face,
            percentile=args.percentile,
            save_debug_images=True
        )
        print(f"Detected roll: {roll}", flush=True)

    posted = False
    if not args.no_discord and DISCORD_WEBHOOK_URL:
        posted = post_to_discord(DISCORD_WEBHOOK_URL, mp4_path, jpg_path, roll, verbose=verbose)
        print("Posted to Discord." if posted else "Failed to post to Discord.", flush=True)
    else:
        vprint(verbose, "Discord step skipped (no webhook or --no-discord).")

    print("Video:", mp4_path, flush=True)
    print("Last frame:", jpg_path, flush=True)

if __name__ == "__main__":
    main()

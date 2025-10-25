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
def record_video(
    ts,
    verbose=False,
    prefer_libav=True,
    fps=FPS,
    w=RES_W,
    h=RES_H,
    duration_ms=DURATION_MS,
    shutter_us=10000,
    gain=1.0,
    awb="incandescent",
    denoise="off"
):
    """
    Record duration_ms and return path to finalized MP4.
    Falls back to raw .h264 + ffmpeg mux if libav isn't available.
    """
    mp4_path  = OUTDIR / f"dice_{ts}.mp4"
    h264_path = OUTDIR / f"dice_{ts}.h264"

    record_timeout = max(10, int(duration_ms/1000) + 8)

    # common capture args
    base_args = [
        "--framerate", str(fps),
        "--width", str(w), "--height", str(h),
        "--bitrate", str(BITRATE),
        *RPICAM_AF_OPTS,
        "--shutter", str(int(shutter_us)),
        "--gain", str(float(gain)),
        "--awb", awb,
        "--denoise", denoise,
    ]

    if prefer_libav:
        vprint(verbose, "Trying direct MP4 with libav muxer…")
        cmd = [
            "rpicam-vid",
            "-t", str(duration_ms),
            *base_args,
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
        *base_args,
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
    """Kept for compatibility; not used by default."""
    jpg_path = OUTDIR / f"dice_{ts}_last.jpg"
    vprint(verbose, "Extracting last frame (~0.1s before end) as JPEG…")
    res = run([
        "ffmpeg", "-y",
        "-sseof", "-0.1",
        "-i", mp4_path,
        "-vframes", "1",
        "-q:v", "2",
        str(jpg_path)
    ], verbose=verbose, timeout=30)
    if res.returncode != 0 or not jpg_path.exists():
        raise RuntimeError(f"Extract last frame failed:\n{res.stderr}")
    vprint(verbose, f"Last frame: {jpg_path}")
    return str(jpg_path)

def extract_best_tail_frame(mp4_path, ts, verbose=False, tail_window_s=0.6, n_frames=10):
    """
    Sample frames from the last tail_window_s seconds and pick the sharpest by Laplacian focus measure.
    Saves to dice_{ts}_last.jpg that the pipeline expects.
    """
    tmpdir = OUTDIR / f"dice_{ts}_tail"
    tmpdir.mkdir(parents=True, exist_ok=True)
    pattern = str(tmpdir / "f_%02d.jpg")

    # choose a reasonable sampling fps for the tail
    sample_fps = max(2, int(np.ceil(n_frames / max(tail_window_s, 0.2))))
    run([
        "ffmpeg","-y","-sseof", f"-{tail_window_s}",
        "-i", mp4_path, "-vf", f"fps={sample_fps}",
        "-vframes", str(n_frames), pattern
    ], verbose=verbose, timeout=45)

    best_path, best_score = None, -1.0
    for p in sorted(tmpdir.glob("f_*.jpg")):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        score = cv2.Laplacian(img, cv2.CV_64F).var()
        if score > best_score:
            best_score, best_path = score, p

    if not best_path:
        raise RuntimeError("No frames extracted for sharpness selection")

    jpg_path = OUTDIR / f"dice_{ts}_last.jpg"
    shutil.copy2(best_path, jpg_path)

    try: shutil.rmtree(tmpdir)
    except: pass

    vprint(verbose, f"Selected sharpest tail frame (var={best_score:.1f}): {jpg_path}")
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

def crop_to_digit_cluster(img_bgr):
    """
    Use MSER to find a glyph-like region, prefer near-center, crop tightly with padding.
    Falls back to a contour-based method if MSER returns nothing or isn't available.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- Try MSER (positional args to avoid keyword incompatibility) ---
    regions = None
    try:
        # delta=5, min_area=60, max_area=5000 (tweak if needed)
        mser = cv2.MSER_create(5, 60, 5000)
        regions, _ = mser.detectRegions(gray)
    except Exception:
        regions = None

    h, w = gray.shape[:2]
    cx, cy = w // 2, h // 2

    def crop_from_regions(regs):
        if not regs:
            return None
        best_rect, best_score = None, 1e12
        for r in regs:
            x, y, w2, h2 = cv2.boundingRect(r.reshape(-1, 1, 2))
            center_dist = ((x + w2 / 2 - cx) ** 2 + (y + h2 / 2 - cy) ** 2) ** 0.5
            aspect_penalty = abs((w2 / max(h2, 1e-3)) - 0.7)  # soft preference for digit-ish boxes
            score = center_dist + 50 * aspect_penalty
            if score < best_score:
                best_score, best_rect = score, (x, y, w2, h2)
        if best_rect is None:
            return None
        x, y, w2, h2 = best_rect
        pad = int(0.25 * max(w2, h2))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img_bgr.shape[1], x + w2 + pad), min(img_bgr.shape[0], y + h2 + pad)
        return img_bgr[y1:y2, x1:x2]

    # If MSER produced something, crop from it
    if regions:
        cropped = crop_from_regions(regions)
        if cropped is not None:
            return cropped

    # --- Fallback: simple contour-based glyph guess near center ---
    # Adaptive threshold to pull strokes; adjust blockSize/C if needed
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 31, 7)
    # clean small noise
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr

    best_rect, best_score = None, 1e12
    for cnt in contours:
        x, y, w2, h2 = cv2.boundingRect(cnt)
        area = w2 * h2
        if area < 60 or area > 8000:  # filter extremes
            continue
        center_dist = ((x + w2 / 2 - cx) ** 2 + (y + h2 / 2 - cy) ** 2) ** 0.5
        aspect_penalty = abs((w2 / max(h2, 1e-3)) - 0.7)
        score = center_dist + 50 * aspect_penalty
        if score < best_score:
            best_score, best_rect = score, (x, y, w2, h2)

    if best_rect is None:
        return img_bgr

    x, y, w2, h2 = best_rect
    pad = int(0.25 * max(w2, h2))
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(img_bgr.shape[1], x + w2 + pad), min(img_bgr.shape[0], y + h2 + pad)
    return img_bgr[y1:y2, x1:x2]

def ocr_digits_single(img_bgr, psm=6, inv=False, force_no_invert=False, verbose=False):
    """Run Tesseract OCR on a single preprocessed image."""
    if not HAVE_TESS:
        vprint(verbose, "OCR skipped: pytesseract/tesseract not installed.")
        return None, ""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if inv:
        gray = 255 - gray
    config = (
        f"--oem 1 --psm {psm} "
        "-c tessedit_char_whitelist=0123456789 "
        "-c classify_bln_numeric_mode=1"
    )
    if force_no_invert:
        config += " -c tessedit_do_invert=0"
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

def enhance_gold_on_dark(img_bgr):
    """
    Specialized preprocessor for metallic gold digits on dark burgundy/purple dice.
    Produces a text-like image with dark digits on light background.
    """
    # 1) HSV mask for yellow/gold
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Hue 10-35 (warm), S 60+, V 70+ — tune if needed
    mask1 = cv2.inRange(hsv, (10, 60, 70), (35, 255, 255))

    # 2) Reinforce with LAB b-channel (yellow high)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    b = lab[:,:,2]
    _, mask2 = cv2.threshold(b, 150, 255, cv2.THRESH_BINARY)  # tune 140–170

    mask = cv2.bitwise_and(mask1, mask2)

    # Clean mask
    k = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ink = cv2.bitwise_and(gray, gray, mask=mask)
    ink = cv2.normalize(ink, None, 0, 255, cv2.NORM_MINMAX)
    _, bw = cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Invert so digits are dark on light
    bw = 255 - bw
    bw = cv2.morphologyEx(bw, cv2.MORPH_DILATE, np.ones((2,2), np.uint8), iterations=1)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

def progressive_ocr(
    jpg_path,
    verbose=False,
    roi_step_sequence=(0.6, 0.8, 1.0),
    max_side_sequence=(480, 640, 960, 1280),
    total_budget_ms=1500,
    allow_zero=False,
    max_face=20,
    percentile=False,
    save_debug_images=True
):
    """
    Progressive numbers-only classifier:
      - Verifies exact _last.jpg path
      - For each (ROI frac, max side):
          * Crop center, downscale
          * (NEW) Digit-cluster crop via MSER
          * Generate multiple preprocess variants (incl. goldmask)
          * OCR with tuned PSMs and inversion strategy
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

    # Try gold-masking first for metallic digits, then other robust modes
    preprocess_modes = ["goldmask", "clahe", "tophat", "morph", "bw"]
    # Favor single-character/line PSMS first, then more general
    psms = [10, 13, 6, 7, 8, 11]  # 10=single char, 13=raw line, 6 block, etc.

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

            # NEW: try to focus on the numeral cluster
            img_digits = crop_to_digit_cluster(img_small)

            if save_debug_images:
                save_debug(img_roi, Path(str(dbg_dir) + f"_dbg/roi_{stage_idx}.jpg"), verbose)
                save_debug(img_small, Path(str(dbg_dir) + f"_dbg/roi_small_{stage_idx}.jpg"), verbose)
                save_debug(img_digits, Path(str(dbg_dir) + f"_dbg/roi_digits_{stage_idx}.jpg"), verbose)

            # Try preprocess variants (normal & inverted where appropriate)
            for mode in preprocess_modes:
                if not within_budget(t0, total_budget_ms): break

                if mode == "goldmask":
                    proc = enhance_gold_on_dark(img_digits)
                    force_no_invert = True  # we've already controlled inversion in this path
                else:
                    proc = enhance_for_ocr(img_digits, mode, verbose)
                    force_no_invert = False

                if save_debug_images:
                    save_debug(proc, Path(str(dbg_dir) + f"_dbg/{mode}_{stage_idx}.png"), verbose)

                for inv in ((False,) if force_no_invert else (False, True)):
                    if not within_budget(t0, total_budget_ms): break
                    for psm in psms:
                        if not within_budget(t0, total_budget_ms): break
                        txt, cfg = ocr_digits_single(
                            proc, psm=psm, inv=inv, force_no_invert=force_no_invert, verbose=verbose
                        )
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
        description="Record Pi Camera, make MP4, grab sharpest tail frame, OCR-only classify numbers, post to Discord."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--no-discord", action="store_true", help="Skip Discord upload even if webhook is set.")
    parser.add_argument("--no-classify", action="store_true", help="Skip classification step.")

    # capture params (literal defaults to avoid global scoping issues)
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default 30).")
    parser.add_argument("--width", type=int, default=1920, help="Video width (default 1920).")
    parser.add_argument("--height", type=int, default=1080, help="Video height (default 1080).")
    parser.add_argument("--duration-ms", type=int, default=7000, help="Duration in ms (default 7000).")

    # NEW: exposure/denoise tweaks for crisp digits
    parser.add_argument("--shutter-us", type=int, default=10000, help="Shutter in microseconds (e.g., 10000 ≈ 1/100s).")
    parser.add_argument("--gain", type=float, default=1.0, help="Analog gain (ISO-ish).")
    parser.add_argument("--awb", type=str, default="incandescent", help="AWB mode (auto, incandescent, daylight, etc.)")
    parser.add_argument("--denoise", type=str, default="off", help="Denoise (off, cdn_off, cdn_fast, cdn_hq)")

    # OCR tuning
    parser.add_argument("--ocr-budget", type=int, default=1500, help="Total OCR time budget (ms).")
    parser.add_argument("--max-face", type=int, default=20, help="Max face value (e.g., 20 for d20, 12 for d12, 100 for percentile).")
    parser.add_argument("--allow-zero", action="store_true", help="Allow 0 as valid result (e.g., some rules for percentile).")
    parser.add_argument("--percentile", action="store_true", help="Treat '00' as 100 (or 0 if --allow-zero).")
    parser.add_argument("--roi-seq", type=str, default="0.6,0.8,1.0", help="ROI center crop fractions in order (comma).")
    parser.add_argument("--maxside-seq", type=str, default="480,640,960,1280", help="Max-side sizes in order (comma).")

    # NEW: best-tail-frame extraction knobs
    parser.add_argument("--tail-window-s", type=float, default=0.6, help="Seconds from end to sample frames.")
    parser.add_argument("--tail-frames", type=int, default=10, help="Number of frames to sample from tail window.")

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
        mp4_path = record_video(
            ts,
            verbose=verbose,
            prefer_libav=True,
            fps=FPS,
            w=RES_W,
            h=RES_H,
            duration_ms=DURATION_MS,
            shutter_us=args.shutter_us,
            gain=args.gain,
            awb=args.awb,
            denoise=args.denoise
        )
    except Exception as e:
        print(f"Capture error: {e}", file=sys.stderr, flush=True); sys.exit(3)

    try:
        # NEW: select sharpest tail frame
        jpg_path = extract_best_tail_frame(
            mp4_path, ts, verbose=verbose, tail_window_s=args.tail_window_s, n_frames=args.tail_frames
        )
        # If you prefer the old single-frame behavior, comment the block above and uncomment below:
        # jpg_path = extract_last_frame(mp4_path, ts, verbose=verbose)
    except Exception as e:
        print(f"Tail-frame extract error: {e}", file=sys.stderr, flush=True); sys.exit(4)

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

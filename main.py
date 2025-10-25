#!/usr/bin/env python3
import os, sys, shutil, subprocess, time, signal, argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests

# ================== DEFAULT CONFIG ==================
DURATION_MS = 7000           # 7 seconds
FPS         = 30
RES_W, RES_H = 1920, 1080
BITRATE     = 8_000_000      # ~8 Mbps

# Autofocus tuned for close-ish work (Module 3)
RPICAM_AF_OPTS = [
    "--autofocus-mode", "auto",
    "--autofocus-range", "macro",
    # "--autofocus-speed", "fast",  # uncomment if you want more AF aggression
]

OUTDIR = Path.home() / "dice_captures"
OUTDIR.mkdir(parents=True, exist_ok=True)

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
# =====================================================

def nowtag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def vprint(verbose, *a, **kw):
    if verbose:
        print(*a, **kw, flush=True)

def run(cmd, verbose=False, timeout=None):
    vprint(verbose, f"→ RUN: {' '.join(cmd)} (timeout={timeout})")
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Command timed out: {' '.join(cmd)}")
    if verbose:
        if res.stdout.strip():
            print("  STDOUT:", res.stdout.strip(), flush=True)
        if res.stderr.strip():
            print("  STDERR:", res.stderr.strip(), flush=True)
    return res

def record_video(ts, verbose=False, prefer_libav=True):
    """
    Record DURATION_MS and return path to finalized MP4.
    Falls back to raw .h264 + ffmpeg mux if libav isn't available.
    """
    mp4_path  = OUTDIR / f"dice_{ts}.mp4"
    h264_path = OUTDIR / f"dice_{ts}.h264"

    # We’ll allow a little extra time over DURATION_MS
    # (recording cmd should never hang indefinitely).
    record_timeout = max(10, int(DURATION_MS/1000) + 8)

    if prefer_libav:
        vprint(verbose, "Trying direct MP4 with libav muxer…")
        cmd = [
            "rpicam-vid",
            "-t", str(DURATION_MS),
            "--framerate", str(FPS),
            "--width", str(RES_W), "--height", str(RES_H),
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
            vprint(verbose, "Direct MP4 failed or empty; will fall back to h264+mux…")

    # Fallback path
    vprint(verbose, "Recording raw H.264 stream…")
    rec = run([
        "rpicam-vid",
        "-t", str(DURATION_MS),
        "--framerate", str(FPS),
        "--width", str(RES_W), "--height", str(RES_H),
        "--bitrate", str(BITRATE),
        *RPICAM_AF_OPTS,
        "-o", str(h264_path),
    ], verbose=verbose, timeout=record_timeout)
    if rec.returncode != 0 or not h264_path.exists():
        raise RuntimeError(f"Recording failed.\n{rec.stderr}")

    vprint(verbose, "Muxing H.264 → MP4 (ffmpeg, stream copy)…")
    mux = run([
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", str(h264_path),
        "-c", "copy",
        str(mp4_path)
    ], verbose=verbose, timeout=60)
    if mux.returncode != 0 or not mp4_path.exists():
        raise RuntimeError(f"MP4 mux failed.\n{mux.stderr}")

    try:
        h264_path.unlink()
    except Exception:
        pass

    vprint(verbose, f"Final MP4: {mp4_path}")
    return str(mp4_path)

def extract_last_frame(mp4_path, ts, verbose=False):
    jpg_path = OUTDIR / f"dice_{ts}_last.jpg"
    vprint(verbose, "Extracting last frame (~0.1s before end) as JPEG…")
    res = run([
        "ffmpeg", "-y",
        "-sseof", "-0.1",   # a hair before end avoids EOF boundary issues
        "-i", mp4_path,
        "-vframes", "1",
        "-q:v", "2",
        str(jpg_path)
    ], verbose=verbose, timeout=30)
    if res.returncode != 0 or not jpg_path.exists():
        raise RuntimeError(f"Extract last frame failed:\n{res.stderr}")
    vprint(verbose, f"Last frame: {jpg_path}")
    return str(jpg_path)

# ---------------- Dice classifier (pips) ----------------
def classify_die_pips(img_bgr, verbose=False):
    """
    Returns 1..6 if found, else None. Verbose prints stages if requested.
    """
    img = img_bgr.copy()
    h, w = img.shape[:2]
    vprint(verbose, f"Classifier: input image {w}x{h}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(gray, 50, 150)
    edged = cv2.dilate(edged, np.ones((3,3), np.uint8), 1)

    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    face_quad = None
    max_area = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < (h*w)*0.01:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4 and area > max_area and cv2.isContourConvex(approx):
            max_area = area
            face_quad = approx

    if face_quad is not None:
        pts = face_quad.reshape(4,2).astype(np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(-1)
        tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]; bl = pts[np.argmax(diff)]
        dst_size = 300
        dst = np.array([[0,0],[dst_size-1,0],[dst_size-1,dst_size-1],[0,dst_size-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(np.array([tl,tr,br,bl], dtype=np.float32), dst)
        face = cv2.warpPerspective(img, M, (dst_size, dst_size))
        vprint(verbose, "Classifier: warped quadrilateral to square.")
    else:
        face = img
        vprint(verbose, "Classifier: no quad found; using full frame.")

    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_gray = cv2.equalizeHist(face_gray)
    th = cv2.adaptiveThreshold(face_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 7)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h2, w2 = th.shape[:2]
    minA = (h2*w2)*0.001
    maxA = (h2*w2)*0.05
    pip_cnt = 0
    for c in cnts:
        A = cv2.contourArea(c)
        if A < minA or A > maxA:
            continue
        P = cv2.arcLength(c, True)
        if P == 0:
            continue
        circularity = 4*np.pi*A/(P*P)
        if circularity > 0.5:
            pip_cnt += 1

    if 1 <= pip_cnt <= 6:
        vprint(verbose, f"Classifier: pips via contours = {pip_cnt}")
        return pip_cnt

    # Fallback to Hough circles
    circles = cv2.HoughCircles(face_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=h2//10,
                               param1=60, param2=18, minRadius=h2//30, maxRadius=h2//6)
    if circles is not None:
        n = circles.shape[1]
        vprint(verbose, f"Classifier: pips via Hough = {n}")
        if 1 <= n <= 6:
            return int(n)

    vprint(verbose, "Classifier: no valid pip count found.")
    return None

def classify_image(jpg_path, verbose=False):
    vprint(verbose, f"Loading image for classification: {jpg_path}")
    img = cv2.imread(jpg_path)
    if img is None:
        vprint(verbose, "Failed to load image.")
        return None
    return classify_die_pips(img, verbose=verbose)

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

def countdown(verbose=False):
    # Clear a beat so you have time to ready the dice
    vprint(verbose, "\nGet ready…")
    for n in [3, 2, 1]:
        print(f"{n}…", flush=True)
        time.sleep(1)
    print("Roll the dice!", flush=True)  # Recording starts immediately after this returns.

def main():
    # MUST be first so any reference below is to the global vars
    global FPS, RES_W, RES_H, DURATION_MS

    parser = argparse.ArgumentParser(
        description="Record Pi Camera, grab last frame, classify pips, (optionally) post to Discord."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--no-discord", action="store_true", help="Skip Discord upload even if webhook is set.")
    parser.add_argument("--no-classify", action="store_true", help="Skip pip classification.")

    # Use literal defaults here to avoid referencing the globals before declaration
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default 30).")
    parser.add_argument("--width", type=int, default=1920, help="Video width (default 1920).")
    parser.add_argument("--height", type=int, default=1080, help="Video height (default 1080).")
    parser.add_argument("--duration-ms", type=int, default=7000, help="Duration in ms (default 7000).")

    args = parser.parse_args()
    verbose = args.verbose

    # Now it's safe to update the globals with CLI overrides
    FPS, RES_W, RES_H, DURATION_MS = args.fps, args.width, args.height, args.duration_ms

    # Basic deps sanity
    if shutil.which("rpicam-vid") is None:
        print("Error: rpicam-vid not found. Install rpicam-apps.", file=sys.stderr)
        sys.exit(1)
    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg not found. Install ffmpeg.", file=sys.stderr)
        sys.exit(1)

    # Ctrl+C graceful exit
    signal.signal(signal.SIGINT, lambda *a: sys.exit(130))

    ts = nowtag()

    # Countdown then IMMEDIATELY start recording
    countdown(verbose=verbose)

    try:
        mp4_path = record_video(ts, verbose=verbose, prefer_libav=True)
    except Exception as e:
        print(f"Capture error: {e}", file=sys.stderr, flush=True)
        sys.exit(2)

    try:
        jpg_path = extract_last_frame(mp4_path, ts, verbose=verbose)
    except Exception as e:
        print(f"Last-frame extract error: {e}", file=sys.stderr, flush=True)
        sys.exit(3)

    roll = None
    if not args.no_classify:
        roll = classify_image(jpg_path, verbose=verbose)
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

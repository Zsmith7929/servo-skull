#!/usr/bin/env python3
import os, sys, shutil, subprocess, time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests

# ================== CONFIG ==================
DURATION_MS = 7000           # 7 seconds
FPS         = 30             # 30 fps (use 60 if lighting is great)
RES_W, RES_H = 1920, 1080    # 1080p
BITRATE     = 8_000_000      # ~8 Mbps

# Lighting/motion tweaks (uncomment to use)
EXTRA_RPICAM_OPTS = [
    # "--shutter", "4000",       # ~1/250s; lower number = faster shutter (needs more light)
    # "--gain", "4.0",           # allow more sensor gain in darker scenes
    "--autofocus-mode", "auto"
]

OUTDIR = Path.home() / "dice_captures"
OUTDIR.mkdir(parents=True, exist_ok=True)

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()

def run(cmd):
    # print(" ".join(cmd))
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def record_video(ts, prefer_libav=True):
    """Record 7s and produce a finalized MP4 (dice_<ts>.mp4). Falls back to h264+ffmpeg mux."""
    mp4_path  = OUTDIR / f"dice_{ts}.mp4"
    h264_path = OUTDIR / f"dice_{ts}.h264"

    if prefer_libav:
        cmd = [
            "rpicam-vid",
            "-t", str(DURATION_MS),
            "--framerate", str(FPS),
            "--width", str(RES_W), "--height", str(RES_H),
            "--bitrate", str(BITRATE),
            *EXTRA_RPICAM_OPTS,
            "--codec", "libav", "--libav-format", "mp4",
            "-o", str(mp4_path),
        ]
        res = run(cmd)
        if res.returncode == 0 and mp4_path.exists() and mp4_path.stat().st_size > 0:
            return str(mp4_path)
        # Fall through to h264 path if libav unsupported

    # Fallback: raw h264 -> mp4 via ffmpeg (copy, no re-encode)
    res = run([
        "rpicam-vid",
        "-t", str(DURATION_MS),
        "--framerate", str(FPS),
        "--width", str(RES_W), "--height", str(RES_H),
        "--bitrate", str(BITRATE),
        *EXTRA_RPICAM_OPTS,
        "-o", str(h264_path),
    ])
    if res.returncode != 0 or not h264_path.exists():
        raise RuntimeError(f"Recording failed:\n{res.stderr}")

    mux = run(["ffmpeg", "-y",
               "-framerate", str(FPS),
               "-i", str(h264_path),
               "-c", "copy", str(mp4_path)])
    if mux.returncode != 0:
        raise RuntimeError(f"MP4 mux failed:\n{mux.stderr}")

    try:
        h264_path.unlink()
    except Exception:
        pass
    return str(mp4_path)

def extract_last_frame(mp4_path, ts):
    """Grab a single JPEG from ~0.1s before the end (robust if last frame is at boundary)."""
    jpg_path = OUTDIR / f"dice_{ts}_last.jpg"
    res = run(["ffmpeg", "-y", "-sseof", "-0.1", "-i", mp4_path, "-vframes", "1", "-q:v", "2", str(jpg_path)])
    if res.returncode != 0 or not jpg_path.exists():
        raise RuntimeError(f"Extract last frame failed:\n{res.stderr}")
    return str(jpg_path)

# ---------------- Dice classifier (pips) ----------------
def classify_die_pips(img_bgr):
    """
    Returns an integer 1..6 if a face is found and pips are counted, else None.
    Pipeline:
      1) find largest quadrilateral (die face), warp to square
      2) adaptive threshold -> binary
      3) find small, circular blobs (pips) -> count
    """
    img = img_bgr.copy()
    h, w = img.shape[:2]

    # 1) find face (largest quad)
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
        # Order the 4 points and warp to square
        pts = face_quad.reshape(4,2).astype(np.float32)
        # order by sum and diff to get tl, tr, br, bl
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(-1)
        tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]; bl = pts[np.argmax(diff)]
        dst_size = 300
        dst = np.array([[0,0],[dst_size-1,0],[dst_size-1,dst_size-1],[0,dst_size-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(np.array([tl,tr,br,bl], dtype=np.float32), dst)
        face = cv2.warpPerspective(img, M, (dst_size, dst_size))
    else:
        # fallback: use full frame
        face = img

    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    # normalize contrast (helps with colored dice / uneven light)
    face_gray = cv2.equalizeHist(face_gray)
    # adaptive threshold to get dark pips on lighter face (invert so pips = 1s)
    th = cv2.adaptiveThreshold(face_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 7)
    # remove small noise
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

    # find candidate blobs
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pip_cnt = 0
    h2, w2 = th.shape[:2]
    minA = (h2*w2)*0.001    # tweak if needed
    maxA = (h2*w2)*0.05
    for c in cnts:
        A = cv2.contourArea(c)
        if A < minA or A > maxA:
            continue
        P = cv2.arcLength(c, True)
        if P == 0: 
            continue
        circularity = 4*np.pi*A/(P*P)       # 1.0 = perfect circle
        if circularity > 0.5:               # permissive threshold
            pip_cnt += 1

    if 1 <= pip_cnt <= 6:
        return pip_cnt
    # fallback: try Hough circles if nothing found (works on very round/high-contrast pips)
    circles = cv2.HoughCircles(face_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=h2//10,
                               param1=60, param2=18, minRadius=h2//30, maxRadius=h2//6)
    if circles is not None:
        n = circles.shape[1]
        if 1 <= n <= 6:
            return int(n)
    return None

def classify_image(jpg_path):
    img = cv2.imread(jpg_path)
    if img is None:
        return None
    return classify_die_pips(img)

# ---------------- Discord upload ----------------
def post_to_discord(webhook_url, mp4_path, jpg_path, roll_result):
    if not webhook_url:
        print("No DISCORD_WEBHOOK_URL configured.", file=sys.stderr)
        return False

    content = f"🎲 **Roll result:** {roll_result if roll_result is not None else 'Unknown'}"
    files = []
    try:
        files.append(("files[0]", (os.path.basename(mp4_path), open(mp4_path, "rb"), "video/mp4")))
    except Exception as e:
        print(f"Could not attach video: {e}", file=sys.stderr)
    try:
        files.append(("files[1]", (os.path.basename(jpg_path), open(jpg_path, "rb"), "image/jpeg")))
    except Exception as e:
        print(f"Could not attach image: {e}", file=sys.stderr)

    data = {"content": content}
    r = requests.post(webhook_url, data=data, files=files, timeout=30)
    ok = 200 <= r.status_code < 300
    if not ok:
        print(f"Discord upload failed ({r.status_code}): {r.text}", file=sys.stderr)
    return ok

def main():
    if shutil.which("rpicam-vid") is None:
        print("rpicam-vid not found. Install rpicam-apps.", file=sys.stderr)
        sys.exit(1)
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found. Install ffmpeg.", file=sys.stderr)
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        mp4_path = record_video(ts, prefer_libav=True)
        jpg_path = extract_last_frame(mp4_path, ts)
    except Exception as e:
        print(f"Capture error: {e}", file=sys.stderr)
        sys.exit(2)

    roll = classify_image(jpg_path)
    print(f"Detected roll: {roll}")

    if DISCORD_WEBHOOK_URL:
        ok = post_to_discord(DISCORD_WEBHOOK_URL, mp4_path, jpg_path, roll)
        print("Posted to Discord." if ok else "Failed to post to Discord.")
    else:
        print("Set DISCORD_WEBHOOK_URL env var to auto-post to Discord.")

    print("Video:", mp4_path)
    print("Last frame:", jpg_path)

if __name__ == "__main__":
    main()

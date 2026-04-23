#!/usr/bin/env python3
"""
Interactive ball-detection parameter viewer.

Usage:
    cd TENNIS/claude
    python interactive_viewer.py

Opens a browser tab with:
  - Video selector
  - Frame scrubber + keyboard arrows (Shift+Arrow = x10)
  - Sliders: frame gap, diff thresh, min/max area, max aspect, score thresh
  - Toggle: compactness vs radius-of-gyration scoring
  - Resolution selector: 320 / 480 / 640 / 960 px wide
  - Solo mode: only the most-recently-touched slider's filter is active
  - Three panels: original | frame-diff (hot) | passing blobs only
  - Tab 2 "Spatial Prior": two-zone probability mask with live heatmap
"""

import json, base64, urllib.parse, threading, webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import cv2
import numpy as np

# ── find TENNIS root ──────────────────────────────────────────────────────────
def find_tennis_root():
    for p in [Path(__file__).resolve().parent,
              Path(__file__).resolve().parent.parent,
              Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]:
        if (p / "videosandaudio").is_dir():
            return p
    raise RuntimeError("Cannot find TENNIS root — expected a 'videosandaudio/' folder")

TENNIS    = find_tennis_root()
VIDEO_DIR = TENNIS / "videosandaudio"

def list_videos():
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    return sorted(p.name for p in VIDEO_DIR.iterdir()
                  if p.suffix.lower() in exts)

# ── per-video capture cache ───────────────────────────────────────────────────
_caps = {}

def get_cap(vname):
    path = str(VIDEO_DIR / vname)
    if path not in _caps:
        _caps[path] = cv2.VideoCapture(path)
    return _caps[path]

# ── reference resolution — court geometry is defined at these coords ──────────
REF_W, REF_H = 320, 180

# ── court geometry (at REF_W x REF_H) ────────────────────────────────────────
COURT_POLY        = np.array([[116,50],[193,52],[293,118],[16,118]], np.int32)
COURT_Y_TOP       = 50
COURT_Y_BOT       = 118
COURT_X_LEFT_TOP  = 116
COURT_X_LEFT_BOT  = 16
COURT_X_RIGHT_TOP = 193
COURT_X_RIGHT_BOT = 293

# ── prior cache ───────────────────────────────────────────────────────────────
_prior_cache: dict = {}

def compute_prior_map(
        court_x_sigma  = 15.0,
        court_y_sigma  = 25.0,
        air_x_left     = 30,
        air_x_right    = 290,
        air_y_top      = 5,
        air_y_bot      = 50,
        air_sigma_x    = 80.0,
        air_sigma_y    = 45.0,
        weight         = 0.8,
    ):
    """
    Returns REF_H x REF_W float32 prior probability map.

    COURT ZONE: trapezoid, Gaussian falloff outside horizontally and below.
    AIR ZONE:   2-D Gaussian anchored at bottom-centre of a top bar.
    Combined:   raw = max(court_zone, air_zone)
                result = (1-weight) + weight * raw
    """
    key = (round(court_x_sigma,2), round(court_y_sigma,2),
           air_x_left, air_x_right, air_y_top, air_y_bot,
           round(air_sigma_x,2), round(air_sigma_y,2), round(weight,3))
    if key in _prior_cache:
        return _prior_cache[key]

    Y, X = np.mgrid[0:REF_H, 0:REF_W].astype(np.float32)

    # Court zone
    t = np.clip((Y - COURT_Y_TOP) / (COURT_Y_BOT - COURT_Y_TOP), 0.0, 1.0)
    xl_trap = COURT_X_LEFT_TOP  + (COURT_X_LEFT_BOT  - COURT_X_LEFT_TOP)  * t
    xr_trap = COURT_X_RIGHT_TOP + (COURT_X_RIGHT_BOT - COURT_X_RIGHT_TOP) * t
    xl = np.where(Y < COURT_Y_TOP, float(COURT_X_LEFT_BOT),  xl_trap)
    xr = np.where(Y < COURT_Y_TOP, float(COURT_X_RIGHT_BOT), xr_trap)
    dx_c = np.maximum(0.0, xl - X) + np.maximum(0.0, X - xr)
    px_c = np.exp(-0.5 * (dx_c / max(court_x_sigma, 1e-3))**2)
    dy_c = np.maximum(0.0, Y - COURT_Y_BOT)
    py_c = np.exp(-0.5 * (dy_c / max(court_y_sigma, 1e-3))**2)
    # Mask to Y >= COURT_Y_TOP so the court zone never bleeds into the air zone
    court_zone = (px_c * py_c * (Y >= COURT_Y_TOP)).astype(np.float32)

    # Air zone
    cx_air = (air_x_left + air_x_right) / 2.0
    dx_a   = X - cx_air
    dy_a   = float(air_y_bot) - Y
    air_gauss = np.exp(
        -0.5 * ((dx_a / max(air_sigma_x, 1e-3))**2 +
                (dy_a / max(air_sigma_y, 1e-3))**2)
    ).astype(np.float32)
    air_mask = ((X >= air_x_left) & (X <= air_x_right) &
                (Y >= air_y_top)  & (Y <= air_y_bot)).astype(np.float32)
    air_zone = (air_gauss * air_mask).astype(np.float32)

    raw    = np.maximum(court_zone, air_zone)
    result = ((1.0 - weight) + weight * raw).astype(np.float32)

    _prior_cache.clear()
    _prior_cache[key] = result
    return result


def make_prior_image(
        court_x_sigma, court_y_sigma,
        air_x_left, air_x_right, air_y_top, air_y_bot,
        air_sigma_x, air_sigma_y,
        weight,
        vname=None, frame_idx=0, frame_blend=0.35):
    """Turbo heatmap of the prior at 3x REF resolution, annotated."""
    prior = compute_prior_map(
        court_x_sigma, court_y_sigma,
        air_x_left, air_x_right, air_y_top, air_y_bot,
        air_sigma_x, air_sigma_y, weight)

    lo, hi = prior.min(), prior.max()
    vis  = ((prior - lo) / (hi - lo + 1e-6) * 255).clip(0, 255).astype(np.uint8)
    heat = cv2.cvtColor(cv2.applyColorMap(vis, cv2.COLORMAP_TURBO),
                        cv2.COLOR_BGR2RGB)

    if vname and frame_blend > 0:
        _, bgr = read_gray_small(vname, frame_idx, REF_W)
        if bgr is not None:
            frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            heat = cv2.addWeighted(frame_rgb, frame_blend,
                                   heat, 1.0 - frame_blend, 0)

    cv2.polylines(heat, [COURT_POLY], True, (230, 230, 230), 1)

    def dash_rect(img, x0, y0, x1, y1, color, step=6):
        for x in range(x0, x1, step):
            if (x//step)%2 == 0:
                cv2.line(img,(x,y0),(min(x+step-1,x1),y0),color,1)
                cv2.line(img,(x,y1),(min(x+step-1,x1),y1),color,1)
        for y in range(y0, y1, step):
            if (y//step)%2 == 0:
                cv2.line(img,(x0,y),(x0,min(y+step-1,y1)),color,1)
                cv2.line(img,(x1,y),(x1,min(y+step-1,y1)),color,1)
    dash_rect(heat, air_x_left, air_y_top, air_x_right, air_y_bot, (255,255,255))

    cx_air = int((air_x_left + air_x_right) / 2)
    cv2.line(heat, (air_x_left, air_y_bot), (air_x_right, air_y_bot), (0,220,220), 1)
    cv2.putText(heat, "anchor (p=max)", (air_x_left+2, air_y_bot-2),
                cv2.FONT_HERSHEY_PLAIN, 0.65, (0,220,220), 1)

    sx = int(min(air_sigma_x, (air_x_right-air_x_left)/2))
    cv2.arrowedLine(heat, (cx_air, air_y_bot), (cx_air+sx, air_y_bot),
                    (0,200,255), 1, tipLength=0.15)
    cv2.putText(heat, f"sx={int(air_sigma_x)}", (cx_air+sx+2, air_y_bot-1),
                cv2.FONT_HERSHEY_PLAIN, 0.6, (0,200,255), 1)

    sy = int(min(air_sigma_y, air_y_bot - air_y_top))
    cv2.arrowedLine(heat, (cx_air, air_y_bot), (cx_air, max(0, air_y_bot-sy)),
                    (0,200,255), 1, tipLength=0.2)
    cv2.putText(heat, f"sy={int(air_sigma_y)}", (cx_air+2, max(6, air_y_bot-sy-1)),
                cv2.FONT_HERSHEY_PLAIN, 0.6, (0,200,255), 1)

    out = cv2.resize(heat, (REF_W * 3, REF_H * 3), interpolation=cv2.INTER_LINEAR)
    return out


# ── frame reading ─────────────────────────────────────────────────────────────
def read_gray_small(vname, idx, proc_w=REF_W):
    """proc_w=0 means native resolution (no resize)."""
    cap = get_cap(vname)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
    ok, frame = cap.read()
    if not ok:
        return None, None
    if proc_w == 0:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), frame
    proc_h = proc_w * 9 // 16
    small = cv2.resize(frame, (proc_w, proc_h))
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), small


# ── detection ─────────────────────────────────────────────────────────────────
def detect(g_curr, g_prev, thresh, min_a, max_a, max_asp, method, score_thresh,
           prior_map=None, proc_w=REF_W, proc_h=REF_H,
           ball_diam=10.0, min_circ=0.2, min_bright=0.0, blur_k=0):
    """
    Returns (diff, bw, labels, passing_blobs, rejected_blobs).
    prior_map: REF_H x REF_W float32; blob centroid scaled to REF coords for lookup.
    method: 'compactness' | 'rog' | 'circularity'
      circularity: score = 0.6*brightness + 0.2*circularity + 0.2*size_score
                   ball_diam sets the reference area; min_circ is a hard filter.
    """
    diff = cv2.absdiff(g_curr, g_prev)
    # Optional pre-blur (smooths blob edges; improves circularity accuracy)
    src = diff
    if blur_k > 0:
        k = int(blur_k) * 2 + 1   # ensure odd
        src = cv2.GaussianBlur(diff, (k, k), 0)
    _, bw = cv2.threshold(src, int(thresh), 255, cv2.THRESH_BINARY)
    # Scale morph kernel with resolution
    mk = max(3, int(3 * proc_w / REF_W))
    if mk % 2 == 0:
        mk += 1
    bw = cv2.morphologyEx(
        bw, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk)))

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bw)

    # Size-affinity target scales with resolution
    size_target = 15.0 * (proc_w / REF_W) ** 2
    # Circularity method: target area from ball_diam reference (also resolution-scaled)
    ball_diam_scaled = ball_diam * (proc_w / REF_W)
    circ_target_area = np.pi * (ball_diam_scaled / 2.0) ** 2

    passing, rejected = [], []
    for i in range(1, n):
        a   = stats[i, cv2.CC_STAT_AREA]
        bwi = stats[i, cv2.CC_STAT_WIDTH]
        bhi = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]
        asp = max(bwi, bhi) / (min(bwi, bhi) + 1e-3)

        circularity = None   # computed lazily for 'circularity' method

        if method == "compactness":
            comp  = a / (bwi * bhi + 1e-3)
            sscr  = max(1.0 - abs(a - size_target) / (size_target * 2.0), 0.1)
            score = float(comp * sscr)

        elif method == "rog":
            ys, xs = np.where(labels == i)
            rog    = float(np.sqrt(np.mean((xs-cx)**2 + (ys-cy)**2))) if len(xs) > 1 else 1.0
            score  = float(a / (np.pi * rog**2 + 1e-3) / 2.0)

        else:  # circularity + brightness
            ys, xs = np.where(labels == i)
            # Circularity via contour perimeter
            mask = np.zeros(bw.shape, np.uint8)
            mask[ys, xs] = 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                perim = cv2.arcLength(max(contours, key=cv2.contourArea), True)
                circularity = float(4.0 * np.pi * a / (perim * perim)) if perim > 0 else 0.0
            else:
                circularity = 0.0
            # Hard circularity gate
            if circularity < min_circ:
                rejected.append(dict(
                    x=float(cx), y=float(cy),
                    x0=int(stats[i, cv2.CC_STAT_LEFT]),
                    y0=int(stats[i, cv2.CC_STAT_TOP]),
                    w=int(bwi), h=int(bhi),
                    area=int(a), asp=float(asp), score=0.0,
                    prior=1.0, circ=round(circularity, 3), _label=i))
                continue
            mean_bright = float(diff[ys, xs].mean()) if len(ys) > 0 else 0.0
            if mean_bright < min_bright:
                rejected.append(dict(
                    x=float(cx), y=float(cy),
                    x0=int(stats[i, cv2.CC_STAT_LEFT]),
                    y0=int(stats[i, cv2.CC_STAT_TOP]),
                    w=int(bwi), h=int(bhi),
                    area=int(a), asp=float(asp), score=0.0,
                    prior=1.0, circ=round(circularity, 3), _label=i))
                continue
            brightness  = mean_bright / 255.0
            size_score  = max(0.0, 1.0 - abs(a - circ_target_area) / max(circ_target_area, 1e-6))
            score       = 0.6 * brightness + 0.2 * circularity + 0.2 * size_score

        prior_val = 1.0
        if prior_map is not None:
            iy = max(0, min(REF_H - 1, int(round(cy * REF_H / proc_h))))
            ix = max(0, min(REF_W - 1, int(round(cx * REF_W / proc_w))))
            prior_val = float(prior_map[iy, ix])
            score *= prior_val

        blob = dict(
            x=float(cx), y=float(cy),
            x0=int(stats[i, cv2.CC_STAT_LEFT]),
            y0=int(stats[i, cv2.CC_STAT_TOP]),
            w=int(bwi), h=int(bhi),
            area=int(a), asp=float(asp), score=score,
            prior=round(prior_val, 3),
            circ=round(circularity, 3) if circularity is not None else None,
            _label=i)

        if min_a <= a <= max_a and asp <= max_asp and score >= score_thresh:
            passing.append(blob)
        else:
            rejected.append(blob)

    passing.sort(key=lambda c: c["score"], reverse=True)
    return diff, bw, labels, passing, rejected


# ── composite image builder ────────────────────────────────────────────────────
DIFF_AMP = 6.0

def get_display_scale(proc_w):
    if proc_w <= 320: return 3
    if proc_w <= 480: return 2
    return 1


def make_composite(vname, frame_idx, gap, thresh, min_a, max_a,
                   max_asp, method, score_thresh,
                   use_prior=False,
                   court_x_sigma=15.0, court_y_sigma=25.0,
                   air_x_left=30, air_x_right=290,
                   air_y_top=5,  air_y_bot=50,
                   air_sigma_x=80.0, air_sigma_y=45.0,
                   pweight=0.8,
                   proc_w=REF_W,
                   ball_diam=10.0, min_circ=0.2, min_bright=0.0, blur_k=0):
    """Returns (composite_rgb_image, passing_blobs_list).
    proc_w=0 means native video resolution."""
    # Resolve native resolution before anything else
    if proc_w == 0:
        cap = get_cap(vname)
        proc_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        proc_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        proc_h = proc_w * 9 // 16
    disp_scale = get_display_scale(proc_w)

    g_curr, bgr_curr = read_gray_small(vname, frame_idx, proc_w)
    g_prev, _        = read_gray_small(vname, max(0, frame_idx - gap), proc_w)

    if g_curr is None or g_prev is None:
        blank = np.zeros((proc_h * disp_scale, proc_w * disp_scale * 3, 3), np.uint8)
        return blank, []

    prior_map = compute_prior_map(
        court_x_sigma, court_y_sigma,
        air_x_left, air_x_right, air_y_top, air_y_bot,
        air_sigma_x, air_sigma_y, pweight) if use_prior else None

    diff, bw, labels, passing, rejected = detect(
        g_curr, g_prev, thresh, min_a, max_a, max_asp, method, score_thresh,
        prior_map=prior_map, proc_w=proc_w, proc_h=proc_h,
        ball_diam=ball_diam, min_circ=min_circ, min_bright=min_bright, blur_k=blur_k)

    # Drawing parameters scale with resolution
    fs = max(0.5, 0.65 * proc_w / REF_W)
    lw = max(1, int(proc_w / REF_W))

    # Panel 1: original (BGR->RGB)
    p1 = cv2.cvtColor(bgr_curr, cv2.COLOR_BGR2RGB).copy()

    # Panel 2: amplified diff -> hot colourmap
    diff_amp = np.clip(diff.astype(np.float32) * DIFF_AMP, 0, 255).astype(np.uint8)
    p2 = cv2.cvtColor(cv2.applyColorMap(diff_amp, cv2.COLORMAP_HOT),
                      cv2.COLOR_BGR2RGB)

    # Panel 3: black canvas, passing blobs only
    p3 = np.zeros((proc_h, proc_w, 3), dtype=np.uint8)

    if passing:
        for rank, b in enumerate(passing):
            t     = 1.0 - rank / len(passing)
            color = (int((1-t)*255), int(t*255), 0)
            p3[labels == b["_label"]] = (200, 200, 200)
            cv2.rectangle(p3, (b["x0"], b["y0"]),
                          (b["x0"]+b["w"], b["y0"]+b["h"]), color, lw)
            cv2.circle(p3, (int(b["x"]), int(b["y"])), max(2, lw+1), color, -1)
            cv2.putText(p3, f"{b['score']:.2f}",
                        (b["x0"], max(b["y0"]-2, 6)),
                        cv2.FONT_HERSHEY_PLAIN, fs, color, lw)

    # Overlay passing blobs on panel 1
    for b in passing:
        cv2.rectangle(p1, (b["x0"], b["y0"]),
                      (b["x0"]+b["w"], b["y0"]+b["h"]), (0, 255, 0), lw)
        cv2.circle(p1, (int(b["x"]), int(b["y"])), max(2, lw+1), (0, 255, 0), -1)

    # Best-blob crosshair on all three panels
    if passing:
        best = passing[0]
        bx, by = int(best["x"]), int(best["y"])
        R      = max(best["w"], best["h"]) // 2 + 4
        YELLOW = (255, 255, 0)
        for panel in (p1, p2, p3):
            cv2.circle(panel, (bx, by), R, YELLOW, lw)
            cv2.line(panel, (bx - R - 3, by), (bx + R + 3, by), YELLOW, lw)
            cv2.line(panel, (bx, by - R - 3), (bx, by + R + 3), YELLOW, lw)

    def up(img):
        return cv2.resize(img, (proc_w * disp_scale, proc_h * disp_scale),
                          interpolation=cv2.INTER_NEAREST)

    composite = np.hstack([up(p1), up(p2), up(p3)])
    return composite, passing


def img_to_b64jpeg(img_rgb):
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode()


# ── trajectory tree ────────────────────────────────────────────────────────────

def collect_tree_blobs(vname, center_frame, gap, n_steps,
                       thresh=10, min_a=1, max_a=400, max_asp=6.0,
                       method="compactness", proc_w=REF_W,
                       ball_diam=10.0, min_circ=0.1, min_bright=0.0, blur_k=0):
    """Collect blobs from n_steps non-overlapping diff pairs around center_frame.

    Step k: diff( center_frame - (2k+1)*gap,  center_frame - 2k*gap )
      k=0 (newest): diff(center_frame-gap, center_frame)
      k=1         : diff(center_frame-3*gap, center_frame-2*gap)
    Returns list of step-dicts (may be fewer if frames run out):
      {'step': k, 'frame_a': int, 'frame_b': int, 'blobs': list}
    Each blob: x, y, x0, y0, w, h, area, score, passing (bool)
    """
    if proc_w == 0:
        cap = get_cap(vname)
        proc_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        proc_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        proc_h = proc_w * 9 // 16

    steps = []
    for k in range(n_steps):
        fb = center_frame - 2 * k * gap   # newer frame of pair
        fa = fb - gap                      # older frame of pair
        if fa < 0:
            break
        g_b, _ = read_gray_small(vname, fb, proc_w)
        g_a, _ = read_gray_small(vname, fa, proc_w)
        if g_a is None or g_b is None:
            break
        _, _, _, passing, rejected = detect(
            g_b, g_a, thresh, min_a, max_a, max_asp, method, 0.0,
            prior_map=None, proc_w=proc_w, proc_h=proc_h,
            ball_diam=ball_diam, min_circ=min_circ,
            min_bright=min_bright, blur_k=blur_k)
        all_blobs = []
        for b in passing:
            b2 = dict(b); b2['passing'] = True; all_blobs.append(b2)
        for b in rejected:
            b2 = dict(b); b2['passing'] = False; all_blobs.append(b2)
        steps.append({'step': k, 'frame_a': fa, 'frame_b': fb, 'blobs': all_blobs})
    return steps


def build_path_dp(steps, link_radius=30, static_radius=8, vel_penalty=0.05):
    """DP path finding over blob tree.

    steps: from collect_tree_blobs (step 0 = newest).
    Reversed internally to chronological order (oldest first).

    Returns:
      chron_steps  - steps oldest-first
      static_flags - list of sets of blob indices that are static, per chron step
      edges        - list of (t, i, t_prev, j) valid connection edges
      best_path    - list of (chron_step_idx, blob_idx) oldest→newest, or []
    """
    if not steps:
        return [], [], [], []

    chron_steps = list(reversed(steps))
    n_t = len(chron_steps)

    # Static: blob i at chron step t has a near-neighbour at the next chron step.
    static_flags = [set() for _ in range(n_t)]
    for t in range(n_t - 1):
        r2 = static_radius * static_radius
        for i, bi in enumerate(chron_steps[t]['blobs']):
            for bj in chron_steps[t + 1]['blobs']:
                dx = bi['x'] - bj['x']
                dy = bi['y'] - bj['y']
                if dx*dx + dy*dy <= r2:
                    static_flags[t].add(i)
                    break

    # Valid edges between consecutive chron steps (within link_radius)
    r2_link = link_radius * link_radius
    edges_by_t = [[] for _ in range(n_t)]
    for t in range(1, n_t):
        for i, bi in enumerate(chron_steps[t]['blobs']):
            if i in static_flags[t]:
                continue
            for j, bj in enumerate(chron_steps[t - 1]['blobs']):
                if j in static_flags[t - 1]:
                    continue
                dx = bi['x'] - bj['x']
                dy = bi['y'] - bj['y']
                if dx*dx + dy*dy <= r2_link:
                    edges_by_t[t].append((i, j))

    edges = [(t, i, t - 1, j) for t in range(1, n_t) for (i, j) in edges_by_t[t]]

    # DP: dp[t][i] = [total_score, pred_j, vx_in, vy_in]
    INF = float('-inf')
    dp = [[None] * len(chron_steps[t]['blobs']) for t in range(n_t)]
    for i, b in enumerate(chron_steps[0]['blobs']):
        if i not in static_flags[0]:
            dp[0][i] = [b['score'], -1, 0.0, 0.0]

    for t in range(1, n_t):
        for i, bi in enumerate(chron_steps[t]['blobs']):
            if i in static_flags[t]:
                continue
            best_val, best_entry = INF, None
            for (ei, ej) in edges_by_t[t]:
                if ei != i or dp[t - 1][ej] is None:
                    continue
                prev_score, _, vx_in, vy_in = dp[t - 1][ej]
                bj = chron_steps[t - 1]['blobs'][ej]
                vx_out = bi['x'] - bj['x']
                vy_out = bi['y'] - bj['y']
                dvx, dvy = vx_out - vx_in, vy_out - vy_in
                accel = (dvx*dvx + dvy*dvy) ** 0.5
                total = prev_score + bi['score'] - vel_penalty * accel
                if total > best_val:
                    best_val = total
                    best_entry = [total, ej, vx_out, vy_out]
            if best_entry is not None:
                dp[t][i] = best_entry

    # Trace best path from newest chron step
    best_path = []
    last_t = n_t - 1
    best_i, best_score = -1, INF
    for i, entry in enumerate(dp[last_t]):
        if entry is not None and entry[0] > best_score:
            best_score, best_i = entry[0], i
    if best_i >= 0:
        t, i, path_rev = last_t, best_i, []
        while t >= 0 and i >= 0:
            path_rev.append((t, i))
            if dp[t][i] is None:
                break
            j = dp[t][i][1]
            t -= 1; i = j
        best_path = list(reversed(path_rev))

    return chron_steps, static_flags, edges, best_path


def make_tree_image(vname, center_frame, proc_w,
                    chron_steps, static_flags, edges, best_path):
    """Render trajectory tree overlay on current frame.  Returns RGB image."""
    if proc_w == 0:
        cap = get_cap(vname)
        proc_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        proc_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        proc_h = proc_w * 9 // 16

    g_curr, bgr_curr = read_gray_small(vname, center_frame, proc_w)
    if bgr_curr is None:
        canvas = np.zeros((proc_h, proc_w, 3), np.uint8)
    else:
        rgb = cv2.cvtColor(bgr_curr, cv2.COLOR_BGR2RGB)
        canvas = cv2.addWeighted(rgb, 0.4, np.zeros_like(rgb), 0.6, 0)

    disp_scale = get_display_scale(proc_w)
    h, w = canvas.shape[:2]
    canvas = cv2.resize(canvas, (w * disp_scale, h * disp_scale),
                        interpolation=cv2.INTER_NEAREST)

    n_t  = len(chron_steps)
    lw   = max(1, disp_scale)
    path_set = set(best_path)

    def bxy(t, i):
        b = chron_steps[t]['blobs'][i]
        return (int(b['x'] * disp_scale), int(b['y'] * disp_scale))

    # Grey lines: all valid edges
    for (t, i, t_prev, j) in edges:
        cv2.line(canvas, bxy(t_prev, j), bxy(t, i), (80, 80, 80), lw)

    # Thick yellow lines: best path
    for k in range(len(best_path) - 1):
        t0, i0 = best_path[k]
        t1, i1 = best_path[k + 1]
        cv2.line(canvas, bxy(t0, i0), bxy(t1, i1), (255, 230, 0), lw * 2 + 1)

    # Blob circles: age-tinted (dim blue=oldest, bright yellow=newest)
    for t, step in enumerate(chron_steps):
        age = t / max(n_t - 1, 1)
        col = (int(50 + age * 200), int(80 + age * 150), int(200 - age * 190))
        for i, blob in enumerate(step['blobs']):
            x, y = bxy(t, i)
            r_px = max(3, int(blob.get('w', 6) * disp_scale // 2))
            if i in static_flags[t]:
                s = max(5, r_px)
                cv2.line(canvas, (x-s, y-s), (x+s, y+s), (210, 50, 50), lw*2)
                cv2.line(canvas, (x-s, y+s), (x+s, y-s), (210, 50, 50), lw*2)
            elif (t, i) in path_set:
                cv2.circle(canvas, (x, y), r_px + 2, (255, 230, 0), lw * 2)
            else:
                cv2.circle(canvas, (x, y), r_px, col, lw)

    return canvas


# ── embedded HTML ─────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Ball Detection Viewer</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #111; color: #ddd; font-family: monospace; font-size: 13px;
       display: flex; flex-direction: column; height: 100vh; overflow: hidden; }

/* tab bar */
#tabbar { background: #141414; border-bottom: 2px solid #333;
          display: flex; align-items: stretch; flex-shrink: 0; }
.tab { padding: 5px 22px; cursor: pointer; color: #666; border-bottom: 2px solid transparent;
       margin-bottom: -2px; font-size: 12px; letter-spacing: .04em; user-select: none; }
.tab:hover { color: #aaa; }
.tab.active { color: #5af; border-bottom-color: #5af; }
#prior-badge { font-size: 10px; padding: 1px 5px; border-radius: 8px; margin-left: 6px;
               background: #333; color: #666; }
#prior-badge.on { background: #1a4a1a; color: #4f4; }

/* top control bar */
#bar { background: #1c1c1c; border-bottom: 1px solid #333;
       padding: 6px 10px; display: flex; gap: 10px; flex-wrap: wrap;
       align-items: flex-end; flex-shrink: 0; }

.group { display: flex; flex-direction: column; gap: 3px; }
.group-title { font-size: 10px; color: #888; letter-spacing: .05em;
               text-transform: uppercase; }
.row { display: flex; gap: 5px; align-items: center; }

input[type=range] { -webkit-appearance: none; height: 4px; border-radius: 2px;
  background: #444; outline: none; cursor: pointer; }
input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none; width: 12px; height: 12px; border-radius: 50%;
  background: #5af; cursor: pointer; }
input[type=range].solo-active { background: #553300; }
input[type=range].solo-active::-webkit-slider-thumb { background: #fa0; }
select { background: #2a2a2a; color: #ddd; border: 1px solid #444;
         border-radius: 3px; padding: 2px 6px; }
.val { min-width: 34px; text-align: right; color: #adf; }

/* frame nav */
.nav-btn { background: #2a2a2a; color: #ddd; border: 1px solid #444;
           border-radius: 3px; padding: 1px 8px; cursor: pointer; font-size: 14px; }
.nav-btn:hover { background: #3a3a3a; }
#frame-sl { width: 200px; }

/* play button */
#play-btn { background: #1a3a1a; color: #4f4; border: 1px solid #3a6a3a;
            border-radius: 3px; padding: 3px 14px; cursor: pointer;
            font-size: 16px; line-height: 1; min-width: 42px; text-align: center; }
#play-btn:hover { background: #2a4a2a; }
#play-btn.playing { background: #3a1a1a; color: #f66; border-color: #6a3a3a; }

/* method/toggle buttons */
.tog-btn { background: #2a2a2a; color: #999; border: 1px solid #444;
           border-radius: 3px; padding: 3px 10px; cursor: pointer; }
.tog-btn:hover { background: #333; }
.tog-btn.on { background: #0a5; color: #fff; border-color: #0c7; }
#btn-solo.on { background: #553300; color: #fa0; border-color: #885500; }

/* main panels */
#main { flex: 1; overflow: hidden; display: flex; flex-direction: column; }
#page-detect { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
#panels { flex: 1; overflow: hidden; display: flex; align-items: center;
          justify-content: center; background: #0a0a0a; }
#composite { max-width: 100%; max-height: 100%; image-rendering: pixelated; }
#panel-labels { background: #161616; padding: 2px 10px; font-size: 10px;
                color: #666; display: flex; flex-shrink: 0; }
#panel-labels span { flex: 1; text-align: center; }

/* prior page */
#page-prior { flex: 1; display: none; flex-direction: column; overflow: hidden; }
#prior-panels { flex: 1; overflow: hidden; display: flex; align-items: center;
                justify-content: center; background: #0a0a0a; }
#prior-img { max-width: 100%; max-height: 100%; image-rendering: pixelated; }
#prior-legend { background: #161616; padding: 3px 12px; font-size: 10px; color: #888;
                flex-shrink: 0; }

/* tree page */
#page-tree { flex: 1; display: none; flex-direction: column; overflow: hidden; }
#tree-panels { flex: 1; overflow: hidden; display: flex; align-items: center;
               justify-content: center; background: #0a0a0a; }
#tree-img { max-width: 100%; max-height: 100%; image-rendering: pixelated; }
#tree-legend { background: #161616; padding: 3px 12px; font-size: 10px; color: #888;
               flex-shrink: 0; }
#tree-badge { font-size: 10px; padding: 1px 5px; border-radius: 8px; margin-left: 6px;
              background: #333; color: #666; }
#tree-badge.on { background: #1a3a4a; color: #5cf; }

/* info bar */
#info { background: #161616; border-top: 1px solid #333;
        padding: 4px 10px; font-size: 11px; color: #aaa; flex-shrink: 0;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
</style>
</head>
<body>

<!-- TAB BAR -->
<div id="tabbar">
  <div class="tab active" id="tab-detect" onclick="switchTab('detect')">Detection</div>
  <div class="tab"        id="tab-prior"  onclick="switchTab('prior')">
    Spatial Prior <span id="prior-badge">OFF</span>
  </div>
  <div class="tab"        id="tab-tree"   onclick="switchTab('tree')">
    Trajectory Tree <span id="tree-badge">OFF</span>
  </div>
</div>

<!-- TOP CONTROL BAR -->
<div id="bar">

  <!-- video selector -->
  <div class="group">
    <div class="group-title">Video</div>
    <select id="vid" onchange="videoChanged()"></select>
  </div>

  <!-- DETECTION TAB controls -->
  <div id="det-controls" style="display:contents">

    <!-- playback -->
    <div class="group">
      <div class="group-title">Playback &nbsp;<span id="fps-display" style="color:#666"></span></div>
      <div class="row">
        <button id="play-btn" onclick="togglePlay()" title="Space">&#9654;</button>
        <select id="speed" title="Speed">
          <option value="0.1">0.1x</option>
          <option value="0.25">0.25x</option>
          <option value="0.5">0.5x</option>
          <option value="1" selected>1x</option>
          <option value="2">2x</option>
          <option value="4">4x</option>
        </select>
      </div>
    </div>

    <!-- frame scrubber -->
    <div class="group">
      <div class="group-title">Frame &nbsp;<span id="flabel" style="color:#adf">0</span>
        &nbsp;/ <span id="ftotal" style="color:#666">?</span></div>
      <div class="row">
        <button class="nav-btn" onclick="step(-100)">&#10218;&#10218;</button>
        <button class="nav-btn" onclick="step(-10)">&#10218;</button>
        <button class="nav-btn" onclick="step(-1)">&#8249;</button>
        <input type="range" id="frame-sl" min="0" max="1000" value="0" oninput="frameInput()">
        <button class="nav-btn" onclick="step(1)">&#8250;</button>
        <button class="nav-btn" onclick="step(10)">&#10219;</button>
        <button class="nav-btn" onclick="step(100)">&#10219;&#10219;</button>
      </div>
    </div>

    <!-- frame gap -->
    <div class="group">
      <div class="group-title">Frame gap</div>
      <div class="row">
        <input type="range" id="gap" min="1" max="10" value="2" oninput="ps('gap')">
        <span class="val" id="gap-v">2</span>
      </div>
    </div>

    <!-- diff threshold -->
    <div class="group">
      <div class="group-title">Diff thresh</div>
      <div class="row">
        <input type="range" id="thresh" min="1" max="80" value="10" oninput="ps('thresh')">
        <span class="val" id="thresh-v">10</span>
      </div>
    </div>

    <!-- min area -->
    <div class="group">
      <div class="group-title">Min area</div>
      <div class="row">
        <input type="range" id="min_a" min="1" max="500" value="2" oninput="ps('min_a')">
        <span class="val" id="min_a-v">2</span>
      </div>
    </div>

    <!-- max area -->
    <div class="group">
      <div class="group-title">Max area</div>
      <div class="row">
        <input type="range" id="max_a" min="5" max="4000" value="80" oninput="ps('max_a')">
        <span class="val" id="max_a-v">80</span>
      </div>
    </div>

    <!-- max aspect -->
    <div class="group">
      <div class="group-title">Max aspect</div>
      <div class="row">
        <input type="range" id="max_asp" min="1" max="10" step="0.5" value="3"
               oninput="ps('max_asp')">
        <span class="val" id="max_asp-v">3.0</span>
      </div>
    </div>

    <!-- score threshold -->
    <div class="group">
      <div class="group-title">Score thresh</div>
      <div class="row">
        <input type="range" id="score_thresh" min="0" max="1.5" step="0.01" value="0.1"
               oninput="ps('score_thresh')">
        <span class="val" id="score_thresh-v">0.10</span>
      </div>
    </div>

    <!-- scoring method -->
    <div class="group">
      <div class="group-title">Scoring</div>
      <div class="row" style="gap:4px">
        <button class="tog-btn on" id="btn-c" onclick="setMethod('compactness')">Compact</button>
        <button class="tog-btn"   id="btn-r"  onclick="setMethod('rog')">RoG</button>
        <button class="tog-btn"   id="btn-circ" onclick="setMethod('circularity')">Circ</button>
      </div>
    </div>

    <!-- pre-blur (useful for all methods, critical for circularity) -->
    <div class="group">
      <div class="group-title">Pre-blur</div>
      <div class="row">
        <input type="range" id="blur_k" min="0" max="9" value="0" step="1"
               oninput="ps('blur_k')">
        <span class="val" id="blur_k-v">0</span>
      </div>
    </div>

    <!-- circularity-method params (shown only when Circ is active) -->
    <div id="circ-params" style="display:none; contents">
      <div class="group">
        <div class="group-title" style="color:#c8f">Ball diam (px)</div>
        <div class="row">
          <input type="range" id="ball_diam" min="1" max="40" value="10" step="0.5"
                 oninput="ps('ball_diam')">
          <span class="val" id="ball_diam-v">10.0</span>
        </div>
      </div>
      <div class="group">
        <div class="group-title" style="color:#c8f">Min circularity</div>
        <div class="row">
          <input type="range" id="min_circ" min="0" max="1" value="0.2" step="0.01"
                 oninput="ps('min_circ')">
          <span class="val" id="min_circ-v">0.20</span>
        </div>
      </div>
      <div class="group">
        <div class="group-title" style="color:#c8f">Min brightness</div>
        <div class="row">
          <input type="range" id="min_bright" min="0" max="255" value="0" step="1"
                 oninput="ps('min_bright')">
          <span class="val" id="min_bright-v">0</span>
        </div>
      </div>
    </div>

    <!-- resolution -->
    <div class="group">
      <div class="group-title">Resolution</div>
      <div class="row">
        <select id="res_w" onchange="resChanged()">
          <option value="320" selected>320 (fast)</option>
          <option value="480">480</option>
          <option value="640">640</option>
          <option value="960">960</option>
          <option value="0">Full (native)</option>
        </select>
      </div>
    </div>

    <!-- solo mode -->
    <div class="group">
      <div class="group-title">Isolation</div>
      <div class="row">
        <button class="tog-btn" id="btn-solo" onclick="toggleSolo()"
                title="Only the last-touched slider filters; all others maximally permissive">
          Solo: OFF
        </button>
      </div>
    </div>

    <!-- spatial prior -->
    <div class="group">
      <div class="group-title">Spatial prior</div>
      <div class="row">
        <button class="tog-btn" id="btn-prior" onclick="togglePrior()">Prior: OFF</button>
      </div>
    </div>

  </div><!-- end det-controls -->

  <!-- PRIOR TAB controls -->
  <div id="prior-controls" style="display:none; contents">

    <!-- frame nav -->
    <div class="group">
      <div class="group-title">Frame &nbsp;<span id="pflabel" style="color:#adf">0</span></div>
      <div class="row">
        <button class="nav-btn" onclick="pstep(-10)">&#10218;</button>
        <button class="nav-btn" onclick="pstep(-1)">&#8249;</button>
        <input type="range" id="pframe-sl" min="0" max="1000" value="0"
               oninput="priorFrameInput()">
        <button class="nav-btn" onclick="pstep(1)">&#8250;</button>
        <button class="nav-btn" onclick="pstep(10)">&#10219;</button>
      </div>
    </div>

    <!-- court zone -->
    <div class="group">
      <div class="group-title" style="color:#8cf">Court sigma-x</div>
      <div class="row">
        <input type="range" id="court_xs" min="0" max="60" value="15" step="1"
               oninput="pp('court_xs')">
        <span class="val" id="court_xs-v">15</span>
      </div>
    </div>
    <div class="group">
      <div class="group-title" style="color:#8cf">Court sigma-below</div>
      <div class="row">
        <input type="range" id="court_ys" min="0" max="80" value="25" step="1"
               oninput="pp('court_ys')">
        <span class="val" id="court_ys-v">25</span>
      </div>
    </div>

    <!-- air zone -->
    <div class="group">
      <div class="group-title" style="color:#fc8">Air left (px)</div>
      <div class="row">
        <input type="range" id="air_xl" min="0" max="160" value="30" step="1"
               oninput="pp('air_xl')">
        <span class="val" id="air_xl-v">30</span>
      </div>
    </div>
    <div class="group">
      <div class="group-title" style="color:#fc8">Air right (px)</div>
      <div class="row">
        <input type="range" id="air_xr" min="160" max="320" value="290" step="1"
               oninput="pp('air_xr')">
        <span class="val" id="air_xr-v">290</span>
      </div>
    </div>
    <div class="group">
      <div class="group-title" style="color:#fc8">Air top cutoff</div>
      <div class="row">
        <input type="range" id="air_yt" min="0" max="60" value="5" step="1"
               oninput="pp('air_yt')">
        <span class="val" id="air_yt-v">5</span>
      </div>
    </div>
    <div class="group">
      <div class="group-title" style="color:#fc8">Air bottom (anchor)</div>
      <div class="row">
        <input type="range" id="air_yb" min="20" max="100" value="50" step="1"
               oninput="pp('air_yb')">
        <span class="val" id="air_yb-v">50</span>
      </div>
    </div>
    <div class="group">
      <div class="group-title" style="color:#fc8">Air sigma-x</div>
      <div class="row">
        <input type="range" id="air_sx" min="5" max="200" value="80" step="1"
               oninput="pp('air_sx')">
        <span class="val" id="air_sx-v">80</span>
      </div>
    </div>
    <div class="group">
      <div class="group-title" style="color:#fc8">Air sigma-y (up)</div>
      <div class="row">
        <input type="range" id="air_sy" min="5" max="120" value="45" step="1"
               oninput="pp('air_sy')">
        <span class="val" id="air_sy-v">45</span>
      </div>
    </div>

    <!-- overall -->
    <div class="group">
      <div class="group-title">Prior weight</div>
      <div class="row">
        <input type="range" id="pweight" min="0" max="1" value="0.8" step="0.01"
               oninput="pp('pweight')">
        <span class="val" id="pweight-v">0.80</span>
      </div>
    </div>
    <div class="group">
      <div class="group-title">Frame blend</div>
      <div class="row">
        <input type="range" id="pblend" min="0" max="1" value="0.35" step="0.05"
               oninput="pp('pblend')">
        <span class="val" id="pblend-v">0.35</span>
      </div>
    </div>
    <div class="group">
      <div class="group-title">Use in detection</div>
      <div class="row">
        <button class="tog-btn" id="btn-prior2" onclick="togglePrior()">Prior: OFF</button>
      </div>
    </div>

  </div><!-- end prior-controls -->

  <!-- TREE TAB controls -->
  <div id="tree-controls" style="display:none; contents">

    <!-- n steps -->
    <div class="group">
      <div class="group-title" style="color:#5cf">Steps</div>
      <div class="row">
        <input type="range" id="n_steps" min="2" max="12" value="5" step="1"
               oninput="tp('n_steps')">
        <span class="val" id="n_steps-v">5</span>
      </div>
    </div>

    <!-- link radius -->
    <div class="group">
      <div class="group-title" style="color:#5cf">Link radius</div>
      <div class="row">
        <input type="range" id="link_r" min="4" max="80" value="30" step="1"
               oninput="tp('link_r')">
        <span class="val" id="link_r-v">30</span>
      </div>
    </div>

    <!-- static radius -->
    <div class="group">
      <div class="group-title" style="color:#5cf">Static radius</div>
      <div class="row">
        <input type="range" id="static_r" min="2" max="40" value="8" step="1"
               oninput="tp('static_r')">
        <span class="val" id="static_r-v">8</span>
      </div>
    </div>

    <!-- vel penalty -->
    <div class="group">
      <div class="group-title" style="color:#5cf">Accel penalty</div>
      <div class="row">
        <input type="range" id="vel_pen" min="0" max="0.5" value="0.05" step="0.005"
               oninput="tp('vel_pen')">
        <span class="val" id="vel_pen-v">0.050</span>
      </div>
    </div>

  </div><!-- end tree-controls -->

</div><!-- end #bar -->

<!-- MAIN CONTENT -->
<div id="main">
  <div id="page-detect">
    <div id="panel-labels">
      <span>Original  &middot;  circle = best blob</span>
      <span>Frame diff (hot)</span>
      <span>Passing blobs only  &middot;  yellow = best</span>
    </div>
    <div id="panels">
      <img id="composite" src="" alt="Loading...">
    </div>
  </div>
  <div id="page-prior">
    <div id="prior-legend">
      Turbo colourmap: blue=low &rarr; red=high  &middot;
      White = court polygon  &middot;  Dashed = air bounds  &middot;  Cyan = anchor line
    </div>
    <div id="prior-panels">
      <img id="prior-img" src="" alt="Loading prior...">
    </div>
  </div>
  <div id="page-tree">
    <div id="tree-legend">
      Blue=oldest blobs &rarr; Yellow=newest &middot; Red X=static noise &middot;
      Grey lines=all edges &middot; Thick yellow=best DP path
    </div>
    <div id="tree-panels">
      <img id="tree-img" src="" alt="Enable tree to load...">
    </div>
  </div>
</div>

<div id="info">Space=play/pause  &middot;  arrows=step  &middot;  Shift=x10  &middot;  Loading&hellip;</div>

<script>
// ═══ STATE ═══════════════════════════════════════════════════════════════════
let videoName = "", totalFrames = 1000, videoFps = 25;
let method    = "compactness";
let usePrior  = false;
let currentTab = "detect";
let debounce = null, priordebounce = null;

// playback
let playing = false, playGen = 0, lastFrameTime = 0, measuredFps = 0;
const fpsAlpha = 0.15;

// solo mode
let soloMode  = false;
let lastParam = null;   // id of last-touched detection slider

// Permissive defaults for solo mode (everything passes)
const SOLO_PERM = {
  gap:          '1',
  thresh:       '1',
  min_a:        '1',
  max_a:        '9999',
  max_asp:      '99',
  score_thresh: '0',
  min_circ:     '0',
  min_bright:   '0',
  blur_k:       '0',
};
const PRIOR_PARAM_IDS = new Set([
  'court_xs','court_ys','air_xl','air_xr','air_yt','air_yb','air_sx','air_sy','pweight'
]);

// ═══ INIT ════════════════════════════════════════════════════════════════════
fetch('/api/videos').then(r=>r.json()).then(d => {
  const sel = document.getElementById('vid');
  d.videos.forEach(v => {
    const o = document.createElement('option');
    o.value = v; o.textContent = v; sel.appendChild(o);
  });
  if (d.videos.length) { videoName = d.videos[0]; loadInfo(); }
});

function videoChanged() {
  pause();
  videoName = document.getElementById('vid').value;
  loadInfo();
}

function loadInfo() {
  fetch('/api/info?video='+encodeURIComponent(videoName))
    .then(r=>r.json()).then(d => {
      totalFrames = d.frames;
      videoFps    = d.fps || 25;
      for (const id of ['frame-sl','pframe-sl'])
        document.getElementById(id).max = totalFrames - 1;
      document.getElementById('ftotal').textContent = totalFrames;
      request();
    });
}

// ═══ TAB SWITCHING ════════════════════════════════════════════════════════════
function switchTab(tab) {
  currentTab = tab;
  document.getElementById('tab-detect').classList.toggle('active', tab==='detect');
  document.getElementById('tab-prior').classList.toggle('active',  tab==='prior');
  document.getElementById('tab-tree').classList.toggle('active',   tab==='tree');
  document.getElementById('page-detect').style.display = tab==='detect' ? 'flex' : 'none';
  document.getElementById('page-prior').style.display  = tab==='prior'  ? 'flex' : 'none';
  document.getElementById('page-tree').style.display   = tab==='tree'   ? 'flex' : 'none';
  document.getElementById('det-controls').style.display   = tab==='detect' ? 'contents' : 'none';
  document.getElementById('prior-controls').style.display = tab==='prior'  ? 'contents' : 'none';
  document.getElementById('tree-controls').style.display  = tab==='tree'   ? 'contents' : 'none';
  if (tab==='prior') schedulePrior();
  else if (tab==='tree') scheduleTree();
  else               request();
}

// ═══ PLAYBACK ════════════════════════════════════════════════════════════════
function togglePlay() { playing ? pause() : play(); }

function play() {
  playing = true; playGen++;
  document.getElementById('play-btn').innerHTML = '&#9646;&#9646;';
  document.getElementById('play-btn').classList.add('playing');
  runPlayLoop(playGen);
}
function pause() {
  playing = false;
  document.getElementById('play-btn').innerHTML = '&#9654;';
  document.getElementById('play-btn').classList.remove('playing');
}

function runPlayLoop(gen) {
  if (!playing || gen !== playGen) return;
  const speed    = parseFloat(document.getElementById('speed').value);
  const targetMs = 1000 / (videoFps * speed);
  const t0       = performance.now();

  const sl = document.getElementById('frame-sl');
  let cur  = +sl.value + 1;
  if (cur >= totalFrames) cur = 0;
  sl.value = cur;
  sync();

  requestFrame(function onDone() {
    if (!playing || gen !== playGen) return;
    const wait = Math.max(0, targetMs - (performance.now() - t0));
    if (lastFrameTime) {
      const inst = 1000 / (t0 - lastFrameTime);
      measuredFps = measuredFps ? measuredFps*(1-fpsAlpha)+inst*fpsAlpha : inst;
      document.getElementById('fps-display').textContent = measuredFps.toFixed(1)+' fps';
    }
    lastFrameTime = t0;
    setTimeout(() => runPlayLoop(gen), wait);
  });
}

// ═══ FRAME NAV ════════════════════════════════════════════════════════════════
function frameInput() { pause(); sync(); schedule(); }

function step(delta) {
  pause();
  const sl = document.getElementById('frame-sl');
  sl.value = Math.max(0, Math.min(totalFrames-1, +sl.value+delta));
  sync(); schedule();
}
function sync() {
  const v = document.getElementById('frame-sl').value;
  document.getElementById('flabel').textContent   = v;
  document.getElementById('pframe-sl').value      = v;
  document.getElementById('pflabel').textContent  = v;
}

document.addEventListener('keydown', e => {
  if (e.target.tagName==='INPUT' || e.target.tagName==='SELECT') return;
  if (e.key===' ')          { e.preventDefault(); togglePlay(); }
  if (e.key==='ArrowRight') { pause(); step(e.shiftKey ? 10 : 1); }
  if (e.key==='ArrowLeft')  { pause(); step(e.shiftKey ? -10 : -1); }
});

// ═══ DETECTION SLIDERS ════════════════════════════════════════════════════════
function ps(id) {
  lastParam = id;
  if (soloMode) updateSoloBtn();
  // highlight the active slider in solo mode
  document.querySelectorAll('input[type=range]').forEach(el => {
    el.classList.toggle('solo-active', soloMode && el.id === id);
  });
  const v = parseFloat(document.getElementById(id).value);
  const fmt2 = new Set(['score_thresh','min_circ','ball_diam','pblend','pweight']);
  const fmt1 = new Set(['max_asp']);
  document.getElementById(id+'-v').textContent =
    fmt2.has(id) ? v.toFixed(2) : fmt1.has(id) ? v.toFixed(1) : String(Math.round(v));
  schedule();
}

function setMethod(m) {
  method = m;
  document.getElementById('btn-c').classList.toggle('on',    m==='compactness');
  document.getElementById('btn-r').classList.toggle('on',    m==='rog');
  document.getElementById('btn-circ').classList.toggle('on', m==='circularity');
  document.getElementById('circ-params').style.display = m==='circularity' ? 'contents' : 'none';
  schedule();
}

function resChanged() {
  request();
}

// ═══ SOLO MODE ════════════════════════════════════════════════════════════════
function toggleSolo() {
  soloMode = !soloMode;
  if (!soloMode) {
    // clear all slider highlights
    document.querySelectorAll('input[type=range]').forEach(el =>
      el.classList.remove('solo-active'));
  } else if (lastParam) {
    document.querySelectorAll('input[type=range]').forEach(el =>
      el.classList.toggle('solo-active', el.id === lastParam));
  }
  updateSoloBtn();
  schedule();
}

function updateSoloBtn() {
  const btn = document.getElementById('btn-solo');
  if (soloMode) {
    btn.classList.add('on');
    btn.textContent = lastParam ? ('Solo: ' + lastParam) : 'Solo: (pick slider)';
  } else {
    btn.classList.remove('on');
    btn.textContent = 'Solo: OFF';
  }
}

// ═══ PRIOR TOGGLE ════════════════════════════════════════════════════════════
function togglePrior() {
  usePrior = !usePrior;
  const label = usePrior ? 'Prior: ON' : 'Prior: OFF';
  document.getElementById('btn-prior').textContent  = label;
  document.getElementById('btn-prior2').textContent = label;
  document.getElementById('btn-prior').classList.toggle('on',  usePrior);
  document.getElementById('btn-prior2').classList.toggle('on', usePrior);
  const badge = document.getElementById('prior-badge');
  badge.textContent = usePrior ? 'ON' : 'OFF';
  badge.classList.toggle('on', usePrior);
  schedule();
}

// ═══ PRIOR PARAMS HELPER ══════════════════════════════════════════════════════
function priorParams() {
  return {
    court_xs: document.getElementById('court_xs').value,
    court_ys: document.getElementById('court_ys').value,
    air_xl:   document.getElementById('air_xl').value,
    air_xr:   document.getElementById('air_xr').value,
    air_yt:   document.getElementById('air_yt').value,
    air_yb:   document.getElementById('air_yb').value,
    air_sx:   document.getElementById('air_sx').value,
    air_sy:   document.getElementById('air_sy').value,
    pweight:  document.getElementById('pweight').value,
  };
}


function requestFrame(onDone) {
  if (!videoName) return;

  // Start with actual slider values
  let gap          = document.getElementById('gap').value;
  let thresh       = document.getElementById('thresh').value;
  let min_a        = document.getElementById('min_a').value;
  let max_a        = document.getElementById('max_a').value;
  let max_asp      = document.getElementById('max_asp').value;
  let score_thresh = document.getElementById('score_thresh').value;
  let use_prior    = usePrior;

  if (soloMode && lastParam) {
    for (const [k, v] of Object.entries(SOLO_PERM)) {
      if (k !== lastParam) {
        if      (k === 'gap')          gap          = v;
        else if (k === 'thresh')       thresh       = v;
        else if (k === 'min_a')        min_a        = v;
        else if (k === 'max_a')        max_a        = v;
        else if (k === 'max_asp')      max_asp      = v;
        else if (k === 'score_thresh') score_thresh = v;
      }
    }
    use_prior = PRIOR_PARAM_IDS.has(lastParam) ? true : false;
  }

  const p = new URLSearchParams({
    video:        videoName,
    frame:        document.getElementById('frame-sl').value,
    gap, thresh, min_a, max_a, max_asp, score_thresh,
    method,
    use_prior:    use_prior ? '1' : '0',
    res_w:        document.getElementById('res_w').value,
    blur_k:       document.getElementById('blur_k').value,
    ball_diam:    document.getElementById('ball_diam').value,
    min_circ:     document.getElementById('min_circ').value,
    min_bright:   document.getElementById('min_bright').value,
    ...priorParams(),
  });

  fetch('/api/frame?' + p).then(r => r.json()).then(d => {
    document.getElementById('composite').src = 'data:image/jpeg;base64,' + d.img;
    const c = d.candidates;
    let info = 'Frame ' + document.getElementById('frame-sl').value +
               (use_prior ? ' [prior ON]' : '') +
               (soloMode  ? ' [SOLO: ' + (lastParam || '?') + ']' : '') +
               '  |  Passing: ' + c.length;
    if (c.length) {
      const b = c[0];
      info += '  \u00b7  BEST score=' + b.score.toFixed(3) +
              (use_prior ? ' (prior=' + b.prior + ')' : '') +
              (b.circ != null ? '  circ=' + b.circ.toFixed(2) : '') +
              '  area=' + b.area + 'px  asp=' + b.asp.toFixed(2) +
              '  @(' + Math.round(b.x) + ',' + Math.round(b.y) + ')';
      if (c.length > 1) info += '  +' + (c.length-1) + ' others';
    } else {
      info += '  \u00b7  no blobs pass filters';
    }
    document.getElementById('info').textContent = info;
    if (onDone) onDone();
  }).catch(e => {
    document.getElementById('info').textContent = 'Error: ' + e;
    if (onDone) onDone();
  });
}

// \u2550\u2550\u2550 PRIOR TAB \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
function priorFrameInput() {
  const v = document.getElementById('pframe-sl').value;
  document.getElementById('pflabel').textContent = v;
  document.getElementById('frame-sl').value      = v;
  document.getElementById('flabel').textContent  = v;
  schedulePrior();
}
function pstep(delta) {
  const sl = document.getElementById('pframe-sl');
  sl.value = Math.max(0, Math.min(totalFrames-1, +sl.value+delta));
  document.getElementById('pframe-sl').value     = sl.value;
  document.getElementById('frame-sl').value      = sl.value;
  document.getElementById('flabel').textContent  = sl.value;
  document.getElementById('pflabel').textContent = sl.value;
  schedulePrior();
}

function pp(id) {
  lastParam = id;
  if (soloMode) updateSoloBtn();
  const v   = parseFloat(document.getElementById(id).value);
  const fmt = (id==='pweight'||id==='pblend') ? v.toFixed(2) : String(Math.round(v));
  document.getElementById(id+'-v').textContent = fmt;
  schedulePrior();
  if (usePrior) schedule();
}

function schedulePrior() {
  clearTimeout(priordebounce);
  priordebounce = setTimeout(requestPrior, 80);
}

function requestPrior() {
  if (!videoName) return;
  const p = new URLSearchParams({
    video:  videoName,
    frame:  document.getElementById('pframe-sl').value,
    pblend: document.getElementById('pblend').value,
    ...priorParams(),
  });
  fetch('/api/prior?' + p).then(r => r.json()).then(d => {
    document.getElementById('prior-img').src = 'data:image/jpeg;base64,' + d.img;
    const pr = priorParams();
    document.getElementById('info').textContent =
      'Court: sx=' + pr.court_xs + '  sy_below=' + pr.court_ys + '    ' +
      'Air: x=[' + pr.air_xl + ',' + pr.air_xr + ']  y=[' + pr.air_yt + ',' + pr.air_yb + ']  ' +
      'sx=' + pr.air_sx + '  sy=' + pr.air_sy + '    weight=' + pr.pweight +
      '  (cyan=anchor  dashed=air bounds  white=court polygon)';
  }).catch(e => {
    document.getElementById('info').textContent = 'Prior error: ' + e;
  });
}

// \u2550\u2550\u2550 TREE \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
let treeDebounce = null;

function tp(id) {
  const v   = parseFloat(document.getElementById(id).value);
  const fmt = id === 'vel_pen' ? v.toFixed(3) : String(Math.round(v));
  document.getElementById(id+'-v').textContent = fmt;
  scheduleTree();
}

function scheduleTree() {
  if (currentTab !== 'tree') return;
  clearTimeout(treeDebounce);
  treeDebounce = setTimeout(requestTree, 120);
}

function requestTree() {
  if (!videoName || currentTab !== 'tree') return;
  const p = new URLSearchParams({
    video:    videoName,
    frame:    document.getElementById('frame-sl').value,
    gap:      document.getElementById('gap').value,
    thresh:   document.getElementById('thresh').value,
    min_a:    document.getElementById('min_a').value,
    max_a:    document.getElementById('max_a').value,
    max_asp:  document.getElementById('max_asp').value,
    method,
    res_w:    document.getElementById('res_w').value,
    blur_k:   document.getElementById('blur_k').value,
    ball_diam:document.getElementById('ball_diam').value,
    min_circ: document.getElementById('min_circ').value,
    min_bright:document.getElementById('min_bright').value,
    n_steps:  document.getElementById('n_steps').value,
    link_r:   document.getElementById('link_r').value,
    static_r: document.getElementById('static_r').value,
    vel_pen:  document.getElementById('vel_pen').value,
  });
  document.getElementById('info').textContent = 'Tree: computing\u2026';
  fetch('/api/tree?' + p).then(r => r.json()).then(d => {
    if (d.img) {
      document.getElementById('tree-img').src = 'data:image/jpeg;base64,' + d.img;
    }
    document.getElementById('info').textContent =
      'Tree: frame ' + document.getElementById('frame-sl').value +
      '  |  steps=' + (d.n_steps||'?') +
      '  blobs=' + (d.n_blobs||'?') +
      '  path_len=' + (d.path_len||0) +
      (d.error ? '  ERR: ' + d.error : '');
  }).catch(e => {
    document.getElementById('info').textContent = 'Tree error: ' + e;
  });
}

</script>
</body>
</html>
"""

# ---- HTTP request handler ----
import json as _json_mod, urllib.parse as _urllib_parse, cv2 as _cv2_mod
from http.server import HTTPServer, BaseHTTPRequestHandler


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, data):
        body = _json_mod.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        prs = _urllib_parse.urlparse(self.path)
        qs  = _urllib_parse.parse_qs(prs.query)

        def q(k, d=""):
            return qs.get(k, [d])[0]

        if prs.path == "/":
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif prs.path == "/api/videos":
            self._json({"videos": list_videos()})

        elif prs.path == "/api/info":
            cap    = get_cap(q("video"))
            frames = int(cap.get(_cv2_mod.CAP_PROP_FRAME_COUNT))
            fps    = float(cap.get(_cv2_mod.CAP_PROP_FPS) or 25.0)
            self._json({"frames": frames, "fps": fps})

        elif prs.path == "/api/frame":
            try:
                res_w = int(q("res_w", "320"))
                comp, cands = make_composite(
                    vname         = q("video"),
                    frame_idx     = int(q("frame",          "0")),
                    gap           = int(q("gap",            "2")),
                    thresh        = int(q("thresh",         "10")),
                    min_a         = int(q("min_a",          "2")),
                    max_a         = int(q("max_a",          "80")),
                    max_asp       = float(q("max_asp",      "3.0")),
                    method        = q("method",             "compactness"),
                    score_thresh  = float(q("score_thresh", "0.1")),
                    use_prior     = q("use_prior", "0") == "1",
                    court_x_sigma = float(q("court_xs",    "15")),
                    court_y_sigma = float(q("court_ys",    "25")),
                    air_x_left    = int(q("air_xl",        "30")),
                    air_x_right   = int(q("air_xr",        "290")),
                    air_y_top     = int(q("air_yt",        "5")),
                    air_y_bot     = int(q("air_yb",        "50")),
                    air_sigma_x   = float(q("air_sx",      "80")),
                    air_sigma_y   = float(q("air_sy",      "45")),
                    pweight       = float(q("pweight",     "0.8")),
                    proc_w        = res_w,
                    ball_diam     = float(q("ball_diam",   "10")),
                    min_circ      = float(q("min_circ",    "0.2")),
                    min_bright    = float(q("min_bright",  "0.0")),
                    blur_k        = int(q("blur_k",        "0")),
                )
                cands_out = [
                    {k: v for k, v in c.items() if k != "_label"}
                    for c in cands
                ]
                self._json({"img": img_to_b64jpeg(comp), "candidates": cands_out})
            except Exception as exc:
                import traceback; traceback.print_exc()
                self._json({"img": "", "candidates": [], "error": str(exc)})

        elif prs.path == "/api/prior":
            try:
                img = make_prior_image(
                    court_x_sigma = float(q("court_xs",  "15")),
                    court_y_sigma = float(q("court_ys",  "25")),
                    air_x_left    = int(q("air_xl",      "30")),
                    air_x_right   = int(q("air_xr",      "290")),
                    air_y_top     = int(q("air_yt",      "5")),
                    air_y_bot     = int(q("air_yb",      "50")),
                    air_sigma_x   = float(q("air_sx",    "80")),
                    air_sigma_y   = float(q("air_sy",    "45")),
                    weight        = float(q("pweight",   "0.8")),
                    vname         = q("video") or None,
                    frame_idx     = int(q("frame",       "0")),
                    frame_blend   = float(q("pblend",    "0.35")),
                )
                self._json({"img": img_to_b64jpeg(img)})
            except Exception as exc:
                import traceback; traceback.print_exc()
                self._json({"img": "", "error": str(exc)})

        elif prs.path == "/api/tree":
            try:
                res_w       = int(q("res_w",    "320"))
                center_frame= int(q("frame",    "0"))
                gap         = int(q("gap",      "2"))
                n_steps     = int(q("n_steps",  "5"))
                link_r      = int(q("link_r",   "30"))
                static_r    = int(q("static_r", "8"))
                vel_pen     = float(q("vel_pen","0.05"))
                steps = collect_tree_blobs(
                    vname        = q("video"),
                    center_frame = center_frame,
                    gap          = gap,
                    n_steps      = n_steps,
                    thresh       = int(q("thresh",      "10")),
                    min_a        = int(q("min_a",       "1")),
                    max_a        = int(q("max_a",       "400")),
                    max_asp      = float(q("max_asp",   "6.0")),
                    method       = q("method",          "compactness"),
                    proc_w       = res_w,
                    ball_diam    = float(q("ball_diam", "10")),
                    min_circ     = float(q("min_circ",  "0.1")),
                    min_bright   = float(q("min_bright","0.0")),
                    blur_k       = int(q("blur_k",      "0")),
                )
                chron_steps, static_flags, edges, best_path = build_path_dp(
                    steps,
                    link_radius   = link_r,
                    static_radius = static_r,
                    vel_penalty   = vel_pen,
                )
                img = make_tree_image(
                    q("video"), center_frame, res_w,
                    chron_steps, static_flags, edges, best_path,
                )
                n_blobs = sum(len(s["blobs"]) for s in chron_steps)
                self._json({
                    "img":      img_to_b64jpeg(img),
                    "n_steps":  len(chron_steps),
                    "n_blobs":  n_blobs,
                    "path_len": len(best_path),
                })
            except Exception as exc:
                import traceback; traceback.print_exc()
                self._json({"img": "", "error": str(exc)})

        else:
            self.send_response(404)
            self.end_headers()


# ---- entry point ----
PORT = 8787

if __name__ == "__main__":
    import threading, webbrowser
    print(f"TENNIS root : {TENNIS}")
    print(f"Video dir   : {VIDEO_DIR}")
    vids = list_videos()
    if not vids:
        print("WARNING: no video files found in VIDEO_DIR above")
    else:
        print(f"Videos      : {vids}")
    url = f"http://localhost:{PORT}"
    print(f"\nStarting server -> {url}")
    print("Tab 1 Detection : Space=play/pause  arrows=step  Shift=x10")
    print("                  Resolution selector: 320/480/640/960/Full(native)")
    print("                  Solo button: isolates last-touched slider")
    print("                  Scoring: Compact | RoG | Circ+Brightness")
    print("Tab 2 Spatial Prior : tune two-zone probability mask")
    print("Tab 3 Trajectory Tree : DP path over blob tree across frames")
    print("Press Ctrl-C to stop.\n")

    server = HTTPServer(("localhost", PORT), Handler)
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")

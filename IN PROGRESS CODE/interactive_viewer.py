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
OUT_DIR   = TENNIS / "claude" / "claude" / "genvideos" / "runanalysis"

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

# ── Constants ────────────────────────────────────────────────────────────────
#    Single source of truth for all slider defaults.
#    HTML value= attributes and handler q() fallbacks must match these.
DEFAULT_PARAMS = {
    # Detection
    "gap":          2,
    "thresh":       18,
    "min_a":        3,
    "max_a":        1551,
    "max_asp":      2.5,
    "score_thresh": 0.0,
    "method":       "circularity",
    "blur_k":       6,
    "ball_diam":    10.0,
    "min_circ":     0.2,
    "min_bright":   0.0,
    "n_diff":       1,
    "res_w":        0,
    # Spatial prior
    "court_xs":     0.1,
    "court_ys":     0.0,
    "court_inset":  8,
    "air_xl":       115,
    "air_xr":       193,
    "air_yt":       38,
    "air_yb":       58,
    "air_sx":       37.0,
    "air_sy":       40.0,
    "pweight":      1.0,
    "pblend":       0.35,
    # Tree (DP)
    "tree_gap":     1,
    "n_steps":      10,
    "link_r":       35,
    "static_r":     2,
    "vel_pen":      0.05,
    # Track
    "n_look":       12,
    "trk_top_k":    3,
    "trk_link_r":   40,
    "trk_vel_pen":  0.03,
    "ransac_px":    8.0,
    # History scatter
    "n_hist":       15,
    "top_k":        5,
}
_DP = DEFAULT_PARAMS   # short alias for handler code

# ── prior cache ───────────────────────────────────────────────────────────────
_prior_cache: dict = {}

def compute_prior_map(
        court_x_sigma  = 15.0,
        court_y_sigma  = 25.0,
        court_inset    = 0,
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
      court_inset > 0 shrinks the "valid" court inward by that many pixels on
      all four walls, so the boundary stripe area experiences the falloff sigma
      rather than being fully inside the prior.  Use this to suppress white-line
      false positives.
    AIR ZONE:   2-D Gaussian anchored at bottom-centre of a top bar.
    Combined:   raw = max(court_zone, air_zone)
                result = (1-weight) + weight * raw
    """
    key = (round(court_x_sigma,2), round(court_y_sigma,2),
           int(court_inset),
           air_x_left, air_x_right, air_y_top, air_y_bot,
           round(air_sigma_x,2), round(air_sigma_y,2), round(weight,3))
    if key in _prior_cache:
        return _prior_cache[key]

    Y, X = np.mgrid[0:REF_H, 0:REF_W].astype(np.float32)

    # Court zone — apply inset to push effective boundary inward
    t = np.clip((Y - COURT_Y_TOP) / (COURT_Y_BOT - COURT_Y_TOP), 0.0, 1.0)
    xl_trap = COURT_X_LEFT_TOP  + (COURT_X_LEFT_BOT  - COURT_X_LEFT_TOP)  * t
    xr_trap = COURT_X_RIGHT_TOP + (COURT_X_RIGHT_BOT - COURT_X_RIGHT_TOP) * t
    xl = np.where(Y < COURT_Y_TOP, float(COURT_X_LEFT_BOT),  xl_trap)
    xr = np.where(Y < COURT_Y_TOP, float(COURT_X_RIGHT_BOT), xr_trap)
    # Shrink effective court by court_inset pixels on left, right, top, and bottom
    xl_eff = xl + court_inset
    xr_eff = np.maximum(xr - court_inset, xl_eff + 1.0)   # guard against inversion
    y_top_eff = COURT_Y_TOP  + court_inset
    y_bot_eff = COURT_Y_BOT  - court_inset
    dx_c = np.maximum(0.0, xl_eff - X) + np.maximum(0.0, X - xr_eff)
    px_c = np.exp(-0.5 * (dx_c / max(court_x_sigma, 1e-3))**2)
    dy_c = np.maximum(0.0, Y - max(y_bot_eff, y_top_eff + 1.0))
    py_c = np.exp(-0.5 * (dy_c / max(court_y_sigma, 1e-3))**2)
    # Mask: above inset top → no court zone (keeps air zone separate)
    court_zone = (px_c * py_c * (Y >= y_top_eff)).astype(np.float32)

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
        court_inset,
        air_x_left, air_x_right, air_y_top, air_y_bot,
        air_sigma_x, air_sigma_y,
        weight,
        vname=None, frame_idx=0, frame_blend=0.35):
    """Turbo heatmap of the prior at 3x REF resolution, annotated."""
    prior = compute_prior_map(
        court_x_sigma, court_y_sigma,
        court_inset,
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


# ── Core CV ──────────────────────────────────────────────────────────────────
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
           ball_diam=10.0, min_circ=0.2, min_bright=0.0, blur_k=0, diff_mode="abs"):
    """
    Returns (diff, bw, labels, passing_blobs, rejected_blobs).
    prior_map: REF_H x REF_W float32; blob centroid scaled to REF coords for lookup.
    method: 'compactness' | 'rog' | 'circularity'
      circularity: score = 0.6*brightness + 0.2*circularity + 0.2*size_score
                   ball_diam sets the reference area; min_circ is a hard filter.
    """
    if diff_mode == "pos":
        diff = np.clip(g_curr.astype(np.int16) - g_prev.astype(np.int16), 0, 255).astype(np.uint8)
    elif diff_mode == "neg":
        diff = np.clip(g_prev.astype(np.int16) - g_curr.astype(np.int16), 0, 255).astype(np.uint8)
    else:
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
    _pw = proc_w if proc_w > 0 else g_curr.shape[1]
    size_target = 15.0 * (_pw / REF_W) ** 2
    # Circularity method: target area from ball_diam reference (also resolution-scaled)
    ball_diam_scaled = ball_diam * (_pw / REF_W)
    circ_target_area = max(1.0, np.pi * (ball_diam_scaled / 2.0) ** 2)

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
            sscr  = max(1.0 - abs(a - size_target) / (max(size_target, 1.0) * 2.0), 0.1)
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
            _ph = proc_h if proc_h > 0 else g_curr.shape[0]
            _pw = proc_w if proc_w > 0 else g_curr.shape[1]
            iy = max(0, min(REF_H - 1, int(round(cy * REF_H / _ph))))
            ix = max(0, min(REF_W - 1, int(round(cx * REF_W / _pw))))
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

        if min_a <= a <= max_a and asp <= max_asp and score > score_thresh:
            passing.append(blob)
        else:
            rejected.append(blob)

    passing.sort(key=lambda c: c["score"], reverse=True)
    return diff, bw, labels, passing, rejected


# ── Rendering ────────────────────────────────────────────────────────────────
DIFF_AMP = 6.0

def get_display_scale(proc_w):
    if proc_w <= 320: return 3
    if proc_w <= 480: return 2
    return 1


def _diff_layer_color(k, n_total):
    """Age-based RGB colour for diff layer k (0=newest/green, n-1=oldest/red, mid=blue)."""
    age = k / max(n_total - 1, 1)   # 0=newest, 1=oldest
    if age <= 0.5:
        t = age * 2                  # 0→1 newest→middle
        return (0, int(255*(1-t)), int(255*t))        # green → blue
    else:
        t = (age - 0.5) * 2         # 0→1 middle→oldest
        return (int(255*t), 0, int(255*(1-t)))        # blue → red


def make_composite(vname, frame_idx, gap, thresh, min_a, max_a,
                   max_asp, method, score_thresh,
                   use_prior=False, diff_mode="abs",
                   court_x_sigma=15.0, court_y_sigma=25.0,
                   court_inset=0,
                   air_x_left=30, air_x_right=290,
                   air_y_top=5,  air_y_bot=50,
                   air_sigma_x=80.0, air_sigma_y=7.0,
                   pweight=1.0,
                   proc_w=REF_W,
                   ball_diam=10.0, min_circ=0.2, min_bright=0.0, blur_k=0,
                   n_diff=1):
    """Returns (composite_rgb_image, passing_blobs_list).
    proc_w=0 means native video resolution.
    n_diff: number of adjacent diff layers to overlay on panel 1 (age-coloured).
    """
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
        court_inset,
        air_x_left, air_x_right, air_y_top, air_y_bot,
        air_sigma_x, air_sigma_y, pweight) if use_prior else None

    # Most-recent diff (k=0) — used for panels 2 & 3 and for the return value
    diff, bw, labels, passing, rejected = detect(
        g_curr, g_prev, thresh, min_a, max_a, max_asp, method, score_thresh,
        prior_map=prior_map, proc_w=proc_w, proc_h=proc_h,
        ball_diam=ball_diam, min_circ=min_circ, min_bright=min_bright, blur_k=blur_k,
        diff_mode=diff_mode)

    # Collect older diff layers if requested (k=1…n_diff-1)
    layers = [(0, passing, rejected)]   # (k, passing, rejected)
    for k in range(1, n_diff):
        fb = max(0, frame_idx - k * gap)
        fa = max(0, fb - gap)
        if fa == fb:
            break
        gb, _ = read_gray_small(vname, fb, proc_w)
        ga, _ = read_gray_small(vname, fa, proc_w)
        if ga is None or gb is None:
            break
        _, _, _, p_k, r_k = detect(
            gb, ga, thresh, min_a, max_a, max_asp, method, score_thresh,
            prior_map=prior_map, proc_w=proc_w, proc_h=proc_h,
            ball_diam=ball_diam, min_circ=min_circ, min_bright=min_bright, blur_k=blur_k,
            diff_mode=diff_mode)
        layers.append((k, p_k, r_k))
    n_layers = len(layers)

    # Drawing parameters scale with resolution
    fs = max(0.5, 0.65 * proc_w / REF_W)
    lw = max(1, int(proc_w / REF_W))

    # Panel 1: original frame with age-coloured blob overlay
    p1 = cv2.cvtColor(bgr_curr, cv2.COLOR_BGR2RGB).copy()

    # Draw older layers first (so newest is on top)
    for (k, p_k, r_k) in reversed(layers):
        col = _diff_layer_color(k, n_layers)
        # Dim rejected blobs slightly
        col_r = tuple(int(c * 0.45) for c in col)
        for b in r_k:
            cv2.circle(p1, (int(b["x"]), int(b["y"])),
                       max(2, int(max(b["w"], b["h"]) // 2)), col_r, lw)
        for b in p_k:
            r_px = max(3, int(max(b["w"], b["h"]) // 2) + 1)
            cv2.circle(p1, (int(b["x"]), int(b["y"])), r_px, col, lw + 1)

    # Panel 2: amplified diff of most-recent pair → viridis colourmap
    diff_amp = np.clip(diff.astype(np.float32) * DIFF_AMP, 0, 255).astype(np.uint8)
    p2 = cv2.cvtColor(cv2.applyColorMap(diff_amp, cv2.COLORMAP_VIRIDIS),
                      cv2.COLOR_BGR2RGB)

    # Panel 3: black canvas, passing blobs of most-recent pair only
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

    # Best-blob crosshair (yellow) on all three panels
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


# ── Trajectory Tree ──────────────────────────────────────────────────────────

def collect_tree_blobs(vname, center_frame, gap, n_steps,
                       thresh=10, min_a=1, max_a=400, max_asp=6.0,
                       method="compactness", proc_w=REF_W,
                       ball_diam=10.0, min_circ=0.1, min_bright=0.0, blur_k=0,
                       score_thresh=0.0, prior_map=None, top_k=0, diff_mode="abs"):
    """Collect blobs from n_steps consecutive BACKWARD diff pairs ending at center_frame.

    Step k: diff(center_frame - (k+1)*gap, center_frame - k*gap)
      k=0 (newest): diff ending at center_frame  → ball position at current frame
      k=n_steps-1 (oldest): diff starting n_steps frames back

    Going backward matches the history scatter view and shows the ball's actual past
    trajectory cleanly — forward diffs pick up post-contact player motion artifacts.

    Returns list newest-first (k=0 first) so build_path_dp can consume directly.
    """
    if proc_w == 0:
        cap = get_cap(vname)
        proc_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        proc_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        proc_h = proc_w * 9 // 16

    # Frame cache
    _fcache = {}
    def _gf(fn):
        if fn not in _fcache:
            _fcache[fn] = read_gray_small(vname, max(0, fn), proc_w)
        return _fcache[fn]

    steps_bwd = []   # newest-first (k=0 first) — ready for build_path_dp directly
    for k in range(n_steps):
        fb = center_frame - k * gap           # newer frame
        fa = center_frame - (k + 1) * gap    # older frame
        if fa < 0:
            break
        g_b, _ = _gf(fb)
        g_a, _ = _gf(fa)
        if g_a is None or g_b is None:
            break
        _, _, _, passing, rejected = detect(
            g_b, g_a, thresh, min_a, max_a, max_asp, method, score_thresh,
            prior_map=None,  # tree uses raw scores for top_k — prior skews ranking
            proc_w=proc_w, proc_h=proc_h,
            ball_diam=ball_diam, min_circ=min_circ,
            min_bright=min_bright, blur_k=blur_k, diff_mode=diff_mode)
        all_blobs = []
        top_passing = (sorted(passing, key=lambda b: b['score'], reverse=True)[:top_k]
                       if top_k > 0 else passing)
        top_set = set(id(b) for b in top_passing)
        for b in passing:
            b2 = dict(b); b2['passing'] = (id(b) in top_set); all_blobs.append(b2)
        for b in rejected:
            b2 = dict(b); b2['passing'] = False; all_blobs.append(b2)
        steps_bwd.append({'step': k, 'frame_a': fa, 'frame_b': fb, 'blobs': all_blobs})
    # Already newest-first — build_path_dp reverses internally to get chron order
    return steps_bwd


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

    # Static: passing blob i at step t has a near-neighbour (passing) at step t+1.
    # Only passing blobs participate — rejected blobs don't affect static detection.
    static_flags = [set() for _ in range(n_t)]
    for t in range(n_t - 1):
        r2 = static_radius * static_radius
        for i, bi in enumerate(chron_steps[t]['blobs']):
            if not bi.get('passing', True):
                continue   # skip rejected blobs
            for bj in chron_steps[t + 1]['blobs']:
                if not bj.get('passing', True):
                    continue
                dx = bi['x'] - bj['x']
                dy = bi['y'] - bj['y']
                if dx*dx + dy*dy <= r2:
                    static_flags[t].add(i)
                    break

    # Valid edges: only between passing, non-static blobs
    r2_link = link_radius * link_radius
    edges_by_t = [[] for _ in range(n_t)]
    for t in range(1, n_t):
        for i, bi in enumerate(chron_steps[t]['blobs']):
            if not bi.get('passing', True):
                continue   # rejected blob — not eligible for DP
            if i in static_flags[t]:
                continue
            for j, bj in enumerate(chron_steps[t - 1]['blobs']):
                if not bj.get('passing', True):
                    continue
                if j in static_flags[t - 1]:
                    continue
                dx = bi['x'] - bj['x']
                dy = bi['y'] - bj['y']
                if dx*dx + dy*dy <= r2_link:
                    edges_by_t[t].append((i, j))

    edges = [(t, i, t - 1, j) for t in range(1, n_t) for (i, j) in edges_by_t[t]]

    # DP: dp[t][i] = [total_score, pred_j, vx_in, vy_in, path_len]
    # Passing, non-static blobs are seeded fresh if no predecessor links to them,
    # so a ball that first appears at step t>0 can still start (and anchor) a chain.
    # Path selection ranks by LENGTH first, then score — so a 7-blob chain of
    # moderate blobs always beats a 2-blob chain of bright blobs.
    INF = float('-inf')
    dp = [[None] * len(chron_steps[t]['blobs']) for t in range(n_t)]
    # Seed t=0
    for i, b in enumerate(chron_steps[0]['blobs']):
        if b.get('passing', True) and i not in static_flags[0]:
            dp[0][i] = [b['score'], -1, 0.0, 0.0, 1]

    for t in range(1, n_t):
        for i, bi in enumerate(chron_steps[t]['blobs']):
            if not bi.get('passing', True) or i in static_flags[t]:
                continue
            best_len, best_val, best_entry = 0, INF, None
            for (ei, ej) in edges_by_t[t]:
                if ei != i or dp[t - 1][ej] is None:
                    continue
                prev_score, _, vx_in, vy_in, prev_len = dp[t - 1][ej]
                bj = chron_steps[t - 1]['blobs'][ej]
                vx_out = bi['x'] - bj['x']
                vy_out = bi['y'] - bj['y']
                # Velocity-consistency gates: if we have an established direction,
                # reject extensions that violate physics.
                if prev_len > 1:
                    spd_in  = (vx_in*vx_in + vy_in*vy_in) ** 0.5
                    spd_out = (vx_out*vx_out + vy_out*vy_out) ** 0.5
                    if spd_in > 1.0 and spd_out > spd_in * 3.0:
                        continue   # speed jumped > 3× — likely a noise blob
                    # Direction-consistency gate: reject turns > ~100° when
                    # we have meaningful established velocity.  This prevents
                    # the DP from jumping to a blob in the wrong direction
                    # even when the speed magnitude is similar.
                    # (cos 100° ≈ -0.17)
                    if spd_in > 2.0 and spd_out > 2.0:
                        cos_theta = (vx_in*vx_out + vy_in*vy_out) / (spd_in * spd_out)
                        if cos_theta < -0.17:
                            continue   # turn > 100° — likely jumped to wrong blob
                dvx, dvy = vx_out - vx_in, vy_out - vy_in
                accel = (dvx*dvx + dvy*dvy) ** 0.5
                total = prev_score + bi['score'] - vel_penalty * accel
                new_len = prev_len + 1
                # Prefer longer chain; break ties by score
                if new_len > best_len or (new_len == best_len and total > best_val):
                    best_len = new_len
                    best_val = total
                    best_entry = [total, ej, vx_out, vy_out, new_len]
            if best_entry is not None:
                dp[t][i] = best_entry
            else:
                # No predecessor within link_r — seed fresh
                dp[t][i] = [bi['score'], -1, 0.0, 0.0, 1]

    # Trace best path: rank by path LENGTH first, then score.
    # A longer chain of moderate blobs beats a shorter chain of bright blobs.
    best_path = []
    best_len = 0
    best_end_score = INF
    best_end_t, best_end_i = -1, -1
    for t in range(n_t):
        for i, entry in enumerate(dp[t]):
            if entry is None:
                continue
            plen = entry[4]
            if plen > best_len or (plen == best_len and entry[0] > best_end_score):
                best_len = plen
                best_end_score = entry[0]
                best_end_t = t
                best_end_i = i
    if best_end_i >= 0:
        t, i, path_rev = best_end_t, best_end_i, []
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
    """Scatter-plot style: small viridis dots + grey edges + thick yellow path.

    Viridis colormap: oldest step (t=0) → purple, newest (t=n-1) → yellow.
    Dot size is fixed (not blob-sized) so the scatter is easy to read.
    """
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
        gray = cv2.cvtColor(bgr_curr, cv2.COLOR_BGR2GRAY)
        dark = (gray.astype(np.float32) * 0.3).astype(np.uint8)
        canvas = cv2.cvtColor(dark, cv2.COLOR_GRAY2RGB)

    disp_scale = get_display_scale(proc_w)
    h, w = canvas.shape[:2]
    canvas = cv2.resize(canvas, (w * disp_scale, h * disp_scale),
                        interpolation=cv2.INTER_NEAREST)

    n_t      = len(chron_steps)
    lw       = max(1, disp_scale)
    DOT_R    = max(2, disp_scale * 2)   # fixed radius — small, matches notebook scatter
    path_set = set(best_path)

    def bxy(t, i):
        b = chron_steps[t]['blobs'][i]
        return (int(b['x'] * disp_scale), int(b['y'] * disp_scale))

    def vcol(t):
        """Return RGB tuple via COLORMAP_VIRIDIS; t=0→purple, t=n-1→yellow."""
        val = int(255 * t / max(n_t - 1, 1))
        lut = np.array([[val]], dtype=np.uint8)
        bgr = cv2.applyColorMap(lut, cv2.COLORMAP_VIRIDIS)[0, 0]
        return (int(bgr[2]), int(bgr[1]), int(bgr[0]))   # BGR → RGB

    # 1. Grey lines: all valid edges (drawn first, behind dots)
    for (t, i, t_prev, j) in edges:
        cv2.line(canvas, bxy(t_prev, j), bxy(t, i), (60, 60, 60), lw)

    # 2. Thick yellow: best DP path
    for k in range(len(best_path) - 1):
        t0, i0 = best_path[k]
        t1, i1 = best_path[k + 1]
        cv2.line(canvas, bxy(t0, i0), bxy(t1, i1), (255, 230, 0), lw * 3 + 1)

    # 3. Dots — oldest drawn first so newest renders on top
    for t in range(n_t):
        step = chron_steps[t]
        col  = vcol(t)
        for i, blob in enumerate(step['blobs']):
            x, y = bxy(t, i)
            if i in static_flags[t]:
                s = max(4, DOT_R)
                cv2.line(canvas, (x-s, y-s), (x+s, y+s), (210, 50, 50), lw * 2)
                cv2.line(canvas, (x-s, y+s), (x+s, y-s), (210, 50, 50), lw * 2)
            elif not blob.get('passing', True):
                pass   # Rejected (failed area/aspect/top_k) — hidden so display == DP input
            elif (t, i) in path_set:
                cv2.circle(canvas, (x, y), DOT_R, col, -1)
                cv2.circle(canvas, (x, y), DOT_R + 2, (255, 230, 0), max(1, lw))
            else:
                cv2.circle(canvas, (x, y), DOT_R, col, -1)

    return canvas

def make_kalman_image(vname, center_frame, proc_w, history, gate_px=40.0):
    """Render Kalman track history overlaid on current frame.

    history: list of per-frame dicts (oldest first), each with keys:
      pred, pos, accepted, confidence, blobs, px_std, py_std, vx, vy,
      hit_count, miss_count.
    Colours: viridis purple (oldest) → yellow (newest).
    Gate circle = predicted gating radius.
    Uncertainty ellipse = 1-sigma from filter covariance diagonal.
    Velocity arrow = predicted motion vector (scaled 3×).
    Confidence bar across bottom.
    """
    if proc_w == 0:
        cap = get_cap(vname)
        proc_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        proc_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        proc_h = proc_w * 9 // 16

    g_curr, _ = read_gray_small(vname, center_frame, proc_w)
    if g_curr is None:
        canvas = np.zeros((proc_h, proc_w, 3), np.uint8)
    else:
        dark = (g_curr.astype(np.float32) * 0.35).astype(np.uint8)
        canvas = cv2.cvtColor(dark, cv2.COLOR_GRAY2RGB)

    disp_scale = get_display_scale(proc_w)
    h, w = canvas.shape[:2]
    canvas = cv2.resize(canvas, (w * disp_scale, h * disp_scale),
                        interpolation=cv2.INTER_NEAREST)

    n = len(history)
    if n == 0:
        return canvas

    lw    = max(1, disp_scale)
    DOT_R = max(2, disp_scale * 2)

    def sc(x, y):
        return (int(x * disp_scale), int(y * disp_scale))

    def vcol(t, n_t):
        val = int(255 * t / max(n_t - 1, 1))
        lut = np.array([[val]], dtype=np.uint8)
        bgr = cv2.applyColorMap(lut, cv2.COLORMAP_VIRIDIS)[0, 0]
        return (int(bgr[2]), int(bgr[1]), int(bgr[0]))

    # Grey lines between consecutive Kalman positions
    prev_p = None
    for h_ in history:
        p = h_["pos"]
        if p and prev_p:
            cv2.line(canvas, sc(*prev_p), sc(*p), (55, 55, 55), lw)
        if p:
            prev_p = p

    # Per-frame dots
    for t, h_ in enumerate(history):
        pos = h_["pos"]
        if pos is None:
            continue
        col = vcol(t, n)
        px, py = sc(*pos)
        if h_["accepted"] is not None:
            cv2.circle(canvas, (px, py), DOT_R, col, -1)
            # Green ring at measurement location
            ax, ay = sc(h_["accepted"]["x"], h_["accepted"]["y"])
            cv2.circle(canvas, (ax, ay), DOT_R + 2, (0, 220, 80), lw)
        else:
            # Miss: dim dot
            dim = (col[0] // 3, col[1] // 3, col[2] // 3)
            cv2.circle(canvas, (px, py), max(1, DOT_R - 1), dim, -1)
            # Small red X for missed frame
            s = max(2, DOT_R - 1)
            cv2.line(canvas, (px - s, py - s), (px + s, py + s), (180, 50, 50), lw)
            cv2.line(canvas, (px - s, py + s), (px + s, py - s), (180, 50, 50), lw)

    # Current (last) frame: gate circle, uncertainty ellipse, velocity arrow
    last = history[-1]
    if last["pred"] is not None:
        pred_x, pred_y = sc(*last["pred"])
        gate_r = max(4, int(gate_px * disp_scale))
        # Gate circle: thin blue ring
        cv2.circle(canvas, (pred_x, pred_y), gate_r, (80, 100, 220), lw)
        # Uncertainty ellipse (1-sigma)
        ex = max(1, int(last["px_std"] * disp_scale * 2))
        ey = max(1, int(last["py_std"] * disp_scale * 2))
        if ex > 1 or ey > 1:
            cv2.ellipse(canvas, (pred_x, pred_y), (ex, ey),
                        0, 0, 360, (180, 180, 60), lw)
        # Velocity arrow (3× scale)
        vx, vy = last["vx"], last["vy"]
        spd = (vx ** 2 + vy ** 2) ** 0.5
        if spd > 0.5:
            tip = sc(last["pred"][0] + vx * 3, last["pred"][1] + vy * 3)
            cv2.arrowedLine(canvas, (pred_x, pred_y), tip,
                            (255, 210, 0), lw * 2, tipLength=0.3)

    # Grey circles for all detected blobs at current frame (not accepted ones)
    acc = last.get("accepted")
    for b in last.get("blobs", []):
        bx, by = sc(b["x"], b["y"])
        if acc is None or abs(b["x"] - acc["x"]) > 0.5 or abs(b["y"] - acc["y"]) > 0.5:
            cv2.circle(canvas, (bx, by), DOT_R, (120, 120, 120), lw)

    # Confidence bar at bottom
    conf = last.get("confidence", 0.0)
    bar_h = max(4, disp_scale * 3)
    bar_w = int(canvas.shape[1] * conf)
    col_conf = vcol(n - 1, n)
    cv2.rectangle(canvas,
                  (0, canvas.shape[0] - bar_h),
                  (max(1, bar_w), canvas.shape[0]),
                  col_conf, -1)

    return canvas


def make_history_scatter(vname, center_frame, proc_w, diff_mode="abs",
                         n_hist=50, top_k=5,
                         thresh=10, min_a=1, max_a=400, max_asp=6.0,
                         method="compactness", ball_diam=10.0,
                         min_circ=0.1, min_bright=0.0, blur_k=0):
    """Notebook-style scatter: top-k passing blobs from n_hist consecutive diffs.

    k=0 newest diff (frame vs frame-1), k=n_hist-1 oldest.
    Viridis: k=n_hist-1(oldest)→purple, k=0(newest)→yellow.
    No DP, no edges — pure position scatter for quick trajectory reading.
    Returns (canvas_RGB, n_layers_collected).
    """
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
        gray = cv2.cvtColor(bgr_curr, cv2.COLOR_BGR2GRAY)
        dark = (gray.astype(np.float32) * 0.3).astype(np.uint8)
        canvas = cv2.cvtColor(dark, cv2.COLOR_GRAY2RGB)

    disp_scale = get_display_scale(proc_w)
    h, w = canvas.shape[:2]
    canvas = cv2.resize(canvas, (w * disp_scale, h * disp_scale),
                        interpolation=cv2.INTER_NEAREST)

    DOT_R = max(2, disp_scale * 2)   # small fixed dot, matches notebook scatter style

    # Collect candidates for each diff layer
    all_layers = []
    for k in range(n_hist):
        fb = center_frame - k    # newer frame of pair
        fa = fb - 1              # older frame of pair
        if fa < 0:
            break
        g_b, _ = read_gray_small(vname, fb, proc_w)
        g_a, _ = read_gray_small(vname, fa, proc_w)
        if g_a is None or g_b is None:
            break
        _, _, _, passing, _ = detect(
            g_b, g_a, thresh, min_a, max_a, max_asp, method, 0.0,
            prior_map=None, proc_w=proc_w, proc_h=proc_h,
            ball_diam=ball_diam, min_circ=min_circ,
            min_bright=min_bright, blur_k=blur_k, diff_mode=diff_mode)
        by_score = sorted(passing, key=lambda b: b['score'], reverse=True)[:top_k]
        all_layers.append((k, by_score))

    n_l = len(all_layers)

    # Draw oldest first (k = n_l-1 → purple) so newest (k=0 → yellow) is on top
    for k, blobs in reversed(all_layers):
        # k=n_l-1 → val=0 (purple); k=0 → val=255 (yellow)
        val = int(255 * (n_l - 1 - k) / max(n_l - 1, 1))
        lut = np.array([[val]], dtype=np.uint8)
        bgr = cv2.applyColorMap(lut, cv2.COLORMAP_VIRIDIS)[0, 0]
        col = (int(bgr[2]), int(bgr[1]), int(bgr[0]))   # BGR → RGB
        for b in blobs:
            x = int(b['x'] * disp_scale)
            y = int(b['y'] * disp_scale)
            cv2.circle(canvas, (x, y), DOT_R, col, -1)

    return canvas, n_l


# ── Track ────────────────────────────────────────────────────────────────────

def collect_track_blobs(vname, center_frame, proc_w, n_look=12, top_k=3,
                        thresh=10, min_a=1, max_a=400, max_asp=6.0,
                        method="compactness", ball_diam=10.0,
                        min_circ=0.1, min_bright=0.0, blur_k=0, score_thresh=0.0,
                        prior_map=None, diff_mode="abs"):
    """Collect top-k passing blobs from consecutive diffs in a symmetric window.

    For each offset delta in [-n_look, ..., +n_look]:
        diff = absdiff(frame center+delta-1, frame center+delta)
    Returns list of layer dicts, oldest first:
        {'rel': int, 'frame_idx': int, 'blobs': [...]}
    rel=0 is the diff whose newer frame is center_frame.
    """
    if proc_w == 0:
        cap = get_cap(vname)
        proc_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        proc_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        proc_h = proc_w * 9 // 16

    cap     = get_cap(vname)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    layers = []
    for delta in range(-n_look, n_look + 1):
        fb = center_frame + delta   # newer frame of the diff pair
        fa = fb - 1
        if fa < 0 or fb >= n_total:
            continue
        g_b, _ = read_gray_small(vname, fb, proc_w)
        g_a, _ = read_gray_small(vname, fa, proc_w)
        if g_a is None or g_b is None:
            continue
        # Same pipeline: detection filters → prior → score_thresh, all inside detect().
        _, _, _, passing, _ = detect(
            g_b, g_a, thresh, min_a, max_a, max_asp, method, score_thresh,
            prior_map=prior_map, proc_w=proc_w, proc_h=proc_h,
            ball_diam=ball_diam, min_circ=min_circ,
            min_bright=min_bright, blur_k=blur_k, diff_mode=diff_mode)
        top = sorted(passing, key=lambda b: b['score'], reverse=True)[:top_k]
        for b in top:
            b['passing'] = True
        layers.append({'rel': delta, 'frame_idx': fb, 'blobs': top})
    return layers


def parabolic_quality(layers, best_path):
    """R² of a least-squares parabola fit through best_path positions (0–1).

    High value (>0.9) means the path looks like a real ball arc.
    Low value means scattered / straight-line / noisy trajectory.
    """
    if len(best_path) < 4:
        return 0.0
    xs = np.array([layers[t]['blobs'][i]['x'] for (t, i) in best_path], float)
    ys = np.array([layers[t]['blobs'][i]['y'] for (t, i) in best_path], float)
    try:
        coeffs  = np.polyfit(xs, ys, 2)
        y_pred  = np.polyval(coeffs, xs)
        ss_res  = float(np.sum((ys - y_pred) ** 2))
        ss_tot  = float(np.sum((ys - ys.mean()) ** 2))
        r2      = 1.0 - ss_res / max(ss_tot, 1e-6)
        return round(max(0.0, min(1.0, r2)), 3)
    except Exception:
        return 0.0


# ── RANSAC arc finder ─────────────────────────────────────────────────────────

def _null_arc():
    return dict(Ax=None, Bx=None, Ay=None, By=None, Cy=None,
                inlier_blobs=[], r2=0.0, coverage=0.0,
                n_inliers=0, n_steps=0, confidence=0.0, ball_at_0=None)


def find_ransac_arc(layers, n_iter=400, inlier_px=8.0, min_inliers=4, min_span=4,
                    min_speed=6.0):
    """RANSAC parabolic arc finder.  Ignores per-blob scores entirely and
    instead finds the set of blobs that are globally consistent with a
    physics-plausible trajectory — solving the problem that small ball blobs
    score lower than large player blobs under compactness scoring.

    Physical model parameterised by t = layer['rel'] (frame offset):
        x(t) = Ax*t + Bx          (constant horizontal velocity in image space)
        y(t) = Ay*t² + By*t + Cy  (parabolic — gravity acts on image-y)

    With weight=1.0 prior and prior ON, only court/air-zone blobs reach this
    function, so we don't need any spatial re-filtering here.

    Args:
        layers    : from collect_track_blobs, each entry has 'rel' and 'blobs'
        n_iter    : RANSAC iterations (400 is fast, <5ms for typical inputs)
        inlier_px : max 2-D distance (px) from predicted position → inlier
        min_inliers: minimum inlier blob count to accept an arc
        min_span  : minimum frame span between the 3 sampled points

    Returns dict:
        Ax,Bx,Ay,By,Cy  arc params (all None if no arc found)
        inlier_blobs     list of (t_float, blob_dict)
        r2               R² of y(t) fit on inliers  [0,1]
        coverage         fraction of time steps that have ≥1 inlier  [0,1]
        confidence       r2 × coverage — the key "ball in play" signal
        ball_at_0        (x,y) predicted at t=0 (current frame), or None
    """
    # ── Build flat candidate list ────────────────────────────────────────────
    all_pts = []          # (t, x, y, blob)
    by_t    = {}          # t → [(t,x,y,blob)]
    for layer in layers:
        t = float(layer['rel'])
        for b in layer['blobs']:
            pt = (t, float(b['x']), float(b['y']), b)
            all_pts.append(pt)
            by_t.setdefault(t, []).append(pt)

    tsteps   = sorted(by_t.keys())
    n_tsteps = len(tsteps)
    if len(all_pts) < min_inliers or n_tsteps < 3:
        return _null_arc()

    rng        = np.random.default_rng(0)
    inlier_px2 = inlier_px ** 2
    best_inliers: list = []
    best_params          = None

    for _ in range(n_iter):
        # Sample 3 DISTINCT time steps with guaranteed minimum span
        idxs = sorted(rng.choice(n_tsteps, size=3, replace=False))
        if tsteps[idxs[-1]] - tsteps[idxs[0]] < min_span:
            continue

        # One random blob per sampled time step
        pts = [by_t[tsteps[i]][rng.integers(len(by_t[tsteps[i]]))]
               for i in idxs]

        ts = np.array([p[0] for p in pts], np.float64)
        xs = np.array([p[1] for p in pts], np.float64)
        ys = np.array([p[2] for p in pts], np.float64)

        # Fit x(t) = Ax*t + Bx  (linear, 2 params, 3 equations → least squares)
        try:
            Ax, Bx = np.polyfit(ts, xs, 1)
        except Exception:
            continue

        # Fit y(t) = Ay*t² + By*t + Cy  (solve exactly from 3 points)
        Vy = np.column_stack([ts * ts, ts, np.ones(3)])
        try:
            if abs(np.linalg.det(Vy)) < 1e-4:
                continue   # near-degenerate (times too close)
            Ay, By, Cy = np.linalg.solve(Vy, ys)
        except Exception:
            continue

        # Gravity sanity check: Ay > 0 means ball curves downward in image
        # (y increases downward in image coords, so gravity → Ay > 0).
        # Accept both signs — camera angle can invert this; don't hard-gate.

        # Count inliers
        inliers = []
        for (t, x, y, b) in all_pts:
            dx = x - (Ax * t + Bx)
            dy = y - (Ay * t * t + By * t + Cy)
            if dx * dx + dy * dy <= inlier_px2:
                inliers.append((t, x, y, b))

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_params  = (Ax, Bx, Ay, By, Cy)

    if best_params is None or len(best_inliers) < min_inliers:
        return _null_arc()

    Ax, Bx, Ay, By, Cy = best_params

    # ── Weighted refinement on all inliers ──────────────────────────────────
    ts_i = np.array([p[0] for p in best_inliers], np.float64)
    xs_i = np.array([p[1] for p in best_inliers], np.float64)
    ys_i = np.array([p[2] for p in best_inliers], np.float64)
    ws_i = np.array([max(p[3].get('score', 1e-3), 1e-3) for p in best_inliers], np.float64)

    try:
        Ax, Bx = np.polyfit(ts_i, xs_i, 1, w=ws_i)
        Ay, By, Cy = np.polyfit(ts_i, ys_i, 2, w=ws_i)
    except Exception:
        pass  # keep RANSAC fit if refinement fails

    # Re-evaluate inliers with refined model
    final_inliers: list = []
    for (t, x, y, b) in all_pts:
        dx = x - (Ax * t + Bx)
        dy = y - (Ay * t * t + By * t + Cy)
        if dx * dx + dy * dy <= inlier_px2:
            final_inliers.append((t, b))

    if len(final_inliers) < min_inliers:
        return _null_arc()

    # ── Quality metrics ──────────────────────────────────────────────────────
    ts_f = np.array([p[0] for p in final_inliers], np.float64)
    ys_f = np.array([p[1]['y'] for p in final_inliers], np.float64)
    y_hat = Ay * ts_f * ts_f + By * ts_f + Cy
    ss_res = float(np.sum((ys_f - y_hat) ** 2))
    ss_tot = float(np.sum((ys_f - ys_f.mean()) ** 2))
    r2     = float(max(0.0, min(1.0, 1.0 - ss_res / max(ss_tot, 1e-6))))

    # Coverage: fraction of time steps with at least one inlier
    inlier_ts  = set(p[0] for p in final_inliers)
    coverage   = len(inlier_ts) / max(n_tsteps, 1)

    # Composite confidence — the "ball in play" signal
    confidence = r2 * coverage

    # Speed check: reject arcs slower than min_speed px/frame (filters player movement).
    # speed = sqrt(Ax^2 + By^2) — the velocity magnitude at t=0.
    arc_speed = (float(Ax) ** 2 + float(By) ** 2) ** 0.5
    if arc_speed < min_speed:
        return _null_arc()

    # Predicted ball position at the current frame (t=0).
    # Only emit ball_at_0 if an inlier blob exists at |t| <= 1 — no extrapolation
    # to frames with zero detections (avoids phantom rings on player-only frames).
    has_inlier_at_0 = any(abs(t) <= 1 for (t, _) in final_inliers)
    ball_at_0 = (float(Bx), float(Cy)) if (confidence > 0.1 and has_inlier_at_0) else None

    return dict(
        Ax=float(Ax), Bx=float(Bx),
        Ay=float(Ay), By=float(By), Cy=float(Cy),
        inlier_blobs=final_inliers,
        r2=round(r2, 3),
        coverage=round(coverage, 3),
        n_inliers=len(final_inliers),
        n_steps=n_tsteps,
        confidence=round(confidence, 3),
        ball_at_0=ball_at_0,
    )


def make_track_image(vname, center_frame, proc_w, layers, best_path, arc_result=None):
    """Render the full-frame track view: current frame + trajectory arc.

    • Background: actual frame at 55% brightness (grayscale).
    • All candidate blobs: tiny dim dots.
    • Best DP path: viridis-colored dots (purple=oldest, yellow=newest)
      connected by a smooth polyline; rel=0 node gets a white ring.
    """
    if proc_w == 0:
        cap = get_cap(vname)
        proc_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        proc_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        proc_h = proc_w * 9 // 16

    _, bgr = read_gray_small(vname, center_frame, proc_w)
    if bgr is None:
        canvas = np.zeros((proc_h, proc_w, 3), np.uint8)
    else:
        gray   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        dark   = (gray.astype(np.float32) * 0.55).astype(np.uint8)
        canvas = cv2.cvtColor(dark, cv2.COLOR_GRAY2RGB)

    disp_scale = get_display_scale(proc_w)
    h, w = canvas.shape[:2]
    canvas = cv2.resize(canvas, (w * disp_scale, h * disp_scale),
                        interpolation=cv2.INTER_NEAREST)

    DOT_R    = max(3, disp_scale * 2)
    lw       = max(1, disp_scale)
    n_l      = len(layers)
    path_set = set(best_path)

    def xy(t, i):
        b = layers[t]['blobs'][i]
        return (int(b['x'] * disp_scale), int(b['y'] * disp_scale))

    def vcol(t):
        # BGR tuple — correct for OpenCV drawing on BGR canvas
        val = int(255 * t / max(n_l - 1, 1))
        bgr_c = cv2.applyColorMap(np.array([[val]], np.uint8), cv2.COLORMAP_VIRIDIS)[0, 0]
        return (int(bgr_c[0]), int(bgr_c[1]), int(bgr_c[2]))

    # 1. Non-path candidates: dim when DP path exists, full brightness in scatter mode
    _scatter_mode = len(best_path) == 0
    for t, layer in enumerate(layers):
        col_f = vcol(t)
        draw_col = col_f if _scatter_mode else (col_f[0]//3, col_f[1]//3, col_f[2]//3)
        dot_r    = max(2, DOT_R - 1)
        for i, b in enumerate(layer['blobs']):
            if (t, i) not in path_set:
                x = int(b['x'] * disp_scale)
                y = int(b['y'] * disp_scale)
                cv2.circle(canvas, (x, y), dot_r, draw_col, -1)

    # 2. Trajectory arc line (path edges) — suppressed when RANSAC is confident
    _ransac_conf = (arc_result or {}).get("confidence", 0.0) or 0.0
    if _ransac_conf < 0.3:
        for k in range(len(best_path) - 1):
            t0, i0 = best_path[k]
            t1, i1 = best_path[k + 1]
            col = vcol(t0)
            cv2.line(canvas, xy(t0, i0), xy(t1, i1), col, lw * 2 + 1)

    # 3. Path nodes colored by time — larger + bright border so they pop
    for (t, i) in best_path:
        col     = vcol(t)
        is_now  = (layers[t]['rel'] == 0)
        px, py  = xy(t, i)
        r = DOT_R + 2 + (2 if is_now else 0)
        cv2.circle(canvas, (px, py), r + 1, (200, 200, 200), 1)  # bright border
        cv2.circle(canvas, (px, py), r, col, -1)
        if is_now:
            cv2.circle(canvas, (px, py), r + 5, (255, 255, 255), lw * 2)

    # ── 4. RANSAC arc overlay ────────────────────────────────────────────────
    # Drawn last so it appears on top of everything.
    # Color brightens with confidence: dim white at 0.1, full cyan-white at 1.0.
    if arc_result and arc_result.get('Ay') is not None:
        conf = arc_result['confidence']
        if conf > 0.05:
            Ax_ = arc_result['Ax'];  Bx_ = arc_result['Bx']
            Ay_ = arc_result['Ay'];  By_ = arc_result['By'];  Cy_ = arc_result['Cy']

            inlier_ts = [t for t, _ in arc_result['inlier_blobs']]
            if len(inlier_ts) >= 2:
                t_lo, t_hi = min(inlier_ts), max(inlier_ts)
                # Extend slightly beyond inliers to show prediction
                t_lo -= 1;  t_hi += 1
                arc_pts = []
                for t_val in np.linspace(t_lo, t_hi, 100):
                    px_arc = int((Ax_ * t_val + Bx_) * disp_scale)
                    py_arc = int((Ay_ * t_val * t_val + By_ * t_val + Cy_) * disp_scale)
                    h_, w_ = canvas.shape[:2]
                    if 0 <= px_arc < w_ and 0 <= py_arc < h_:
                        arc_pts.append((px_arc, py_arc))

                # Arc brightness scales with confidence
                brightness = int(min(255, 80 + conf * 175))
                arc_col = (brightness, brightness, 255)   # blue-white → pure white
                arc_lw  = max(1, lw + 1)
                for k in range(len(arc_pts) - 1):
                    cv2.line(canvas, arc_pts[k], arc_pts[k + 1], arc_col, arc_lw)

            # Predicted ball position at t=0 (current frame)
            if arc_result.get('ball_at_0'):
                bx0 = int(arc_result['ball_at_0'][0] * disp_scale)
                by0 = int(arc_result['ball_at_0'][1] * disp_scale)
                h_, w_ = canvas.shape[:2]
                if 0 <= bx0 < w_ and 0 <= by0 < h_:
                    ring_col = (255, 255, 255)   # white marker
                    for r_ring in (DOT_R + 4, DOT_R + 8, DOT_R + 13):
                        cv2.circle(canvas, (bx0, by0), r_ring, ring_col, lw)
                    cv2.line(canvas, (bx0 - 16, by0), (bx0 + 16, by0), ring_col, lw)
                    cv2.line(canvas, (bx0, by0 - 16), (bx0, by0 + 16), ring_col, lw)

            # Draw RANSAC inlier blobs: bright cyan outlines (distinct from DP path)
            for (t_in, b_in) in arc_result['inlier_blobs']:
                xi = int(b_in['x'] * disp_scale)
                yi = int(b_in['y'] * disp_scale)
                cv2.circle(canvas, (xi, yi), DOT_R + 2, (255, 220, 0), lw)

    return canvas



# ── export (confidence scan + segment writer) ─────────────────────────────────
_export_state: dict = {
    "running": False, "phase": "", "progress": 0, "total": 0,
    "output": "", "output2": "", "error": "",
    "n_ball": 0, "n_dead": 0,
}


# ── Kalman ball tracker ───────────────────────────────────────────────────────
class KalmanBallTracker:
    """Constant-velocity Kalman filter for ball tracking.

    State  x = [px, py, vx, vy]
    Obs    z = [px, py]          (blob centroid)

    Much faster than RANSAC for full-video export because it processes
    each frame exactly once in a single sequential pass.
    """
    def __init__(self, process_noise=5.0, measurement_noise=8.0,
                 gate_px=40.0, max_miss=8):
        dt = 1.0
        self.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], np.float64)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], np.float64)
        q = process_noise ** 2
        r = measurement_noise ** 2
        self.Q = np.diag([q*0.25, q*0.25, q, q])   # position less noisy than vel
        self.R = np.eye(2) * r
        self.gate_px   = gate_px
        self.max_miss  = max_miss
        self._reset()

    def _reset(self):
        self.x          = None
        self.P          = None
        self.hit_count  = 0
        self.miss_count = 0
        self.confidence = 0.0

    def init(self, px, py, vx=0.0, vy=0.0):
        self.x = np.array([px, py, vx, vy], np.float64)
        self.P = np.eye(4) * 200.0
        self.hit_count  = 1
        self.miss_count = 0
        self.confidence = 0.0

    def predict(self):
        if self.x is None:
            return
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, meas):
        """meas = [px, py].  Returns True if accepted."""
        z = np.asarray(meas, np.float64)
        if self.x is None:
            self.init(z[0], z[1])
            return True

        innov = z - self.H @ self.x
        dist  = float(np.sqrt(innov @ innov))

        # Gate: ignore blobs too far from prediction once track is established
        if dist > self.gate_px and self.hit_count >= 4:
            return False

        S      = self.H @ self.P @ self.H.T + self.R
        K      = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        self.P = (np.eye(4) - K @ self.H) @ self.P

        self.hit_count  += 1
        self.miss_count  = 0

        # Confidence: innovation proximity × track maturity
        innov_sigma = float(np.sqrt(max(np.linalg.det(S), 1e-6)))
        prox_conf   = max(0.0, 1.0 - dist / max(innov_sigma, 1.0) / 3.0)
        age_conf    = min(1.0, self.hit_count / 10.0)
        self.confidence = round(prox_conf * age_conf, 3)
        return True

    def miss(self):
        self.miss_count += 1
        self.confidence  = round(self.confidence * 0.65, 3)
        if self.miss_count > self.max_miss:
            self._reset()

    def position(self):
        return (float(self.x[0]), float(self.x[1])) if self.x is not None else None

    def velocity(self):
        return (float(self.x[2]), float(self.x[3])) if self.x is not None else (0.0, 0.0)


def scan_confidence_kalman(video, proc_w, thresh, min_a, max_a, max_asp,
                            method, ball_diam, min_circ, min_bright, blur_k,
                            score_thresh, prior_map,
                            kf_proc_noise=5.0, kf_meas_noise=8.0,
                            kf_gate_px=40.0, kf_max_miss=8,
                            total_frames_limit=None, progress_cb=None,
                            diff_mode="abs"):
    """Single sequential pass — reads each frame once.
    Returns list of (frame_idx, confidence) pairs covering [0, total_frames_limit).
    """
    proc_h  = proc_w * 9 // 16
    cap     = cv2.VideoCapture(str(VIDEO_DIR / video))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames_limit:
        total = min(total, int(total_frames_limit))

    tracker   = KalmanBallTracker(kf_proc_noise, kf_meas_noise, kf_gate_px, kf_max_miss)
    results   = []
    prev_gray = None

    for fi in range(total):
        if progress_cb and fi % 50 == 0:
            progress_cb(fi, total)

        ok, bgr = cap.read()
        if not ok or bgr is None:
            results.append((fi, 0.0))
            prev_gray = None
            continue

        bgr_s = cv2.resize(bgr, (proc_w, proc_h))
        gray  = cv2.cvtColor(bgr_s, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            _, _, _, passing, _ = detect(
                gray, prev_gray, thresh, min_a, max_a, max_asp,
                method, score_thresh,
                prior_map=prior_map, proc_w=proc_w, proc_h=proc_h,
                ball_diam=ball_diam, min_circ=min_circ, diff_mode=diff_mode,
                min_bright=min_bright, blur_k=blur_k,
            )
            tracker.predict()
            if passing:
                pred = tracker.position()
                if pred is not None:
                    dists  = [((b['x']-pred[0])**2+(b['y']-pred[1])**2)**0.5
                               for b in passing]
                    best   = passing[int(np.argmin(dists))]
                else:
                    best = max(passing, key=lambda b: b['score'])
                if not tracker.update([best['x'], best['y']]):
                    tracker.miss()
            else:
                tracker.miss()

        results.append((fi, tracker.confidence))
        prev_gray = gray

    cap.release()
    return results


def scan_and_export(video, res_w, n_look, top_k, thresh, min_a, max_a, max_asp,
                    method, ball_diam, min_circ, min_bright, blur_k, score_thresh,
                    use_prior, court_xs, court_ys, air_xl, air_xr, air_yt, air_yb,
                    air_sx, air_sy, pweight, ransac_px, ransac_spd,
                    conf_thresh, max_minutes, min_seg_sec,
                    mode, out_dir, tracker_mode="ransac",
                    kf_proc_noise="5.0", kf_meas_noise="8.0",
                    kf_gate_px="40.0", kf_max_miss="8", **_kwargs):
    """Background job: scan confidence timeline then write ball / dead-time MP4."""
    global _export_state
    import threading
    _export_state.update(running=True, phase="scanning", progress=0, total=0,
                         output="", output2="", error="", n_ball=0, n_dead=0)
    try:
        proc_w    = int(res_w)
        proc_h    = proc_w * 9 // 16
        n_look_   = int(n_look)
        step      = max(1, n_look_ // 2)          # scan stride
        inlier_px = float(ransac_px)
        min_spd   = float(ransac_spd)
        conf_thr  = float(conf_thresh)
        min_seg_f = int(float(min_seg_sec) * 25)  # approx frames

        prior_map = compute_prior_map(
            court_x_sigma = float(court_xs),
            court_y_sigma = float(court_ys),
            air_x_left    = int(air_xl),
            air_x_right   = int(air_xr),
            air_y_top     = int(air_yt),
            air_y_bot     = int(air_yb),
            air_sigma_x   = float(air_sx),
            air_sigma_y   = float(air_sy),
            weight        = float(pweight),
        ) if use_prior else None

        # Open a private cap
        cap = cv2.VideoCapture(str(VIDEO_DIR / video))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {video}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0

        if float(max_minutes) > 0:
            limit = int(float(max_minutes) * 60 * fps)
            total_frames = min(total_frames, limit)

        # ── Confidence scan: Kalman (fast, sequential) or RANSAC (window) ──
        tracker_mode_ = str(tracker_mode).lower()

        if tracker_mode_ == "kalman":
            # Single sequential pass — reads each frame once (~25x faster)
            _export_state.update(phase="scanning (Kalman)", total=total_frames)
            def _kalman_cb(fi, tot):
                _export_state["progress"] = fi
            pairs = scan_confidence_kalman(
                video=video, proc_w=proc_w,
                thresh=int(thresh), min_a=int(min_a), max_a=int(max_a),
                max_asp=float(max_asp), method=method,
                ball_diam=float(ball_diam), min_circ=float(min_circ),
                min_bright=float(min_bright), blur_k=int(blur_k),
                score_thresh=float(score_thresh), prior_map=prior_map,
                kf_proc_noise=float(kf_proc_noise),
                kf_meas_noise=float(kf_meas_noise),
                kf_gate_px=float(kf_gate_px),
                kf_max_miss=int(kf_max_miss),
                total_frames_limit=total_frames,
                progress_cb=_kalman_cb,
            )
            cap.release()
            conf_full = np.zeros(total_frames, np.float32)
            for fi, cv in pairs:
                if fi < total_frames:
                    conf_full[fi] = cv

        else:
            # RANSAC window scan (original approach)
            scan_frames = list(range(n_look_, total_frames - n_look_, step))
            _export_state["total"] = len(scan_frames)
            conf_sparse = {}
            for idx, cf in enumerate(scan_frames):
                _export_state["progress"] = idx
                layers = collect_track_blobs(
                    vname=video, center_frame=cf, proc_w=proc_w,
                    n_look=n_look_, top_k=int(top_k),
                    thresh=int(thresh), min_a=int(min_a), max_a=int(max_a),
                    max_asp=float(max_asp), method=method,
                    ball_diam=float(ball_diam), min_circ=float(min_circ),
                    min_bright=float(min_bright), blur_k=int(blur_k),
                    score_thresh=float(score_thresh), prior_map=prior_map,
                )
                arc = find_ransac_arc(layers, n_iter=200,
                                      inlier_px=inlier_px, min_speed=min_spd)
                conf_sparse[cf] = arc["confidence"]
                if _export_state.get("cancel"):
                    cap.release(); return
            cap.release()
            if not conf_sparse:
                _export_state.update(running=False, error="No frames scanned")
                return
            sorted_keys = sorted(conf_sparse)
            conf_full = np.zeros(total_frames, np.float32)
            for fk in sorted_keys:
                conf_full[fk] = conf_sparse[fk]
            prev_k = sorted_keys[0]
            for k in sorted_keys[1:]:
                v0, v1 = conf_full[prev_k], conf_full[k]
                span   = k - prev_k
                for j in range(span + 1):
                    conf_full[prev_k + j] = v0 + (v1-v0)*j/max(span,1)
                prev_k = k
            conf_full[:sorted_keys[0]]  = conf_full[sorted_keys[0]]
            conf_full[sorted_keys[-1]:] = conf_full[sorted_keys[-1]]

        # Boolean mask per frame
        is_ball = conf_full > conf_thr

        # Build contiguous segments; merge gaps < step
        def extract_segments(mask, total):
            segs = []
            in_seg = False
            st = 0
            for i in range(total):
                if mask[i] and not in_seg:
                    st = i; in_seg = True
                elif not mask[i] and in_seg:
                    segs.append((st, i - 1)); in_seg = False
            if in_seg:
                segs.append((st, total - 1))
            # Merge small gaps
            merged = []
            for seg in segs:
                if merged and seg[0] - merged[-1][1] <= step:
                    merged[-1] = (merged[-1][0], seg[1])
                else:
                    merged.append(list(seg))
            # Filter short segments
            return [s for s in merged if s[1] - s[0] + 1 >= min_seg_f]

        ball_segs = extract_segments(is_ball,  total_frames)
        dead_segs = extract_segments(~is_ball, total_frames)

        n_ball_seg = len(ball_segs)
        n_dead_seg = len(dead_segs)
        _export_state.update(n_ball=n_ball_seg, n_dead=n_dead_seg)

        def write_segments(segs, out_path, label):
            if not segs:
                return ""
            _export_state.update(phase=f"writing {label}", progress=0, total=len(segs))
            # Use AVI+XVID: avoids libavcodec async-lock assertion that mp4v/H264
            # triggers when VideoCapture is opened in a background thread.
            out_path = out_path.replace(".mp4", ".avi")
            cap2 = cv2.VideoCapture(str(VIDEO_DIR / video))
            cap2.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # disable read-ahead buffering
            fw   = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh   = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (fw, fh))
            for si, (fa, fb) in enumerate(segs):
                _export_state["progress"] = si
                cap2.set(cv2.CAP_PROP_POS_FRAMES, fa)
                for fi in range(fa, fb + 1):
                    ok, frame = cap2.read()
                    if not ok:
                        break
                    writer.write(frame)
                if _export_state.get("cancel"):
                    break
            cap2.release(); writer.release()
            return out_path

        stem   = video.rsplit(".", 1)[0]
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out1   = str(OUT_DIR / f"{stem}_ball.avi")
        out2   = str(OUT_DIR / f"{stem}_deadtime.avi")

        if mode in ("ball", "both"):
            write_segments(ball_segs, out1, "ball video")
            _export_state["output"] = out1 if ball_segs else ""
        if mode in ("dead", "both"):
            write_segments(dead_segs, out2, "dead-time video")
            _export_state["output2"] = out2 if dead_segs else ""

        import os as _os
        _export_state.update(
            running=False, phase="done", progress=_export_state["total"],
            basename1=_os.path.basename(_export_state.get("output","")) if _export_state.get("output") else "",
            basename2=_os.path.basename(_export_state.get("output2","")) if _export_state.get("output2") else "",
        )

    except Exception as exc:
        import traceback; traceback.print_exc()
        _export_state.update(running=False, error=str(exc))

# ── render state (shared between handler thread and render thread) ────────────
_render_state: dict = {
    "running": False, "progress": 0, "total": 0, "output": "", "filepath": "", "error": ""
}

def render_tracked_video(video, res_w="320", n_look="12", top_k="3",
                          link_r="40", vel_pen="0.03", static_r="8",
                          thresh="10", min_a="1", max_a="400",
                          max_asp="6.0", method="compactness",
                          ball_diam="10", min_circ="0.1", min_bright="0.0",
                          blur_k="0", score_thresh="0.0",
                          use_prior="0",
                          court_xs="0.1", court_ys="0", court_inset="0",
                          air_xl="115", air_xr="193", air_yt="38", air_yb="58",
                          air_sx="37", air_sy="40", pweight="1.0",
                          **_kwargs):
    """Render the full video with track overlay to MP4.

    Fully self-contained: opens its own private VideoCapture objects so it
    never races with the shared _caps used by the HTTP handler thread.
    """
    global _render_state
    _render_state = {"running": True, "progress": 0, "total": 0, "output": "", "filepath": "", "error": ""}
    try:
        proc_w     = int(res_w);    proc_h    = proc_w * 9 // 16
        n_look_    = int(n_look);   top_k_    = int(top_k)
        link_r_    = int(link_r);   static_r_ = int(static_r)
        vel_pen_   = float(vel_pen)
        thresh_    = int(thresh);   min_a_    = int(min_a)
        max_a_     = int(max_a);    max_asp_  = float(max_asp)
        ball_diam_ = float(ball_diam); min_circ_ = float(min_circ)
        min_bright_= float(min_bright); blur_k_  = int(blur_k)
        score_th_  = float(score_thresh)
        render_prior_map = compute_prior_map(
            court_x_sigma = float(court_xs),
            court_y_sigma = float(court_ys),
            court_inset   = int(court_inset),
            air_x_left    = int(air_xl),
            air_x_right   = int(air_xr),
            air_y_top     = int(air_yt),
            air_y_bot     = int(air_yb),
            air_sigma_x   = float(air_sx),
            air_sigma_y   = float(air_sy),
            weight        = float(pweight),
        ) if use_prior == "1" else None

        # Private caps — never share with the HTTP handler thread
        _pcaps = {}
        def _open(vname):
            p = str(VIDEO_DIR / vname)
            if p not in _pcaps:
                _pcaps[p] = cv2.VideoCapture(p)
            return _pcaps[p]

        def _read(vname, idx):
            cap_ = _open(vname)
            cap_.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap_.read()
            if not ok or bgr is None:
                return None, None
            bgr_s = cv2.resize(bgr, (proc_w, proc_h))
            return cv2.cvtColor(bgr_s, cv2.COLOR_BGR2GRAY), bgr_s

        cap0  = _open(video)
        total = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap0.get(cv2.CAP_PROP_FPS) or 25.0
        _render_state["total"] = total

        renders_dir = TENNIS / "claude" / "renders"
        renders_dir.mkdir(parents=True, exist_ok=True)
        out_path = renders_dir / (Path(video).stem + "_tracked.mp4")

        disp_scale = get_display_scale(proc_w)
        out_w = proc_w * disp_scale
        out_h = proc_h * disp_scale

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))

        def _vcol(t, n_l):
            val = int(255 * t / max(n_l - 1, 1))
            c = cv2.applyColorMap(np.array([[val]], np.uint8), cv2.COLORMAP_VIRIDIS)[0, 0]
            return (int(c[0]), int(c[1]), int(c[2]))  # BGR

        for fi in range(total):
            _render_state["progress"] = fi
            try:
                # Collect blobs for +-n_look window using private caps
                layers = []
                for delta in range(-n_look_, n_look_ + 1):
                    fb = fi + delta; fa = fb - 1
                    if fa < 0 or fb >= total:
                        continue
                    g_b, _ = _read(video, fb)
                    g_a, _ = _read(video, fa)
                    if g_b is None or g_a is None:
                        continue
                    _, _, _, passing, _ = detect(
                        g_b, g_a,
                        thresh=thresh_, min_a=min_a_, max_a=max_a_, max_asp=max_asp_,
                        method=method, score_thresh=score_th_,
                        prior_map=render_prior_map,
                        proc_w=proc_w, proc_h=proc_h,
                        ball_diam=ball_diam_, min_circ=min_circ_,
                        min_bright=min_bright_, blur_k=blur_k_,
                    )
                    top = passing[:top_k_]
                    layers.append({"rel": delta, "frame_idx": fb, "blobs": top})

                # Build canvas from current frame
                _, bgr_fi = _read(video, fi)
                if bgr_fi is not None:
                    canvas = cv2.resize(
                        (bgr_fi.astype(np.float32) * 0.65).astype(np.uint8),
                        (out_w, out_h), interpolation=cv2.INTER_NEAREST)
                else:
                    canvas = np.zeros((out_h, out_w, 3), np.uint8)

                if layers:
                    steps_nf = list(reversed(layers))
                    chron, _, _, best = build_path_dp(
                        steps_nf, link_radius=link_r_,
                        static_radius=static_r_, vel_penalty=vel_pen_)
                    n_l = len(chron)
                    path_set = set(best)
                    DOT_R = max(3, disp_scale * 2)
                    lw    = max(1, disp_scale)

                    def _xy(t, i):
                        b = chron[t]["blobs"][i]
                        return (int(b["x"] * disp_scale), int(b["y"] * disp_scale))

                    for t, step in enumerate(chron):
                        for i, b in enumerate(step["blobs"]):
                            if (t, i) not in path_set:
                                cv2.circle(canvas,
                                    (int(b["x"]*disp_scale), int(b["y"]*disp_scale)),
                                    max(2, DOT_R-1), (200, 200, 200), 1)

                    for k in range(len(best) - 1):
                        t0, i0 = best[k]; t1, i1 = best[k+1]
                        cv2.line(canvas, _xy(t0,i0), _xy(t1,i1), _vcol(t0,n_l), lw*2+1)

                    for t, i in best:
                        col = _vcol(t, n_l)
                        is_now = (chron[t]["rel"] == 0)
                        px, py = _xy(t, i)
                        r = DOT_R + 2 + (2 if is_now else 0)
                        cv2.circle(canvas, (px, py), r+1, (200,200,200), 1)
                        cv2.circle(canvas, (px, py), r, col, -1)
                        if is_now:
                            cv2.circle(canvas, (px, py), r+5, (255,255,255), lw*2)

                writer.write(canvas)
            except Exception:
                import traceback; traceback.print_exc()
                writer.write(np.zeros((out_h, out_w, 3), np.uint8))

        writer.release()
        for c in _pcaps.values(): c.release()
        _render_state["output"]   = str(out_path)
        _render_state["filepath"] = str(out_path)
    except Exception as exc:
        import traceback; traceback.print_exc()
        _render_state["error"] = str(exc)
    finally:
        _render_state["running"] = False


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
#tree-play-btn { background: #1a3a1a; color: #4f4; border: 1px solid #3a6a3a;
                 border-radius: 3px; padding: 3px 14px; cursor: pointer;
                 font-size: 16px; line-height: 1; min-width: 42px; text-align: center; }
#tree-play-btn:hover { background: #2a4a2a; }
#tree-play-btn.playing { background: #3a1a1a; color: #f66; border-color: #6a3a3a; }

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
    Spatial Prior <span id="prior-badge" class="on">ON</span>
  </div>
  <div class="tab"        id="tab-tree"   onclick="switchTab('tree')">
    Trajectory Tree
  </div>
  <div class="tab"        id="tab-track"  onclick="switchTab('track')">
    Track
  </div>
  <div style="margin-left:auto; display:flex; align-items:center; padding:0 8px; gap:6px">
    <button class="tog-btn" onclick="saveParams()" title="Save current slider values to params.json" style="padding:3px 10px; font-size:11px">&#128190; Save</button>
    <button class="tog-btn" onclick="loadParams()" title="Load slider values from params.json" style="padding:3px 10px; font-size:11px">&#128193; Load</button>
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
        <input type="range" id="thresh" min="1" max="120" value="18" oninput="ps('thresh')">
        <span class="val" id="thresh-v">18</span>
      </div>
    </div>

    <!-- min area -->
    <div class="group">
      <div class="group-title">Min area</div>
      <div class="row">
        <input type="range" id="min_a" min="1" max="500" value="3" oninput="ps('min_a')">
        <span class="val" id="min_a-v">3</span>
      </div>
    </div>

    <!-- max area -->
    <div class="group">
      <div class="group-title">Max area</div>
      <div class="row">
        <input type="range" id="max_a" min="5" max="4000" value="1551" oninput="ps('max_a')">
        <span class="val" id="max_a-v">16</span>
      </div>
    </div>

    <!-- max aspect -->
    <div class="group">
      <div class="group-title">Max aspect</div>
      <div class="row">
        <input type="range" id="max_asp" min="1" max="10" step="0.5" value="2.5"
               oninput="ps('max_asp')">
        <span class="val" id="max_asp-v">3.0</span>
      </div>
    </div>

    <!-- score threshold -->
    <div class="group">
      <div class="group-title">Score thresh</div>
      <div class="row">
        <input type="range" id="score_thresh" min="0" max="1.5" step="0.01" value="0"
               oninput="ps('score_thresh')">
        <span class="val" id="score_thresh-v">0</span>
      </div>
    </div>

    <!-- scoring method -->
    <div class="group">
      <div class="group-title">Scoring</div>
      <div class="row" style="gap:4px">
        <button class="tog-btn"    id="btn-c"    onclick="setMethod('compactness')">Compact</button>
        <button class="tog-btn"    id="btn-r"    onclick="setMethod('rog')">RoG</button>
        <button class="tog-btn on" id="btn-circ" onclick="setMethod('circularity')">Circ</button>
      </div>
    </div>

    <!-- pre-blur (useful for all methods, critical for circularity) -->
    <div class="group">
      <div class="group-title">Pre-blur</div>
      <div class="row">
        <input type="range" id="blur_k" min="0" max="9" value="6" step="1"
               oninput="ps('blur_k')">
        <span class="val" id="blur_k-v">6</span>
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
          <option value="320">320 (fast)</option>
          <option value="480">480</option>
          <option value="640">640</option>
          <option value="960">960</option>
          <option value="0" selected>Full (native)</option>
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

    <!-- diff layers -->
    <div class="group">
      <div class="group-title">Diff layers</div>
      <div class="row">
        <input type="range" id="n_diff" min="1" max="8" value="1" step="1"
               oninput="ps('n_diff')">
        <span class="val" id="n_diff-v">1</span>
      </div>
    </div>

    <!-- spatial prior -->
    <div class="group">
      <div class="group-title">Spatial prior</div>
      <div class="row">
        <button class="tog-btn on" id="btn-prior" onclick="togglePrior()">Prior: ON</button>
        <button class="tog-btn" id="btn-diff" onclick="toggleDiffMode()">Diff: Unsigned</button>
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
        <input type="range" id="court_xs" min="0" max="60" value="0.1" step="0.1"
               oninput="pp('court_xs')">
        <span class="val" id="court_xs-v">0.1</span>
      </div>
    </div>
    <div class="group">
      <div class="group-title" style="color:#8cf">Court sigma-below</div>
      <div class="row">
        <input type="range" id="court_ys" min="0" max="80" value="0" step="1"
               oninput="pp('court_ys')">
        <span class="val" id="court_ys-v">0</span>
      </div>
    </div>
    <div class="group">
      <div class="group-title" style="color:#8cf">Court inset (px)</div>
      <div class="row">
        <input type="range" id="court_inset" min="0" max="30" value="8" step="1"
               oninput="pp('court_inset')">
        <span class="val" id="court_inset-v">8</span>
      </div>
    </div>

    <!-- air zone -->
    <div class="group">
      <div class="group-title" style="color:#fc8">Air left (px)</div>
      <div class="row">
        <input type="range" id="air_xl" min="0" max="160" value="115" step="1"
               oninput="pp('air_xl')">
        <span class="val" id="air_xl-v">115</span>
      </div>
    </div>
    <div class="group">
      <div class="group-title" style="color:#fc8">Air right (px)</div>
      <div class="row">
        <input type="range" id="air_xr" min="160" max="320" value="193" step="1"
               oninput="pp('air_xr')">
        <span class="val" id="air_xr-v">193</span>
      </div>
    </div>
    <div class="group">
      <div class="group-title" style="color:#fc8">Air top cutoff</div>
      <div class="row">
        <input type="range" id="air_yt" min="0" max="60" value="38" step="1"
               oninput="pp('air_yt')">
        <span class="val" id="air_yt-v">38</span>
      </div>
    </div>
    <div class="group">
      <div class="group-title" style="color:#fc8">Air bottom (anchor)</div>
      <div class="row">
        <input type="range" id="air_yb" min="20" max="100" value="58" step="1"
               oninput="pp('air_yb')">
        <span class="val" id="air_yb-v">58</span>
      </div>
    </div>
    <div class="group">
      <div class="group-title" style="color:#fc8">Air sigma-x</div>
      <div class="row">
        <input type="range" id="air_sx" min="5" max="200" value="37" step="1"
               oninput="pp('air_sx')">
        <span class="val" id="air_sx-v">37</span>
      </div>
    </div>
    <div class="group">
      <div class="group-title" style="color:#fc8">Air sigma-y (up)</div>
      <div class="row">
        <input type="range" id="air_sy" min="1" max="120" value="40" step="1"
               oninput="pp('air_sy')">
        <span class="val" id="air_sy-v">40</span>
      </div>
    </div>

    <!-- overall -->
    <div class="group">
      <div class="group-title">Prior weight</div>
      <div class="row">
        <input type="range" id="pweight" min="0" max="1" value="1" step="0.01"
               oninput="pp('pweight')">
        <span class="val" id="pweight-v">1.00</span>
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
        <button class="tog-btn on" id="btn-prior2" onclick="togglePrior()">Prior: ON</button>
      </div>
    </div>

  </div><!-- end prior-controls -->

  <!-- TREE TAB controls -->
  <div id="tree-controls" style="display:none; contents">

    <!-- enable + mode + playback -->
    <div class="group">
      <div class="group-title">Tree</div>
      <div class="row" style="gap:6px">
        <button class="tog-btn" id="btn-tree-enable" onclick="toggleTreeEnabled()">Enable: OFF</button>
        <button id="tree-play-btn" onclick="toggleTreePlay()" title="Play tree forward">&#9654;</button>
        <select id="tree-speed" title="Speed">
          <option value="0.25">0.25x</option>
          <option value="0.5">0.5x</option>
          <option value="1" selected>1x</option>
          <option value="2">2x</option>
        </select>
      </div>
      <div class="row" style="gap:4px;margin-top:3px">
        <button class="tog-btn" id="btn-mode-dp"   onclick="setTreeMode('dp')"      style="background:#1a3a5a;color:#5af;border-color:#5af">DP Tree</button>
        <button class="tog-btn" id="btn-mode-hist" onclick="setTreeMode('history')" style="background:#1c1c1c;color:#888">Scatter</button>
        <button class="tog-btn" id="btn-mode-kf"   onclick="setTreeMode('kalman')"  style="background:#1c1c1c;color:#888">Kalman</button>
      </div>
    </div>

    <!-- frame nav (mirrors detection tab) -->
    <div class="group">
      <div class="group-title">Frame &nbsp;<span id="tflabel" style="color:#adf">0</span></div>
      <div class="row">
        <button class="nav-btn" onclick="tstep(-10)">&#10218;</button>
        <button class="nav-btn" onclick="tstep(-1)">&#8249;</button>
        <input type="range" id="tframe-sl" min="0" max="1000" value="0"
               oninput="tframeInput()">
        <button class="nav-btn" onclick="tstep(1)">&#8250;</button>
        <button class="nav-btn" onclick="tstep(10)">&#10219;</button>
      </div>
    </div>

    <!-- DP Tree controls -->
    <div id="dp-controls">
      <!-- tree-specific frame gap (default 1 for frame-by-frame tracking) -->
      <div class="group">
        <div class="group-title" style="color:#5cf">Frame gap</div>
        <div class="row">
          <input type="range" id="tree_gap" min="1" max="8" value="1" step="1"
                 oninput="tp('tree_gap')">
          <span class="val" id="tree_gap-v">1</span>
        </div>
      </div>
      <!-- n steps -->
      <div class="group">
        <div class="group-title" style="color:#5cf">Steps</div>
        <div class="row">
          <input type="range" id="n_steps" min="2" max="30" value="10" step="1"
                 oninput="tp('n_steps')">
          <span class="val" id="n_steps-v">10</span>
        </div>
      </div>

      <!-- link radius -->
      <div class="group">
        <div class="group-title" style="color:#5cf">Link radius</div>
        <div class="row">
          <input type="range" id="link_r" min="4" max="80" value="35" step="1"
                 oninput="tp('link_r')">
          <span class="val" id="link_r-v">35</span>
        </div>
      </div>

      <!-- static radius -->
      <div class="group">
        <div class="group-title" style="color:#5cf">Static radius</div>
        <div class="row">
          <input type="range" id="static_r" min="2" max="40" value="2" step="1"
                 oninput="tp('static_r')">
          <span class="val" id="static_r-v">2</span>
          <button class="tog-btn on" id="btn-static" onclick="toggleStatic()" title="Disable to stop static-blob filtering">Static: ON</button>
        </div>
      </div>

      <!-- top_k cap -->
      <div class="group">
        <div class="group-title" style="color:#5cf">Top K blobs</div>
        <div class="row">
          <input type="range" id="tree_top_k" min="0" max="12" value="3" step="1"
                 oninput="tp('tree_top_k')">
          <span class="val" id="tree_top_k-v">3</span>
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
    </div><!-- end dp-controls -->

    <!-- Kalman Track controls -->
    <div id="kf-controls" style="display:none">
      <div class="group">
        <div class="group-title" style="color:#c5f">History frames</div>
        <div class="row">
          <input type="range" id="kf_history" min="10" max="120" value="30" step="5"
                 oninput="kfp('kf_history')">
          <span class="val" id="kf_history-v">30</span>
        </div>
        <div style="font-size:9px;color:#666;margin-top:2px">Frames to run filter backward from current</div>
      </div>
      <div class="group">
        <div class="group-title" style="color:#c5f">Process noise</div>
        <div class="row">
          <input type="range" id="kf_proc_noise" min="0.5" max="30" value="5" step="0.5"
                 oninput="kfp('kf_proc_noise')">
          <span class="val" id="kf_proc_noise-v">5.0</span>
        </div>
        <div style="font-size:9px;color:#666;margin-top:2px">How much the ball can deviate from constant-velocity</div>
      </div>
      <div class="group">
        <div class="group-title" style="color:#c5f">Meas noise</div>
        <div class="row">
          <input type="range" id="kf_meas_noise" min="0.5" max="30" value="8" step="0.5"
                 oninput="kfp('kf_meas_noise')">
          <span class="val" id="kf_meas_noise-v">8.0</span>
        </div>
        <div style="font-size:9px;color:#666;margin-top:2px">How noisy blob centroids are (px). Higher = smoother but lags</div>
      </div>
      <div class="group">
        <div class="group-title" style="color:#c5f">Gate px</div>
        <div class="row">
          <input type="range" id="kf_gate_px" min="5" max="120" value="40" step="1"
                 oninput="kfp('kf_gate_px')">
          <span class="val" id="kf_gate_px-v">40</span>
        </div>
        <div style="font-size:9px;color:#666;margin-top:2px">Max px from prediction to accept a blob (blue ring = gate)</div>
      </div>
      <div class="group">
        <div class="group-title" style="color:#c5f">Max miss</div>
        <div class="row">
          <input type="range" id="kf_max_miss" min="1" max="20" value="8" step="1"
                 oninput="kfp('kf_max_miss')">
          <span class="val" id="kf_max_miss-v">8</span>
        </div>
        <div style="font-size:9px;color:#666;margin-top:2px">Consecutive missed frames before track is dropped and reset</div>
      </div>
    </div><!-- end kf-controls -->

    <!-- History Scatter controls -->
    <div id="hist-controls" style="display:none">
      <!-- n_hist frames -->
      <div class="group">
        <div class="group-title" style="color:#fa5">Hist frames</div>
        <div class="row">
          <input type="range" id="n_hist" min="5" max="120" value="15" step="5"
                 oninput="tp('n_hist')">
          <span class="val" id="n_hist-v">15</span>
        </div>
      </div>

      <!-- top_k -->
      <div class="group">
        <div class="group-title" style="color:#fa5">Top-K per diff</div>
        <div class="row">
          <input type="range" id="top_k" min="1" max="15" value="5" step="1"
                 oninput="tp('top_k')">
          <span class="val" id="top_k-v">5</span>
        </div>
      </div>
    </div><!-- end hist-controls -->

  </div><!-- end tree-controls -->

  <!-- TRACK TAB controls -->
  <div id="track-controls" style="display:none; contents">

    <!-- enable + playback -->
    <div class="group">
      <div class="group-title">Track</div>
      <div class="row" style="gap:6px">
        <button class="tog-btn" id="btn-track-enable" onclick="toggleTrackEnabled()">Enable: OFF</button>
        <button id="track-play-btn" onclick="toggleTrackPlay()" title="Play track">&#9654;</button>
        <button class="tog-btn on" id="btn-track-dp" onclick="toggleTrackDP()" title="Use DP tree to pre-filter blobs before RANSAC. OFF = scatter blobs → RANSAC directly">DP: ON</button>
        <select id="track-speed" title="Speed">
          <option value="0.25">0.25x</option>
          <option value="0.5">0.5x</option>
          <option value="1" selected>1x</option>
          <option value="2">2x</option>
        </select>
      </div>
    </div>

    <!-- frame nav -->
    <div class="group">
      <div class="group-title">Frame &nbsp;<span id="kflabel" style="color:#adf">0</span></div>
      <div class="row">
        <button class="nav-btn" onclick="kstep(-10)">&#10218;</button>
        <button class="nav-btn" onclick="kstep(-1)">&#8249;</button>
        <input type="range" id="kframe-sl" min="0" max="1000" value="0"
               oninput="kframeInput()">
        <button class="nav-btn" onclick="kstep(1)">&#8250;</button>
        <button class="nav-btn" onclick="kstep(10)">&#10219;</button>
      </div>
    </div>

    <!-- look-ahead/behind window -->
    <div class="group">
      <div class="group-title" style="color:#af5">Window ±</div>
      <div class="row">
        <input type="range" id="n_look" min="3" max="30" value="12" step="1"
               oninput="kp('n_look')">
        <span class="val" id="n_look-v">12</span>
      </div>
    </div>

    <!-- top-k per diff -->
    <div class="group">
      <div class="group-title" style="color:#af5">Top-K</div>
      <div class="row">
        <input type="range" id="trk_top_k" min="1" max="8" value="3" step="1"
               oninput="kp('trk_top_k')">
        <span class="val" id="trk_top_k-v">3</span>
      </div>
    </div>

    <!-- link radius -->
    <div class="group">
      <div class="group-title" style="color:#af5">Link r</div>
      <div class="row">
        <input type="range" id="trk_link_r" min="4" max="80" value="40" step="1"
               oninput="kp('trk_link_r')">
        <span class="val" id="trk_link_r-v">40</span>
      </div>
    </div>

    <!-- vel penalty -->
    <div class="group">
      <div class="group-title" style="color:#af5">Accel pen</div>
      <div class="row">
        <input type="range" id="trk_vel_pen" min="0" max="0.5" value="0.03" step="0.005"
               oninput="kp('trk_vel_pen')">
        <span class="val" id="trk_vel_pen-v">0.030</span>
      </div>
    </div>

    <!-- RANSAC inlier radius -->
    <div class="group">
      <div class="group-title" style="color:#5fa">RANSAC px</div>
      <div class="row">
        <input type="range" id="ransac_px" min="2" max="20" value="8" step="0.5"
               oninput="kp('ransac_px')">
        <span class="val" id="ransac_px-v">8</span>
      </div>
    </div>
    <div class="group">
      <div class="group-title" style="color:#5fa">Min speed (px/frame)</div>
      <div class="row">
        <input type="range" id="ransac_spd" min="0" max="30" value="6" step="0.5"
               oninput="kp('ransac_spd')">
        <span class="val" id="ransac_spd-v">6</span>
      </div>
    </div>

    <!-- confidence display -->
    <div class="group">
      <div class="group-title" style="color:#5fa">Ball confidence</div>
      <div class="row">
        <div id="confidence-bar-wrap" style="width:80px; height:10px; background:#222;
             border:1px solid #444; border-radius:3px; overflow:hidden">
          <div id="confidence-bar" style="width:0%; height:100%;
               background:linear-gradient(90deg,#f44,#fa0,#4f4); transition:width 0.2s"></div>
        </div>
        <span id="confidence-val" style="color:#adf; min-width:36px; text-align:right; font-size:12px">–</span>
      </div>
    </div>

    <!-- render to mp4 -->
    <div class="group">
      <div class="group-title" style="color:#f9a">Render</div>
      <div class="row" style="gap:6px; flex-wrap:wrap">
        <button class="tog-btn" id="btn-render-mp4" onclick="startRender()"
                style="color:#f9a; border-color:#c57">Render MP4</button>
        <span id="render-status" style="color:#aaa; font-size:11px; max-width:200px; white-space:normal"></span>
      </div>
    </div>

    <!-- export menu button -->
    <div class="group">
      <div class="group-title" style="color:#7df">Export</div>
      <div class="row">
        <button class="tog-btn" onclick="toggleExportPanel()"
                style="color:#7df; border-color:#37a; padding:3px 10px">&#9881; Export menu</button>
      </div>
    </div>


  </div><!-- end track-controls -->

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
      <span id="legend-dp">Purple=oldest &rarr; Yellow=newest (viridis) &middot; Red&nbsp;X=static &middot; Grey=edges &middot; Thick&nbsp;yellow=DP&nbsp;path</span>
      <span id="legend-hist" style="display:none">Purple=oldest frame &rarr; Yellow=newest (viridis) &middot; top-K passing blobs per diff, no DP</span>
    </div>
    <div id="tree-panels">
      <img id="tree-img" src="" alt="Enable tree to load...">
    </div>
  </div>
  <div id="page-track" style="display:none; flex-direction:column; flex:1; overflow:hidden">
    <div id="track-legend" style="padding:4px 10px; font-size:11px; color:#888; flex-shrink:0">
      Purple=oldest &rarr; Yellow=newest (viridis) &middot;
      White&nbsp;ring=current&nbsp;frame &middot;
      Arc=DP&nbsp;best&nbsp;path &middot;
      Quality=parabolic&nbsp;R&sup2;
    </div>
    <div id="track-panels" style="flex:1; overflow:hidden; display:flex; align-items:center; justify-content:center; background:#0a0a0a">
      <img id="track-img" src="" alt="Enable track to load...">
    </div>
  </div>
</div>

<div id="info">Space=play/pause  &middot;  arrows=step  &middot;  Shift=x10  &middot;  Loading&hellip;</div>

<script>
// ═══ STATE ═══════════════════════════════════════════════════════════════════
let videoName = "", totalFrames = 1000, videoFps = 25;
let method    = "compactness";
let usePrior  = true;
let diffMode  = "abs";   // "abs" = unsigned  |  "pos" = signed (forward only)
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
  'court_xs','court_ys','court_inset','air_xl','air_xr','air_yt','air_yb','air_sx','air_sy','pweight'
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
      for (const id of ['frame-sl','pframe-sl','tframe-sl','kframe-sl'])
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
  document.getElementById('tab-track').classList.toggle('active',  tab==='track');
  document.getElementById('page-detect').style.display = tab==='detect' ? 'flex'   : 'none';
  document.getElementById('page-prior').style.display  = tab==='prior'  ? 'flex'   : 'none';
  document.getElementById('page-tree').style.display   = tab==='tree'   ? 'flex'   : 'none';
  document.getElementById('page-track').style.display  = tab==='track'  ? 'flex'   : 'none';
  document.getElementById('det-controls').style.display    = tab==='detect' ? 'contents' : 'none';
  document.getElementById('prior-controls').style.display  = tab==='prior'  ? 'contents' : 'none';
  document.getElementById('tree-controls').style.display   = tab==='tree'   ? 'contents' : 'none';
  document.getElementById('track-controls').style.display  = tab==='track'  ? 'contents' : 'none';
  sync();   // ensure all tab frame sliders match the canonical frame-sl
  if      (tab==='prior') schedulePrior();
  else if (tab==='tree')  scheduleTree();
  else if (tab==='track') scheduleTrack();
  else { treePause(); trackPause(); request(); }
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
  sync();
  if      (currentTab === 'tree')  scheduleTree();
  else if (currentTab === 'track') scheduleTrack();
  else if (currentTab === 'prior') schedulePrior();
  else schedule();
}
function sync() {
  const v = document.getElementById('frame-sl').value;
  document.getElementById('flabel').textContent   = v;
  document.getElementById('pframe-sl').value      = v;
  document.getElementById('pflabel').textContent  = v;
  document.getElementById('tframe-sl').value      = v;
  document.getElementById('tflabel').textContent  = v;
  document.getElementById('kframe-sl').value      = v;
  document.getElementById('kflabel').textContent  = v;
}

// Tree-tab frame nav (mirrors main frame-sl)
function tframeInput() {
  const v = document.getElementById('tframe-sl').value;
  document.getElementById('frame-sl').value = v;
  sync();   // propagates to ALL frame sliders + labels
  scheduleTree();
}
function tstep(delta) {
  const sl = document.getElementById('tframe-sl');
  sl.value = Math.max(0, Math.min(totalFrames-1, +sl.value+delta));
  document.getElementById('frame-sl').value = sl.value;
  sync();   // propagates to ALL frame sliders + labels
  scheduleTree();
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
  document.querySelectorAll('input[type=range]').forEach(el => {
    el.classList.toggle('solo-active', soloMode && el.id === id);
  });
  const v = parseFloat(document.getElementById(id).value);
  const fmt2 = new Set(['score_thresh','min_circ','ball_diam','pblend','pweight']);
  const fmt1 = new Set(['max_asp']);
  const disp = fmt2.has(id) ? v.toFixed(2) : fmt1.has(id) ? v.toFixed(1) : String(Math.round(v));
  const vEl = document.getElementById(id+'-v');
  if (vEl) vEl.textContent = disp;
  scheduleActive();
}

function setMethod(m) {
  method = m;
  document.getElementById('btn-c').classList.toggle('on',    m==='compactness');
  document.getElementById('btn-r').classList.toggle('on',    m==='rog');
  document.getElementById('btn-circ').classList.toggle('on', m==='circularity');
  document.getElementById('circ-params').style.display = m==='circularity' ? 'contents' : 'none';
  scheduleActive();
}

function resChanged() {
  scheduleActive();
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
  scheduleActive();
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
function toggleDiffMode() {
  diffMode = diffMode === "abs" ? "pos" : "abs";
  const label = diffMode === "pos" ? "Diff: Signed" : "Diff: Unsigned";
  document.getElementById("btn-diff").textContent  = label;
  document.getElementById("btn-diff").classList.toggle("on", diffMode === "pos");
  requestFrame();
}

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
  scheduleActive();
}

// ═══ PRIOR PARAMS HELPER ══════════════════════════════════════════════════════
function priorParams() {
  return {
    court_xs:    document.getElementById('court_xs').value,
    court_ys:    document.getElementById('court_ys').value,
    court_inset: document.getElementById('court_inset').value,
    air_xl:      document.getElementById('air_xl').value,
    air_xr:      document.getElementById('air_xr').value,
    air_yt:      document.getElementById('air_yt').value,
    air_yb:      document.getElementById('air_yb').value,
    air_sx:      document.getElementById('air_sx').value,
    air_sy:      document.getElementById('air_sy').value,
    pweight:     document.getElementById('pweight').value,
  };
}


function schedule() {
  clearTimeout(debounce);
  debounce = setTimeout(() => requestFrame(null), 60);
}
function request() { requestFrame(null); }

// Fire the refresh for whichever tab is currently visible.
// Called after any parameter change so tree/track stay in sync.
function scheduleActive() {
  if      (currentTab === 'detect') schedule();
  else if (currentTab === 'prior')  schedulePrior();
  else if (currentTab === 'tree')   scheduleTree();
  else if (currentTab === 'track')  scheduleTrack();
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
    n_diff:       document.getElementById('n_diff').value,
    diff_mode:    diffMode,
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
  document.getElementById('frame-sl').value = v;
  sync();   // propagates to ALL frame sliders + labels
  schedulePrior();
}
function pstep(delta) {
  const sl = document.getElementById('pframe-sl');
  sl.value = Math.max(0, Math.min(totalFrames-1, +sl.value+delta));
  document.getElementById('frame-sl').value = sl.value;
  sync();   // propagates to ALL frame sliders + labels
  schedulePrior();
}

function pp(id) {
  lastParam = id;
  if (soloMode) updateSoloBtn();
  const v   = parseFloat(document.getElementById(id).value);
  const fmt = (id==='pweight'||id==='pblend') ? v.toFixed(2) : String(Math.round(v));
  document.getElementById(id+'-v').textContent = fmt;
  schedulePrior();
  if (usePrior) {
    // Propagate prior-param changes to whichever data view is active
    if      (currentTab === 'detect') schedule();
    else if (currentTab === 'tree')   scheduleTree();
    else if (currentTab === 'track')  scheduleTrack();
  }
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
let treeEnabled  = false;
let treePlaying  = false;
let treePlayGen  = 0;
let treeMode     = 'dp';   // 'dp' | 'history'
let treeFetching    = false;  // true while a fetch is in-flight
let treePlayDirty   = false;  // frame advanced while fetch was in-flight

function toggleTreeEnabled() {
  treeEnabled = !treeEnabled;
  const btn = document.getElementById('btn-tree-enable');
  btn.textContent = treeEnabled ? 'Enable: ON' : 'Enable: OFF';
  btn.classList.toggle('on', treeEnabled);
  if (treeEnabled) scheduleTree();
  else {
    treePause();
    treeFetching = false;
    document.getElementById('tree-img').src = '';
  }
}

function setTreeMode(mode) {
  treeMode = mode;
  const dp   = document.getElementById('btn-mode-dp');
  const hist = document.getElementById('btn-mode-hist');
  const kf   = document.getElementById('btn-mode-kf');
  const dpC  = document.getElementById('dp-controls');
  const hC   = document.getElementById('hist-controls');
  const kfC  = document.getElementById('kf-controls');
  const ldp  = document.getElementById('legend-dp');
  const lh   = document.getElementById('legend-hist');
  // Reset all buttons + panels
  [dp,hist,kf].forEach(b => { if(b){b.style.background='#1c1c1c';b.style.color='#888';b.style.borderColor='';} });
  [dpC,hC,kfC].forEach(c => { if(c) c.style.display='none'; });
  if (ldp) ldp.style.display = 'none';
  if (lh)  lh.style.display  = 'none';
  if (mode === 'dp') {
    dp.style.background = '#1a3a5a'; dp.style.color = '#5af'; dp.style.borderColor = '#5af';
    dpC.style.display = '';
    if (ldp) ldp.style.display = '';
  } else if (mode === 'history') {
    hist.style.background = '#3a2a10'; hist.style.color = '#fa5'; hist.style.borderColor = '#fa5';
    hC.style.display = '';
    if (lh) lh.style.display = '';
  } else {  // kalman
    kf.style.background = '#2a1a4a'; kf.style.color = '#c5f'; kf.style.borderColor = '#a5f';
    kfC.style.display = '';
  }
  scheduleTree();
}

function toggleTreePlay() { treePlaying ? treePause() : treePlay(); }

function treePlay() {
  if (!treeEnabled) { toggleTreeEnabled(); }
  treePlaying = true; treePlayGen++;
  document.getElementById('tree-play-btn').innerHTML = '&#9646;&#9646;';
  document.getElementById('tree-play-btn').classList.add('playing');
  runTreeLoop(treePlayGen);
}
function treePause() {
  treePlaying = false;
  document.getElementById('tree-play-btn').innerHTML = '&#9654;';
  document.getElementById('tree-play-btn').classList.remove('playing');
}

function runTreeLoop(gen) {
  if (!treePlaying || gen !== treePlayGen) return;
  const speed    = parseFloat(document.getElementById('tree-speed').value);
  const targetMs = Math.max(50, 1000 / (videoFps * speed));

  const sl = document.getElementById('tframe-sl');
  let cur  = +sl.value + 1;
  if (cur >= totalFrames) cur = 0;
  sl.value = cur;
  document.getElementById('frame-sl').value     = cur;
  document.getElementById('tflabel').textContent = cur;
  document.getElementById('flabel').textContent  = cur;

  // Fire-and-forget: frame counter advances at target FPS;
  // image updates whenever the server finishes (may lag behind for slow modes).
  requestTree(null);
  setTimeout(() => runTreeLoop(gen), targetMs);
}

function tp(id) {
  const v   = parseFloat(document.getElementById(id).value);
  const fmt = id === 'vel_pen' ? v.toFixed(3) : id === 'tree_top_k' && v === 0 ? '0 (all)' : String(Math.round(v));
  document.getElementById(id+'-v').textContent = fmt;
  scheduleTree();
}

function kfp(id) {
  const v = parseFloat(document.getElementById(id).value);
  document.getElementById(id + '-v').textContent = (v % 1 === 0) ? String(v) : v.toFixed(1);
  scheduleTree();
}

function scheduleTree() {
  if (currentTab !== 'tree' || !treeEnabled) return;
  clearTimeout(treeDebounce);
  treeDebounce = setTimeout(() => requestTree(null), 120);
}

function requestTree(onDone) {
  if (!videoName || currentTab !== 'tree' || !treeEnabled) {
    if (onDone) onDone(); return;
  }
  // If a request is in-flight, mark dirty so we re-request on completion.
  if (treeFetching) { if (onDone === null) { treePlayDirty = true; return; } }

  const frame   = document.getElementById('tframe-sl').value;
  const baseP   = {
    video:        videoName,
    frame:        frame,
    thresh:       document.getElementById('thresh').value,
    min_a:        document.getElementById('min_a').value,
    max_a:        document.getElementById('max_a').value,
    max_asp:      document.getElementById('max_asp').value,
    method,
    res_w:        document.getElementById('res_w').value,
    blur_k:       document.getElementById('blur_k').value,
    ball_diam:    document.getElementById('ball_diam').value,
    min_circ:     document.getElementById('min_circ').value,
    min_bright:   document.getElementById('min_bright').value,
    score_thresh: document.getElementById('score_thresh').value,
    use_prior:    usePrior ? '1' : '0',
    diff_mode:    diffMode,
    ...priorParams(),
  };

  let endpoint, extraP, infoPrefix;
  if (treeMode === 'history') {
    endpoint  = '/api/history';
    extraP    = {
      n_hist: document.getElementById('n_hist').value,
      top_k:  document.getElementById('top_k').value,
    };
    infoPrefix = 'Scatter';
  } else if (treeMode === 'kalman') {
    endpoint  = '/api/kalman_frame';
    extraP    = {
      kf_proc_noise: document.getElementById('kf_proc_noise').value,
      kf_meas_noise: document.getElementById('kf_meas_noise').value,
      kf_gate_px:    document.getElementById('kf_gate_px').value,
      kf_max_miss:   document.getElementById('kf_max_miss').value,
      n_history:     document.getElementById('kf_history').value,
    };
    infoPrefix = 'Kalman';
  } else {
    endpoint  = '/api/tree';
    extraP    = {
      gap:        document.getElementById('tree_gap').value,
      n_steps:    document.getElementById('n_steps').value,
      link_r:     document.getElementById('link_r').value,
      static_r:   document.getElementById('static_r').value,
      vel_pen:    document.getElementById('vel_pen').value,
      tree_top_k: document.getElementById('tree_top_k').value,
      static_r:   staticEnabled ? document.getElementById('static_r').value : '0',
    };
    infoPrefix = 'Tree';
  }

  const p = new URLSearchParams(Object.assign({}, baseP, extraP));
  document.getElementById('info').textContent = infoPrefix + ': computing\u2026';
  treeFetching = true;
  treePlayDirty = false;
  fetch(endpoint + '?' + p).then(r => r.json()).then(d => {
    treeFetching = false;
    if (d.img) {
      document.getElementById('tree-img').src = 'data:image/jpeg;base64,' + d.img;
    }
    if (treeMode === 'history') {
      document.getElementById('info').textContent =
        'Scatter: frame ' + frame +
        '  |  layers=' + (d.n_layers||'?') +
        (d.error ? '  ERR: ' + d.error : '');
    } else if (treeMode === 'kalman') {
      document.getElementById('info').textContent =
        'Kalman: frame ' + frame +
        '  |  conf=' + ((d.confidence||0).toFixed ? (d.confidence||0).toFixed(3) : d.confidence||0) +
        '  hits=' + (d.hit_count||0) +
        '  miss=' + (d.miss_count||0) +
        '  vel=(' + (d.vx||0) + ',' + (d.vy||0) + ')' +
        (d.error ? '  ERR: ' + d.error : '');
    } else {
      document.getElementById('info').textContent =
        'Tree: frame ' + frame +
        '  |  steps=' + (d.n_steps||'?') +
        '  blobs=' + (d.n_blobs||'?') +
        '  path_len=' + (d.path_len||0) +
        (d.error ? '  ERR: ' + d.error : '');
    }
    if (onDone) onDone();
    // If frame advanced while we were fetching, immediately request the new frame
    if (treePlayDirty) { treePlayDirty = false; requestTree(null); }
  }).catch(e => {
    treeFetching = false;
    document.getElementById('info').textContent = infoPrefix + ' error: ' + e;
    if (onDone) onDone();
    if (treePlayDirty) { treePlayDirty = false; requestTree(null); }
  });
}


// ════════════════════════════════════════════════════════════════════════════
let trackDebounce = null;
let trackEnabled  = false;
let trackUseDP    = true;   // false = scatter blobs → RANSAC (skip DP)
let staticEnabled = true;   // false = static_radius=0
let trackPlaying  = false;
let trackPlayGen  = 0;
let trackFetching = false;

function toggleTrackDP() {
  trackUseDP = !trackUseDP;
  const btn = document.getElementById('btn-track-dp');
  btn.textContent = trackUseDP ? 'DP: ON' : 'DP: OFF';
  btn.classList.toggle('on', trackUseDP);
  scheduleTrack();
}

function toggleStatic() {
  staticEnabled = !staticEnabled;
  const btn = document.getElementById('btn-static');
  btn.textContent = staticEnabled ? 'Static: ON' : 'Static: OFF';
  btn.classList.toggle('on', staticEnabled);
  requestTree(null);
}

function toggleTrackEnabled() {
  trackEnabled = !trackEnabled;
  const btn = document.getElementById('btn-track-enable');
  btn.textContent = trackEnabled ? 'Enable: ON' : 'Enable: OFF';
  btn.classList.toggle('on', trackEnabled);
  if (trackEnabled) scheduleTrack();
  else { trackPause(); trackFetching = false; document.getElementById('track-img').src = ''; }
}

function toggleTrackPlay() { trackPlaying ? trackPause() : trackPlay(); }
function trackPlay() {
  if (!trackEnabled) toggleTrackEnabled();
  trackPlaying = true; trackPlayGen++;
  document.getElementById('track-play-btn').innerHTML = '&#9646;&#9646;';
  document.getElementById('track-play-btn').classList.add('playing');
  runTrackLoop(trackPlayGen);
}
function trackPause() {
  trackPlaying = false;
  document.getElementById('track-play-btn').innerHTML = '&#9654;';
  document.getElementById('track-play-btn').classList.remove('playing');
}

function runTrackLoop(gen) {
  if (!trackPlaying || gen !== trackPlayGen) return;
  const speed    = parseFloat(document.getElementById('track-speed').value);
  const minMs    = Math.max(33, 1000 / (videoFps * speed));
  const t0       = Date.now();
  const sl = document.getElementById('kframe-sl');
  let cur = +sl.value + 1;
  if (cur >= totalFrames) cur = 0;
  sl.value = cur;
  document.getElementById('frame-sl').value      = cur;
  document.getElementById('kflabel').textContent  = cur;
  document.getElementById('flabel').textContent   = cur;
  // Wait for the response before advancing to next frame (frame-accurate playback)
  requestTrack(() => {
    if (!trackPlaying || gen !== trackPlayGen) return;
    const elapsed = Date.now() - t0;
    const wait    = Math.max(0, minMs - elapsed);
    setTimeout(() => runTrackLoop(gen), wait);
  });
}

function kp(id) {
  const v   = parseFloat(document.getElementById(id).value);
  const fmt = id === 'trk_vel_pen' ? v.toFixed(3)
            : id === 'ransac_px'   ? v.toFixed(1)
            : String(Math.round(v));
  document.getElementById(id+'-v').textContent = fmt;
  scheduleTrack();
}
function kframeInput() {
  const v = document.getElementById('kframe-sl').value;
  document.getElementById('frame-sl').value = v;
  sync();   // propagates to ALL frame sliders + labels
  scheduleTrack();
}
function kstep(delta) {
  const sl = document.getElementById('kframe-sl');
  sl.value = Math.max(0, Math.min(totalFrames-1, +sl.value+delta));
  document.getElementById('frame-sl').value = sl.value;
  sync();   // propagates to ALL frame sliders + labels
  scheduleTrack();
}

function scheduleTrack() {
  if (currentTab !== 'track' || !trackEnabled) return;
  clearTimeout(trackDebounce);
  trackDebounce = setTimeout(() => requestTrack(null), 120);
}

// ═══ RENDER MP4 ══════════════════════════════════════════════════════════════
let renderPollTimer = null;

function startRender() {
  if (_isRendering) return;
  const btn = document.getElementById('btn-render-mp4');
  const status = document.getElementById('render-status');
  const p = new URLSearchParams({
    video:        videoName,
    res_w:        document.getElementById('res_w').value,
    n_look:       document.getElementById('n_look').value,
    top_k:        document.getElementById('trk_top_k').value,
    link_r:       document.getElementById('trk_link_r').value,
    vel_pen:      document.getElementById('trk_vel_pen').value,
    static_r:     '8',
    thresh:       document.getElementById('thresh').value,
    min_a:        document.getElementById('min_a').value,
    max_a:        document.getElementById('max_a').value,
    max_asp:      document.getElementById('max_asp').value,
    method,
    ball_diam:    document.getElementById('ball_diam').value,
    min_circ:     document.getElementById('min_circ').value,
    min_bright:   document.getElementById('min_bright').value,
    blur_k:       document.getElementById('blur_k').value,
    score_thresh: document.getElementById('score_thresh').value,
    use_prior:    usePrior ? '1' : '0',
    ...priorParams(),
  });
  fetch('/api/render_start?' + p).then(r => r.json()).then(d => {
    if (!d.ok) { status.textContent = 'Error: ' + (d.error || '?'); return; }
    btn.disabled = true;
    btn.style.opacity = '0.5';
    status.textContent = 'Starting\u2026';
    clearInterval(renderPollTimer);
    renderPollTimer = setInterval(pollRenderStatus, 800);
  }).catch(e => { status.textContent = 'Error: ' + e; });
}

let _isRendering = false;
function pollRenderStatus() {
  fetch('/api/render_status').then(r => r.json()).then(d => {
    const btn = document.getElementById('btn-render-mp4');
    const status = document.getElementById('render-status');
    if (d.running) {
      _isRendering = true;
      const pct = d.total > 0 ? Math.round(100 * d.progress / d.total) : 0;
      status.textContent = `Rendering\u2026 ${pct}% (${d.progress}/${d.total} frames)`;
    } else {
      _isRendering = false;
      clearInterval(renderPollTimer);
      btn.disabled = false;
      btn.style.opacity = '';
      if (d.error) {
        status.textContent = '\u274c Error: ' + d.error;
      } else if (d.output) {
        status.textContent = '\u2713 Done \u2192 ' + d.output;
      } else {
        status.textContent = '';
      }
    }
  });
}

// ═══ SAVE / LOAD PARAMS ══════════════════════════════════════════════════════
const _ALL_SLIDER_IDS = [
  'gap','thresh','min_a','max_a','max_asp','score_thresh',
  'blur_k','ball_diam','min_circ','min_bright','n_diff',
  'court_xs','court_ys','court_inset','air_xl','air_xr','air_yt','air_yb',
  'air_sx','air_sy','pweight','pblend',
  'n_look','trk_top_k','trk_link_r','trk_vel_pen','ransac_px',
  'tree_gap','n_steps','link_r','static_r','vel_pen','tree_top_k',
  'n_hist','top_k',
  'kf_proc_noise','kf_meas_noise','kf_gate_px','kf_max_miss','kf_history',
];

function saveParams() {
  const params = {};
  for (const id of _ALL_SLIDER_IDS) {
    const el = document.getElementById(id);
    if (el) params[id] = el.value;
  }
  params.method    = method;
  params.res_w     = document.getElementById('res_w').value;
  params.use_prior = usePrior ? '1' : '0';
  const p = new URLSearchParams(params);
  fetch('/api/save_params?' + p).then(r => r.json()).then(d => {
    document.getElementById('info').textContent = d.ok
      ? 'Params saved to params.json.'
      : 'Save failed: ' + (d.error || '?');
  }).catch(e => { document.getElementById('info').textContent = 'Save error: ' + e; });
}

function loadParams() {
  fetch('/api/load_params').then(r => r.json()).then(d => {
    if (d.error) {
      document.getElementById('info').textContent = 'Load failed: ' + d.error;
      return;
    }
    for (const id of _ALL_SLIDER_IDS) {
      if (d[id] !== undefined) {
        const el = document.getElementById(id);
        if (el) { el.value = d[id]; ps(id); }
      }
    }
    if (d.method)    setMethod(d.method);
    if (d.res_w) {
      const sel = document.getElementById('res_w');
      if (sel) { sel.value = d.res_w; resChanged(); }
    }
    if (d.use_prior !== undefined) {
      const shouldBe = d.use_prior === '1';
      if (usePrior !== shouldBe) togglePrior();
    }
    document.getElementById('info').textContent = 'Params loaded from params.json.';
    scheduleActive();
  }).catch(e => { document.getElementById('info').textContent = 'Load error: ' + e; });
}

function requestTrack(onDone) {
  if (!videoName || currentTab !== 'track' || !trackEnabled) {
    if (onDone) onDone(); return;
  }
  if (trackFetching && onDone === null) return;
  const frame = document.getElementById('kframe-sl').value;
  const p = new URLSearchParams({
    video:        videoName,
    frame:        frame,
    thresh:       document.getElementById('thresh').value,
    min_a:        document.getElementById('min_a').value,
    max_a:        document.getElementById('max_a').value,
    max_asp:      document.getElementById('max_asp').value,
    method,
    res_w:        document.getElementById('res_w').value,
    blur_k:       document.getElementById('blur_k').value,
    ball_diam:    document.getElementById('ball_diam').value,
    min_circ:     document.getElementById('min_circ').value,
    min_bright:   document.getElementById('min_bright').value,
    score_thresh: document.getElementById('score_thresh').value,
    n_look:       document.getElementById('n_look').value,
    top_k:        document.getElementById('trk_top_k').value,
    link_r:       document.getElementById('trk_link_r').value,
    vel_pen:      document.getElementById('trk_vel_pen').value,
    ransac_px:    document.getElementById('ransac_px').value,
    ransac_spd:   document.getElementById('ransac_spd').value,
    use_dp:       trackUseDP ? '1' : '0',
    use_prior:    usePrior ? '1' : '0',
    ...priorParams(),
  });
  document.getElementById('info').textContent = 'Track: computing\u2026';
  trackFetching = true;
  fetch('/api/track?' + p).then(r => r.json()).then(d => {
    trackFetching = false;
    if (d.img) document.getElementById('track-img').src = 'data:image/jpeg;base64,' + d.img;

    // Update confidence bar
    const conf = d.confidence !== undefined ? d.confidence : 0;
    document.getElementById('confidence-bar').style.width = Math.round(conf * 100) + '%';
    const confLabel = conf > 0.65 ? '\u2713 IN PLAY' : conf > 0.30 ? '~ unsure' : '\u00d7 not ball';
    document.getElementById('confidence-val').textContent = conf.toFixed(2);
    document.getElementById('confidence-val').style.color =
      conf > 0.65 ? '#4f4' : conf > 0.30 ? '#fa0' : '#f66';

    const dpQ  = d.quality     !== undefined ? '  dp_r\u00b2=' + d.quality.toFixed(2)    : '';
    const arcQ = d.arc_r2      !== undefined ? '  arc_r\u00b2=' + d.arc_r2.toFixed(2)     : '';
    const cov  = d.arc_cov     !== undefined ? '  cov='  + d.arc_cov.toFixed(2)           : '';
    const nin  = d.arc_inliers !== undefined ? '  inliers=' + d.arc_inliers               : '';
    document.getElementById('info').textContent =
      'Track frame ' + frame +
      '  |  conf=' + conf.toFixed(3) + ' [' + confLabel + ']' +
      arcQ + cov + nin + dpQ +
      '  dp_path=' + (d.path_len||0) + '/' + (d.n_layers||'?') +
      (d.error ? '  ERR: ' + d.error : '');
    if (onDone) onDone();
  }).catch(e => {
    trackFetching = false;
    document.getElementById('info').textContent = 'Track error: ' + e;
    if (onDone) onDone();
  });
}


  // ── Export panel ──────────────────────────────────────────────────────────
  let exportPollTimer = null;

  function toggleExportPanel() {
    const panel   = document.getElementById('export-panel');
    const overlay = document.getElementById('export-overlay');
    const visible = panel.style.display !== 'none';
    panel.style.display   = visible ? 'none' : 'block';
    overlay.style.display = visible ? 'none' : 'block';
    setExportTracker('kalman');   // highlight default on open
  }

  function exportParams() {
    // Collect all current detection / prior / RANSAC params
    const get = id => { const el = document.getElementById(id); return el ? el.value : ''; };
    return {
      video:        videoName,
      res_w:        get('res_w'),
      n_look:       get('n_look'),
      top_k:        get('trk_top_k'),
      thresh:       get('thresh'),
      min_a:        get('min_a'),
      max_a:        get('max_a'),
      max_asp:      get('max_asp'),
      method:       method,
      ball_diam:    get('ball_diam'),
      min_circ:     get('min_circ'),
      min_bright:   get('min_bright'),
      blur_k:       get('blur_k'),
      score_thresh: get('score_thresh'),
      use_prior:    usePrior ? '1' : '0',
      ...priorParams(),
      ransac_px:    get('ransac_px'),
      ransac_spd:   '6.0',
      conf_thresh:  document.getElementById('exp-conf-thr').value,
      max_minutes:  document.getElementById('exp-max-min').value,
      min_seg_sec:  document.getElementById('exp-min-seg').value,
      tracker_mode: _exportTracker,
    };
  }

  let _exportTracker = "kalman";   // default: fast Kalman

  function setExportTracker(mode) {
    _exportTracker = mode;
    const btns = { ransac: document.getElementById("exp-trk-ransac"),
                   kalman: document.getElementById("exp-trk-kalman") };
    if (btns.ransac && btns.kalman) {
      btns.ransac.style.background = mode==="ransac" ? "#0d2a1a" : "#111";
      btns.ransac.style.border     = mode==="ransac" ? "2px solid #3a7" : "1px solid #444";
      btns.ransac.style.color      = mode==="ransac" ? "#5e5" : "#aaa";
      btns.kalman.style.background = mode==="kalman" ? "#0a1a2a" : "#111";
      btns.kalman.style.border     = mode==="kalman" ? "2px solid #37a" : "1px solid #444";
      btns.kalman.style.color      = mode==="kalman" ? "#7df" : "#aaa";
    }
  }


  function startExport(mode) {
    if (!videoName) { alert('Select a video first.'); return; }
    const p = new URLSearchParams({...exportParams(), mode});
    document.getElementById('exp-progress-wrap').style.display = 'block';
    document.getElementById('exp-cancel-btn').style.display    = 'inline-block';
    document.getElementById('exp-outputs').textContent         = '';
    document.getElementById('exp-status').textContent          = 'Starting…';
    document.getElementById('exp-progress-bar').style.width    = '0%';
    fetch('/api/export_start?' + p).then(r => r.json()).then(d => {
      if (!d.ok) { document.getElementById('exp-status').textContent = 'Error: ' + d.error; return; }
      clearInterval(exportPollTimer);
      exportPollTimer = setInterval(pollExport, 800);
    });
  }

  function cancelExport() {
    fetch('/api/export_cancel');
    clearInterval(exportPollTimer);
    document.getElementById('exp-status').textContent = 'Cancelled.';
    document.getElementById('exp-cancel-btn').style.display = 'none';
  }

  function pollExport() {
    fetch('/api/export_status').then(r => r.json()).then(d => {
      const pct = d.total > 0 ? Math.round(100 * d.progress / d.total) : 0;
      document.getElementById('exp-progress-bar').style.width = pct + '%';
      let msg = '';
      if (d.phase === 'scanning') {
        msg = 'Scanning confidence… frame ' + d.progress + '/' + d.total;
      } else if (d.phase && d.phase.startsWith('writing')) {
        msg = d.phase + ' — segment ' + d.progress + '/' + d.total;
      } else if (d.phase === 'done') {
        msg = 'Done!  ' + d.n_ball + ' ball segment(s), ' + d.n_dead + ' dead-time segment(s).';
      }
      if (d.error) msg = 'Error: ' + d.error;
      document.getElementById('exp-status').textContent = msg;

      if (!d.running) {
        clearInterval(exportPollTimer);
        document.getElementById('exp-cancel-btn').style.display = 'none';
        // Show output file names
        let links = '';
        if (d.output)  links += d.output + String.fromCharCode(10);
        if (d.output2) links += d.output2 + String.fromCharCode(10);
        document.getElementById('exp-outputs').textContent = links;
      }
    });
  }

</script>
</body>
<!-- ── EXPORT PANEL ──────────────────────────────────────────────────────── -->
<div id="export-panel" style="
  display:none; position:fixed; top:50%; left:50%; transform:translate(-50%,-50%);
  z-index:999; background:#1a1a2e; border:1px solid #37a; border-radius:8px;
  padding:18px 22px; width:360px; box-shadow:0 8px 32px rgba(0,0,0,0.7);
  font-size:13px; color:#ccc;">

  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px">
    <span style="font-size:15px; font-weight:600; color:#7df">&#9881; Export Options</span>
    <button onclick="toggleExportPanel()" style="background:none;border:none;color:#888;
            font-size:18px;cursor:pointer;line-height:1">&#10005;</button>
  </div>

  <!-- Analysis limit -->
  <div style="margin-bottom:10px">
    <label style="color:#aaa; display:block; margin-bottom:4px">
      Analyse first
      <input id="exp-max-min" type="number" min="0" max="9999" step="0.5" value="0"
             style="width:58px; margin:0 5px; background:#111; color:#eee;
                    border:1px solid #444; border-radius:3px; padding:2px 5px">
      minutes &nbsp;<span style="color:#666; font-size:11px">(0 = full video)</span>
    </label>
  </div>

  <!-- Confidence threshold -->
  <div style="margin-bottom:6px">
    <label style="color:#aaa; display:block; margin-bottom:4px">
      Ball confidence threshold
    </label>
    <div style="display:flex; align-items:center; gap:8px">
      <input id="exp-conf-thr" type="range" min="0" max="1" step="0.01" value="0.5"
             oninput="document.getElementById('exp-conf-v').textContent=parseFloat(this.value).toFixed(2)"
             style="flex:1">
      <span id="exp-conf-v" style="color:#7df; min-width:30px">0.50</span>
    </div>
  </div>

  <!-- Min segment duration -->
  <div style="margin-bottom:14px">
    <label style="color:#aaa; display:block; margin-bottom:4px">
      Min segment duration (s)
    </label>
    <div style="display:flex; align-items:center; gap:8px">
      <input id="exp-min-seg" type="range" min="0.1" max="5" step="0.1" value="0.5"
             oninput="document.getElementById('exp-seg-v').textContent=parseFloat(this.value).toFixed(1)"
             style="flex:1">
      <span id="exp-seg-v" style="color:#7df; min-width:30px">0.5</span>
    </div>
  </div>

  <!-- Tracker method -->
  <div style="margin-bottom:14px">
    <label style="color:#aaa; display:block; margin-bottom:6px">Tracker method</label>
    <div style="display:flex; gap:6px">
      <button id="exp-trk-ransac" onclick="setExportTracker('ransac')" style="flex:1; border-radius:4px; padding:5px; cursor:pointer; font-size:12px; background:#0d2a1a; border:2px solid #3a7; color:#5e5">RANSAC (accurate)</button>
      <button id="exp-trk-kalman" onclick="setExportTracker('kalman')" style="flex:1; border-radius:4px; padding:5px; cursor:pointer; font-size:12px; background:#111; border:1px solid #444; color:#aaa">Kalman (fast ~25x)</button>
    </div>
    <div style="font-size:10px; color:#666; margin-top:4px" id="exp-trk-hint">RANSAC: window scan, physics arc &nbsp;|&nbsp; Kalman: single pass, sequential</div>
  </div>


  <!-- Action buttons -->
  <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px">
    <button onclick="startExport('ball')"
            style="flex:1; background:#0a2a1a; border:1px solid #3a7; color:#5e5;
                   border-radius:4px; padding:6px; cursor:pointer; font-size:12px">
      &#9654; Ball detections only
    </button>
    <button onclick="startExport('dead')"
            style="flex:1; background:#2a1a0a; border:1px solid #a73; color:#e85;
                   border-radius:4px; padding:6px; cursor:pointer; font-size:12px">
      &#9654; Dead time only
    </button>
    <button onclick="startExport('both')"
            style="flex:1 0 100%; background:#0d0d2a; border:1px solid #37a; color:#7df;
                   border-radius:4px; padding:6px; cursor:pointer; font-size:12px">
      &#9654; Export both videos
    </button>
  </div>

  <!-- Cancel -->
  <div style="margin-bottom:10px">
    <button id="exp-cancel-btn" onclick="cancelExport()"
            style="display:none; background:#2a0a0a; border:1px solid #a33; color:#f77;
                   border-radius:4px; padding:4px 10px; cursor:pointer; font-size:12px">
      &#9632; Cancel
    </button>
  </div>

  <!-- Progress -->
  <div id="exp-progress-wrap" style="display:none">
    <div style="background:#111; border-radius:3px; overflow:hidden; height:8px; margin-bottom:6px">
      <div id="exp-progress-bar"
           style="height:100%; width:0%; background:linear-gradient(90deg,#37a,#7df); transition:width 0.3s"></div>
    </div>
    <div id="exp-status" style="color:#aaa; font-size:11px; white-space:pre-wrap"></div>
  </div>

  <!-- Output links -->
  <div id="exp-outputs" style="margin-top:8px; font-size:11px; color:#8c8"></div>
</div>
<div id="export-overlay" onclick="toggleExportPanel(); event.stopPropagation()"
     style="display:none; position:fixed; inset:0; z-index:998; background:rgba(0,0,0,0.4)"></div>


</html>
"""

# ── HTTP Handler ─────────────────────────────────────────────────────────────


def _prior_from_q(q):
    """Build a compute_prior_map from a handler query-string accessor q(key, default)."""
    return compute_prior_map(
        court_x_sigma = float(q("court_xs",     str(_DP["court_xs"]))),
        court_y_sigma = float(q("court_ys",     str(_DP["court_ys"]))),
        court_inset   = int(q("court_inset",    str(_DP["court_inset"]))),
        air_x_left    = int(q("air_xl",         str(_DP["air_xl"]))),
        air_x_right   = int(q("air_xr",         str(_DP["air_xr"]))),
        air_y_top     = int(q("air_yt",         str(_DP["air_yt"]))),
        air_y_bot     = int(q("air_yb",         str(_DP["air_yb"]))),
        air_sigma_x   = float(q("air_sx",       str(_DP["air_sx"]))),
        air_sigma_y   = float(q("air_sy",       str(_DP["air_sy"]))),
        weight        = float(q("pweight",      str(_DP["pweight"]))),
    )


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        prs = urllib.parse.urlparse(self.path)
        qs  = urllib.parse.parse_qs(prs.query)

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
            try:
                self._json({"videos": list_videos()})
            except Exception as exc:
                import traceback; traceback.print_exc()
                self._json({"videos": [], "error": str(exc)})

        elif prs.path == "/api/info":
            try:
                cap    = get_cap(q("video"))
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
                self._json({"frames": frames, "fps": fps})
            except Exception as exc:
                import traceback; traceback.print_exc()
                self._json({"frames": 1000, "fps": 25.0, "error": str(exc)})

        elif prs.path == "/api/frame":
            try:
                res_w = int(q("res_w", str(_DP["res_w"])))
                comp, cands = make_composite(
                    vname         = q("video"),
                    frame_idx     = int(q("frame",          "0")),
                    gap           = int(q("gap",            str(_DP["gap"]))),
                    thresh        = int(q("thresh",         str(_DP["thresh"]))),
                    min_a         = int(q("min_a",          str(_DP["min_a"]))),
                    max_a         = int(q("max_a",          str(_DP["max_a"]))),
                    max_asp       = float(q("max_asp",      str(_DP["max_asp"]))),
                    method        = q("method",             str(_DP["method"])),
                    score_thresh  = float(q("score_thresh", str(_DP["score_thresh"]))),
                    use_prior     = q("use_prior", "0") == "1",
                    court_x_sigma = float(q("court_xs",    str(_DP["court_xs"]))),
                    court_y_sigma = float(q("court_ys",    str(_DP["court_ys"]))),
                    court_inset   = int(q("court_inset",   str(_DP["court_inset"]))),
                    air_x_left    = int(q("air_xl",        str(_DP["air_xl"]))),
                    air_x_right   = int(q("air_xr",        str(_DP["air_xr"]))),
                    air_y_top     = int(q("air_yt",        str(_DP["air_yt"]))),
                    air_y_bot     = int(q("air_yb",        str(_DP["air_yb"]))),
                    air_sigma_x   = float(q("air_sx",      str(_DP["air_sx"]))),
                    air_sigma_y   = float(q("air_sy",      str(_DP["air_sy"]))),
                    pweight       = float(q("pweight",     str(_DP["pweight"]))),
                    proc_w        = res_w,
                    ball_diam     = float(q("ball_diam",   str(_DP["ball_diam"]))),
                    min_circ      = float(q("min_circ",    str(_DP["min_circ"]))),
                    min_bright    = float(q("min_bright",  str(_DP["min_bright"]))),
                    blur_k        = int(q("blur_k",        str(_DP["blur_k"]))),
                    n_diff        = int(q("n_diff",        str(_DP["n_diff"]))),
                    diff_mode     = q("diff_mode",         "abs"),
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
                    court_x_sigma = float(q("court_xs",    str(_DP["court_xs"]))),
                    court_y_sigma = float(q("court_ys",    str(_DP["court_ys"]))),
                    court_inset   = int(q("court_inset",   str(_DP["court_inset"]))),
                    air_x_left    = int(q("air_xl",        str(_DP["air_xl"]))),
                    air_x_right   = int(q("air_xr",      str(_DP["air_xr"]))),
                    air_y_top     = int(q("air_yt",      str(_DP["air_yt"]))),
                    air_y_bot     = int(q("air_yb",      str(_DP["air_yb"]))),
                    air_sigma_x   = float(q("air_sx",    str(_DP["air_sx"]))),
                    air_sigma_y   = float(q("air_sy",    str(_DP["air_sy"]))),
                    weight        = float(q("pweight",   str(_DP["pweight"]))),
                    vname         = q("video") or None,
                    frame_idx     = int(q("frame",       "0")),
                    frame_blend   = float(q("pblend",    str(_DP["pblend"]))),
                )
                self._json({"img": img_to_b64jpeg(img)})
            except Exception as exc:
                import traceback; traceback.print_exc()
                self._json({"img": "", "error": str(exc)})

        elif prs.path == "/api/tree":
            try:
                res_w        = int(q("res_w",      str(_DP["res_w"])))
                center_frame = int(q("frame",      "0"))
                gap          = int(q("gap",        str(_DP["tree_gap"])))
                n_steps      = int(q("n_steps",    str(_DP["n_steps"])))
                link_r       = int(q("link_r",     str(_DP["link_r"])))
                static_r     = int(q("static_r",   str(_DP["static_r"])))
                vel_pen      = float(q("vel_pen",  str(_DP["vel_pen"])))
                use_prior    = q("use_prior", "0") == "1"
                tree_prior   = _prior_from_q(q) if use_prior else None
                steps = collect_tree_blobs(
                    vname        = q("video"),
                    center_frame = center_frame,
                    gap          = gap,
                    n_steps      = n_steps,
                    top_k        = int(q("tree_top_k", "0")),
                    thresh       = int(q("thresh",        str(_DP["thresh"]))),
                    min_a        = int(q("min_a",          str(_DP["min_a"]))),
                    max_a        = int(q("max_a",          str(_DP["max_a"]))),
                    max_asp      = float(q("max_asp",      str(_DP["max_asp"]))),
                    method       = q("method",             str(_DP["method"])),
                    proc_w       = res_w,
                    ball_diam    = float(q("ball_diam",    str(_DP["ball_diam"]))),
                    min_circ     = float(q("min_circ",     str(_DP["min_circ"]))),
                    min_bright   = float(q("min_bright",   str(_DP["min_bright"]))),
                    blur_k       = int(q("blur_k",         str(_DP["blur_k"]))),
                    score_thresh = float(q("score_thresh", str(_DP["score_thresh"]))),
                    prior_map    = tree_prior,
                    diff_mode    = q("diff_mode", "abs"),
                )
                # Scale radii so slider values are resolution-independent
                # (calibrated for REF_W=320; scale up at native res)
                _vname = q("video")
                _aw = res_w if res_w > 0 else int(get_cap(_vname).get(cv2.CAP_PROP_FRAME_WIDTH))
                _rsc = _aw / REF_W
                chron_steps, static_flags, edges, best_path = build_path_dp(
                    steps,
                    link_radius   = max(4, int(link_r   * _rsc)),
                    static_radius = max(1, int(static_r * _rsc)),
                    vel_penalty   = vel_pen,
                )
                img = make_tree_image(
                    _vname, center_frame, res_w,
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

        elif prs.path == "/api/history":
            try:
                res_w        = int(q("res_w",    str(_DP["res_w"])))
                center_frame = int(q("frame",    "0"))
                img, n_layers = make_history_scatter(
                    vname        = q("video"),
                    center_frame = center_frame,
                    proc_w       = res_w,
                    n_hist       = int(q("n_hist",     str(_DP["n_hist"]))),
                    top_k        = int(q("top_k",       str(_DP["top_k"]))),
                    thresh       = int(q("thresh",      str(_DP["thresh"]))),
                    min_a        = int(q("min_a",        str(_DP["min_a"]))),
                    max_a        = int(q("max_a",        str(_DP["max_a"]))),
                    max_asp      = float(q("max_asp",    str(_DP["max_asp"]))),
                    method       = q("method",           str(_DP["method"])),
                    ball_diam    = float(q("ball_diam",  str(_DP["ball_diam"]))),
                    min_circ     = float(q("min_circ",   str(_DP["min_circ"]))),
                    min_bright   = float(q("min_bright", str(_DP["min_bright"]))),
                    blur_k       = int(q("blur_k",       str(_DP["blur_k"]))),
                    diff_mode    = q("diff_mode", "abs"),
                )
                self._json({
                    "img":      img_to_b64jpeg(img),
                    "n_layers": n_layers,
                })
            except Exception as exc:
                import traceback; traceback.print_exc()
                self._json({"img": "", "error": str(exc)})

        elif prs.path == "/api/track":
            try:
                res_w        = int(q("res_w",    str(_DP["res_w"])))
                center_frame = int(q("frame",    "0"))
                link_r       = int(q("link_r",   str(_DP["trk_link_r"])))
                static_r     = int(q("static_r", str(_DP["static_r"])))
                vel_pen      = float(q("vel_pen", str(_DP["trk_vel_pen"])))
                use_prior    = q("use_prior", "0") == "1"
                track_prior  = _prior_from_q(q) if use_prior else None
                layers = collect_track_blobs(
                    vname        = q("video"),
                    center_frame = center_frame,
                    proc_w       = res_w,
                    n_look       = int(q("n_look",        str(_DP["n_look"]))),
                    top_k        = int(q("top_k",          str(_DP["trk_top_k"]))),
                    thresh       = int(q("thresh",         str(_DP["thresh"]))),
                    min_a        = int(q("min_a",           str(_DP["min_a"]))),
                    max_a        = int(q("max_a",           str(_DP["max_a"]))),
                    max_asp      = float(q("max_asp",       str(_DP["max_asp"]))),
                    method       = q("method",              str(_DP["method"])),
                    ball_diam    = float(q("ball_diam",     str(_DP["ball_diam"]))),
                    min_circ     = float(q("min_circ",      str(_DP["min_circ"]))),
                    min_bright   = float(q("min_bright",    str(_DP["min_bright"]))),
                    blur_k       = int(q("blur_k",          str(_DP["blur_k"]))),
                    score_thresh = float(q("score_thresh",  "0.0")),
                    prior_map    = track_prior,
                    diff_mode    = q("diff_mode", "abs"),
                )
                # build_path_dp expects steps newest-first; layers are oldest-first
                use_dp = q("use_dp", "1") == "1"
                if use_dp:
                    steps_nf = list(reversed(layers))
                    _taw = res_w if res_w > 0 else int(get_cap(q("video")).get(cv2.CAP_PROP_FRAME_WIDTH))
                    _trsc = _taw / REF_W
                    chron_steps, static_flags, edges, best_path = build_path_dp(
                        steps_nf,
                        link_radius   = max(4, int(link_r   * _trsc)),
                        static_radius = max(1, int(static_r * _trsc)),
                        vel_penalty   = vel_pen,
                    )
                else:
                    # Scatter mode: skip DP, use all top-k blobs directly
                    chron_steps = layers   # oldest-first, same format
                    static_flags = [set() for _ in layers]
                    edges = []
                    best_path = []
                # chron_steps is oldest-first (same order as layers)
                arc = find_ransac_arc(
                    chron_steps,
                    n_iter      = int(q("ransac_iter",  "400")),
                    inlier_px   = float(q("ransac_px",  "8.0")),
                    min_inliers = int(q("ransac_min",    "4")),
                    min_span    = int(q("ransac_span",   "4")),
                    min_speed   = float(q("ransac_spd", "6.0")),
                )
                img = make_track_image(
                    q("video"), center_frame, res_w,
                    chron_steps, best_path,
                    arc_result=arc,
                )
                quality = parabolic_quality(chron_steps, best_path)
                self._json({
                    "img":        img_to_b64jpeg(img),
                    "path_len":   len(best_path),
                    "n_layers":   len(layers),
                    "quality":    quality,
                    "confidence": arc["confidence"],
                    "arc_r2":     arc["r2"],
                    "arc_cov":    arc["coverage"],
                    "arc_inliers":arc["n_inliers"],
                    "ball_at_0":  arc["ball_at_0"],
                })
            except Exception as exc:
                import traceback; traceback.print_exc()
                self._json({"img": "", "error": str(exc)})

        elif prs.path == "/api/kalman_frame":
            try:
                res_w        = int(q("res_w",        str(_DP["res_w"])))
                center_frame = int(q("frame",        "0"))
                n_history    = int(q("n_history",    "30"))
                kf_proc      = float(q("kf_proc_noise", "5.0"))
                kf_meas      = float(q("kf_meas_noise", "8.0"))
                kf_gate      = float(q("kf_gate_px",    "40.0"))
                kf_maxm      = int(q("kf_max_miss",     "8"))
                thresh       = int(q("thresh",       str(_DP["thresh"])))
                min_a        = int(q("min_a",        str(_DP["min_a"])))
                max_a        = int(q("max_a",        str(_DP["max_a"])))
                max_asp      = float(q("max_asp",    str(_DP["max_asp"])))
                method_      = q("method",           str(_DP["method"]))
                ball_diam    = float(q("ball_diam",  str(_DP["ball_diam"])))
                min_circ     = float(q("min_circ",   str(_DP["min_circ"])))
                min_bright   = float(q("min_bright", str(_DP["min_bright"])))
                blur_k       = int(q("blur_k",       str(_DP["blur_k"])))
                score_thresh = float(q("score_thresh", "0.0"))
                diff_mode_   = q("diff_mode", "abs")
                use_prior    = q("use_prior", "0") == "1"
                kf_prior     = _prior_from_q(q) if use_prior else None
                vname        = q("video")
                proc_h       = res_w * 9 // 16

                kf = KalmanBallTracker(
                    process_noise=kf_proc, measurement_noise=kf_meas,
                    gate_px=kf_gate, max_miss=kf_maxm)

                start_frame = max(0, center_frame - n_history)
                history     = []
                prev_gray   = None

                for fi in range(start_frame, center_frame + 1):
                    g, _ = read_gray_small(vname, fi, res_w)
                    if g is None:
                        prev_gray = None
                        continue
                    if prev_gray is not None:
                        _, _, _, passing, _ = detect(
                            g, prev_gray, thresh, min_a, max_a, max_asp,
                            method_, score_thresh,
                            prior_map=kf_prior, proc_w=res_w, proc_h=proc_h,
                            ball_diam=ball_diam, min_circ=min_circ, diff_mode=diff_mode_,
                            min_bright=min_bright, blur_k=blur_k)
                        kf.predict()
                        pred_pos = kf.position()
                        px_std = float(np.sqrt(max(float(kf.P[0,0]),0))) if kf.P is not None else 0.0
                        py_std = float(np.sqrt(max(float(kf.P[1,1]),0))) if kf.P is not None else 0.0
                        vx, vy = kf.velocity()
                        accepted = None
                        all_blobs = [dict(b) for b in passing]
                        if passing:
                            best = max(passing, key=lambda b: b["score"])
                            if kf.update([best["x"], best["y"]]):
                                accepted = best
                            else:
                                kf.miss()
                        else:
                            kf.miss()
                        history.append({
                            "frame":      fi,
                            "pred":       pred_pos,
                            "pos":        kf.position(),
                            "accepted":   accepted,
                            "confidence": kf.confidence,
                            "blobs":      all_blobs,
                            "px_std":     px_std,
                            "py_std":     py_std,
                            "vx":         vx,
                            "vy":         vy,
                            "hit_count":  kf.hit_count,
                            "miss_count": kf.miss_count,
                        })
                    prev_gray = g

                img  = make_kalman_image(vname, center_frame, res_w, history, kf_gate)
                last = history[-1] if history else {}
                self._json({
                    "img":        img_to_b64jpeg(img),
                    "confidence": last.get("confidence", 0.0),
                    "hit_count":  last.get("hit_count", 0),
                    "miss_count": last.get("miss_count", 0),
                    "vx":         round(last.get("vx", 0.0), 1),
                    "vy":         round(last.get("vy", 0.0), 1),
                    "n_history":  len(history),
                })
            except Exception as exc:
                import traceback; traceback.print_exc()
                self._json({"img": "", "error": str(exc)})

        elif prs.path == "/api/render_start":
            try:
                if _render_state.get("running"):
                    self._json({"ok": False, "error": "Render already running"})
                else:
                    kwargs = {k: v[0] for k, v in qs.items()}
                    t = threading.Thread(
                        target=render_tracked_video, kwargs=kwargs, daemon=True)
                    t.start()
                    self._json({"ok": True})
            except Exception as exc:
                import traceback; traceback.print_exc()
                self._json({"ok": False, "error": str(exc)})

        elif prs.path == "/api/render_status":
            self._json(dict(_render_state))

        elif prs.path == "/api/save_params":
            try:
                params = {k: v[0] for k, v in qs.items()}
                pfile  = TENNIS / "claude" / "params.json"
                pfile.parent.mkdir(parents=True, exist_ok=True)
                pfile.write_text(json.dumps(params, indent=2))
                self._json({"ok": True})
            except Exception as exc:
                import traceback; traceback.print_exc()
                self._json({"ok": False, "error": str(exc)})

        elif prs.path == "/api/load_params":
            try:
                pfile = TENNIS / "claude" / "params.json"
                if pfile.exists():
                    params = json.loads(pfile.read_text())
                    self._json({"ok": True, "params": params})
                else:
                    self._json({"ok": False, "error": "No saved params found"})
            except Exception as exc:
                import traceback; traceback.print_exc()
                self._json({"ok": False, "error": str(exc)})

        elif prs.path == "/api/export_start":
            try:
                if _export_state.get("running"):
                    self._json({"ok": False, "error": "Export already running"})
                else:
                    _export_state["cancel"] = False
                    kw = {k: v[0] for k, v in qs.items()}
                    kw.setdefault("use_prior",   "0")
                    kw.setdefault("mode",        "both")
                    kw.setdefault("conf_thresh", "0.5")
                    kw.setdefault("max_minutes", "0")
                    kw.setdefault("min_seg_sec", "0.5")
                    kw.setdefault("ransac_spd",  "6.0")
                    kw["use_prior"] = kw["use_prior"] == "1"
                    kw["out_dir"]   = str(OUT_DIR)
                    import threading
                    threading.Thread(target=scan_and_export, kwargs=kw,
                                     daemon=True).start()
                    self._json({"ok": True})
            except Exception as exc:
                import traceback; traceback.print_exc()
                self._json({"ok": False, "error": str(exc)})

        elif prs.path == "/api/export_status":
            self._json(dict(_export_state))

        elif prs.path == "/api/export_cancel":
            _export_state["cancel"] = True
            self._json({"ok": True})

        else:
            self.send_response(404)
            self.end_headers()


# ── entry point ───────────────────────────────────────────────────────────────
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
    print("")
    print(f"Starting server -> {url}")
    print("Tab 1 Detection  : tune blob filters")
    print("Tab 2 Prior      : tune spatial prior")
    print("Tab 3 Tree       : DP trajectory tree  (gap=1, prior-aware)")
    print("Tab 4 Track      : RANSAC arc + ball confidence")
    print("Press Ctrl-C to stop.")

    server = HTTPServer(("localhost", PORT), Handler)
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")

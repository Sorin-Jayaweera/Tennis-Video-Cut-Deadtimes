"""
fulldeadtimecutter.py  —  Combined audio + ball-vision tennis clip cutter.

Pipeline per video:
  1. Extract audio → bandpass → STFT → NCC matched filter → thwack peaks
     → group into rally segments (with pre/post buffers)
  2. Ball validation via RANSAC arc fitting:
       For each rally segment, stride through frames with a sliding window,
       running collect_track_blobs (same detection filters + spatial prior as
       the Detection screen in interactive_viewer) then find_ransac_arc.
       A segment is kept if any window achieves confidence ≥ ball_conf_thr.
  3. Write  <stem>_active.mp4   (validated rallies + buffer)
            <stem>_deadtime.mp4 (everything else)

Usage:
    python fulldeadtimecutter.py [--port 8789]

Web UI at http://localhost:8789
Audio params template must already exist in RunDirectory/params/
(run thwack only processing/01_make_params.ipynb once to build it).

NOTE: After validating this sequential version works, parallelize ball
tracking across audio segments (one thread per rally segment). Do NOT
implement until user confirms sequential version is correct.
"""

import sys, os, json, threading, time, shutil, tempfile, subprocess, traceback
import urllib.parse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

# ─── Locate TENNIS root ───────────────────────────────────────────────────────
def find_tennis_root():
    for p in [Path(__file__).resolve().parent,
              Path(__file__).resolve().parent.parent,
              Path.cwd(), Path.cwd().parent]:
        if (p / "videosandaudio").is_dir():
            return p
    return Path(__file__).resolve().parent.parent   # fallback

TENNIS     = find_tennis_root()
PARAMS_DIR = Path(__file__).resolve().parent / "params"
sys.path.insert(0, str(TENNIS / "claude"))

try:
    import interactive_viewer as _iv
    _IV_OK = True
except Exception as _e:
    _iv = None
    _IV_OK = False
    print(f"WARNING: could not import interactive_viewer: {_e}")
    print("  Ball validation will be skipped (all audio segments kept).")

# ─── Constants ────────────────────────────────────────────────────────────────
DEFAULT_PORT = 8789
VIDEO_EXTS   = {".mp4", ".avi", ".mov", ".mkv"}

DEFAULT_PARAMS = {
    # ── Audio detection ───────────────────────────────────────────────────────
    "k_mad":          6.0,    # MAD multiplier — lower = more detections
    "min_gap_ms":     250,    # min ms between two thwack detections
    "group_gap_s":    5.0,    # gap (s) that splits detections into separate rallies
    "pre_buffer_s":   1.0,    # seconds before first hit in a segment
    "post_single_s":  1.0,    # extra seconds after an isolated single hit
    "post_group_s":   0.5,    # extra seconds after last hit in a rally
    # ── Ball validation — detection filters ───────────────────────────────────
    # These mirror the Detection tab sliders in interactive_viewer.
    "res_w":          640,    # processing width (0 = native)
    "thresh":         18,     # frame-diff threshold
    "min_a":          3,      # min blob area (px²)
    "max_a":          800,    # max blob area
    "max_asp":        2.0,    # max aspect ratio
    "method":         "circularity",   # compactness | rog | circularity
    "ball_diam":      10.0,   # reference ball diameter at REF_W (px)
    "min_circ":       0.2,    # minimum circularity (circularity method only)
    "min_bright":     0.0,    # minimum mean brightness in diff blob
    "blur_k":         9,      # Gaussian blur kernel half-size (0 = off)
    "score_thresh":   0.0,    # minimum blob score
    "gap":            1,      # frame gap for computing frame diff
    # ── Spatial prior ─────────────────────────────────────────────────────────
    "pweight":        1.0,    # prior weight (0 = off, 1 = full)
    "court_xs":       0.1,    # court x falloff sigma
    "court_ys":       0.0,    # court y falloff sigma (0 = hard cut at bottom)
    "court_inset":    8,      # shrink court polygon inward by this many px
    "air_xl":         115,    # air zone x-left  (at REF_W=320)
    "air_xr":         193,    # air zone x-right
    "air_yt":         38,     # air zone y-top
    "air_yb":         58,     # air zone y-bottom (anchor row of max probability)
    "air_sx":         37.0,   # air zone Gaussian sigma x
    "air_sy":         40.0,   # air zone Gaussian sigma y
    # ── RANSAC arc fitting ─────────────────────────────────────────────────────
    "n_look":         10,     # ±frame window for collect_track_blobs
    "top_k":          3,      # top-k blobs per diff layer fed to RANSAC
    "ransac_px":      16.0,   # inlier pixel threshold for arc fitting
    "min_inliers":    4,      # minimum RANSAC inliers to accept an arc
    "min_span":       4,      # minimum frame span across 3 RANSAC sample points
    "scan_stride":    10,     # frames between sliding RANSAC windows (0 = n_look)
    # ── Thresholds & buffers ──────────────────────────────────────────────────
    "ball_conf_thr":  0.20,   # min RANSAC confidence (r2 × coverage) to keep seg
    "buf_sec":        0.25,   # extra seconds added each side of kept segments
    # ── Debug time range (0 = full video) ─────────────────────────────────────
    # When debug_end_s > 0, ball validation only runs on segments overlapping
    # [debug_start_s, debug_end_s]. Segments outside that window are kept as-is.
    "debug_start_s":  0.0,
    "debug_end_s":    0.0,
}

# ─── Mutable state ────────────────────────────────────────────────────────────
_params    = dict(DEFAULT_PARAMS)
_video_dir = TENNIS / "videosandaudio"
_job = {
    "running":  False,
    "cancel":   False,
    "phase":    "idle",
    "progress": 0,
    "total":    0,
    "log":      [],
    "results":  {},
    "error":    "",
}
_job_lock = threading.Lock()

# ─── Utility ──────────────────────────────────────────────────────────────────
def _log(msg):
    ts   = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with _job_lock:
        _job["log"].append(line)
        if len(_job["log"]) > 600:
            _job["log"] = _job["log"][-600:]

def list_videos():
    try:
        import cv2
        out = []
        for p in sorted(_video_dir.iterdir()):
            if p.suffix.lower() not in VIDEO_EXTS:
                continue
            size_mb = p.stat().st_size / 1024 / 1024
            cap     = cv2.VideoCapture(str(p))
            fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            out.append({
                "name":         p.name,
                "size_mb":      round(size_mb, 1),
                "duration_min": round(frames / fps / 60, 1),
            })
        return out
    except Exception:
        return []

def find_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    raise RuntimeError("ffmpeg not found — run: pip install imageio-ffmpeg")


# ─── Phase 1: Audio pipeline ──────────────────────────────────────────────────
def _bandpass(audio, sr, lo=1000.0, hi=8000.0, order=4):
    from scipy.signal import butter, sosfiltfilt
    nyq = 0.5 * sr
    sos = butter(order, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, audio).astype("float32")

def _ncc(template, spec):
    from scipy.signal import correlate2d
    t = (template - template.mean()) / (template.std() + 1e-8)
    s = (spec     - spec.mean())     / (spec.std()     + 1e-8)
    return correlate2d(s, t, mode="valid")

def run_audio_pipeline(video_path, p):
    """
    Returns (rally_segs, video_duration_s).
    rally_segs = [[start_s, end_s], ...]  sorted and merged.
    """
    import numpy as np, librosa
    from scipy.signal import find_peaks

    if not PARAMS_DIR.exists() or not (PARAMS_DIR / "template.npy").exists():
        raise RuntimeError(
            f"Audio params not found at {PARAMS_DIR}\n"
            "Run RunDirectory/thwack only processing/01_make_params.ipynb first.")

    template   = np.load(PARAMS_DIR / "template.npy")
    sr_templ   = int(np.load(PARAMS_DIR / "sr.npy")[0])
    hop_length = int(np.load(PARAMS_DIR / "hop_length.npy")[0])
    n_fft      = int(np.load(PARAMS_DIR / "n_fft.npy")[0])
    pre_ms     = float(np.load(PARAMS_DIR / "pre_ms.npy")[0])

    ffmpeg = find_ffmpeg()
    tmp    = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    _log("  Extracting audio from video ...")
    res = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error",
         "-i", str(video_path), "-vn", "-ac", "1", "-ar", str(sr_templ),
         "-acodec", "pcm_s16le", "-f", "wav", tmp.name],
        capture_output=True)
    if res.returncode != 0:
        os.unlink(tmp.name)
        raise RuntimeError(res.stderr.decode(errors="replace"))

    audio_raw, sr_check = librosa.load(tmp.name, sr=None)
    os.unlink(tmp.name)
    if sr_check != sr_templ:
        raise RuntimeError(f"Audio sample rate {sr_check} != template sr {sr_templ}")

    video_dur = len(audio_raw) / sr_check
    _log(f"  Audio: {video_dur / 60:.1f} min @ {sr_check} Hz")

    audio_bp = _bandpass(audio_raw, sr_check)

    _log("  Running matched filter (STFT + NCC) ...")
    mag  = np.abs(librosa.stft(audio_bp, n_fft=n_fft, hop_length=hop_length))
    corr = _ncc(template, mag)
    r    = corr[0].astype("float32")

    med = np.median(r)
    mad = np.median(np.abs(r - med))
    thr = med + float(p["k_mad"]) * 1.4826 * (mad + 1e-12)
    gap = max(1, int(round(float(p["min_gap_ms"]) / 1000 * sr_check / hop_length)))
    idx, _ = find_peaks(r, height=thr, distance=gap)
    times  = idx * (hop_length / sr_check) + pre_ms / 1000
    _log(f"  Detected {len(times)} thwacks  (threshold={thr:.3f})")

    if len(times) == 0:
        return [], video_dur

    # Group detections into rally segments
    gg    = float(p["group_gap_s"])
    pre   = float(p["pre_buffer_s"])
    post1 = float(p["post_single_s"])
    postN = float(p["post_group_s"])

    sorted_t   = sorted(times)
    groups, cur = [], [sorted_t[0]]
    for t in sorted_t[1:]:
        if t - cur[-1] <= gg:
            cur.append(t)
        else:
            groups.append(cur)
            cur = [t]
    groups.append(cur)

    segs = []
    for g in groups:
        s = max(0.0,       g[0] - pre)
        e = min(video_dur, g[-1] + (post1 if len(g) == 1 else postN))
        segs.append([s, e])

    segs.sort()
    merged = [segs[0]]
    for s, e in segs[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    total_s = sum(e - s for s, e in merged)
    _log(f"  Rally segments: {len(merged)}  "
         f"({total_s / 60:.1f} min = {total_s / video_dur * 100:.0f}% of video)")
    return merged, video_dur


# ─── Phase 2: Ball validation (RANSAC arc fitting) ────────────────────────────

def _build_prior(p):
    """Build prior map from current params. Returns (prior_map, error_str_or_None)."""
    if not _IV_OK or float(p["pweight"]) <= 0:
        return None, None
    try:
        pm = _iv.compute_prior_map(
            court_x_sigma = float(p["court_xs"]),
            court_y_sigma = float(p["court_ys"]),
            court_inset   = int(p["court_inset"]),
            air_x_left    = int(p["air_xl"]),
            air_x_right   = int(p["air_xr"]),
            air_y_top     = int(p["air_yt"]),
            air_y_bot     = int(p["air_yb"]),
            air_sigma_x   = float(p["air_sx"]),
            air_sigma_y   = float(p["air_sy"]),
            weight        = float(p["pweight"]),
        )
        return pm, None
    except Exception as e:
        return None, str(e)


def validate_with_ball(video_name, rally_segs, video_dur, p):
    """
    Validate each rally segment by running a strided RANSAC arc scan:

      For each segment [seg_start, seg_end]:
        - Convert to frame indices [fa, fb]
        - Stride through frames at interval `scan_stride`
        - At each position, call collect_track_blobs (same filters + prior as
          the Detection screen) then find_ransac_arc
        - If any window yields confidence >= ball_conf_thr, keep the segment
        - Expand kept segments by buf_sec on each side

    debug_start_s / debug_end_s:
      When debug_end_s > 0, only segments overlapping [debug_start_s, debug_end_s]
      are validated. Segments outside that range are kept without ball-checking
      (so you can test a small slice quickly without re-cutting the whole video).

    Returns (kept_segs, dropped_segs).
    """
    import cv2

    if not _IV_OK:
        _log("  interactive_viewer not available — keeping all segments (no ball check)")
        return [list(s) for s in rally_segs], []

    # Prior map
    prior_map, prior_err = _build_prior(p)
    if prior_err:
        _log(f"  WARNING: prior map failed ({prior_err}) — scanning without prior")

    # Video metadata
    video_path = _iv.VIDEO_DIR / video_name
    cap_tmp    = cv2.VideoCapture(str(video_path))
    fps        = cap_tmp.get(cv2.CAP_PROP_FPS) or 25.0
    n_total    = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
    proc_w     = int(p["res_w"])
    if proc_w == 0:
        proc_w = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_tmp.release()

    # Scan parameters
    n_look  = max(1, int(p["n_look"]))
    top_k   = max(1, int(p["top_k"]))
    raw_stride = int(p.get("scan_stride", 0))
    stride  = raw_stride if raw_stride > 0 else n_look

    ball_thr = float(p["ball_conf_thr"])
    buf_s    = float(p["buf_sec"])

    # Debug range
    dbg_s = float(p.get("debug_start_s", 0.0))
    dbg_e = float(p.get("debug_end_s",   0.0))
    use_debug_range = dbg_e > 0.0

    # Detection kwargs forwarded to collect_track_blobs
    det_kw = dict(
        n_look      = n_look,
        top_k       = top_k,
        thresh      = int(p["thresh"]),
        min_a       = int(p["min_a"]),
        max_a       = int(p["max_a"]),
        max_asp     = float(p["max_asp"]),
        method      = str(p["method"]),
        ball_diam   = float(p["ball_diam"]),
        min_circ    = float(p["min_circ"]),
        min_bright  = float(p["min_bright"]),
        blur_k      = int(p["blur_k"]),
        score_thresh = float(p["score_thresh"]),
        gap         = int(p["gap"]),
        prior_map   = prior_map,
    )

    # RANSAC kwargs
    ransac_kw = dict(
        n_iter      = 400,
        inlier_px   = float(p["ransac_px"]),
        min_inliers = int(p["min_inliers"]),
        min_span    = int(p["min_span"]),
    )

    _log(f"  RANSAC ball scan: {len(rally_segs)} segs, "
         f"n_look={n_look}, stride={stride}, res={proc_w}px, "
         f"prior={'on' if prior_map is not None else 'off'}")
    if use_debug_range:
        _log(f"  DEBUG RANGE: only validating segments overlapping "
             f"{dbg_s:.1f}s – {dbg_e:.1f}s (others kept as-is)")

    kept, dropped = [], []
    total_segs    = len(rally_segs)

    for seg_i, seg in enumerate(rally_segs):
        if _job["cancel"]:
            break

        seg_s = float(seg[0])
        seg_e = float(seg[1])

        # ── Debug range: skip validation outside the window ───────────────────
        if use_debug_range:
            in_range = (seg_e >= dbg_s) and (seg_s <= dbg_e)
            if not in_range:
                kept.append([max(0.0, seg_s - buf_s), min(video_dur, seg_e + buf_s)])
                _log(f"  [{seg_i+1}/{total_segs}] {seg_s:.1f}–{seg_e:.1f}s  "
                     f"→ outside debug range, kept")
                continue

        # ── Convert to frame indices ──────────────────────────────────────────
        fa = max(0,           int(seg_s * fps))
        fb = min(n_total - 1, int(seg_e * fps))

        if fb - fa < 2 * n_look:
            # Segment too short to meaningfully scan; run a single window at center
            centers = [max(n_look, min((fa + fb) // 2, n_total - n_look - 1))]
        else:
            # Stride: start at fa+n_look (need context before), stop at fb-n_look
            start = fa + n_look
            end   = fb - n_look
            centers = list(range(start, end + 1, stride))
            if not centers or centers[-1] < end:
                centers.append(end)   # always include the tail

        # ── Strided RANSAC scan ───────────────────────────────────────────────
        max_conf  = 0.0
        n_windows = 0
        hit_frame = None

        for pos in centers:
            if _job["cancel"]:
                break

            center = max(n_look, min(pos, n_total - n_look - 1))

            # Progress: express as (seg_i * 1000 + window_within_seg)
            with _job_lock:
                _job["progress"] = seg_i * 1000 + n_windows
                _job["total"]    = total_segs * 1000

            try:
                layers = _iv.collect_track_blobs(video_name, center, proc_w, **det_kw)
                arc    = _iv.find_ransac_arc(layers, **ransac_kw)
                conf   = float(arc["confidence"])
            except Exception as exc:
                _log(f"    WARNING frame {center}: {exc}")
                conf = 0.0

            n_windows += 1
            if conf > max_conf:
                max_conf  = conf
                hit_frame = center

            if max_conf >= ball_thr:
                break   # early exit — no need to scan the rest of the segment

        # ── Decision ─────────────────────────────────────────────────────────
        if max_conf >= ball_thr:
            expanded_s = max(0.0,        seg_s - buf_s)
            expanded_e = min(video_dur,  seg_e + buf_s)
            kept.append([expanded_s, expanded_e])
            _log(f"  [{seg_i+1}/{total_segs}] {seg_s:.1f}–{seg_e:.1f}s  "
                 f"✓ KEPT   conf={max_conf:.3f}  (frame {hit_frame}, "
                 f"{n_windows} window{'s' if n_windows != 1 else ''})")
        else:
            dropped.append(seg)
            _log(f"  [{seg_i+1}/{total_segs}] {seg_s:.1f}–{seg_e:.1f}s  "
                 f"✗ DROPPED  max_conf={max_conf:.3f}  "
                 f"({n_windows} window{'s' if n_windows != 1 else ''})")

    # ── Merge overlapping kept segments ───────────────────────────────────────
    kept.sort()
    if len(kept) > 1:
        merged = [list(kept[0])]
        for s, e in kept[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        kept = merged

    _log(f"  Ball validation complete: {len(kept)} kept, {len(dropped)} dropped")
    return kept, dropped


# ─── Phase 3: Write output videos ────────────────────────────────────────────
def write_videos(video_path, active_segs, video_dur, out_dir, stem):
    """
    Cut active segments and dead-time (complement) using ffmpeg -c copy.
    Returns (active_path, deadtime_path).
    """
    ffmpeg = find_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dead time = complement of active segments
    dead_segs, prev = [], 0.0
    for s, e in sorted(active_segs):
        if s > prev + 0.1:
            dead_segs.append([prev, s])
        prev = e
    if prev < video_dur - 0.1:
        dead_segs.append([prev, video_dur])

    def cut_segs(segs, out_path, label):
        if not segs:
            _log(f"  No {label} segments — skipping")
            return None
        tmp_dir = out_dir / f"_tmp_{label}"
        tmp_dir.mkdir(exist_ok=True)
        seg_paths = []
        _log(f"  Cutting {len(segs)} {label} clips ...")
        for i, (s, e) in enumerate(segs):
            sp = tmp_dir / f"seg_{i:04d}.mp4"
            res = subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error",
                 "-ss", f"{s:.3f}", "-to", f"{e:.3f}",
                 "-i", str(video_path), "-c", "copy", str(sp)],
                capture_output=True)
            if res.returncode == 0 and sp.exists():
                seg_paths.append(sp)
            else:
                _log(f"    WARNING seg {i} failed: "
                     f"{res.stderr.decode(errors='replace')[:120]}")
            with _job_lock:
                _job["progress"] = i + 1
                _job["total"]    = len(segs)

        if not seg_paths:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return None

        lst = tmp_dir / "list.txt"
        lst.write_text("".join(f"file '{p.resolve()}'\n" for p in seg_paths),
                       encoding="utf-8")
        res = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error",
             "-f", "concat", "-safe", "0",
             "-i", str(lst), "-c", "copy", str(out_path)],
            capture_output=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if res.returncode != 0:
            _log(f"  Concat failed: {res.stderr.decode(errors='replace')[:200]}")
            return None
        total_s = sum(e - s for s, e in segs)
        mb      = out_path.stat().st_size / 1e6
        _log(f"  {label}: {len(segs)} clips · {total_s / 60:.1f} min "
             f"· {mb:.1f} MB → {out_path.name}")
        return str(out_path)

    _log("  Writing active video ...")
    with _job_lock: _job["phase"] = "writing active video"
    active_path = cut_segs(active_segs, out_dir / f"{stem}_active.mp4",   "active")

    _log("  Writing deadtime video ...")
    with _job_lock: _job["phase"] = "writing deadtime video"
    dead_path   = cut_segs(dead_segs,   out_dir / f"{stem}_deadtime.mp4", "deadtime")

    return active_path, dead_path


# ─── Main job (background thread) ────────────────────────────────────────────
def run_job(video_name, params_override, out_folder_str):
    with _job_lock:
        _job.update(running=True, cancel=False, phase="starting",
                    progress=0, total=0, log=[], results={}, error="")

    p = dict(DEFAULT_PARAMS)
    p.update(params_override)

    stem    = Path(video_name).stem
    out_dir = (Path(out_folder_str) if out_folder_str
               else _video_dir / stem)

    try:
        video_path = _video_dir / video_name
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        _log(f"=== {video_name} ===")
        _log(f"Output dir: {out_dir}")

        # ── Phase 1: Audio ────────────────────────────────────────────────────
        with _job_lock: _job["phase"] = "audio detection"
        rally_segs, video_dur = run_audio_pipeline(video_path, p)

        if _job["cancel"]: _log("Cancelled."); _done("cancelled"); return

        if not rally_segs:
            _log("No rally segments found — check K_MAD or audio template.")
            _done("done — no audio detections", error="No thwack detections found")
            return

        # ── Phase 2: Ball validation ──────────────────────────────────────────
        with _job_lock: _job["phase"] = "ball scan"
        active_segs, dropped_segs = validate_with_ball(
            video_name, rally_segs, video_dur, p)

        if _job["cancel"]: _log("Cancelled."); _done("cancelled"); return

        if not active_segs:
            _log("All segments dropped — try lowering ball_conf_thr "
                 "or adjusting detection filters.")
            _done("done — no ball segments", error="Ball validation dropped all segments")
            return

        # ── Phase 3: Write videos ─────────────────────────────────────────────
        with _job_lock: _job["phase"] = "writing videos"
        active_path, dead_path = write_videos(
            video_path, active_segs, video_dur, out_dir, stem)

        with _job_lock:
            _job["results"] = {
                "active":    active_path or "",
                "deadtime":  dead_path   or "",
                "n_audio":   len(rally_segs),
                "n_kept":    len(active_segs),
                "n_dropped": len(dropped_segs),
            }
        _done("done")
        _log("=== Complete ===")

    except Exception as exc:
        traceback.print_exc()
        _done("error", error=str(exc))
        _log(f"ERROR: {exc}")


def _done(phase, error=""):
    with _job_lock:
        _job["running"] = False
        _job["phase"]   = phase
        _job["error"]   = error


# ─── Embedded HTML ────────────────────────────────────────────────────────────
_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Full Deadtime Cutter</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#111;color:#ccc;font-family:'Segoe UI',sans-serif;font-size:13px}
h1{color:#7df;padding:14px 18px 10px;font-size:17px;border-bottom:1px solid #222}
h2{color:#888;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px}
.wrap{display:flex;height:calc(100vh - 49px)}
.col{padding:14px;overflow-y:auto;border-right:1px solid #222}
.col-l{width:270px;flex-shrink:0}
.col-m{width:320px;flex-shrink:0}
.col-r{flex:1;min-width:0}
input[type=text],input[type=number]{
  background:#1a1a1a;border:1px solid #333;color:#ddd;
  padding:5px 8px;border-radius:4px;font-size:12px}
input[type=range]{width:100%;accent-color:#37a}
select{background:#1a1a1a;border:1px solid #333;color:#ddd;
       padding:5px 7px;border-radius:4px;font-size:12px}
button{cursor:pointer;border:none;border-radius:4px;padding:7px 14px;font-size:12px}
.btn-go{background:#0d2a1a;border:2px solid #3a7;color:#5e5}
.btn-go:hover{background:#1a3d25}
.btn-stop{background:#2a0d0d;border:2px solid #a33;color:#e55}
.btn-stop:hover{background:#3d1a1a}
.btn-sm{background:#1e1e1e;border:1px solid #333;color:#aaa;padding:5px 10px}
.btn-sm:hover{background:#2a2a2a}
.vi{padding:8px 10px;border-radius:5px;cursor:pointer;margin-bottom:4px;
    border:1px solid #222;background:#1a1a1a}
.vi:hover{background:#222}
.vi.sel{background:#0d2a1a;border-color:#3a7}
.vi .vn{color:#eee;font-weight:500;font-size:12px}
.vi .vm{color:#666;font-size:11px;margin-top:2px}
.row{display:flex;align-items:center;gap:6px;margin-bottom:9px}
.row label{color:#888;font-size:11px;min-width:155px;flex-shrink:0}
.row .v{color:#7df;font-size:11px;min-width:38px;text-align:right;white-space:nowrap}
.sep{height:1px;background:#1e1e1e;margin:12px 0}
#log{background:#0a0a0a;border:1px solid #1e1e1e;border-radius:4px;
     padding:8px;height:220px;overflow-y:auto;
     font-family:monospace;font-size:11px;color:#9a9;white-space:pre-wrap}
.badge{display:inline-block;padding:3px 10px;border-radius:10px;font-size:11px;
       background:#1a2a1a;color:#5e5;border:1px solid #3a7}
.badge.err{background:#2a1a1a;color:#e55;border-color:#a33}
.badge.warn{background:#2a1e0a;color:#fa5;border-color:#a73}
progress{width:100%;height:7px;accent-color:#3a7;border-radius:4px}
.rbox{background:#0d1a0d;border:1px solid #2a5;border-radius:6px;
      padding:12px;margin-top:10px}
.rpath{color:#7df;font-size:11px;word-break:break-all;margin-top:4px}
.stat{display:inline-block;background:#1a2a1a;border-radius:4px;
      padding:3px 8px;font-size:11px;color:#7df;margin-right:5px;margin-bottom:5px}
.sec{margin-bottom:18px}
.adv-toggle{color:#4a8a4a;font-size:10px;cursor:pointer;text-decoration:underline;
            margin-top:6px;display:block}
.adv-toggle:hover{color:#7df}
.adv{display:none}
.adv.open{display:block}
.hint{color:#555;font-size:10px;margin-top:-6px;margin-bottom:8px}
.dbg-on label{color:#fa8}
</style>
</head>
<body>
<h1>⚾ Full Deadtime Cutter</h1>
<div class="wrap">

<!-- LEFT: folder + video list -->
<div class="col col-l">
  <div class="sec">
    <h2>Video Folder</h2>
    <div style="display:flex;gap:5px;margin-bottom:6px">
      <input id="fi" type="text" placeholder="Paste folder path…" style="flex:1">
      <button class="btn-sm" onclick="setFolder()">Set</button>
    </div>
    <div id="fstat" style="color:#555;font-size:11px"></div>
  </div>
  <div class="sep"></div>
  <div class="sec">
    <h2>Videos</h2>
    <div id="vlist"><div style="color:#444">Set a folder above</div></div>
  </div>
  <div class="sep"></div>
  <div class="sec">
    <h2>Output Folder <span style="color:#444;text-transform:none;font-size:10px">(optional)</span></h2>
    <input id="outf" type="text" placeholder="Default: video-folder/stem/" style="width:100%">
    <div style="color:#444;font-size:10px;margin-top:4px">Leave blank → auto subfolder per video</div>
  </div>
</div>

<!-- MID: params -->
<div class="col col-m">

  <!-- Audio -->
  <div class="sec">
    <h2>Audio Detection</h2>
    <div class="row">
      <label>Sensitivity (K-MAD) ↓ = more</label>
      <input type="range" min="1" max="15" step="0.5" value="6" id="k_mad"
             oninput="$('k_mad_v').textContent=parseFloat(this.value).toFixed(1)">
      <span class="v" id="k_mad_v">6.0</span>
    </div>
    <div class="row">
      <label>Min gap between hits (ms)</label>
      <input type="number" min="50" max="2000" step="50" value="250" id="min_gap_ms" style="width:72px">
    </div>
    <div class="row">
      <label>Rally gap (s)</label>
      <input type="number" min="1" max="30" step="0.5" value="5" id="group_gap_s" style="width:72px">
    </div>
    <div class="row">
      <label>Pre-buffer (s)</label>
      <input type="number" min="0" max="10" step="0.25" value="1" id="pre_buffer_s" style="width:72px">
    </div>
    <div class="row">
      <label>Post-buffer single hit (s)</label>
      <input type="number" min="0" max="10" step="0.25" value="1" id="post_single_s" style="width:72px">
    </div>
    <div class="row">
      <label>Post-buffer rally (s)</label>
      <input type="number" min="0" max="10" step="0.25" value="0.5" id="post_group_s" style="width:72px">
    </div>
  </div>

  <div class="sep"></div>

  <!-- Ball validation -->
  <div class="sec">
    <h2>Ball Validation (RANSAC)</h2>

    <div class="row">
      <label>Processing resolution</label>
      <select id="res_w">
        <option value="320">320px (fast)</option>
        <option value="480">480px</option>
        <option value="640" selected>640px</option>
        <option value="960">960px</option>
        <option value="0">Native</option>
      </select>
    </div>

    <div class="row">
      <label>Confidence threshold</label>
      <input type="range" min="0" max="1" step="0.05" value="0.2" id="ball_conf_thr"
             oninput="$('ball_conf_v').textContent=parseFloat(this.value).toFixed(2)">
      <span class="v" id="ball_conf_v">0.20</span>
    </div>
    <div class="hint">RANSAC confidence = r² × coverage. 0.20 is a good starting point.</div>

    <div class="row">
      <label>Buffer around kept segs (s)</label>
      <input type="range" min="0" max="3" step="0.05" value="0.25" id="buf_sec"
             oninput="$('buf_v').textContent=parseFloat(this.value).toFixed(2)">
      <span class="v" id="buf_v">0.25</span>
    </div>

    <div class="row">
      <label>Frame diff threshold</label>
      <input type="number" min="1" max="60" step="1" value="18" id="thresh" style="width:72px">
    </div>
    <div class="row">
      <label>Max aspect ratio</label>
      <input type="number" min="1" max="5" step="0.1" value="2.0" id="max_asp" style="width:72px">
    </div>
    <div class="row">
      <label>Gaussian blur k</label>
      <input type="number" min="0" max="21" step="2" value="9" id="blur_k" style="width:72px">
    </div>
    <div class="row">
      <label>Spatial prior weight</label>
      <input type="range" min="0" max="1" step="0.05" value="1.0" id="pweight"
             oninput="$('pweight_v').textContent=parseFloat(this.value).toFixed(2)">
      <span class="v" id="pweight_v">1.00</span>
    </div>
    <div class="row">
      <label>RANSAC window (±frames)</label>
      <input type="number" min="3" max="30" step="1" value="10" id="n_look" style="width:72px">
    </div>
    <div class="row">
      <label>Scan stride (frames)</label>
      <input type="number" min="1" max="60" step="1" value="10" id="scan_stride" style="width:72px">
    </div>

    <span class="adv-toggle" onclick="toggleAdv('adv-ball')">▶ Advanced detection params</span>
    <div id="adv-ball" class="adv">
      <div style="height:8px"></div>
      <div class="row">
        <label>Frame diff gap (frames)</label>
        <input type="number" min="1" max="10" step="1" value="1" id="gap" style="width:72px">
      </div>
      <div class="row">
        <label>Blob method</label>
        <select id="method">
          <option value="circularity" selected>Circularity</option>
          <option value="compactness">Compactness</option>
          <option value="rog">Radius of gyration</option>
        </select>
      </div>
      <div class="row">
        <label>Min blob area (px²)</label>
        <input type="number" min="1" max="200" step="1" value="3" id="min_a" style="width:72px">
      </div>
      <div class="row">
        <label>Max blob area (px²)</label>
        <input type="number" min="50" max="5000" step="50" value="800" id="max_a" style="width:72px">
      </div>
      <div class="row">
        <label>Ball diameter ref (px)</label>
        <input type="number" min="2" max="50" step="0.5" value="10" id="ball_diam" style="width:72px">
      </div>
      <div class="row">
        <label>Min circularity</label>
        <input type="number" min="0" max="1" step="0.05" value="0.2" id="min_circ" style="width:72px">
      </div>
      <div class="row">
        <label>Min brightness</label>
        <input type="number" min="0" max="255" step="1" value="0" id="min_bright" style="width:72px">
      </div>
      <div class="row">
        <label>Score threshold</label>
        <input type="number" min="0" max="1" step="0.01" value="0" id="score_thresh" style="width:72px">
      </div>
      <div class="row">
        <label>Top-K blobs/layer</label>
        <input type="number" min="1" max="10" step="1" value="3" id="top_k" style="width:72px">
      </div>
      <div class="row">
        <label>RANSAC inlier px</label>
        <input type="number" min="2" max="60" step="1" value="16" id="ransac_px" style="width:72px">
      </div>
      <div class="row">
        <label>Min RANSAC inliers</label>
        <input type="number" min="2" max="20" step="1" value="4" id="min_inliers" style="width:72px">
      </div>
      <div class="row">
        <label>Min RANSAC span</label>
        <input type="number" min="2" max="20" step="1" value="4" id="min_span" style="width:72px">
      </div>
    </div>
  </div>

  <div class="sep"></div>

  <!-- Debug range -->
  <div class="sec" id="dbg-sec">
    <h2>Debug Time Range
      <span style="color:#555;text-transform:none;font-size:10px;margin-left:4px">(0 = full video)</span>
    </h2>
    <div class="hint">When End > 0, only validates segments in this window. Others are kept unchanged.</div>
    <div class="row" id="dbg-row-s">
      <label>Start (seconds)</label>
      <input type="number" min="0" step="1" value="0" id="debug_start_s" style="width:80px"
             oninput="updateDebugHint()">
    </div>
    <div class="row" id="dbg-row-e">
      <label>End (seconds, 0 = off)</label>
      <input type="number" min="0" step="1" value="0" id="debug_end_s" style="width:80px"
             oninput="updateDebugHint()">
    </div>
    <div id="dbg-hint" style="color:#555;font-size:10px;margin-top:4px"></div>
  </div>

</div>

<!-- RIGHT: run + results + log -->
<div class="col col-r" style="border-right:none">
  <div class="sec">
    <h2>Selected Video</h2>
    <div id="sel-name" style="color:#555;margin-bottom:10px">— none —</div>
    <div style="display:flex;gap:8px">
      <button class="btn-go"  id="btn-run"    onclick="startJob()">▶ Run Pipeline</button>
      <button class="btn-stop" id="btn-cancel" onclick="cancelJob()" style="display:none">✕ Cancel</button>
    </div>
  </div>
  <div class="sep"></div>
  <div class="sec">
    <h2>Progress</h2>
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
      <span class="badge" id="phase-badge">idle</span>
      <span id="prog-txt" style="color:#555;font-size:11px"></span>
    </div>
    <progress id="pbar" value="0" max="100" style="margin-bottom:10px"></progress>
    <div id="result-area"></div>
  </div>
  <div class="sep"></div>
  <div class="sec">
    <h2>Log</h2>
    <div id="log"></div>
  </div>
</div>

</div>
<script>
const $ = id => document.getElementById(id);
let _sel = null, _poll = null;

function toggleAdv(id) {
  const el = $(id);
  const tog = el.previousElementSibling;
  el.classList.toggle('open');
  tog.textContent = (el.classList.contains('open') ? '▼' : '▶') +
                    tog.textContent.slice(1);
}

function updateDebugHint() {
  const s = parseFloat($('debug_start_s').value) || 0;
  const e = parseFloat($('debug_end_s').value)   || 0;
  const hint = $('dbg-hint');
  const rows = [$('dbg-row-s'), $('dbg-row-e')];
  if (e > 0) {
    hint.textContent = `Validating ${(e-s).toFixed(0)}s window (${s.toFixed(0)}s – ${e.toFixed(0)}s).`;
    hint.style.color = '#fa8';
    rows.forEach(r => r.classList.add('dbg-on'));
  } else {
    hint.textContent = 'Full video mode (debug range off).';
    hint.style.color = '#555';
    rows.forEach(r => r.classList.remove('dbg-on'));
  }
}

function setFolder() {
  const f = $('fi').value.trim();
  if (!f) return;
  fetch('/api/set_folder', {method:'POST',
    headers:{'Content-Type':'application/json'}, body:JSON.stringify({folder:f})})
  .then(r=>r.json()).then(d=>{
    $('fstat').textContent = d.ok ? `${d.count} video(s) found` : ('Error: '+d.error);
    loadVideos();
  });
}

function loadVideos() {
  fetch('/api/videos').then(r=>r.json()).then(d=>{
    const el = $('vlist');
    if (!d.videos.length){el.innerHTML='<div style="color:#444">No videos found</div>';return;}
    el.innerHTML = d.videos.map(v=>`
      <div class="vi ${v.name===_sel?'sel':''}" onclick="selVid('${v.name.replace(/'/g,"\\'")}')">
        <div class="vn">${v.name}</div>
        <div class="vm">${v.size_mb} MB · ${v.duration_min} min</div>
      </div>`).join('');
  });
}

function selVid(name){
  _sel=name; $('sel-name').textContent=name; $('sel-name').style.color='#7df'; loadVideos();
}

function getParams(){
  return {
    k_mad:          parseFloat($('k_mad').value),
    min_gap_ms:     parseInt($('min_gap_ms').value),
    group_gap_s:    parseFloat($('group_gap_s').value),
    pre_buffer_s:   parseFloat($('pre_buffer_s').value),
    post_single_s:  parseFloat($('post_single_s').value),
    post_group_s:   parseFloat($('post_group_s').value),
    res_w:          parseInt($('res_w').value),
    ball_conf_thr:  parseFloat($('ball_conf_thr').value),
    buf_sec:        parseFloat($('buf_sec').value),
    thresh:         parseInt($('thresh').value),
    max_asp:        parseFloat($('max_asp').value),
    blur_k:         parseInt($('blur_k').value),
    pweight:        parseFloat($('pweight').value),
    n_look:         parseInt($('n_look').value),
    scan_stride:    parseInt($('scan_stride').value),
    // Advanced
    gap:            parseInt($('gap').value),
    method:         $('method').value,
    min_a:          parseInt($('min_a').value),
    max_a:          parseInt($('max_a').value),
    ball_diam:      parseFloat($('ball_diam').value),
    min_circ:       parseFloat($('min_circ').value),
    min_bright:     parseFloat($('min_bright').value),
    score_thresh:   parseFloat($('score_thresh').value),
    top_k:          parseInt($('top_k').value),
    ransac_px:      parseFloat($('ransac_px').value),
    min_inliers:    parseInt($('min_inliers').value),
    min_span:       parseInt($('min_span').value),
    // Debug range
    debug_start_s:  parseFloat($('debug_start_s').value) || 0,
    debug_end_s:    parseFloat($('debug_end_s').value)   || 0,
  };
}

function startJob(){
  if(!_sel){alert('Select a video first');return;}
  fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({video:_sel, params:getParams(), out_folder:$('outf').value.trim()})})
  .then(r=>r.json()).then(d=>{
    if(!d.ok){alert(d.error);return;}
    $('btn-run').style.display='none'; $('btn-cancel').style.display='';
    $('result-area').innerHTML='';
    startPoll();
  });
}

function cancelJob(){fetch('/api/cancel',{method:'POST'}).then(r=>r.json());}

function startPoll(){
  if(_poll) clearInterval(_poll);
  _poll = setInterval(pollStatus, 1200);
}

function pollStatus(){
  fetch('/api/status').then(r=>r.json()).then(d=>{
    const badge=$('phase-badge');
    badge.textContent = d.phase||'idle';
    badge.className   = 'badge'+(d.phase==='error'?' err':d.phase==='cancelled'?' warn':'');
    const pct = d.total>0 ? Math.round(100*d.progress/d.total) : 0;
    $('pbar').value        = pct;
    $('prog-txt').textContent = d.total>0 ? `${d.progress} / ${d.total}` : '';
    const log=$('log');
    log.textContent = (d.log_tail||[]).join('\n');
    log.scrollTop   = log.scrollHeight;
    if(!d.running){
      clearInterval(_poll); _poll=null;
      $('btn-run').style.display=''; $('btn-cancel').style.display='none';
      if(d.results && d.results.active){
        const r=d.results;
        $('result-area').innerHTML=`<div class="rbox">
          <div style="margin-bottom:8px">
            <span class="stat">${r.n_audio} audio segs</span>
            <span class="stat">${r.n_kept} with ball</span>
            <span class="stat">${r.n_dropped} dropped</span>
          </div>
          <div style="color:#6a8;font-size:11px">Active:</div>
          <div class="rpath">${r.active}</div>
          <div style="color:#6a8;font-size:11px;margin-top:8px">Deadtime:</div>
          <div class="rpath">${r.deadtime}</div>
        </div>`;
      }
      if(d.error){
        $('result-area').innerHTML+=`<div style="color:#e55;margin-top:8px;font-size:12px">
          Error: ${d.error}</div>`;
      }
    }
  });
}

// Init
updateDebugHint();
fetch('/api/status').then(r=>r.json()).then(d=>{ if(d.running) startPoll(); });
fetch('/api/folder').then(r=>r.json()).then(d=>{
  if(d.folder){ $('fi').value=d.folder; loadVideos(); }
});
</script>
</body>
</html>
"""

# ─── HTTP handler ─────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass

    def _send(self, data, ctype="application/json", status=200):
        if isinstance(data, dict):
            body = json.dumps(data).encode()
        elif isinstance(data, str):
            body = data.encode()
        else:
            body = data
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            self._send(_HTML, "text/html; charset=utf-8")
        elif path == "/api/videos":
            self._send({"videos": list_videos()})
        elif path == "/api/status":
            with _job_lock:
                s = dict(_job)
                s["log_tail"] = s["log"][-60:]
            self._send(s)
        elif path == "/api/folder":
            self._send({"folder": str(_video_dir)})
        else:
            self._send({"error": "not found"}, status=404)

    def do_POST(self):
        global _video_dir
        path = urllib.parse.urlparse(self.path).path
        body = self._body()

        if path == "/api/set_folder":
            d = Path(body.get("folder", ""))
            if not d.is_dir():
                self._send({"ok": False, "error": f"Not a directory: {d}"}); return
            _video_dir = d
            if _IV_OK:
                try:
                    _iv.VIDEO_DIR = d
                except Exception:
                    pass
            self._send({"ok": True, "count": len(list_videos())})

        elif path == "/api/run":
            with _job_lock:
                if _job["running"]:
                    self._send({"ok": False, "error": "Already running"}); return
            video      = body.get("video", "")
            params_in  = body.get("params", {})
            out_folder = body.get("out_folder", "").strip()
            if not video:
                self._send({"ok": False, "error": "No video specified"}); return
            threading.Thread(
                target=run_job,
                args=(video, params_in, out_folder),
                daemon=True).start()
            self._send({"ok": True})

        elif path == "/api/cancel":
            with _job_lock: _job["cancel"] = True
            self._send({"ok": True})

        else:
            self._send({"error": "not found"}, status=404)


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, webbrowser
    parser = argparse.ArgumentParser(description="Full Deadtime Cutter server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"HTTP port (default {DEFAULT_PORT})")
    args = parser.parse_args()

    url = f"http://localhost:{args.port}"
    print(f"Full Deadtime Cutter → {url}")
    print(f"TENNIS root  : {TENNIS}")
    print(f"Params dir   : {PARAMS_DIR}")
    print(f"IV import    : {'OK — RANSAC ball validation active' if _IV_OK else 'FAILED — ball validation unavailable'}")
    print()
    print("Ball validation pipeline:")
    print("  Audio thwacks → rally segments")
    print("  → collect_track_blobs (detection filters + spatial prior)")
    print("  → find_ransac_arc (parabolic arc fitting)")
    print("  → keep segment if confidence ≥ ball_conf_thr")
    print()
    print("REMINDER: Once this sequential version is validated, parallelize")
    print("  ball tracking across audio segments (one thread per rally). ~10x speedup.")
    print()
    print("Press Ctrl-C to stop.")

    server = HTTPServer(("localhost", args.port), Handler)
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")

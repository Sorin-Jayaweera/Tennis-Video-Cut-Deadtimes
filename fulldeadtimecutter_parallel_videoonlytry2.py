"""
vision_only_cutter.py — Vision-Only Parallel Tennis Clip Cutter with Debug Mode.

Pipeline per video:
  1. Slice video into overlapping chunks (5s native / 10s lower-res).
  2. Each worker opens its own VideoCapture (no lock contention).
  3. Runs blob detection WITH spatial prior hard-gate (filters player/crowd blobs)
     then RANSAC arc fitting to find exact timestamps where ball is in play.
  4. Group timestamps → Dropout Gap clustering → Pre/Post buffers → merge.
  5. Debug Mode: silent audio NCC pass for ground-truth thwacks +
     matplotlib confidence-timeline plot saved to output dir.
  6. Write <stem>_active.mp4 (fast -c copy unless debug re-encode requested).

Key fixes over initial draft (Gemini):
  - spatial prior (prior_hard_gate=True) gates out-of-court blobs BEFORE RANSAC
  - proc_w resolved to actual frame width so morph kernel scales correctly
  - adaptive chunk_size + n_workers cap prevents OOM at native resolution
  - min_speed=7.5 rejects slow background "arcs" (was 4.0)
  - log_message suppressor; errors are logged not silently swallowed

REMINDER (from earlier session): Parallelism is now implemented here via
  per-chunk ThreadPoolExecutor.  Future speedup: process multiple VIDEOS in
  parallel (one sub-pool per video) — implement after validating quality.
"""

import sys, os, json, threading, time, shutil, tempfile, subprocess
import urllib.parse
import numpy as np
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── Locate TENNIS root ───────────────────────────────────────────────────────
def find_tennis_root():
    for p in [Path(__file__).resolve().parent,
              Path(__file__).resolve().parent.parent,
              Path.cwd(), Path.cwd().parent]:
        if (p / "videosandaudio").is_dir():
            return p
    return Path(__file__).resolve().parent.parent

TENNIS     = find_tennis_root()
PARAMS_DIR = Path(__file__).resolve().parent / "params"
PRIORS_DIR = PARAMS_DIR / "priors"
PRIORS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(TENNIS / "claude"))

try:
    import interactive_viewer as _iv
    _IV_OK = True
except Exception as _e:
    _iv = None
    _IV_OK = False
    print(f"WARNING: could not import interactive_viewer: {_e}")

# ─── Constants ────────────────────────────────────────────────────────────────
DEFAULT_PORT = 8790
VIDEO_EXTS   = {".mp4", ".avi", ".mov", ".mkv"}

DEFAULT_PARAMS = {
    # ── Vision grouping ───────────────────────────────────────────────────────
    "dropout_gap":    3.0,    # max gap (s) between detections in the same rally
    "pre_buffer":     1.0,    # seconds before first detection in a cluster
    "post_buffer":    1.5,    # seconds after last detection in a cluster
    "debug_mode":     False,  # run audio pass + save matplotlib plot
    # ── Parallel chunking ─────────────────────────────────────────────────────
    "n_workers":      4,      # parallel workers (auto-capped based on res_w)
    # ── Detection filters (mirror Detection tab in interactive_viewer) ────────
    "res_w":          0,      # 0 = native; 320/640 for faster runs
    "thresh":         18,     # frame-diff threshold
    "min_a":          3,      # min blob area (px², at actual res)
    "max_a":          800,    # max blob area
    "max_asp":        4.0,    # max aspect ratio~
    "method":         "circularity",
    "ball_diam":      10.0,   # reference ball diameter at REF_W=320
    "min_circ":       0.05,   # min circularity (circularity method)
    "min_bright":     0.0,
    "blur_k":         9,      # Gaussian pre-blur half-size (0=off)
    "score_thresh":   0.0,
    "gap":            2,      # frame-diff gap
    "top_k":          5,      # top-k blobs per diff layer → RANSAC
    # ── Spatial prior ─────────────────────────────────────────────────────────
    # Coordinates are at REF_W=320 × REF_H=180.
    # prior_hard_gate=True rejects blobs with prior < 0.5 before RANSAC.
    "pweight":        1.0,
    "court_xs":       0.1,
    "court_ys":       0.0,
    "court_inset":    8,
    "air_xl":         115,
    "air_xr":         193,
    "air_yt":         38,
    "air_yb":         58,
    "air_sx":         37.0,
    "air_sy":         40.0,
    # ── RANSAC arc fitting ─────────────────────────────────────────────────────
    "n_look":         10,     # ±frame window for each RANSAC call
    "ransac_px":      16.0,   # inlier pixel threshold
    "min_inliers":    4,
    "min_span":       4,
    "scan_stride":    10,     # frames between RANSAC windows within a chunk
    "ransac_spd":     7.5,    # min arc speed (px/frame) — filters slow background
    "ball_conf_thr":  0.30,   # min RANSAC confidence (r² × coverage)
    # ── Audio-first gating ────────────────────────────────────────────────────
    # When audio_gated_scan=True, run silent_audio_pass FIRST and only scan
    # video chunks that intersect a ±radius_s window around each thwack.
    # On dead-time-rich footage this drops vision work by 5–10×.
    "audio_gated_scan":   False,
    "audio_gate_radius_s": 5.0,
}

# ─── Mutable state ────────────────────────────────────────────────────────────
_job = {
    "running": False, "cancel": False, "phase": "idle",
    "progress": 0, "total": 0, "log": [], "results": [], "error": "",
    "curr_video": "", "batch_idx": 0, "batch_total": 0,
}
_job_lock  = threading.Lock()
_video_dir = TENNIS / "videosandaudio"

def _log(msg):
    ts   = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with _job_lock:
        _job["log"].append(line)
        if len(_job["log"]) > 800:
            _job["log"] = _job["log"][-800:]

def find_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    raise RuntimeError("ffmpeg not found — pip install imageio-ffmpeg")

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
            out.append({"name": p.name, "size_mb": round(size_mb, 1),
                        "duration_min": round(frames / fps / 60, 1)})
        return out
    except Exception:
        return []


# ─── Spatial prior ────────────────────────────────────────────────────────────
def _build_prior(p):
    """Compute REF_H×REF_W prior map from params. Returns None if pweight=0."""
    if not _IV_OK or float(p.get("pweight", 1.0)) <= 0:
        return None
    try:
        return _iv.compute_prior_map(
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
    except Exception as e:
        _log(f"  WARNING: prior map failed ({e}) — running without prior")
        return None


# ─── Prior preview PNG rendering ─────────────────────────────────────────────
def _render_prior_preview_png(p, vname=None):
    """
    Render a 320×180 prior heatmap blended with the actual video frame.

    Mirrors interactive_viewer.make_prior_image():
      result = addWeighted(frame, 0.35, TURBO_heatmap, 0.65, 0)

    Prior is normalised min→max (not clipped 0–1) so the full dynamic
    range is visible.  Returns JPEG bytes (opaque), or None on failure.
    The canvas layer above this shows court outlines + drag handles with
    a transparent background so no black is introduced.
    """
    if not _IV_OK:
        return None
    try:
        import cv2
        prior = _iv.compute_prior_map(
            court_x_sigma = float(p.get("court_xs", 0.1)),
            court_y_sigma = float(p.get("court_ys", 0.0)),
            court_inset   = int(  p.get("court_inset", 8)),
            air_x_left    = int(  p.get("air_xl", 115)),
            air_x_right   = int(  p.get("air_xr", 193)),
            air_y_top     = int(  p.get("air_yt", 38)),
            air_y_bot     = int(  p.get("air_yb", 58)),
            air_sigma_x   = float(p.get("air_sx", 37.0)),
            air_sigma_y   = float(p.get("air_sy", 40.0)),
            weight        = float(p.get("pweight", 1.0)),
        ).astype("float32")

        # Normalise min→max (matches interactive_viewer, shows full range)
        lo, hi = float(prior.min()), float(prior.max())
        vis = ((prior - lo) / (hi - lo + 1e-6) * 255).clip(0, 255).astype("uint8")
        heat_bgr = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)  # (180,320,3) BGR

        # Blend with real video frame if available: 35% frame + 65% heatmap
        if vname and _video_dir:
            try:
                cap = cv2.VideoCapture(str(_video_dir / vname))
                ok_f, frame = cap.read()
                cap.release()
                if ok_f:
                    frame_small = cv2.resize(frame, (320, 180))
                    heat_bgr = cv2.addWeighted(frame_small, 0.35, heat_bgr, 0.65, 0)
            except Exception:
                pass  # fall through to heatmap-only

        ok, buf = cv2.imencode(".jpg", heat_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return buf.tobytes() if ok else None
    except Exception as e:
        _log(f"  _render_prior_preview_png: {e}")
        return None


# ─── Debug: silent audio pass ─────────────────────────────────────────────────
def run_silent_audio_pass(video_path):
    """NCC matched-filter thwack detector. Returns sorted list of timestamps (s)."""
    import librosa
    from scipy.signal import find_peaks, butter, sosfiltfilt, correlate2d

    _log("  [Debug] Running silent audio pass for ground-truth thwacks...")
    if not (PARAMS_DIR / "template.npy").exists():
        _log("  [Debug] ERROR: Audio template not found in RunDirectory/params/")
        return []

    template   = np.load(PARAMS_DIR / "template.npy")
    sr_templ   = int(np.load(PARAMS_DIR / "sr.npy")[0])
    hop_length = int(np.load(PARAMS_DIR / "hop_length.npy")[0])
    n_fft      = int(np.load(PARAMS_DIR / "n_fft.npy")[0])
    pre_ms     = float(np.load(PARAMS_DIR / "pre_ms.npy")[0])

    ffmpeg = find_ffmpeg()
    tmp    = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(video_path),
         "-vn", "-ac", "1", "-ar", str(sr_templ),
         "-acodec", "pcm_s16le", "-f", "wav", tmp.name],
        capture_output=True)

    try:
        audio_raw, sr_check = librosa.load(tmp.name, sr=None)
    finally:
        os.unlink(tmp.name)

    nyq    = 0.5 * sr_check
    sos    = butter(4, [1000.0/nyq, 8000.0/nyq], btype="bandpass", output="sos")
    abp    = sosfiltfilt(sos, audio_raw).astype("float32")
    mag    = np.abs(librosa.stft(abp, n_fft=n_fft, hop_length=hop_length))
    t      = (template - template.mean()) / (template.std() + 1e-8)
    s      = (mag      - mag.mean())      / (mag.std()      + 1e-8)
    r      = correlate2d(s, t, mode="valid")[0].astype("float32")

    med    = np.median(r)
    mad    = np.median(np.abs(r - med))
    thr    = med + 6.0 * 1.4826 * (mad + 1e-12)
    gap_fr = max(1, int(round(250 / 1000 * sr_check / hop_length)))
    idx, _ = find_peaks(r, height=thr, distance=gap_fr)
    audio_times = sorted(idx * (hop_length / sr_check) + pre_ms / 1000)
    _log(f"  [Debug] Found {len(audio_times)} audio thwacks.")
    return audio_times


# ─── Debug: matplotlib confidence plot ───────────────────────────────────────
def save_debug_plot(out_dir, stem, video_ts, audio_ts, active_segs, ball_conf_thr, video_dur):
    """
    Save a 2-panel confidence timeline PNG to out_dir/<stem>_debug_plot.png.

    Top panel:  scatter of (time, confidence) for every RANSAC detection.
                Active segments shown as green bands.
    Bottom panel: density view — audio thwack bars (red) vs video dots (green),
                  active segment bands (teal).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 7),
                                        gridspec_kw={"height_ratios": [3, 1]},
                                        sharex=True)
        fig.patch.set_facecolor("#111111")

        # ── Top: confidence scatter ─────────────────────────────────────────
        ax1.set_facecolor("#0a0a0a")
        if video_ts:
            ts_arr    = np.array([t for t, _ in video_ts])
            conf_arr  = np.array([c for _, c in video_ts])
            sc = ax1.scatter(ts_arr, conf_arr, c=conf_arr, cmap="YlGn",
                             vmin=0, vmax=1, s=12, alpha=0.8, zorder=3)
            plt.colorbar(sc, ax=ax1, pad=0.01, shrink=0.8,
                         label="RANSAC confidence").ax.yaxis.label.set_color("#aaa")

        ax1.axhline(ball_conf_thr, color="#fa8040", linewidth=1.2,
                    linestyle="--", label=f"conf threshold = {ball_conf_thr}")
        for i, (s, e) in enumerate(active_segs):
            ax1.axvspan(s, e, alpha=0.18, color="#33aa77",
                        label="active segment" if i == 0 else None)

        ax1.set_ylabel("RANSAC Confidence", color="#aaa", fontsize=11)
        ax1.set_ylim(-0.02, 1.05)
        ax1.set_xlim(0, video_dur)
        ax1.tick_params(colors="#888")
        for sp in ax1.spines.values():
            sp.set_color("#333")
        ax1.legend(facecolor="#1a1a1a", edgecolor="#444",
                   labelcolor="#ccc", fontsize=9)
        ax1.set_title(f"{stem}  —  Vision-Only RANSAC Analysis  "
                      f"({len(active_segs)} active segs | "
                      f"{sum(e-s for s,e in active_segs):.0f}s active)",
                      color="#7ddff0", fontsize=13, pad=8)

        # ── Bottom: density view ────────────────────────────────────────────
        ax2.set_facecolor("#0a0a0a")
        if audio_ts:
            for t in audio_ts:
                ax2.axvline(t, color="#ff4444", linewidth=0.9, alpha=0.85)
        if video_ts:
            for t, c in video_ts:
                ax2.axvline(t, color="#44ff44",
                            linewidth=0.5, alpha=max(0.1, 0.3 + 0.6 * c))
        for i, (s, e) in enumerate(active_segs):
            ax2.axvspan(s, e, alpha=0.30, color="#33aa77",
                        label="active seg" if i == 0 else None)

        ax2.set_xlim(0, video_dur)
        ax2.set_yticks([])
        ax2.set_xlabel("Time (seconds)", color="#aaa", fontsize=11)
        ax2.tick_params(colors="#888")
        for sp in ax2.spines.values():
            sp.set_color("#333")
        ax2.text(0.01, 0.85, "▐ Audio thwacks (red)  |  ▐ Vision detections (green)",
                 transform=ax2.transAxes, color="#999", fontsize=8.5)

        plt.tight_layout(rect=[0, 0, 1, 1])
        plot_path = out_dir / f"{stem}_debug_plot.png"
        plt.savefig(str(plot_path), dpi=110, bbox_inches="tight",
                    facecolor="#111111")
        plt.close(fig)
        _log(f"  [Debug] Plot saved → {plot_path.name}")
        return str(plot_path)

    except ImportError:
        _log("  [Debug] matplotlib not installed — skipping plot. "
             "Run: pip install matplotlib")
        return None
    except Exception as exc:
        _log(f"  [Debug] Plot failed: {exc}")
        return None


# ─── Core: parallel chunk scanner ─────────────────────────────────────────────
def _scan_video_chunk(args):
    """
    Worker function — runs in a thread pool.

    Each call:
      1. Opens its OWN cv2.VideoCapture (no shared-cap locking).
      2. Reads chunk frames sequentially into a dict (fast codec path).
      3. Strides through frames, running detect() with spatial prior hard-gate
         then find_ransac_arc() to find confident ball timestamps.
      4. Returns list of (timestamp_s, confidence) pairs.
    """
    start_s, end_s, fps, video_name, p, det_kw, ransac_kw, prior_map = args

    if _job["cancel"]:
        return []

    import cv2
    ball_timestamps = []
    ball_thr   = float(p["ball_conf_thr"])
    stride     = int(p.get("scan_stride", 10))
    n_look     = int(p["n_look"])
    gap        = int(p["gap"])
    req_res_w  = int(p["res_w"])

    frame_start = int(start_s * fps)
    frame_end   = int(end_s   * fps)

    # ── Open a private reader for this thread ─────────────────────────────────
    cap = cv2.VideoCapture(str(_video_dir / video_name))
    if not cap.isOpened():
        return []

    # Resolve actual resolution (needed for correct morph kernel scaling)
    if req_res_w > 0:
        actual_w = req_res_w
        actual_h = req_res_w * 9 // 16
    else:
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read buffer: extend before chunk start and after chunk end for RANSAC context
    read_start = max(0, frame_start - n_look - gap)
    read_end   = frame_end + n_look + 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, read_start)

    # ── Load chunk frames sequentially into RAM ───────────────────────────────
    chunk_frames = {}
    for fi in range(read_start, read_end):
        if _job["cancel"]:
            break
        ret, frame = cap.read()
        if not ret:
            break
        if req_res_w > 0:
            frame = cv2.resize(frame, (actual_w, actual_h))
        chunk_frames[fi] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cap.release()

    if _job["cancel"] or not chunk_frames:
        return []

    # ── Strided RANSAC scan ───────────────────────────────────────────────────
    for center in range(frame_start, frame_end, stride):
        if _job["cancel"]:
            break
        try:
            layers = []
            for delta in range(-n_look, n_look + 1):
                fb = center + delta
                fa = fb - gap
                if fa not in chunk_frames or fb not in chunk_frames:
                    continue

                g_b = chunk_frames[fb]
                g_a = chunk_frames[fa]

                # detect() with:
                #   proc_w = actual_w → morph kernel scales correctly at native res
                #   prior_map         → spatial prior hard-gate (pre-filters RANSAC)
                #   prior_hard_gate=True → rejects blobs outside court/air zone
                _, _, _, passing, rejected = _iv.detect(
                    g_curr        = g_b,
                    g_prev        = g_a,
                    thresh        = det_kw["thresh"],
                    min_a         = det_kw["min_a"],
                    max_a         = det_kw["max_a"],
                    max_asp       = det_kw["max_asp"],
                    method        = det_kw["method"],
                    score_thresh  = det_kw["score_thresh"],
                    prior_map     = prior_map,
                    prior_hard_gate = True,
                    proc_w        = actual_w,
                    proc_h        = actual_h,
                    ball_diam     = det_kw["ball_diam"],
                    min_circ      = det_kw["min_circ"],
                    min_bright    = det_kw["min_bright"],
                    blur_k        = det_kw["blur_k"],
                )

                by_score  = sorted(passing, key=lambda b: b["score"], reverse=True)
                top       = by_score[:det_kw["top_k"]]
                top_ids   = {id(b) for b in top}

                blobs_out = []
                for b in by_score:
                    b2 = dict(b); b2["passing"] = True; b2["top_k"] = (id(b) in top_ids)
                    blobs_out.append(b2)
                for b in rejected:
                    b2 = dict(b); b2["passing"] = False; b2["top_k"] = False
                    blobs_out.append(b2)

                layers.append({"rel": delta, "frame_idx": fb, "blobs": blobs_out})

            arc  = _iv.find_ransac_arc(layers, **ransac_kw)
            conf = float(arc["confidence"])
            if conf >= ball_thr:
                ball_timestamps.append((center / fps, conf))

        except Exception as exc:
            # Log the first error per chunk to aid debugging; don't spam
            if not ball_timestamps:
                _log(f"  [chunk {start_s:.0f}s] warn: {exc}")

    with _job_lock:
        _job["progress"] += 1

    return ball_timestamps


# ─── Audio-gating helpers ────────────────────────────────────────────────────
def _merge_audio_windows(audio_ts, radius_s, video_dur):
    """
    Build merged candidate windows around each audio thwack.
    Returns sorted, non-overlapping list of (start_s, end_s).
    Empty audio_ts → returns [(0, video_dur)] (i.e. scan everything).
    """
    if not audio_ts:
        return [(0.0, float(video_dur))]
    raw = [(max(0.0, float(t) - float(radius_s)),
            min(float(video_dur), float(t) + float(radius_s)))
           for t in audio_ts]
    raw.sort()
    merged = [list(raw[0])]
    for s, e in raw[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(float(s), float(e)) for s, e in merged]


def _windows_to_chunks(windows, chunk_size, overlap, video_dur):
    """Slice each candidate window into overlapping chunks."""
    advance = max(0.1, chunk_size - overlap)
    chunks = []
    for cs, ce in windows:
        cs = max(0.0, float(cs))
        ce = min(float(video_dur), float(ce))
        if ce - cs <= chunk_size:
            chunks.append((cs, ce))
            continue
        curr = cs
        while curr < ce:
            chunks.append((curr, min(curr + chunk_size, ce)))
            curr += advance
    return chunks


# ─── Main vision timestamp extractor ─────────────────────────────────────────
def extract_vision_timestamps(video_name, video_dur, p, candidate_windows=None):
    """
    Slice video into overlapping chunks and scan in parallel.
    Returns sorted list of (timestamp_s, confidence) pairs.

    If candidate_windows is provided (list of (start_s, end_s)), only chunks
    inside those windows are scanned (audio-gating fast path).
    """
    import cv2

    cap_tmp  = cv2.VideoCapture(str(_video_dir / video_name))
    fps      = cap_tmp.get(cv2.CAP_PROP_FPS) or 25.0
    cap_tmp.release()

    # ── Adaptive chunking: smaller chunks at native res to limit RAM ──────────
    req_res_w = int(p["res_w"])
    if req_res_w == 0:
        # Native resolution: each chunk ~250 frames × ~2MB = ~500MB per worker
        chunk_size = 5.0
        overlap    = 1.0
        max_workers = min(int(p.get("n_workers", 4)), 4)
    else:
        chunk_size = 10.0
        overlap    = 2.0
        max_workers = min(int(p.get("n_workers", 6)),
                          max(1, os.cpu_count() or 4))

    if candidate_windows:
        chunks = _windows_to_chunks(candidate_windows, chunk_size, overlap, video_dur)
        gated_dur = sum(e - s for s, e in candidate_windows)
        _log(f"  Audio-gated: {len(candidate_windows)} windows "
             f"covering {gated_dur/60:.1f} min "
             f"({gated_dur/video_dur*100:.0f}% of video)")
    else:
        chunks = []
        curr   = 0.0
        while curr < video_dur:
            chunks.append((curr, min(curr + chunk_size, video_dur)))
            curr += (chunk_size - overlap)

    # ── Compute prior once; all threads share the reference (no copy) ─────────
    prior_map = _build_prior(p)
    prior_str = "on" if prior_map is not None else "OFF (pweight=0)"

    det_kw = dict(
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
        top_k       = int(p["top_k"]),
    )
    ransac_kw = dict(
        n_iter      = 400,
        inlier_px   = float(p["ransac_px"]),
        min_inliers = int(p["min_inliers"]),
        min_span    = int(p["min_span"]),
        min_speed   = float(p["ransac_spd"]),
    )

    _log(f"  Parallel Vision Scan: {len(chunks)} chunks, "
         f"{max_workers} workers, "
         f"res={'native' if req_res_w == 0 else f'{req_res_w}px'}, "
         f"prior={prior_str}, min_speed={p['ransac_spd']}")

    with _job_lock:
        _job["progress"] = 0
        _job["total"]    = len(chunks)

    tasks = [(s, e, fps, video_name, p, det_kw, ransac_kw, prior_map)
             for s, e in chunks]

    all_detections = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futs = [executor.submit(_scan_video_chunk, t) for t in tasks]
        for fut in as_completed(futs):
            try:
                all_detections.extend(fut.result())
            except Exception as exc:
                _log(f"  [worker error] {exc}")

    # Deduplicate (overlapping chunks can produce duplicate timestamps)
    seen = set()
    deduped = []
    for ts, conf in sorted(all_detections, key=lambda x: x[0]):
        key = round(ts, 2)   # 10ms bucket
        if key not in seen:
            seen.add(key)
            deduped.append((ts, conf))

    _log(f"  Raw detections: {len(all_detections)} → {len(deduped)} unique")
    return deduped


# ─── FFmpeg export ────────────────────────────────────────────────────────────
def _complement_segs(active_segs, video_dur):
    """Return the complement of `active_segs` over [0, video_dur]."""
    if not active_segs:
        return [(0.0, float(video_dur))]
    segs = sorted([(float(s), float(e)) for s, e in active_segs])
    out = []
    prev_end = 0.0
    for s, e in segs:
        if s > prev_end + 1e-6:
            out.append((prev_end, s))
        prev_end = max(prev_end, e)
    if prev_end < video_dur - 1e-6:
        out.append((prev_end, float(video_dur)))
    # Strip ultra-short slivers (< 0.1s) — they aren't useful and confuse ffmpeg
    return [(s, e) for s, e in out if e - s >= 0.1]


def _cut_and_concat(video_path, segs, out_dir, stem, suffix,
                    debug_mode, audio_ts, label, label_color="white",
                    phase_name=None):
    """
    Cut `segs` from `video_path` and concat into a single file
    `<stem>_<suffix>.mp4`. With debug_mode=True, burn `label` text at top-left.
    Returns the output path or None.
    """
    ffmpeg  = find_ffmpeg()
    tmp_dir = out_dir / f"_tmp_{suffix}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    seg_paths = []

    if phase_name:
        with _job_lock:
            _job["total"]    = len(segs)
            _job["progress"] = 0
            _job["phase"]    = phase_name

    for i, (s, e) in enumerate(segs):
        sp = tmp_dir / f"seg_{i:04d}.mp4"
        if debug_mode:
            vf = [f"drawtext=text='{label}':x=20:y=20:fontsize=28:"
                  f"fontcolor={label_color}:box=1:boxcolor=black@0.7"]
            if any(s <= t <= e for t in audio_ts):
                vf.append("drawtext=text='AUDIO THWACK':x=20:y=60:fontsize=28:"
                           "fontcolor=red:box=1:boxcolor=white@0.9:"
                           "enable='lt(mod(t\\,1)\\,0.5)'")
            subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error",
                 "-ss", f"{s:.3f}", "-to", f"{e:.3f}", "-i", str(video_path),
                 "-vf", ",".join(vf), "-c:v", "libx264",
                 "-preset", "fast", "-crf", "23", "-c:a", "aac", str(sp)],
                capture_output=True)
        else:
            subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error",
                 "-ss", f"{s:.3f}", "-to", f"{e:.3f}", "-i", str(video_path),
                 "-c", "copy", str(sp)],
                capture_output=True)

        if sp.exists():
            seg_paths.append(sp)
        else:
            _log(f"  WARNING: {suffix} segment {i} ({s:.1f}–{e:.1f}s) export failed")
        if phase_name:
            with _job_lock:
                _job["progress"] += 1

    out_path = out_dir / f"{stem}_{suffix}.mp4"
    if seg_paths:
        lst = tmp_dir / "list.txt"
        lst.write_text("".join(f"file '{p.resolve()}'\n" for p in seg_paths),
                       encoding="utf-8")
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error",
             "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
             str(out_path)],
            capture_output=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return str(out_path) if out_path.exists() else None


def write_active_video(video_path, active_segs, out_dir, stem,
                       debug_mode, audio_ts, video_ts, video_dur=None):
    """
    Cut active segments → <stem>_active.mp4.

    In debug_mode, also writes:
      - <stem>_deadtime.mp4 — the complement of active_segs (everything
        rejected — useful for spot-checking missed rallies).
      - <stem>_debug_data.json — audio_ts, video_ts, active_segs, deadtime_segs.

    Returns dict {"active": <path>, "deadtime": <path or "">}.
    """
    active_path = _cut_and_concat(
        video_path, active_segs, out_dir, stem, "active",
        debug_mode, audio_ts,
        label="RANSAC Active", label_color="white",
        phase_name="writing active video")

    deadtime_path = ""
    deadtime_segs = []
    if debug_mode and video_dur is not None:
        deadtime_segs = _complement_segs(active_segs, video_dur)
        if deadtime_segs:
            total_dead = sum(e - s for s, e in deadtime_segs)
            _log(f"  Writing deadtime video: {len(deadtime_segs)} segs "
                 f"({total_dead/60:.1f} min = "
                 f"{100*total_dead/video_dur:.0f}% of video)")
            deadtime_path = _cut_and_concat(
                video_path, deadtime_segs, out_dir, stem, "deadtime",
                debug_mode, audio_ts,
                label="DEAD TIME", label_color="#ffaa55",
                phase_name="writing deadtime video") or ""

    if debug_mode:
        debug_json = out_dir / f"{stem}_debug_data.json"
        debug_json.write_text(json.dumps({
            "audio_ts": list(audio_ts),
            "video_ts": [t for t, _ in video_ts],
            "active_segs": active_segs,
            "deadtime_segs": deadtime_segs,
        }), encoding="utf-8")

    return {"active": active_path or "", "deadtime": deadtime_path}


# ─── Batch job (background thread) ───────────────────────────────────────────
def run_batch_job(video_names, params_override, out_folder_str):
    import cv2
    with _job_lock:
        _job.update(running=True, cancel=False, phase="starting",
                    progress=0, total=0, log=[], results=[], error="",
                    batch_idx=0, batch_total=len(video_names))

    p = dict(DEFAULT_PARAMS)
    p.update(params_override)

    for idx, video_name in enumerate(video_names):
        if _job["cancel"]:
            break
        with _job_lock:
            _job["batch_idx"]  = idx + 1
            _job["curr_video"] = video_name

        stem      = Path(video_name).stem
        out_dir   = (Path(out_folder_str) if out_folder_str
                     else _video_dir / stem)
        video_path = _video_dir / video_name
        out_dir.mkdir(parents=True, exist_ok=True)

        _log(f"\n=== [{idx+1}/{len(video_names)}] {video_name} ===")
        _log(f"    Output dir : {out_dir}")

        cap = cv2.VideoCapture(str(video_path))
        fps       = cap.get(cv2.CAP_PROP_FPS) or 25.0
        video_dur = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
        cap.release()
        _log(f"    Duration   : {video_dur/60:.1f} min  @  {fps:.1f} fps")

        try:
            # ── Audio pass (used by debug mode and audio-gated scan) ──────────
            audio_ts = []
            need_audio = bool(p.get("debug_mode")) or bool(p.get("audio_gated_scan"))
            if need_audio:
                with _job_lock:
                    _job["phase"] = "audio pass"
                audio_ts = run_silent_audio_pass(video_path)

            # ── Build candidate windows if audio gating is on ─────────────────
            candidate_windows = None
            if p.get("audio_gated_scan"):
                radius = float(p.get("audio_gate_radius_s", 5.0))
                candidate_windows = _merge_audio_windows(audio_ts, radius, video_dur)
                gated_s = sum(e - s for s, e in candidate_windows)
                if not audio_ts:
                    _log("  Audio gating: no thwacks found — falling back to FULL scan")
                    candidate_windows = None
                elif gated_s < 0.05 * video_dur:
                    _log(f"  Audio gating: only {gated_s/60:.1f} min covered "
                         f"(<5% of video) — falling back to FULL scan")
                    candidate_windows = None
                else:
                    _log(f"  Audio gating ON: scanning {gated_s/60:.1f} min "
                         f"of {video_dur/60:.1f} min "
                         f"({100*gated_s/video_dur:.0f}%) "
                         f"around {len(audio_ts)} thwacks (±{radius}s)")

            # ── Parallel vision scan ──────────────────────────────────────────
            with _job_lock:
                _job["phase"] = "video scan"
            video_ts = extract_vision_timestamps(
                video_name, video_dur, p,
                candidate_windows=candidate_windows)

            if not video_ts:
                _log("  No ball detections found. "
                     "Try lowering ball_conf_thr or ransac_spd.")
                with _job_lock:
                    _job["results"].append({"video": video_name,
                                            "active": "", "n_kept": 0})
                continue

            # ── Cluster → buffer → merge ──────────────────────────────────────
            dropout = float(p["dropout_gap"])
            pre     = float(p["pre_buffer"])
            post    = float(p["post_buffer"])

            groups, cur = [], [video_ts[0][0]]
            for t, _ in video_ts[1:]:
                if t - cur[-1] <= dropout:
                    cur.append(t)
                else:
                    groups.append(cur)
                    cur = [t]
            groups.append(cur)

            segs = [[max(0.0, g[0] - pre), min(video_dur, g[-1] + post)]
                    for g in groups]
            segs.sort()
            merged = [list(segs[0])]
            for s, e in segs[1:]:
                if s <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])

            total_active = sum(e - s for s, e in merged)
            _log(f"  {len(merged)} active segments  "
                 f"({total_active/60:.1f} min = "
                 f"{total_active/video_dur*100:.0f}% of video)")

            # ── Export ────────────────────────────────────────────────────────
            export_paths = write_active_video(
                video_path, merged, out_dir, stem,
                p["debug_mode"], audio_ts, video_ts, video_dur=video_dur)
            active_path   = export_paths.get("active", "")
            deadtime_path = export_paths.get("deadtime", "")

            # ── Debug plot ────────────────────────────────────────────────────
            plot_path = None
            if p["debug_mode"]:
                plot_path = save_debug_plot(
                    out_dir, stem, video_ts, audio_ts,
                    merged, float(p["ball_conf_thr"]), video_dur)

            with _job_lock:
                _job["results"].append({
                    "video":    video_name,
                    "active":   active_path,
                    "deadtime": deadtime_path,
                    "plot":     plot_path or "",
                    "n_kept":   len(merged),
                    "active_s": round(total_active, 1),
                })

        except Exception as exc:
            import traceback
            _log(f"  ERROR on {video_name}: {exc}")
            _log(traceback.format_exc()[-400:])
            with _job_lock:
                _job["results"].append({"video": video_name,
                                        "active": "", "n_kept": 0})

    with _job_lock:
        _job["running"] = False
        _job["phase"]   = "cancelled" if _job["cancel"] else "done"
    _log("=== Batch complete ===")


# ─── Embedded HTML UI ─────────────────────────────────────────────────────────
_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Vision-Only Tennis Cutter</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#111;color:#ccc;font-family:'Segoe UI',sans-serif;font-size:13px;padding:16px}
h1{color:#7df;border-bottom:1px solid #333;padding-bottom:10px;margin-bottom:16px}
h2{color:#888;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
.wrap{display:flex;height:calc(100vh - 90px);gap:0}
.col{padding:12px;overflow-y:auto;border-right:1px solid #222}
.col-l{width:290px;flex-shrink:0}
.col-m{width:330px;flex-shrink:0}
.col-r{flex:1;min-width:0;border-right:none}
.row{display:flex;align-items:center;gap:8px;margin-bottom:9px}
.row label{color:#888;font-size:11px;min-width:130px;flex-shrink:0}
input[type=text],input[type=number]{background:#1a1a1a;border:1px solid #333;color:#ddd;padding:4px 7px;border-radius:4px;font-size:12px}
input[type=range]{width:100%;accent-color:#37a}
input[type=checkbox]{accent-color:#37a;transform:scale(1.2)}
select{background:#1a1a1a;border:1px solid #333;color:#ddd;padding:4px 6px;border-radius:4px;font-size:12px}
button{cursor:pointer;border:none;border-radius:4px;padding:7px 14px;font-size:12px}
.btn-go{background:#0d2a1a;border:2px solid #3a7;color:#5e5}
.btn-go:hover{background:#1a3d25}
.btn-stop{background:#2a0d0d;border:2px solid #a33;color:#e55}
.btn-sm{background:#1e1e1e;border:1px solid #333;color:#aaa;padding:4px 9px}
.btn-sm:hover{background:#2a2a2a}
.vi{padding:7px 9px;border-radius:5px;cursor:pointer;margin-bottom:4px;border:1px solid #222;background:#1a1a1a;display:flex;align-items:center}
.vi:hover{background:#222}
.vi .vn{color:#eee;font-weight:500;font-size:12px;word-break:break-all}
.vi .vm{color:#666;font-size:10px;margin-top:2px}
.sep{height:1px;background:#1e1e1e;margin:10px 0}
#log{background:#0a0a0a;border:1px solid #1e1e1e;border-radius:4px;padding:8px;height:240px;overflow-y:auto;font-family:monospace;font-size:11px;color:#9a9;white-space:pre-wrap}
.badge{display:inline-block;padding:3px 9px;border-radius:10px;font-size:11px;background:#1a2a1a;color:#5e5;border:1px solid #3a7}
.badge.err{background:#2a1a1a;color:#e55;border-color:#a33}
.badge.warn{background:#2a1e0a;color:#fa5;border-color:#a73}
progress{width:100%;height:7px;accent-color:#3a7;border-radius:4px}
.rbox{background:#0d1a0d;border:1px solid #2a5;border-radius:6px;padding:10px;margin-top:8px}
.rpath{color:#7df;font-size:11px;word-break:break-all;margin-top:3px}
.stat{display:inline-block;background:#1a2a1a;border-radius:4px;padding:2px 7px;font-size:11px;color:#7df;margin-right:4px;margin-bottom:4px}
details{background:#151515;border:1px solid #2a2a2a;border-radius:5px;padding:10px;margin-bottom:8px}
summary{color:#7df;cursor:pointer;font-size:12px;font-weight:bold;margin-bottom:8px;outline:none}
#vid-thumb{width:100%;display:block}
#preview-wrap{position:relative;display:none;border-radius:4px;overflow:hidden;border:1px solid #444;margin-top:6px;background:#000}
.tabs{display:flex;gap:8px;margin-bottom:14px}
.tab{padding:7px 14px;background:#222;cursor:pointer;border-radius:4px;font-size:12px}
.tab.active{background:#2a4a6a;color:#7df;border:1px solid #37a}
canvas{background:#000;border:1px solid #444;width:100%;height:120px;margin-top:8px}
.hint{color:#555;font-size:10px;margin-top:-6px;margin-bottom:8px}
</style>
</head>
<body>
<h1>🎾 Vision-Only Tennis Cutter</h1>
<div class="tabs">
  <div class="tab active" onclick="switchTab('main',this)">Processing Setup</div>
  <div class="tab" onclick="switchTab('debug',this)">Debug Timeline</div>
</div>

<div id="main-tab" class="wrap">

  <!-- LEFT: folder + video list -->
  <div class="col col-l">
    <div style="margin-bottom:10px">
      <h2>Video Folder</h2>
      <div style="display:flex;gap:5px;margin-bottom:5px">
        <input id="fi" type="text" placeholder="Paste folder path…" style="flex:1">
        <button class="btn-sm" onclick="setFolder()">Set</button>
      </div>
      <div id="fstat" style="color:#555;font-size:10px"></div>
    </div>
    <div class="sep"></div>
    <div>
      <h2>Videos</h2>
      <div style="display:flex;gap:5px;margin-bottom:7px">
        <button class="btn-sm" onclick="selectAll(true)">Select All</button>
        <button class="btn-sm" onclick="selectAll(false)">Clear</button>
      </div>
      <div id="vlist"><div style="color:#444">Set a folder above</div></div>
    </div>
    <div class="sep"></div>
    <div>
      <h2>Output Folder <span style="color:#444;text-transform:none;font-size:10px">(optional)</span></h2>
      <input id="outf" type="text" placeholder="Default: videofolder/stem/" style="width:100%">
    </div>
  </div>

  <!-- MID: params -->
  <div class="col col-m">

    <details open>
      <summary>🎯 Core Cut Parameters</summary>
      <div class="row">
        <label>Dropout Gap (s)</label>
        <input type="number" id="dropout_gap" value="3.0" step="0.5" style="width:75px">
      </div>
      <div class="hint">If no ball detected for this long, point is over.</div>
      <div class="row">
        <label>Pre-Buffer (s)</label>
        <input type="number" id="pre_buffer" value="1.0" step="0.25" style="width:75px">
      </div>
      <div class="row">
        <label>Post-Buffer (s)</label>
        <input type="number" id="post_buffer" value="1.5" step="0.25" style="width:75px">
      </div>
      <div class="row">
        <label>Confidence Threshold</label>
        <input type="number" id="ball_conf_thr" value="0.30" step="0.05" style="width:75px">
      </div>
      <div class="hint">RANSAC confidence = r² × coverage. Keep ~0.30.</div>
      <div class="row">
        <label>Min Speed (px/frame)</label>
        <input type="number" id="ransac_spd" value="7.5" step="0.5" style="width:75px">
      </div>
      <div class="hint">Rejects slow "arcs" (players, background motion). 7.5 is good.</div>
      <div class="row" style="margin-top:10px;background:#1a1a00;padding:9px;border:1px solid #763;border-radius:4px">
        <input type="checkbox" id="debug_mode">
        <label for="debug_mode" style="color:#fa8;font-size:11px;min-width:0">
          Enable Debug Mode<br>
          <span style="color:#666;font-size:10px">(audio pass + matplotlib plot + burned text)</span>
        </label>
      </div>
      <div class="row" style="margin-top:8px;background:#001a1a;padding:9px;border:1px solid #267;border-radius:4px">
        <input type="checkbox" id="audio_gated_scan">
        <label for="audio_gated_scan" style="color:#7df;font-size:11px;min-width:0">
          Audio-First Gating &nbsp;<span style="color:#999;font-size:10px">(5–10× speedup)</span><br>
          <span style="color:#666;font-size:10px">Run audio pass first; only scan video around thwacks.</span>
        </label>
      </div>
      <div class="row">
        <label>Audio Gate Radius (s)</label>
        <input type="number" id="audio_gate_radius_s" value="5.0" step="0.5" style="width:75px">
      </div>
      <div class="hint">±seconds around each audio thwack to scan. 5s is generous.</div>
    </details>

    <details>
      <summary>⚙️ Parallel Processing</summary>
      <div class="row">
        <label>Resolution</label>
        <select id="res_w">
          <option value="320">320px (fastest)</option>
          <option value="640">640px</option>
          <option value="0" selected>Native (best quality)</option>
        </select>
      </div>
      <div class="hint">Native = 5s chunks / 4 workers max to avoid OOM.</div>
      <div class="row">
        <label>Workers</label>
        <input type="number" id="n_workers" value="4" min="1" max="16" step="1" style="width:75px">
      </div>
      <div class="hint">At native res, max 4 is enforced regardless of this setting.</div>
      <div class="row">
        <label>Scan Stride (frames)</label>
        <input type="number" id="scan_stride" value="10" step="1" style="width:75px">
      </div>
      <div class="row">
        <label>RANSAC Window (±frames)</label>
        <input type="number" id="n_look" value="10" step="1" style="width:75px">
      </div>
    </details>

    <details>
      <summary>🎯 Spatial Prior Controls</summary>
      <div class="hint" style="margin-bottom:8px">
        Select a video then click <b style="color:#7df">🎯 Edit Prior</b> in the
        Preview panel to drag handles directly on the court image.
      </div>

      <div class="row">
        <label>Court Inset (px)</label>
        <input type="number" id="court_inset" value="8" min="0" max="40" step="1" style="width:75px"
               oninput="redrawPriorCanvas()">
      </div>
      <div class="row">
        <label>Air Zone L / R</label>
        <input type="number" id="air_xl" value="115" min="0" max="320" step="1" style="width:60px"
               oninput="redrawPriorCanvas()">
        <input type="number" id="air_xr" value="193" min="0" max="320" step="1" style="width:60px"
               oninput="redrawPriorCanvas()">
      </div>
      <div class="row">
        <label>Air Zone T / B</label>
        <input type="number" id="air_yt" value="38"  min="0" max="180" step="1" style="width:60px"
               oninput="redrawPriorCanvas()">
        <input type="number" id="air_yb" value="58"  min="0" max="180" step="1" style="width:60px"
               oninput="redrawPriorCanvas()">
      </div>
      <div class="row">
        <label>Court σ x / y</label>
        <input type="number" id="court_xs" value="0.1" step="0.1" style="width:60px"
               oninput="redrawPriorCanvas()">
        <input type="number" id="court_ys" value="0.0" step="0.1" style="width:60px"
               oninput="redrawPriorCanvas()">
      </div>
      <div class="row">
        <label>Air σ x / y</label>
        <input type="number" id="air_sx" value="37.0" step="1.0" style="width:60px"
               oninput="redrawPriorCanvas()">
        <input type="number" id="air_sy" value="40.0" step="1.0" style="width:60px"
               oninput="redrawPriorCanvas()">
      </div>

      <div class="sep"></div>

      <div class="row">
        <label>Save Prior As:</label>
        <input type="text" id="prior-save-name" placeholder="court name"
               style="flex:1;min-width:0">
        <button class="btn-sm" onclick="savePrior()">💾 Save</button>
      </div>
      <div class="row">
        <label>Load Prior:</label>
        <select id="prior-load-select" style="flex:1;min-width:0"
                onchange="loadPrior(this.value)">
          <option value="">— pick preset —</option>
        </select>
        <button class="btn-sm" onclick="reloadPriors()">↻</button>
      </div>
      <div class="row">
        <button class="btn-sm" onclick="previewPrior()">🔍 Live Preview</button>
      </div>
    </details>

    <details>
      <summary>🔬 Advanced Detection</summary>
      <div class="row">
        <label>Diff Gap (frames)</label>
        <input type="number" id="gap" value="2" step="1" style="width:75px">
      </div>
      <div class="row">
        <label>Diff Threshold</label>
        <input type="number" id="thresh" value="18" step="1" style="width:75px">
      </div>
      <div class="row">
        <label>Max Aspect Ratio</label>
        <input type="number" id="max_asp" value="4.0" step="0.5" style="width:75px">
      </div>
      <div class="row">
        <label>Min Circularity</label>
        <input type="number" id="min_circ" value="0.05" step="0.01" style="width:75px">
      </div>
      <div class="row">
        <label>Min Area (px²)</label>
        <input type="number" id="min_a" value="3" step="1" style="width:75px">
      </div>
      <div class="row">
        <label>Max Area (px²)</label>
        <input type="number" id="max_a" value="800" step="50" style="width:75px">
      </div>
      <div class="row">
        <label>Top-K Blobs/Layer</label>
        <input type="number" id="top_k" value="5" step="1" style="width:75px">
      </div>
      <div class="row">
        <label>RANSAC Inlier px</label>
        <input type="number" id="ransac_px" value="16.0" step="1.0" style="width:75px">
      </div>
      <div class="row">
        <label>Spatial Prior Weight</label>
        <input type="number" id="pweight" value="1.0" step="0.1" min="0" max="1" style="width:75px">
      </div>
      <div class="hint">1.0 = hard-gate blobs outside court/air zone. 0 = no prior.</div>
      <div class="row">
        <label>Blur k</label>
        <input type="number" id="blur_k" value="9" step="2" style="width:75px">
      </div>
    </details>

  </div>

  <!-- RIGHT: selection + run + results + log -->
  <div class="col col-r">
    <div style="margin-bottom:10px">
      <h2 style="display:flex;align-items:center;gap:8px">Preview
        <button class="btn-sm" id="prior-edit-btn" onclick="togglePriorEdit()"
                style="display:none;font-size:10px;padding:2px 8px">🎯 Edit Prior</button>
      </h2>
      <div id="sel-name" style="color:#555;margin-bottom:4px">— click a video to preview —</div>
      <!-- Layered preview: video frame → heatmap overlay → drag-handle canvas -->
      <div id="preview-wrap">
        <img id="vid-thumb" src="">
        <img id="prior-preview-img"
             style="position:absolute;top:0;left:0;width:100%;height:100%;
                    pointer-events:none;display:none;z-index:1">
        <canvas id="prior-canvas" width="320" height="180"
                style="position:absolute;top:0;left:0;width:100%;height:100%;
                       cursor:crosshair;display:none;z-index:2"></canvas>
      </div>
      <div id="prior-edit-hint" style="display:none;margin-top:4px;color:#666;font-size:10px;
                                        gap:6px;align-items:center">
        Drag handles to align ·
        <button class="btn-sm" onclick="togglePreview()" style="padding:1px 6px;font-size:10px">Toggle Overlay</button>
        <span id="prior-msg" style="color:#6af"></span>
      </div>
    </div>
    <div style="display:flex;gap:8px;margin-bottom:12px">
      <button class="btn-go" id="btn-run" onclick="startBatch()">▶ Process Selected</button>
      <button class="btn-stop" id="btn-cancel" onclick="cancelJob()" style="display:none">✕ Cancel</button>
    </div>
    <div class="sep"></div>
    <div style="margin-bottom:12px">
      <h2>Progress</h2>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
        <span class="badge" id="phase-badge">idle</span>
        <span id="batch-txt" style="color:#7df;font-size:11px;font-weight:bold"></span>
        <span id="prog-txt" style="color:#555;font-size:11px"></span>
      </div>
      <progress id="pbar" value="0" max="100" style="margin-bottom:8px"></progress>
      <div id="result-area"></div>
    </div>
    <div class="sep"></div>
    <h2>Log</h2>
    <div id="log"></div>
  </div>

</div>

<!-- DEBUG TAB -->
<div id="debug-tab" style="display:none;padding:12px">
  <p style="color:#888;margin-bottom:12px">
    Load a <code style="color:#7df">_debug_data.json</code> from a Debug Mode run to see the audio/vision timeline.
  </p>
  <input type="file" id="jsonUpload" accept=".json" onchange="handleJSON(event)"
         style="background:#1a1a1a;border:1px solid #333;color:#ddd;padding:5px;border-radius:4px">
  <canvas id="timelineCanvas" width="1200" height="120"></canvas>
  <div style="margin-top:8px;color:#aaa;font-size:11px">
    <span style="color:#f44">▐ Audio Thwack</span>
    &nbsp;|&nbsp;
    <span style="color:#4f4">● Vision Detection (higher = more confident)</span>
    &nbsp;|&nbsp;
    <span style="color:#3a7;opacity:0.5">█ Active Segment</span>
  </div>
  <div id="timeline-stats" style="margin-top:8px;color:#666;font-size:11px"></div>
</div>

<script>
const $ = id => document.getElementById(id);
let _poll = null;

function switchTab(tab, el) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  $('main-tab').style.display  = tab === 'main'  ? 'flex'  : 'none';
  $('debug-tab').style.display = tab === 'debug' ? 'block' : 'none';
}

function setFolder() {
  const f = $('fi').value.trim();
  if (!f) return;
  fetch('/api/set_folder', {method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({folder:f})})
  .then(r=>r.json()).then(d=>{
    $('fstat').textContent = d.ok ? `${d.count} video(s) found` : 'Error: '+d.error;
    loadVideos();
  });
}

function loadVideos() {
  fetch('/api/videos').then(r=>r.json()).then(d=>{
    const el = $('vlist');
    if (!d.videos.length) { el.innerHTML='<div style="color:#444">No videos found</div>'; return; }
    el.innerHTML = d.videos.map(v=>`
      <div class="vi" onclick="selVid('${v.name.replace(/'/g,"\\'")}')">
        <input type="checkbox" class="vid-cb" value="${v.name}"
               style="margin-right:8px;flex-shrink:0" onclick="event.stopPropagation()">
        <div style="flex:1">
          <div class="vn">${v.name}</div>
          <div class="vm">${v.size_mb} MB · ${v.duration_min} min</div>
        </div>
      </div>`).join('');
  });
}

function selectAll(state) {
  document.querySelectorAll('.vid-cb').forEach(cb => cb.checked = state);
}

function selVid(name) {
  $('sel-name').textContent = name;
  $('sel-name').style.color = '#7df';
  const img = $('vid-thumb');
  img.src = '/api/thumb?video=' + encodeURIComponent(name);
  img.onload = () => { $('preview-wrap').style.display = 'block'; };
  $('prior-edit-btn').style.display = 'inline-block';
  // Close prior editor when switching videos
  if (_priorEditActive) togglePriorEdit();
}

function getParams() {
  return {
    dropout_gap:  parseFloat($('dropout_gap').value),
    pre_buffer:   parseFloat($('pre_buffer').value),
    post_buffer:  parseFloat($('post_buffer').value),
    debug_mode:   $('debug_mode').checked,
    audio_gated_scan:    $('audio_gated_scan').checked,
    audio_gate_radius_s: parseFloat($('audio_gate_radius_s').value),
    res_w:        parseInt($('res_w').value),
    n_workers:    parseInt($('n_workers').value),
    ransac_spd:   parseFloat($('ransac_spd').value),
    ball_conf_thr: parseFloat($('ball_conf_thr').value),
    gap:          parseInt($('gap').value),
    thresh:       parseInt($('thresh').value),
    max_asp:      parseFloat($('max_asp').value),
    min_circ:     parseFloat($('min_circ').value),
    min_a:        parseInt($('min_a').value),
    max_a:        parseInt($('max_a').value),
    top_k:        parseInt($('top_k').value),
    ransac_px:    parseFloat($('ransac_px').value),
    pweight:      parseFloat($('pweight').value),
    blur_k:       parseInt($('blur_k').value),
    scan_stride:  parseInt($('scan_stride').value),
    n_look:       parseInt($('n_look').value),
    method:       'circularity',
    min_bright:   0.0,
    score_thresh: 0.0,
    ball_diam:    10.0,
    min_inliers:  4,
    min_span:     4,
    court_xs:     parseFloat($('court_xs').value),
    court_ys:     parseFloat($('court_ys').value),
    court_inset:  parseInt(  $('court_inset').value),
    air_xl:       parseInt(  $('air_xl').value),
    air_xr:       parseInt(  $('air_xr').value),
    air_yt:       parseInt(  $('air_yt').value),
    air_yb:       parseInt(  $('air_yb').value),
    air_sx:       parseFloat($('air_sx').value),
    air_sy:       parseFloat($('air_sy').value),
  };
}

function startBatch() {
  const cbs  = document.querySelectorAll('.vid-cb:checked');
  const vids = Array.from(cbs).map(cb => cb.value);
  if (!vids.length) { alert('Select at least one video'); return; }
  fetch('/api/run', {method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({videos:vids, params:getParams(),
                         out_folder:$('outf').value.trim()})})
  .then(r=>r.json()).then(d=>{
    if (!d.ok) { alert(d.error); return; }
    $('btn-run').style.display='none'; $('btn-cancel').style.display='';
    $('result-area').innerHTML='';
    startPoll();
  });
}

function cancelJob() { fetch('/api/cancel',{method:'POST'}).then(r=>r.json()); }

function startPoll() {
  if (_poll) clearInterval(_poll);
  _poll = setInterval(pollStatus, 1200);
}

function pollStatus() {
  fetch('/api/status').then(r=>r.json()).then(d=>{
    const badge = $('phase-badge');
    badge.textContent = d.phase || 'idle';
    badge.className   = 'badge' + (d.phase==='error' ? ' err'
                                 : d.phase==='cancelled' ? ' warn' : '');
    if (d.batch_total > 0)
      $('batch-txt').textContent = `Video ${d.batch_idx}/${d.batch_total}: ${d.curr_video}`;
    const pct = d.total > 0 ? Math.round(100*d.progress/d.total) : 0;
    $('pbar').value = pct;
    $('prog-txt').textContent = d.total > 0 ? `${d.progress} / ${d.total} chunks` : '';
    const log = $('log');
    log.textContent = (d.log_tail||[]).join('\n');
    log.scrollTop   = log.scrollHeight;

    if (!d.running) {
      clearInterval(_poll); _poll = null;
      $('btn-run').style.display=''; $('btn-cancel').style.display='none';
      if (d.results && d.results.length) {
        $('result-area').innerHTML = d.results.map(r => `
          <div class="rbox">
            <div style="font-weight:bold;color:#ddd;margin-bottom:5px">${r.video}</div>
            <span class="stat">${r.n_kept} active segs</span>
            <span class="stat">${r.active_s||'?'}s active</span>
            <div style="color:#6a8;font-size:11px;margin-top:6px">Active video:</div>
            <div class="rpath">${r.active || '—'}</div>
            ${r.deadtime ? `<div style="color:#fa8;font-size:11px;margin-top:4px">Deadtime video:</div>
            <div class="rpath">${r.deadtime}</div>` : ''}
            ${r.plot ? `<div style="color:#6a8;font-size:11px;margin-top:4px">Debug plot:</div>
            <div class="rpath">${r.plot}</div>` : ''}
          </div>`).join('');
      }
      if (d.error)
        $('result-area').innerHTML += `<div style="color:#e55;margin-top:8px;font-size:12px">Error: ${d.error}</div>`;
    }
  });
}

function handleJSON(event) {
  const reader = new FileReader();
  reader.onload = e => drawTimeline(JSON.parse(e.target.result));
  reader.readAsText(event.target.files[0]);
}

function drawTimeline(data) {
  const canvas = $('timelineCanvas');
  const ctx    = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const audio_ts = data.audio_ts || [];
  const video_ts = data.video_ts || [];
  const segs     = data.active_segs || [];
  const maxTime  = Math.max(...audio_ts, ...video_ts, ...(segs.map(s=>s[1])), 10) + 2;
  const sc       = canvas.width / maxTime;

  // Active segments
  ctx.fillStyle = 'rgba(51,170,119,0.2)';
  segs.forEach(s => ctx.fillRect(s[0]*sc, 0, (s[1]-s[0])*sc, canvas.height));

  // Audio thwacks
  ctx.strokeStyle = '#ff4444';
  ctx.lineWidth   = 1.5;
  audio_ts.forEach(t => { ctx.beginPath(); ctx.moveTo(t*sc,10); ctx.lineTo(t*sc,60); ctx.stroke(); });

  // Vision detections (height = confidence)
  ctx.fillStyle = '#44ff44';
  video_ts.forEach(t => {
    ctx.beginPath();
    ctx.arc(t*sc, 90, 2.5, 0, Math.PI*2);
    ctx.fill();
  });

  $('timeline-stats').textContent =
    `Duration: ${maxTime.toFixed(0)}s | Audio hits: ${audio_ts.length} | ` +
    `Vision detections: ${video_ts.length} | Active segs: ${segs.length}`;
}

// ── Spatial Prior Editor ────────────────────────────────────────────────────
const REF_W = 320, REF_H = 180;
// Static court trapezoid (matches interactive_viewer.COURT_POLY)
const COURT_POLY = [[116,50],[193,52],[293,118],[16,118]];
// Handle hit radius in REF coords
const HANDLE_R = 6;
let _dragHandle = null;        // {kind: 'air-tl'|'air-tr'|'air-bl'|'air-br'|'inset'}
let _previewVisible = false;
let _priorEditActive = false;
let _previewTimer = null;      // debounce handle for auto-refreshing heatmap

function togglePriorEdit() {
  _priorEditActive = !_priorEditActive;
  const canvas = $('prior-canvas');
  const hint   = $('prior-edit-hint');
  const btn    = $('prior-edit-btn');
  canvas.style.display = _priorEditActive ? 'block' : 'none';
  hint.style.display   = _priorEditActive ? 'flex'  : 'none';
  btn.textContent = _priorEditActive ? '✕ Close Editor' : '🎯 Edit Prior';
  btn.style.background   = _priorEditActive ? '#2a0d10' : '';
  btn.style.borderColor  = _priorEditActive ? '#a33'    : '';
  btn.style.color        = _priorEditActive ? '#f77'    : '';
  if (_priorEditActive) {
    redrawPriorCanvas();
    previewPrior();          // auto-show heatmap when editor opens
  } else {
    // hide overlay when editor closes
    $('prior-preview-img').style.display = 'none';
    _previewVisible = false;
  }
}

function _priorParams() {
  return {
    court_xs:    parseFloat($('court_xs').value)    || 0,
    court_ys:    parseFloat($('court_ys').value)    || 0,
    court_inset: parseInt(  $('court_inset').value) || 0,
    air_xl:      parseInt(  $('air_xl').value)      || 0,
    air_xr:      parseInt(  $('air_xr').value)      || 0,
    air_yt:      parseInt(  $('air_yt').value)      || 0,
    air_yb:      parseInt(  $('air_yb').value)      || 0,
    air_sx:      parseFloat($('air_sx').value)      || 1,
    air_sy:      parseFloat($('air_sy').value)      || 1,
    pweight:     parseFloat($('pweight') ? $('pweight').value : '1.0') || 1.0,
  };
}

function _canvasCoords(evt) {
  const c = $('prior-canvas');
  const r = c.getBoundingClientRect();
  // Map mouse → REF coords (canvas internal is 320x180)
  const sx = (evt.clientX - r.left) / r.width  * REF_W;
  const sy = (evt.clientY - r.top)  / r.height * REF_H;
  return [Math.max(0, Math.min(REF_W-1, sx)),
          Math.max(0, Math.min(REF_H-1, sy))];
}

function _hitTest(x, y) {
  const p = _priorParams();
  const handles = [
    ['air-tl', p.air_xl, p.air_yt],
    ['air-tr', p.air_xr, p.air_yt],
    ['air-bl', p.air_xl, p.air_yb],
    ['air-br', p.air_xr, p.air_yb],
    ['inset',  COURT_POLY[0][0] + p.court_inset, COURT_POLY[0][1] + p.court_inset],
  ];
  let best = null, bestD = HANDLE_R;
  for (const [k, hx, hy] of handles) {
    const d = Math.hypot(x - hx, y - hy);
    if (d < bestD) { bestD = d; best = k; }
  }
  return best;
}

function redrawPriorCanvas() {
  const c   = $('prior-canvas');
  if (!c) return;
  const ctx = c.getContext('2d');
  const p   = _priorParams();

  ctx.clearRect(0, 0, REF_W, REF_H);

  // No background fill: keep canvas transparent so the heatmap preview img
  // (sibling under this canvas in the prior-stack) shows through. The parent
  // div has #0a0a0a as a fallback for when the overlay is hidden.

  // Court trapezoid (static outer reference — white with shadow for contrast)
  ctx.shadowColor = 'rgba(0,0,0,0.8)';
  ctx.shadowBlur  = 2;
  ctx.strokeStyle = 'rgba(255,255,255,0.6)';
  ctx.lineWidth   = 1.5;
  ctx.beginPath();
  ctx.moveTo(COURT_POLY[0][0], COURT_POLY[0][1]);
  for (let i = 1; i < COURT_POLY.length; i++)
    ctx.lineTo(COURT_POLY[i][0], COURT_POLY[i][1]);
  ctx.closePath();
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Inset trapezoid (dashed cyan — the active gate boundary)
  const ins = p.court_inset;
  const insetPoly = [
    [COURT_POLY[0][0] + ins, COURT_POLY[0][1] + ins],
    [COURT_POLY[1][0] - ins, COURT_POLY[1][1] + ins],
    [COURT_POLY[2][0] - ins, COURT_POLY[2][1] - ins],
    [COURT_POLY[3][0] + ins, COURT_POLY[3][1] - ins],
  ];
  ctx.setLineDash([4, 3]);
  ctx.shadowColor = 'rgba(0,0,0,0.9)';
  ctx.shadowBlur  = 3;
  ctx.strokeStyle = '#0ff';
  ctx.lineWidth   = 1.5;
  ctx.beginPath();
  ctx.moveTo(insetPoly[0][0], insetPoly[0][1]);
  for (let i = 1; i < insetPoly.length; i++)
    ctx.lineTo(insetPoly[i][0], insetPoly[i][1]);
  ctx.closePath();
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.shadowBlur = 0;

  // Air zone rectangle
  ctx.strokeStyle = 'rgba(255, 220, 50, 1.0)';
  ctx.fillStyle   = 'rgba(255, 220, 50, 0.15)';
  ctx.lineWidth   = 1.5;
  ctx.fillRect(p.air_xl, p.air_yt, p.air_xr - p.air_xl, p.air_yb - p.air_yt);
  ctx.strokeRect(p.air_xl, p.air_yt, p.air_xr - p.air_xl, p.air_yb - p.air_yt);

  // Air zone Gaussian falloff iso-ellipse (1σ ring)
  const cx = (p.air_xl + p.air_xr) / 2;
  ctx.strokeStyle = 'rgba(255, 200, 80, 0.4)';
  ctx.beginPath();
  ctx.ellipse(cx, p.air_yb, p.air_sx, p.air_sy, 0, 0, Math.PI * 2);
  ctx.stroke();

  // Handles
  function dot(x, y, kind) {
    ctx.fillStyle = (_dragHandle === kind) ? '#fff' : '#fa8';
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 0.5;
    ctx.stroke();
  }
  dot(p.air_xl, p.air_yt, 'air-tl');
  dot(p.air_xr, p.air_yt, 'air-tr');
  dot(p.air_xl, p.air_yb, 'air-bl');
  dot(p.air_xr, p.air_yb, 'air-br');
  // Inset handle as a diamond
  ctx.save();
  ctx.translate(insetPoly[0][0], insetPoly[0][1]);
  ctx.rotate(Math.PI / 4);
  ctx.fillStyle = (_dragHandle === 'inset') ? '#fff' : '#7df';
  ctx.fillRect(-3, -3, 6, 6);
  ctx.strokeStyle = '#000';
  ctx.strokeRect(-3, -3, 6, 6);
  ctx.restore();

  // Auto-refresh heatmap overlay whenever params change (debounced 350 ms)
  if (_priorEditActive) {
    clearTimeout(_previewTimer);
    _previewTimer = setTimeout(previewPrior, 350);
  }
}

function _attachPriorMouse() {
  const c = $('prior-canvas');
  if (!c || c._wired) return;
  c._wired = true;
  c.addEventListener('mousedown', e => {
    const [x, y] = _canvasCoords(e);
    _dragHandle = _hitTest(x, y);
    if (_dragHandle) e.preventDefault();
    redrawPriorCanvas();
  });
  window.addEventListener('mousemove', e => {
    if (!_dragHandle) return;
    const [x, y] = _canvasCoords(e);
    const xi = Math.round(x), yi = Math.round(y);
    if (_dragHandle === 'air-tl') { $('air_xl').value = xi; $('air_yt').value = yi; }
    if (_dragHandle === 'air-tr') { $('air_xr').value = xi; $('air_yt').value = yi; }
    if (_dragHandle === 'air-bl') { $('air_xl').value = xi; $('air_yb').value = yi; }
    if (_dragHandle === 'air-br') { $('air_xr').value = xi; $('air_yb').value = yi; }
    if (_dragHandle === 'inset')  {
      // Inset = distance from top-left of static court polygon
      const ins = Math.max(0, Math.min(40, Math.round(
        Math.max(xi - COURT_POLY[0][0], yi - COURT_POLY[0][1]))));
      $('court_inset').value = ins;
    }
    redrawPriorCanvas();
  });
  window.addEventListener('mouseup', () => {
    if (_dragHandle) {
      _dragHandle = null;
      redrawPriorCanvas();
      // If overlay is on, refresh it
      if (_previewVisible) previewPrior();
    }
  });
}

function reloadPriors() {
  fetch('/api/priors').then(r=>r.json()).then(d=>{
    const sel = $('prior-load-select');
    if (!sel) return;
    sel.innerHTML = '<option value="">— pick preset —</option>'
      + (d.priors||[]).map(p => `<option value="${p.name}">${p.name}</option>`).join('');
  }).catch(()=>{});
}

function loadPrior(name) {
  if (!name) return;
  fetch('/api/priors').then(r=>r.json()).then(d=>{
    const item = (d.priors||[]).find(p => p.name === name);
    if (!item) return;
    const p = item.params || {};
    for (const k of ['court_xs','court_ys','court_inset',
                     'air_xl','air_xr','air_yt','air_yb',
                     'air_sx','air_sy','pweight']) {
      if (p[k] !== undefined && $(k)) $(k).value = p[k];
    }
    redrawPriorCanvas();
    $('prior-msg').textContent = 'Loaded: ' + name;
    if (_previewVisible) previewPrior();
  });
}

function savePrior() {
  const name = $('prior-save-name').value.trim();
  if (!name) { alert('Enter a name first'); return; }
  const params = _priorParams();
  fetch('/api/priors', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({name, params})})
  .then(r=>r.json()).then(d=>{
    if (d.ok) {
      $('prior-msg').textContent = 'Saved as ' + d.name;
      reloadPriors();
    } else {
      $('prior-msg').textContent = 'Save error: ' + (d.error||'?');
    }
  });
}

function previewPrior() {
  const p = _priorParams();
  // Include current video so server can blend the actual frame into the heatmap
  const vname = $('sel-name').textContent;
  if (vname && vname !== '— click a video to preview —') p.video = vname;
  const qs = new URLSearchParams(p).toString();
  const img = $('prior-preview-img');
  img.src = '/api/prior_preview?' + qs + '&_=' + Date.now();
  img.style.display = 'block';
  _previewVisible = true;
}

function togglePreview() {
  const img = $('prior-preview-img');
  if (_previewVisible) {
    img.style.display = 'none';
    _previewVisible = false;
  } else {
    previewPrior();
  }
}

// Init
_attachPriorMouse();
redrawPriorCanvas();
reloadPriors();
fetch('/api/status').then(r=>r.json()).then(d=>{ if(d.running) startPoll(); });
fetch('/api/folder').then(r=>r.json()).then(d=>{
  if (d.folder) { $('fi').value=d.folder; loadVideos(); }
});
</script>
</body>
</html>
"""

# ─── HTTP Handler ─────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass   # suppress per-request console spam

    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try: self.wfile.write(body)
        except Exception: pass

    def do_GET(self):
        prs  = urllib.parse.urlparse(self.path)
        path = prs.path

        if path == "/":
            body = _HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try: self.wfile.write(body)
            except Exception: pass

        elif path == "/api/status":
            with _job_lock:
                s = dict(_job)
                s["log_tail"] = s["log"][-60:]
            self._json(s)

        elif path == "/api/videos":
            self._json({"videos": list_videos()})

        elif path == "/api/folder":
            self._json({"folder": str(_video_dir)})

        elif path == "/api/priors":
            try:
                items = []
                for jp in sorted(PRIORS_DIR.glob("*.json")):
                    try:
                        data = json.loads(jp.read_text(encoding="utf-8"))
                        items.append({"name": jp.stem,
                                      "params": data.get("params", data)})
                    except Exception:
                        items.append({"name": jp.stem, "params": {}})
                self._json({"ok": True, "priors": items})
            except Exception as e:
                self._json({"ok": False, "error": str(e)})

        elif path == "/api/prior_preview":
            qs = urllib.parse.parse_qs(prs.query)
            # Each query param is a list — collapse to scalar
            p = {k: v[0] for k, v in qs.items()}
            vname = p.pop("video", None)   # separate video name from prior params
            jpg = _render_prior_preview_png(p, vname)
            if jpg is None:
                self.send_response(500); self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try: self.wfile.write(jpg)
            except Exception: pass

        elif path == "/api/thumb":
            qs    = urllib.parse.parse_qs(prs.query)
            vname = qs.get("video", [""])[0]
            try:
                import cv2
                cap = cv2.VideoCapture(str(_video_dir / vname))
                ok, frame = cap.read()
                cap.release()
                if ok:
                    h, w = frame.shape[:2]
                    if w > 800:
                        frame = cv2.resize(frame, (800, int(h * 800 / w)))
                    _, buf = cv2.imencode(".jpg", frame,
                                         [cv2.IMWRITE_JPEG_QUALITY, 85])
                    body = buf.tobytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404); self.end_headers()
            except Exception:
                self.send_response(500); self.end_headers()

        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        global _video_dir
        path = urllib.parse.urlparse(self.path).path
        n    = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n)) if n else {}

        if path == "/api/set_folder":
            d = Path(body.get("folder", ""))
            if not d.is_dir():
                self._json({"ok": False, "error": f"Not a directory: {d}"}); return
            _video_dir = d
            if _IV_OK:
                try: _iv.VIDEO_DIR = d
                except Exception: pass
            self._json({"ok": True, "count": len(list_videos())})

        elif path == "/api/run":
            with _job_lock:
                if _job["running"]:
                    self._json({"ok": False, "error": "Already running"}); return
            videos = body.get("videos", [])
            if not videos:
                self._json({"ok": False, "error": "No videos specified"}); return
            threading.Thread(
                target=run_batch_job,
                args=(videos, body.get("params", {}), body.get("out_folder", "")),
                daemon=True).start()
            self._json({"ok": True})

        elif path == "/api/cancel":
            with _job_lock: _job["cancel"] = True
            self._json({"ok": True})

        elif path == "/api/priors":
            name = str(body.get("name", "")).strip()
            params = body.get("params", {}) or {}
            if not name:
                self._json({"ok": False, "error": "Empty name"}); return
            # Sanitise to a safe filename — allow letters/digits/dash/underscore/space
            safe = "".join(c for c in name
                           if c.isalnum() or c in (" ", "-", "_")).strip()
            if not safe:
                self._json({"ok": False, "error": "Name has no safe chars"}); return
            try:
                PRIORS_DIR.mkdir(parents=True, exist_ok=True)
                fp = PRIORS_DIR / f"{safe}.json"
                fp.write_text(json.dumps({"name": safe, "params": params},
                                          indent=2), encoding="utf-8")
                self._json({"ok": True, "name": safe, "path": str(fp)})
            except Exception as e:
                self._json({"ok": False, "error": str(e)})

        else:
            self._json({"error": "not found"}, 404)


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, webbrowser
    parser = argparse.ArgumentParser(description="Vision-Only Tennis Cutter")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    url = f"http://localhost:{args.port}"
    print(f"🎾 Vision-Only Tennis Cutter  →  {url}")
    print(f"   TENNIS root : {TENNIS}")
    print(f"   IV import   : {'OK' if _IV_OK else 'FAILED'}")
    print()
    print("   Key improvements vs initial draft:")
    print("   - spatial prior hard-gate filters player/crowd blobs before RANSAC")
    print("   - proc_w resolved to actual width → morph kernel scales at native res")
    print("   - adaptive workers/chunk-size prevents OOM at native resolution")
    print("   - min_speed=7.5 (filters slow background arcs)")
    print("   - debug mode saves matplotlib confidence plot")
    print()
    print("   REMINDER: Future speedup → parallelize across VIDEOS (one pool per video).")
    print("   Press Ctrl-C to stop.")

    server = HTTPServer(("localhost", args.port), Handler)
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")

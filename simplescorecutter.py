"""
simple_score_cutter.py  —  Simplified audio + blob-score tennis clip cutter.

Pipeline per video:
  1. Extract audio → bandpass → STFT → NCC matched filter → thwack peaks
     → group into rally segments (with pre/post buffers)
  2. Blob validation (PARALLELIZED):
       For each rally segment, stride through frames.
       Run circularity blob detection.
       Keep the segment if ANY frame contains a blob with score > 0.3.
  3. Write  <stem>_active.mp4   (validated rallies + buffer)
            <stem>_deadtime.mp4 (everything else)

Usage:
    python simple_score_cutter.py [--port 8790]
"""

import sys, os, json, threading, time, shutil, tempfile, subprocess, traceback
import urllib.parse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from concurrent.futures import ThreadPoolExecutor

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
sys.path.insert(0, str(TENNIS / "claude"))

try:
    import interactive_viewer as _iv
    _IV_OK = True
except Exception as _e:
    _iv = None
    _IV_OK = False
    print(f"WARNING: could not import interactive_viewer: {_e}")
    print("  Blob validation will be skipped.")

# ─── Constants ────────────────────────────────────────────────────────────────
DEFAULT_PORT = 8790
VIDEO_EXTS   = {".mp4", ".avi", ".mov", ".mkv"}

DEFAULT_PARAMS = {
    # ── Audio detection ───────────────────────────────────────────────────────
    "k_mad":          6.0,
    "min_gap_ms":     250,
    "group_gap_s":    5.0,
    "pre_buffer_s":   1.0,
    "post_single_s":  1.0,
    "post_group_s":   0.5,
    # ── Blob validation ───────────────────────────────────────────────────────
    "res_w":          0,      # Default to native resolution
    "thresh":         18,
    "min_a":          3,
    "max_a":          800,
    "max_asp":        3.0,    # Increased to handle streaks
    "method":         "circularity",
    "ball_diam":      10.0,
    "min_circ":       0.1,    # Lowered to handle streaks
    "min_bright":     0.0,
    "blur_k":         9,
    "score_thresh":   0.3,    # HARD REQUIREMENT: > 0.3 score
    "gap":            1,
    "scan_stride":    2,      # Check every 2nd frame in segment
    "buf_sec":        0.25,
    # ── Spatial prior ─────────────────────────────────────────────────────────
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
    # ── Debug time range ──────────────────────────────────────────────────────
    "debug_start_s":  0.0,
    "debug_end_s":    0.0,
}

# ─── Mutable state ────────────────────────────────────────────────────────────
_params    = dict(DEFAULT_PARAMS)
_video_dir = TENNIS / "videosandaudio"
_job = {
    "running":     False,
    "cancel":      False,
    "phase":       "idle",
    "progress":    0,
    "total":       0,
    "log":         [],
    "results":     [],
    "error":       "",
    "batch_idx":   0,
    "batch_total": 0,
    "curr_video":  "",
}
_job_lock = threading.Lock()

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
    if ff: return ff
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
    import numpy as np, librosa
    from scipy.signal import find_peaks

    if not PARAMS_DIR.exists() or not (PARAMS_DIR / "template.npy").exists():
        raise RuntimeError("Audio params not found at params/. Run make_params.ipynb first.")

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
    _log(f"  Detected {len(times)} thwacks")

    if len(times) == 0: return [], video_dur

    gg, pre = float(p["group_gap_s"]), float(p["pre_buffer_s"])
    post1, postN = float(p["post_single_s"]), float(p["post_group_s"])

    sorted_t   = sorted(times)
    groups, cur = [], [sorted_t[0]]
    for t in sorted_t[1:]:
        if t - cur[-1] <= gg: cur.append(t)
        else: groups.append(cur); cur = [t]
    groups.append(cur)

    segs = [[max(0.0, g[0] - pre), min(video_dur, g[-1] + (post1 if len(g) == 1 else postN))] for g in groups]
    segs.sort()
    
    merged = [segs[0]]
    for s, e in segs[1:]:
        if s <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], e)
        else: merged.append([s, e])

    return merged, video_dur


# ─── Phase 2: Simple Blob Validation (Parallelized) ───────────────────────────

def _build_prior(p):
    if not _IV_OK or float(p["pweight"]) <= 0: return None, None
    try:
        pm = _iv.compute_prior_map(
            court_x_sigma=float(p["court_xs"]), court_y_sigma=float(p["court_ys"]),
            court_inset=int(p["court_inset"]), air_x_left=int(p["air_xl"]),
            air_x_right=int(p["air_xr"]), air_y_top=int(p["air_yt"]),
            air_y_bot=int(p["air_yb"]), air_sigma_x=float(p["air_sx"]),
            air_sigma_y=float(p["air_sy"]), weight=float(p["pweight"]),
        )
        return pm, None
    except Exception as e:
        return None, str(e)

def _scan_segment_task(args):
    """Worker function: strides through frames, exits True if ANY score > 0.3"""
    seg_i, seg, video_dur, p, prior_map, proc_w, proc_h, n_total, fps, video_name, use_debug_range, dbg_s, dbg_e = args
    
    if _job["cancel"]: return None, None
    
    seg_s, seg_e = float(seg[0]), float(seg[1])
    buf_s = float(p["buf_sec"])
    gap = int(p["gap"])

    if use_debug_range and not ((seg_e >= dbg_s) and (seg_s <= dbg_e)):
        return "keep", [max(0.0, seg_s - buf_s), min(video_dur, seg_e + buf_s)]

    fa = max(gap, int(seg_s * fps))
    fb = min(n_total - 1, int(seg_e * fps))
    stride = max(1, int(p.get("scan_stride", 2)))
    
    hit_found = False

    for pos in range(fa, fb + 1, stride):
        if _job["cancel"]: return None, None
        
        try:
            g_curr, _ = _iv.read_gray_small(video_name, pos, proc_w)
            g_prev, _ = _iv.read_gray_small(video_name, pos - gap, proc_w)
            
            if g_curr is None or g_prev is None:
                continue

            # interactive_viewer's detect function returns 'passing' only for blobs > score_thresh
            _, _, _, passing, _ = _iv.detect(
                g_curr, g_prev,
                thresh=int(p["thresh"]),
                min_a=int(p["min_a"]),
                max_a=int(p["max_a"]),
                max_asp=float(p["max_asp"]),
                method=str(p["method"]),
                score_thresh=float(p["score_thresh"]), # This enforces > 0.3
                prior_map=prior_map,
                proc_w=proc_w,
                proc_h=proc_h,
                ball_diam=float(p["ball_diam"]),
                min_circ=float(p["min_circ"]),
                min_bright=float(p["min_bright"]),
                blur_k=int(p["blur_k"]),
                diff_mode="abs",
                prior_hard_gate=False 
            )
            
            if len(passing) > 0:
                hit_found = True
                break
                
        except Exception as e:
            continue

    with _job_lock:
        _job["progress"] += 1

    if hit_found:
        return "keep", [max(0.0, seg_s - buf_s), min(video_dur, seg_e + buf_s)]
    else:
        return "drop", seg

def validate_with_ball_simple(video_name, rally_segs, video_dur, p):
    import cv2
    if not _IV_OK: return [list(s) for s in rally_segs], []

    prior_map, prior_err = _build_prior(p)
    cap_tmp = cv2.VideoCapture(str(_iv.VIDEO_DIR / video_name))
    fps     = cap_tmp.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
    
    proc_w = int(p["res_w"])
    if proc_w == 0:
        proc_w = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
    proc_h = proc_w * 9 // 16
    cap_tmp.release()

    dbg_s, dbg_e = float(p.get("debug_start_s", 0.0)), float(p.get("debug_end_s", 0.0))
    use_debug_range = dbg_e > 0.0

    _log(f"  Parallel Simple Scan (Req score > {p['score_thresh']}): {len(rally_segs)} segments...")
    
    with _job_lock:
        _job["progress"] = 0
        _job["total"] = len(rally_segs)

    tasks = [
        (i, seg, video_dur, p, prior_map, proc_w, proc_h, n_total, fps, video_name, use_debug_range, dbg_s, dbg_e)
        for i, seg in enumerate(rally_segs)
    ]

    kept, dropped = [], []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for status, seg_result in executor.map(_scan_segment_task, tasks):
            if status == "keep": kept.append(seg_result)
            elif status == "drop": dropped.append(seg_result)

    kept.sort()
    if len(kept) > 1:
        merged = [list(kept[0])]
        for s, e in kept[1:]:
            if s <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], e)
            else: merged.append([s, e])
        kept = merged

    _log(f"  Validation complete: {len(kept)} kept, {len(dropped)} dropped")
    return kept, dropped


# ─── Phase 3: Write output videos ────────────────────────────────────────────
def write_videos(video_path, active_segs, video_dur, out_dir, stem):
    ffmpeg = find_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)

    dead_segs, prev = [], 0.0
    for s, e in sorted(active_segs):
        if s > prev + 0.1: dead_segs.append([prev, s])
        prev = e
    if prev < video_dur - 0.1: dead_segs.append([prev, video_dur])

    def cut_segs(segs, out_path, label):
        if not segs: return None
        tmp_dir = out_dir / f"_tmp_{label}"
        tmp_dir.mkdir(exist_ok=True)
        seg_paths = []
        for i, (s, e) in enumerate(segs):
            sp = tmp_dir / f"seg_{i:04d}.mp4"
            subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-ss", f"{s:.3f}", "-to", f"{e:.3f}",
                 "-i", str(video_path), "-c", "copy", str(sp)], capture_output=True)
            if sp.exists(): seg_paths.append(sp)
            with _job_lock:
                _job["progress"] = i + 1
                _job["total"] = len(segs)

        if not seg_paths: return None

        lst = tmp_dir / "list.txt"
        lst.write_text("".join(f"file '{p.resolve()}'\n" for p in seg_paths), encoding="utf-8")
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", str(lst), "-c", "copy", str(out_path)], capture_output=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return str(out_path) if out_path.exists() else None

    with _job_lock: _job["phase"] = "writing active video"
    active_path = cut_segs(active_segs, out_dir / f"{stem}_active.mp4", "active")

    with _job_lock: _job["phase"] = "writing deadtime video"
    dead_path = cut_segs(dead_segs, out_dir / f"{stem}_deadtime.mp4", "deadtime")

    return active_path, dead_path


# ─── Batch Job Wrapper ────────────────────────────────────────────────────────
def run_batch_job(video_names, params_override, out_folder_str):
    with _job_lock:
        _job.update(running=True, cancel=False, phase="starting", progress=0, total=0, 
                    log=[], results=[], error="", batch_idx=0, batch_total=len(video_names))

    p = dict(DEFAULT_PARAMS)
    p.update(params_override)

    for idx, video_name in enumerate(video_names):
        if _job["cancel"]: break
        
        with _job_lock:
            _job["batch_idx"] = idx + 1
            _job["curr_video"] = video_name

        stem = Path(video_name).stem
        out_dir = Path(out_folder_str) if out_folder_str else _video_dir / stem
        video_path = _video_dir / video_name
        
        if not video_path.exists():
            _log(f"ERROR: Video not found: {video_path}")
            continue

        _log(f"\n=== Processing {video_name} ({idx+1}/{len(video_names)}) ===")
        
        try:
            with _job_lock: _job["phase"] = "audio detection"
            rally_segs, video_dur = run_audio_pipeline(video_path, p)
            if _job["cancel"]: break
            
            if not rally_segs:
                _log("No rally segments found.")
                continue

            with _job_lock: _job["phase"] = "blob scan"
            active_segs, dropped_segs = validate_with_ball_simple(video_name, rally_segs, video_dur, p)
            if _job["cancel"]: break

            if not active_segs:
                _log("All segments dropped by validation.")
                continue

            with _job_lock: _job["phase"] = "writing videos"
            active_path, dead_path = write_videos(video_path, active_segs, video_dur, out_dir, stem)

            with _job_lock:
                _job["results"].append({
                    "video":     video_name,
                    "active":    active_path or "",
                    "deadtime":  dead_path or "",
                    "n_audio":   len(rally_segs),
                    "n_kept":    len(active_segs),
                    "n_dropped": len(dropped_segs),
                })
        except Exception as exc:
            traceback.print_exc()
            _log(f"ERROR on {video_name}: {exc}")

    with _job_lock:
        _job["running"] = False
        _job["phase"] = "cancelled" if _job["cancel"] else "done"


# ─── Embedded HTML ────────────────────────────────────────────────────────────
_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Simple Score Deadtime Cutter</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#111;color:#ccc;font-family:'Segoe UI',sans-serif;font-size:13px}
h1{color:#7df;padding:14px 18px 10px;font-size:17px;border-bottom:1px solid #222}
h2{color:#888;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px}
.wrap{display:flex;height:calc(100vh - 49px)}
.col{padding:14px;overflow-y:auto;border-right:1px solid #222}
.col-l{width:290px;flex-shrink:0}
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
    border:1px solid #222;background:#1a1a1a; display:flex; align-items:center;}
.vi:hover{background:#222}
.vi.sel{background:#0d2a1a;border-color:#3a7}
.vi .vn{color:#eee;font-weight:500;font-size:12px; word-break: break-all;}
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
details { background: #151515; border: 1px solid #333; border-radius: 6px; padding: 10px; margin-bottom: 10px; }
summary { color: #7df; cursor: pointer; font-size: 13px; font-weight: bold; outline: none; margin-bottom: 8px;}
.hint{color:#555;font-size:10px;margin-top:-6px;margin-bottom:8px}
.dbg-on label{color:#fa8}
#vid-thumb { width: 100%; border-radius: 4px; margin-top: 10px; border: 1px solid #444; display: none; }
</style>
</head>
<body>
<h1>🎾🏸 Simple Score Deadtime Cutter</h1>
<div class="wrap">

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
    <div style="margin-bottom: 8px; display:flex; gap:6px;">
        <button class="btn-sm" onclick="selectAll(true)">Select All</button>
        <button class="btn-sm" onclick="selectAll(false)">Clear</button>
    </div>
    <div id="vlist"><div style="color:#444">Set a folder above</div></div>
  </div>
  <div class="sep"></div>
  <div class="sec">
    <h2>Output Folder <span style="color:#444;text-transform:none;font-size:10px">(optional)</span></h2>
    <input id="outf" type="text" placeholder="Default: video-folder/stem/" style="width:100%">
  </div>
</div>

<div class="col col-m">

  <details>
    <summary>Audio Detection Parameters</summary>
    <div class="row">
      <label>Sensitivity (K-MAD) ↓ = more</label>
      <input type="range" min="1" max="15" step="0.5" value="6" id="k_mad" oninput="$('k_mad_v').textContent=parseFloat(this.value).toFixed(1)">
      <span class="v" id="k_mad_v">6.0</span>
    </div>
    <div class="row"><label>Min gap between hits (ms)</label><input type="number" min="50" max="2000" step="50" value="250" id="min_gap_ms" style="width:72px"></div>
    <div class="row"><label>Rally gap (s)</label><input type="number" min="1" max="30" step="0.5" value="5" id="group_gap_s" style="width:72px"></div>
    <div class="row"><label>Pre-buffer (s)</label><input type="number" min="0" max="10" step="0.25" value="1" id="pre_buffer_s" style="width:72px"></div>
    <div class="row"><label>Post-buffer single hit (s)</label><input type="number" min="0" max="10" step="0.25" value="1" id="post_single_s" style="width:72px"></div>
    <div class="row"><label>Post-buffer rally (s)</label><input type="number" min="0" max="10" step="0.25" value="0.5" id="post_group_s" style="width:72px"></div>
  </details>

  <details open>
    <summary>Blob Validation (Simple Score)</summary>
    <div class="row">
      <label>Processing resolution</label>
      <select id="res_w">
        <option value="320">320px (fast)</option>
        <option value="480">480px</option>
        <option value="640">640px</option>
        <option value="960">960px</option>
        <option value="0" selected>Native (Full)</option>
      </select>
    </div>
    
    <div class="row">
      <label>Required Score Threshold</label>
      <input type="range" min="0" max="1" step="0.05" value="0.3" id="score_thresh" oninput="$('score_thresh_v').textContent=parseFloat(this.value).toFixed(2)">
      <span class="v" id="score_thresh_v">0.30</span>
    </div>

    <div class="row">
      <label>Scan Stride (frames)</label>
      <input type="number" min="1" max="10" step="1" value="2" id="scan_stride" style="width:72px">
    </div>

    <div class="row">
      <label>Buffer around kept segs (s)</label>
      <input type="range" min="0" max="3" step="0.05" value="0.25" id="buf_sec" oninput="$('buf_v').textContent=parseFloat(this.value).toFixed(2)">
      <span class="v" id="buf_v">0.25</span>
    </div>
    <div class="row"><label>Frame diff threshold</label><input type="number" min="1" max="60" step="1" value="18" id="thresh" style="width:72px"></div>
    <div class="row"><label>Max aspect ratio</label><input type="number" min="1" max="5" step="0.1" value="3.0" id="max_asp" style="width:72px"></div>
    <div class="row"><label>Gaussian blur k</label><input type="number" min="0" max="21" step="2" value="9" id="blur_k" style="width:72px"></div>
    <div class="row">
      <label>Spatial prior weight</label>
      <input type="range" min="0" max="1" step="0.05" value="1.0" id="pweight" oninput="$('pweight_v').textContent=parseFloat(this.value).toFixed(2)">
      <span class="v" id="pweight_v">1.00</span>
    </div>

    <div style="margin-top:10px; border-top:1px solid #333; padding-top:8px">
      <div style="color:#888; font-size:10px; margin-bottom:6px">ADVANCED DETECTION</div>
      <div class="row"><label>Frame diff gap (frames)</label><input type="number" min="1" max="10" step="1" value="1" id="gap" style="width:72px"></div>
      <div class="row">
        <label>Blob method</label>
        <select id="method">
          <option value="circularity" selected>Circularity</option>
          <option value="compactness">Compactness</option>
          <option value="rog">Radius of gyration</option>
        </select>
      </div>
      <div class="row"><label>Min blob area (px²)</label><input type="number" min="1" max="200" step="1" value="3" id="min_a" style="width:72px"></div>
      <div class="row"><label>Max blob area (px²)</label><input type="number" min="50" max="5000" step="50" value="800" id="max_a" style="width:72px"></div>
      <div class="row"><label>Ball diameter ref (px)</label><input type="number" min="2" max="50" step="0.5" value="10" id="ball_diam" style="width:72px"></div>
      <div class="row"><label>Min circularity</label><input type="number" min="0" max="1" step="0.05" value="0.1" id="min_circ" style="width:72px"></div>
      <div class="row"><label>Min brightness</label><input type="number" min="0" max="255" step="1" value="0" id="min_bright" style="width:72px"></div>
    </div>
  </details>

  <details id="dbg-sec">
    <summary>Debug Time Range (0 = full video)</summary>
    <div class="hint">When End > 0, only validates segments in this window. Others are kept unchanged.</div>
    <div class="row" id="dbg-row-s"><label>Start (seconds)</label><input type="number" min="0" step="1" value="0" id="debug_start_s" style="width:80px" oninput="updateDebugHint()"></div>
    <div class="row" id="dbg-row-e"><label>End (seconds, 0 = off)</label><input type="number" min="0" step="1" value="0" id="debug_end_s" style="width:80px" oninput="updateDebugHint()"></div>
    <div id="dbg-hint" style="color:#555;font-size:10px;margin-top:4px"></div>
  </details>

</div>

<div class="col col-r" style="border-right:none">
  <div class="sec">
    <h2>Selection Preview</h2>
    <div id="sel-name" style="color:#555;margin-bottom:10px">— none clicked —</div>
    <img id="vid-thumb" src="">
    <div style="display:flex;gap:8px; margin-top: 15px">
      <button class="btn-go"  id="btn-run"    onclick="startBatch()">▶ Process Selected Batch</button>
      <button class="btn-stop" id="btn-cancel" onclick="cancelJob()" style="display:none">✕ Cancel Batch</button>
    </div>
  </div>
  <div class="sep"></div>
  <div class="sec">
    <h2>Progress</h2>
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
      <span class="badge" id="phase-badge">idle</span>
      <span id="batch-txt" style="color:#7df; font-weight:bold; font-size:11px"></span>
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
let _poll = null;

function updateDebugHint() {
  const s = parseFloat($('debug_start_s').value) || 0, e = parseFloat($('debug_end_s').value) || 0;
  const hint = $('dbg-hint'), rows = [$('dbg-row-s'), $('dbg-row-e')];
  if (e > 0) {
    hint.textContent = `Validating ${(e-s).toFixed(0)}s window (${s.toFixed(0)}s – ${e.toFixed(0)}s).`;
    hint.style.color = '#fa8'; rows.forEach(r => r.classList.add('dbg-on'));
  } else {
    hint.textContent = 'Full video mode (debug range off).';
    hint.style.color = '#555'; rows.forEach(r => r.classList.remove('dbg-on'));
  }
}

function setFolder() {
  const f = $('fi').value.trim();
  if (!f) return;
  fetch('/api/set_folder', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({folder:f})})
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
      <div class="vi" onclick="selVid('${v.name.replace(/'/g,"\\'")}')">
        <input type="checkbox" class="vid-cb" value="${v.name}" style="margin-right:8px; accent-color: #37a; transform: scale(1.2)" onclick="event.stopPropagation()">
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

function selVid(name){
  $('sel-name').textContent=name; 
  $('sel-name').style.color='#7df';
  $('vid-thumb').style.display='block';
  $('vid-thumb').src='/api/thumb?video=' + encodeURIComponent(name);
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
    buf_sec:        parseFloat($('buf_sec').value),
    thresh:         parseInt($('thresh').value),
    max_asp:        parseFloat($('max_asp').value),
    blur_k:         parseInt($('blur_k').value),
    pweight:        parseFloat($('pweight').value),
    scan_stride:    parseInt($('scan_stride').value),
    gap:            parseInt($('gap').value),
    method:         $('method').value,
    min_a:          parseInt($('min_a').value),
    max_a:          parseInt($('max_a').value),
    ball_diam:      parseFloat($('ball_diam').value),
    min_circ:       parseFloat($('min_circ').value),
    min_bright:     parseFloat($('min_bright').value),
    score_thresh:   parseFloat($('score_thresh').value),
    debug_start_s:  parseFloat($('debug_start_s').value) || 0,
    debug_end_s:    parseFloat($('debug_end_s').value)   || 0,
  };
}

function startBatch(){
  const cbs = document.querySelectorAll('.vid-cb:checked');
  const vids = Array.from(cbs).map(cb => cb.value);
  if (vids.length === 0) { alert('Select at least one video to process'); return; }
  
  fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({videos:vids, params:getParams(), out_folder:$('outf').value.trim()})})
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
    
    if (d.batch_total > 0) {
        $('batch-txt').textContent = `Video ${d.batch_idx}/${d.batch_total}: ${d.curr_video}`;
    }
    
    const pct = d.total>0 ? Math.round(100*d.progress/d.total) : 0;
    $('pbar').value = pct;
    $('prog-txt').textContent = d.total>0 ? `${d.progress} / ${d.total}` : '';
    
    const log=$('log');
    log.textContent = (d.log_tail||[]).join('\n');
    log.scrollTop   = log.scrollHeight;
    
    if(!d.running){
      clearInterval(_poll); _poll=null;
      $('btn-run').style.display=''; $('btn-cancel').style.display='none';
      
      if(d.results && d.results.length > 0){
        let html = '';
        d.results.forEach(r => {
            html += `<div class="rbox">
              <div style="font-weight:bold; margin-bottom: 5px; color:#ddd">${r.video}</div>
              <div style="margin-bottom:8px">
                <span class="stat">${r.n_audio} audio segs</span>
                <span class="stat">${r.n_kept} with ball</span>
                <span class="stat">${r.n_dropped} dropped</span>
              </div>
              <div style="color:#6a8;font-size:11px">Active:</div>
              <div class="rpath">${r.active || "None"}</div>
              <div style="color:#6a8;font-size:11px;margin-top:8px">Deadtime:</div>
              <div class="rpath">${r.deadtime || "None"}</div>
            </div>`;
        });
        $('result-area').innerHTML = html;
      }
      if(d.error) $('result-area').innerHTML+=`<div style="color:#e55;margin-top:8px;font-size:12px">Error: ${d.error}</div>`;
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
        body = json.dumps(data).encode() if isinstance(data, dict) else (data.encode() if isinstance(data, str) else data)
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try: self.wfile.write(body)
        except Exception: pass

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def do_GET(self):
        prs = urllib.parse.urlparse(self.path)
        path = prs.path
        if path == "/": self._send(_HTML, "text/html; charset=utf-8")
        elif path == "/api/videos": self._send({"videos": list_videos()})
        elif path == "/api/status":
            with _job_lock:
                s = dict(_job)
                s["log_tail"] = s["log"][-60:]
            self._send(s)
        elif path == "/api/folder": self._send({"folder": str(_video_dir)})
        elif path == "/api/thumb":
            qs = urllib.parse.parse_qs(prs.query)
            vname = qs.get("video", [""])[0]
            if not vname:
                self.send_response(404); self.end_headers(); return
            try:
                import cv2
                cap = cv2.VideoCapture(str(_video_dir / vname))
                ok, frame = cap.read()
                cap.release()
                if ok:
                    _, buf = cv2.imencode(".jpg", frame)
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.end_headers()
                    self.wfile.write(buf.tobytes())
                else:
                    self.send_response(404); self.end_headers()
            except Exception:
                self.send_response(500); self.end_headers()
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
                try: _iv.VIDEO_DIR = d
                except: pass
            self._send({"ok": True, "count": len(list_videos())})

        elif path == "/api/run":
            with _job_lock:
                if _job["running"]:
                    self._send({"ok": False, "error": "Already running"}); return
            videos     = body.get("videos", [])
            params_in  = body.get("params", {})
            out_folder = body.get("out_folder", "").strip()
            if not videos:
                self._send({"ok": False, "error": "No videos specified"}); return
            threading.Thread(
                target=run_batch_job,
                args=(videos, params_in, out_folder),
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
    parser = argparse.ArgumentParser(description="Simple Score Cutter server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"HTTP port (default {DEFAULT_PORT})")
    args = parser.parse_args()

    url = f"http://localhost:{args.port}"
    print(f"🎾🏸 Simple Score Deadtime Cutter → {url}")
    print(f"TENNIS root  : {TENNIS}")
    print(f"Params dir   : {PARAMS_DIR}")
    print(f"IV import    : {'OK — Simple blob validation active' if _IV_OK else 'FAILED — ball validation unavailable'}")
    print()
    print("Press Ctrl-C to stop.")

    server = HTTPServer(("localhost", args.port), Handler)
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")
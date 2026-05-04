# Speed Optimization Analysis — Tennis Cutter Pipeline

Notes only. Implementation paused per user request 2026-05-03.

## Status of completed runs

- **Raja vs Wijemanne.mp4** (~60 min): did NOT fully run. No active segments, no debug plot, no debug_data.json found at `videosandaudio/Raja vs Wijemanne/`. The output directory is empty. Likely cancelled or crashed mid-scan.
- **Raja vs Wijemanne Edited.mp4** (~9 min, already-edited): ran in debug mode but produced 0 active segments. `_debug_data.json` shows 475 audio thwacks and 1193 video detections but `active_segs: []` (empty array — the JSON only contains `audio_ts` and `video_ts` keys, no `active_segs` key). This means either the clustering / ball_conf_thr filter killed everything, or the run was cut short before the cluster→buffer→merge pass wrote the segments. The `_active.mp4` was apparently produced separately (37 clips per task description).
- **New videos in `videosandaudio/`** beyond the two Raja files: `Kim _ Raja vs Parsons _ Whittington.mp4`, `Kodumuri vs Ghafarshad.mp4`, `Mareedu _ Settles vs Petcov _ Abarca.mp4`, `Pham vs Marcanth.mp4` — all ~Apr 11 originals, freshly added on May 3.

## Current cost model

Pipeline current configuration:
- Native res, 5s chunks, 1s overlap → ~720 chunks for a 60-min video
- 4 workers max at native, stride=10 frames, n_look=10 → ~21 detect() calls per RANSAC window
- ~9000 RANSAC windows for 60 min @ 25fps
- Each detect() runs a frame-diff + morphology + connected-components on a ~1920×1080 grayscale frame
- Disk I/O is dominant: each chunk reads ~125 frames sequentially, but the 4 workers all hit disk simultaneously

## Optimization options ranked by expected speedup

### 1. Audio-first gating (highest impact: estimated 5–10× speedup)

Run `run_silent_audio_pass()` BEFORE the vision scan. Use the resulting thwack timestamps to build narrow candidate windows (±5s each), merge overlapping windows. Only chunk and scan video within those windows.

For a typical match: ~500–2000 thwacks → after merging ±5s windows, candidate time covers 30–50% of video duration (in dead-time-rich footage), or 70–80% in tight matches. Even at 70%, that is a 30% video reduction; at 30%, it's a 70% reduction. For Raja vs Wijemanne Edited (already-edited, dense thwacks), audio gating would save little. For raw matches like Pham vs Marcanth, savings should be substantial.

**Caveats:**
- Audio NCC pass costs ~30–60s per hour of video — small overhead.
- Risk: if audio detection misses a thwack, the rally there gets skipped entirely. Mitigation: use a generous ±8s window, OR fall back to full scan if audio_ts is suspiciously small (< some threshold per minute).
- This changes the pipeline semantics — audio is no longer "ground truth for debug only" but a hard gate. Should be opt-in via a checkbox `audio_gated_scan`.

**Implementation sketch:**
```python
# In run_batch_job, before extract_vision_timestamps:
if p.get("audio_gated_scan", False):
    audio_ts = run_silent_audio_pass(video_path)  # ~30s
    # Build candidate windows
    win = float(p.get("audio_gate_radius_s", 5.0))
    raw = sorted([(max(0, t - win), min(video_dur, t + win)) for t in audio_ts])
    # Merge overlaps
    candidate_windows = []
    for s, e in raw:
        if candidate_windows and s <= candidate_windows[-1][1]:
            candidate_windows[-1] = (candidate_windows[-1][0], max(candidate_windows[-1][1], e))
        else:
            candidate_windows.append((s, e))
    # Pass into extract_vision_timestamps; replace `chunks` derivation with intersection
```

### 2. Two-pass coarse-to-fine (estimated 2–3× speedup)

First pass: 320px low-res, stride=20 (4× fewer RANSAC windows × ~5× faster per window) → ~20× cheaper, ~30s for a 60 min video. Identify candidate windows where conf > 0.2 (looser threshold).

Second pass: 640px or native, stride=10, scan only candidate windows ±2s.

This is more conservative than audio gating because it relies on the same vision modality and won't miss rallies that audio missed.

### 3. Larger stride (low-cost, ~2× speedup, some recall risk)

Currently stride=10 means RANSAC every 0.4s (at 25fps). Tennis ball physics: a single bounce-to-bounce arc lasts 0.5–1.5s (12–37 frames). At stride=20, we'd still hit ≥1 RANSAC window per arc in most cases.

Risk: short volleys (~0.4s service returns) could be missed. Mitigation: combine with audio gating to recover those.

Quick win: try stride=15 first.

### 4. Frame caching to shared memory (~1.3–1.5× speedup, large code change)

Currently each chunk worker does `cap.set(POS_FRAMES, ...)` then reads ~125 frames. Disk seek + decode is the bottleneck at native res.

Mmap-friendly approach: pre-decode the full video once into a uint8 array on disk (~`H × W × N_frames × 1` bytes = ~30GB for 60 min @ 25fps @ 1080p — too large for grayscale). Even at 320px: `180 × 320 × 90000 × 1` = ~5GB — feasible but slow to prepare.

Better: use `pyav` or `decord` for frame-accurate random access without ffmpeg seek overhead. Defer this.

### 5. Chunk sizing (marginal, <1.2× speedup)

Currently 5s native / 10s low-res chunks. Larger chunks reduce overlap waste but increase peak RAM per worker. Probably already near optimal. Not worth touching.

## Recommended implementation order (when unpaused)

1. **Audio-first gating with `audio_gated_scan` checkbox** — biggest payoff, smallest code change. Default off. Add a `audio_gate_radius_s` numeric input (default 5.0).
2. **Stride=15 default** with `scan_stride` already exposed in UI — just bump the default value.
3. **Two-pass coarse-to-fine** — bigger refactor, defer until #1 and #2 are validated.
4. **Frame caching** — only if disk I/O profiling actually shows it's the bottleneck.

## Things to investigate before implementing

- Why did Raja vs Wijemanne.mp4 not finish? Check job logs / look for crash. Could be OOM at native res — relevant for sizing decisions.
- Is the 0-active-seg result on Raja vs Wijemanne Edited a real bug or expected (because the video is already edited and all segments are dense)? The 1193 video_ts with conf > 0.30 should produce SOME active_segs even with `dropout_gap=3.0`.

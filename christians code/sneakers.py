"""
Sneaker Noise Detection — full pipeline library.

All processing, visualization, and video I/O lives here.
The notebook imports from this module and is for testing and display only.

Frequency reference (from train_thwack.ipynb):
  thwack band : 1 000 – 6 000 Hz  (racket resonance; bandpass used to EXCLUDE sneakers)
  sneaker band: 6 000 – 11 025 Hz (the ">6 kHz hiss" the thwack bandpass removes)
"""

import numpy as np
import librosa
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import subprocess
import warnings
from scipy.signal import stft, find_peaks
from scipy.stats import multivariate_normal
import IPython.display as ipd

warnings.filterwarnings('ignore')

DEFAULT_SR  = 22050
DEFAULT_WIN = 2048
DEFAULT_HOP = 512
SNEAKER_BAND_HZ = (6000, 11025)   # above thwack bandpass — where squeak energy lives
THWACK_BAND_HZ  = (1000, 6000)    # racket resonance, from train_thwack.ipynb


# ============================================================================
# AUDIO / VIDEO LOADING
# ============================================================================

def load_audio(video_path, sr=DEFAULT_SR, mono=True):
    """Load audio from a video/audio file via librosa."""
    try:
        y, sr_out = librosa.load(video_path, sr=sr, mono=mono)
        return y, sr_out
    except Exception as e:
        print(f"Error loading {video_path}: {e}")
        return None, sr


def load_video_audio(video_path, sr=DEFAULT_SR):
    """Load video, extract audio and metadata. Returns (y, sr, video_info)."""
    y, sr_out = load_audio(video_path, sr=sr)
    video_info = {'path': video_path, 'duration': len(y) / sr_out if y is not None else 0}
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        clip = VideoFileClip(video_path)
        video_info.update({'duration': clip.duration, 'fps': clip.fps, 'size': clip.size})
        clip.close()
    except Exception as e:
        print(f"moviepy unavailable ({e}); using audio-derived duration")
    if y is not None:
        print(f"Loaded: {video_info['duration']/60:.1f} min  ({sr_out} Hz)")
    return y, sr_out, video_info


def get_time_frames_to_seconds(frame_idx, hop_length, sr):
    """STFT frame index → seconds."""
    return frame_idx * hop_length / sr


def get_frame_from_seconds(time_seconds, hop_length, sr):
    """Seconds → STFT frame index."""
    return int(np.round(time_seconds * sr / hop_length))


# ============================================================================
# ANNOTATION I/O
# ============================================================================

def load_annotations(annotation_file, delimiter=None):
    """Load sneaker event CSV (start_time, end_time). Handles decimal and MM:SS times."""
    def parse_time(t):
        t = str(t).strip()
        if ':' in t:
            parts = t.split(':')
            return float(parts[0]) * 60 + float(parts[1])
        return float(t)

    if delimiter is None:
        with open(annotation_file, 'r') as f:
            first = f.readline()
        delimiter = next((s for s in [',', '\t', ' '] if s in first), ',')

    df = pd.read_csv(annotation_file, delimiter=delimiter)
    df.columns = [c.strip() for c in df.columns]
    df['start_time'] = df['start_time'].apply(parse_time)
    df['end_time']   = df['end_time'].apply(parse_time)
    if 'label' not in df.columns:
        df['label'] = 'sneaker'
    print(f"Loaded {len(df)} annotations  ({df['start_time'].min():.1f}s – {df['end_time'].max():.1f}s)")
    return df


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_stft_spectrogram(y, sr=DEFAULT_SR, win_size=DEFAULT_WIN, hop_length=DEFAULT_HOP):
    """STFT magnitude spectrogram. Returns (S_mag, freqs, times)."""
    freqs, times, S = stft(y, fs=sr, nperseg=win_size, noverlap=win_size - hop_length)
    return np.abs(S), freqs, times


def extract_mfcc(y, sr=DEFAULT_SR, n_mfcc=13, win_size=DEFAULT_WIN, hop_length=DEFAULT_HOP):
    """MFCCs. Returns (n_mfcc, n_frames)."""
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=win_size, hop_length=hop_length)


def extract_spectral_features(S_mag):
    """Centroid, bandwidth, zero crossings, energy per frame. Returns (n_frames, 4)."""
    n_frames = S_mag.shape[1]
    features = np.zeros((n_frames, 4))
    bin_idx = np.arange(S_mag.shape[0])
    for i in range(n_frames):
        spec = S_mag[:, i]
        norm = spec / (np.sum(spec) + 1e-8)
        c = np.sum(bin_idx * norm)
        features[i, 0] = c
        features[i, 1] = np.sqrt(np.sum((bin_idx - c) ** 2 * norm))
        features[i, 2] = np.sum(np.abs(np.diff(np.sign(spec))))
        features[i, 3] = np.sum(spec)
    return features


def extract_band_energy_features(S_mag, freqs, bands=None):
    """Energy fraction in each frequency band per frame. Returns (n_frames, n_bands).

    Default bands:
      col 0 — low     0–1 kHz   (crowd noise / rumble)
      col 1 — thwack  1–6 kHz   (racket resonance band from train_thwack.ipynb)
      col 2 — sneaker 6–11 kHz  (squeak band; excluded from thwack model)
    """
    if bands is None:
        bands = [(0, 1000), THWACK_BAND_HZ, SNEAKER_BAND_HZ]
    total = np.sum(S_mag, axis=0) + 1e-8
    out = np.zeros((S_mag.shape[1], len(bands)))
    for j, (lo, hi) in enumerate(bands):
        lo_bin = np.searchsorted(freqs, lo)
        hi_bin = np.searchsorted(freqs, hi)
        out[:, j] = np.sum(S_mag[lo_bin:hi_bin, :], axis=0) / total
    return out


def log_compression(x, gamma=10000):
    """log(1 + gamma * x)."""
    return np.log(1 + gamma * x)


def normalize_l2(features):
    """L2 normalize column-wise."""
    norms = np.linalg.norm(features, axis=0, keepdims=True)
    norms[norms == 0] = 1
    return features / norms


def extract_audio_segment_features(y, sr, start_time, end_time,
                                    win_size=DEFAULT_WIN, hop_length=DEFAULT_HOP):
    """Combined feature vector for a time segment: MFCC + spectral + band energy stats."""
    s, e = int(start_time * sr), int(end_time * sr)
    segment = y[s:e]
    if len(segment) == 0:
        return None
    S_mag, freqs, _ = extract_stft_spectrogram(segment, sr, win_size, hop_length)
    mfcc       = extract_mfcc(segment, sr, n_mfcc=13, win_size=win_size, hop_length=hop_length)
    spec_feats = extract_spectral_features(S_mag)
    band_feats = extract_band_energy_features(S_mag, freqs)
    return np.concatenate([
        np.mean(mfcc, axis=1),    np.std(mfcc, axis=1),
        np.mean(spec_feats, axis=0), np.std(spec_feats, axis=0),
        np.mean(band_feats, axis=0), np.std(band_feats, axis=0),
    ])


# ============================================================================
# SPECTRAL ANALYSIS & TEMPLATE
# ============================================================================

def compute_mean_annotation_spectrum(y, sr, annotations_df,
                                      n_fft=DEFAULT_WIN, hop_length=DEFAULT_HOP):
    """Mean STFT magnitude averaged over all annotated frames. Returns (spectrum, freqs)."""
    spectra = []
    for _, row in annotations_df.iterrows():
        s, e = int(row['start_time'] * sr), int(row['end_time'] * sr)
        seg = y[s:e]
        if len(seg) < n_fft:
            continue
        S = np.abs(librosa.stft(seg, n_fft=n_fft, hop_length=hop_length))
        spectra.append(np.mean(S, axis=1))
    if not spectra:
        return None, None
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return np.mean(np.stack(spectra, axis=0), axis=0), freqs


def compute_mean_background_spectrum(y, sr, annotations_df,
                                      n_fft=DEFAULT_WIN, hop_length=DEFAULT_HOP,
                                      n_samples=20, sample_dur=2.0):
    """Mean STFT magnitude from background regions (well away from all annotations)."""
    duration = len(y) / sr
    ann_iv = list(zip(annotations_df['start_time'], annotations_df['end_time']))
    margin = 1.0
    candidates = []
    t = margin
    while t + sample_dur < duration:
        if all(t + sample_dur < s - margin or t > e + margin for s, e in ann_iv):
            candidates.append(t)
        t += sample_dur + 0.5
    if not candidates:
        return None, None
    np.random.shuffle(candidates)
    spectra = []
    for t in candidates[:n_samples]:
        seg = y[int(t * sr):int((t + sample_dur) * sr)]
        S = np.abs(librosa.stft(seg, n_fft=n_fft, hop_length=hop_length))
        spectra.append(np.mean(S, axis=1))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return np.mean(np.stack(spectra, axis=0), axis=0), freqs


def build_sneaker_template(y, sr, annotations_df, pre_ms=50, post_ms=200,
                            n_fft=DEFAULT_WIN, hop_length=DEFAULT_HOP):
    """Magnitude-averaged, peak-aligned template from annotated sneaker events.
    Adapted from average_aligned_magnitude_stft in train_thwack.ipynb.
    Magnitude averaging avoids phase-cancellation at high frequencies.
    """
    pre_s, post_s = int(pre_ms / 1000 * sr), int(post_ms / 1000 * sr)
    mags, max_frames = [], 0
    for _, row in annotations_df.iterrows():
        seg_start = int(row['start_time'] * sr)
        seg_end   = int(row['end_time']   * sr)
        segment   = y[seg_start:seg_end]
        if len(segment) == 0:
            continue
        center = seg_start + np.argmax(np.abs(segment))
        ws, we = center - pre_s, center + post_s
        if ws < 0 or we > len(y):
            continue
        aligned  = y[ws:we].astype(np.float32)
        aligned /= (np.linalg.norm(aligned) + 1e-8)
        mag = np.abs(librosa.stft(aligned, n_fft=n_fft, hop_length=hop_length))
        mags.append(mag)
        max_frames = max(max_frames, mag.shape[1])
    if not mags:
        return None
    padded   = [np.pad(m, ((0, 0), (0, max_frames - m.shape[1]))) for m in mags]
    template = np.mean(np.stack(padded, axis=0), axis=0).astype(np.float32)
    print(f"Template: {template.shape}  from {len(mags)} annotations")
    return template


def listen_to_annotations(y, sr, annotations_df, pre_ms=100, post_ms=400, gap_ms=300):
    """Concatenate all annotated clips with silence gaps. Returns audio array."""
    silence = np.zeros(int(gap_ms / 1000 * sr), dtype=np.float32)
    pieces  = []
    for _, row in annotations_df.iterrows():
        s = max(0,      int(row['start_time'] * sr) - int(pre_ms  / 1000 * sr))
        e = min(len(y), int(row['end_time']   * sr) + int(post_ms / 1000 * sr))
        snip  = y[s:e].astype(np.float32)
        snip /= (np.max(np.abs(snip)) + 1e-8)
        pieces += [snip, silence]
    return np.concatenate(pieces)


def listen_to_detections(y, sr, roi_times, n=10, pre_ms=100, post_ms=400,
                          gap_ms=300, which="spread"):
    """Concatenate n audio snippets around detected ROIs for in-notebook playback.
    Returns (audio_array, chosen_start_times).
    """
    if not roi_times:
        raise ValueError("No detections.")
    n = min(n, len(roi_times))
    if   which == "first":  idxs = np.arange(n)
    elif which == "last":   idxs = np.arange(len(roi_times) - n, len(roi_times))
    elif which == "spread": idxs = np.linspace(0, len(roi_times) - 1, n).astype(int)
    elif which == "random": idxs = np.sort(np.random.choice(len(roi_times), n, replace=False))
    silence = np.zeros(int(gap_ms / 1000 * sr), dtype=np.float32)
    pieces, chosen = [], []
    for i in idxs:
        start, end = roi_times[i]
        s = max(0,      int(start * sr) - int(pre_ms  / 1000 * sr))
        e = min(len(y), int(end   * sr) + int(post_ms / 1000 * sr))
        snip  = y[s:e].astype(np.float32)
        snip /= (np.max(np.abs(snip)) + 1e-8)
        pieces += [snip, silence]
        chosen.append(start)
    return np.concatenate(pieces), np.array(chosen)


# ============================================================================
# DETECTION
# ============================================================================

def detect_sneaker_frames(S_mag, freqs, times, energy_threshold=None, peak_height_ratio=0.5):
    """v1: total-energy detection (kept for comparison with v2)."""
    frame_energy = np.sum(S_mag, axis=0)
    if energy_threshold is None:
        energy_threshold = np.percentile(frame_energy, 75)
    detected = np.where(frame_energy > energy_threshold)[0]
    scores   = frame_energy / (np.max(frame_energy) + 1e-8)
    return detected, scores


def detect_sneaker_frames_v2(S_mag, freqs, times,
                              sneaker_band_hz=SNEAKER_BAND_HZ,
                              energy_threshold=None,
                              ratio_threshold=None):
    """v2: frequency-selective detection using sneaker-band energy ratio.

    Score = fraction of frame energy in sneaker_band_hz (>6 kHz).
    Requires both a high ratio AND sufficient absolute energy so quiet
    high-frequency noise doesn't trigger.

    Returns (detected_frames, frame_scores, band_ratio).
    """
    lo_bin = np.searchsorted(freqs, sneaker_band_hz[0])
    hi_bin = np.searchsorted(freqs, sneaker_band_hz[1])

    band_energy  = np.sum(S_mag[lo_bin:hi_bin, :], axis=0)
    total_energy = np.sum(S_mag, axis=0) + 1e-8
    band_ratio   = band_energy / total_energy

    if ratio_threshold is None:
        ratio_threshold = np.percentile(band_ratio, 80)
    if energy_threshold is None:
        energy_threshold = np.percentile(total_energy, 50)

    detected = np.where(
        (band_ratio   > ratio_threshold) &
        (total_energy > energy_threshold)
    )[0]

    # Score: ratio weighted by log-energy so loud+HF frames rank highest
    med_e = np.percentile(total_energy, 50) + 1e-8
    scores = band_ratio * np.log1p(total_energy / med_e)

    return detected, scores, band_ratio


def group_frames_to_events(event_times_sec, min_gap_sec=0.3, min_dur_sec=0.1):
    """Connected-component grouping of frame times (seconds) into (start, end) events.

    Pass the time values of detected frames directly — e.g. times[detected_frames].
    Frames within min_gap_sec of each other belong to the same event.
    Events shorter than min_dur_sec are discarded.
    """
    event_times_sec = np.asarray(event_times_sec)
    if len(event_times_sec) == 0:
        return []
    sorted_t = np.sort(event_times_sec)
    events, start, end = [], sorted_t[0], sorted_t[0]
    for t in sorted_t[1:]:
        if t - end <= min_gap_sec:
            end = t
        else:
            if end - start >= min_dur_sec:
                events.append((start, end))
            start = end = t
    if end - start >= min_dur_sec:
        events.append((start, end))
    return events


# ============================================================================
# HMM SMOOTHING  (adapted from HW7 — Viterbi / state prediction)
# ============================================================================

def estimate_hmm_emissions(band_ratio, annotations_df, times):
    """Estimate Gaussian emission parameters for each HMM state from annotations.

    Frames inside annotated windows → sneaker state observations.
    All other frames → background state observations.
    Returns (mu_s, std_s, mu_b, std_b).
    """
    sneaker_mask = np.zeros(len(times), dtype=bool)
    for _, row in annotations_df.iterrows():
        sneaker_mask |= (times >= row['start_time']) & (times <= row['end_time'])

    r_s = band_ratio[sneaker_mask]
    r_b = band_ratio[~sneaker_mask]

    if len(r_s) == 0:
        return 0.50, 0.10, 0.30, 0.08

    mu_s, std_s = float(np.mean(r_s)), max(float(np.std(r_s)), 0.02)
    mu_b, std_b = float(np.mean(r_b)), max(float(np.std(r_b)), 0.02)
    print(f"HMM emissions — sneaker: μ={mu_s:.3f} σ={std_s:.3f}  "
          f"background: μ={mu_b:.3f} σ={std_b:.3f}")
    return mu_s, std_s, mu_b, std_b


def viterbi_smooth(observations, mu_s, std_s, mu_b, std_b,
                    p_stay_background=0.995, p_stay_sneaker=0.90):
    """Two-state Viterbi decoder for HMM smoothing of per-frame scores.

    States: 0=background, 1=sneaker.
    Emission model: Gaussian per state fitted to band_ratio observations.
    Transition matrix enforces temporal continuity — once in the sneaker
    state the model stays there for ~1/( 1-p_stay_sneaker ) frames before
    transitioning back.

    Adapted from the Viterbi / forced-alignment approach in HW7
    (speech recognition with HMM).

    Returns integer state sequence (n_frames,).
    """
    from scipy.stats import norm

    n = len(observations)
    log_T = np.log(np.array([
        [p_stay_background,      1.0 - p_stay_background],
        [1.0 - p_stay_sneaker,   p_stay_sneaker         ],
    ]) + 1e-12)

    # Log emission probabilities: shape (n_frames, 2)
    log_E = np.stack([
        norm.logpdf(observations, mu_b, std_b),
        norm.logpdf(observations, mu_s, std_s),
    ], axis=1)

    # Initialise (assume video starts in background)
    V   = np.empty((n, 2), dtype=np.float64)
    ptr = np.empty((n, 2), dtype=np.int8)
    V[0] = np.log([0.99, 0.01]) + log_E[0]

    # Forward pass — vectorised over states
    for t in range(1, n):
        trans    = V[t - 1, :, np.newaxis] + log_T   # (2_prev, 2_next)
        ptr[t]   = np.argmax(trans, axis=0)
        V[t]     = np.max(trans, axis=0) + log_E[t]

    # Backtrack
    states = np.empty(n, dtype=np.int8)
    states[-1] = int(np.argmax(V[-1]))
    for t in range(n - 2, -1, -1):
        states[t] = ptr[t + 1, states[t + 1]]

    return states


def detect_sneaker_hmm(S_mag, freqs, times, annotations_df=None,
                        sneaker_band_hz=SNEAKER_BAND_HZ,
                        p_stay_background=0.995, p_stay_sneaker=0.90):
    """HMM-smoothed sneaker detection.

    Pipeline (mirrors HW7's approach of using state transitions to smooth
    frame-level decisions):
      1. Compute per-frame band_ratio (v2 frequency-selective score)
      2. Fit Gaussian emission model from annotations (if available)
      3. Viterbi decode → optimal two-state sequence
      4. Return state=1 frame indices, full state sequence, and band_ratio

    The transition probabilities enforce that transients cluster together:
    p_stay_sneaker=0.90 means ~10 consecutive frames (~0.24 s) is the
    expected minimum event length, and isolated single-frame spikes are
    suppressed.
    """
    lo = np.searchsorted(freqs, sneaker_band_hz[0])
    hi = np.searchsorted(freqs, sneaker_band_hz[1])
    band_energy  = np.sum(S_mag[lo:hi, :], axis=0)
    total_energy = np.sum(S_mag,           axis=0) + 1e-8
    band_ratio   = band_energy / total_energy

    if annotations_df is not None:
        mu_s, std_s, mu_b, std_b = estimate_hmm_emissions(band_ratio, annotations_df, times)
    else:
        mu_s, std_s = 0.50, 0.10
        mu_b, std_b = 0.30, 0.08

    states          = viterbi_smooth(band_ratio, mu_s, std_s, mu_b, std_b,
                                     p_stay_background, p_stay_sneaker)
    detected_frames = np.where(states == 1)[0]
    return detected_frames, states, band_ratio


def merge_overlapping_intervals(intervals, gap_threshold=0.1):
    """Merge intervals within gap_threshold seconds of each other."""
    if not intervals:
        return []
    merged = [list(sorted(intervals)[0])]
    for start, end in sorted(intervals)[1:]:
        if start - merged[-1][1] <= gap_threshold:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [tuple(m) for m in merged]


def extract_regions_of_interest(y, sr, detected_frame_indices, times,
                                 min_duration=0.2, context_seconds=0.5):
    """Legacy ROI extraction. Use group_frames_to_events + add_buffer_to_detections instead."""
    if len(detected_frame_indices) == 0:
        return []
    det_times = times[detected_frame_indices]
    gaps      = np.diff(det_times)
    splits    = list(np.where(gaps > 0.1)[0] + 1) + [len(det_times)]
    regions, prev = [], 0
    for sp in splits:
        rt = det_times[prev:sp]
        if len(rt):
            s = max(0,          rt[0]  - context_seconds)
            e = min(len(y)/sr,  rt[-1] + context_seconds)
            if e - s >= min_duration:
                regions.append((s, e))
        prev = sp
    return merge_overlapping_intervals(regions)


# ============================================================================
# TRAINING / INFERENCE
# ============================================================================

def prepare_training_features(y, sr, annotations_df, win_size=DEFAULT_WIN, hop_length=DEFAULT_HOP):
    """Feature matrix + labels from annotated segments. Returns (X, y_labels)."""
    feats, labels = [], []
    for _, row in annotations_df.iterrows():
        f = extract_audio_segment_features(y, sr, row['start_time'], row['end_time'],
                                            win_size, hop_length)
        if f is not None:
            feats.append(f)
            labels.append(1 if str(row['label']).lower() == 'sneaker' else 0)
    return np.array(feats), np.array(labels)


def simple_gaussian_classifier(X_train, y_train, X_test):
    """Gaussian classifier per class; classify by log-likelihood ratio.
    Returns (predictions, probabilities).
    """
    def _fit(X):
        mean = np.mean(X, axis=0)
        cov  = np.cov(X.T)
        try:
            return multivariate_normal(mean=mean, cov=cov)
        except np.linalg.LinAlgError:
            return multivariate_normal(mean=mean, cov=cov + 0.01 * np.eye(X.shape[1]))

    rv_s = _fit(X_train[y_train == 1])
    rv_n = _fit(X_train[y_train == 0])
    l_s  = rv_s.pdf(X_test)
    l_n  = rv_n.pdf(X_test)
    probs = l_s / (l_s + l_n + 1e-10)
    return (probs > 0.5).astype(int), probs


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_detections(y, sr, roi_times, title='Sneaker Noise Detections'):
    """Waveform with detected ROIs shaded in red."""
    t = np.arange(len(y)) / sr
    fig, ax = plt.subplots(figsize=(16, 3))
    ax.plot(t, y, color='steelblue', lw=0.4, alpha=0.8)
    for s, e in roi_times:
        ax.axvspan(s, e, alpha=0.3, color='red')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'{title}  ({len(roi_times)} regions)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_sneaker_spectrum(mean_sneaker, mean_background, freqs,
                                sneaker_band_hz=SNEAKER_BAND_HZ,
                                thwack_band_hz=THWACK_BAND_HZ):
    """3-panel plot: sneaker spectrum, background spectrum, and their ratio."""
    def _db(x):
        return 20 * np.log10(x / (x.max() + 1e-9) + 1e-9)

    db_s  = _db(mean_sneaker)
    db_bg = _db(mean_background)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    for ax, data, color, label in zip(
        axes[:2],
        [db_s, db_bg],
        ['#d62728', 'steelblue'],
        ['Annotated sneaker events', 'Background (non-sneaker)'],
    ):
        ax.semilogx(freqs + 1, data, color=color, lw=1.2)
        ax.axvspan(*thwack_band_hz,  alpha=0.10, color='orange', label='thwack 1–6 kHz')
        ax.axvspan(*sneaker_band_hz, alpha=0.15, color='green',  label='sneaker 6–11 kHz')
        ax.set_xlim(100, freqs[-1]); ax.set_ylim(-60, 5)
        ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('dB (normalised)')
        ax.set_title(label); ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

    ratio_db = db_s - db_bg
    axes[2].semilogx(freqs + 1, ratio_db, color='purple', lw=1.2)
    axes[2].axhline(0, color='black', lw=0.8, ls='--')
    axes[2].axvspan(*thwack_band_hz,  alpha=0.10, color='orange', label='thwack band')
    axes[2].axvspan(*sneaker_band_hz, alpha=0.15, color='green',  label='sneaker band')
    axes[2].set_xlim(100, freqs[-1])
    axes[2].set_xlabel('Frequency (Hz)'); axes[2].set_ylabel('Sneaker − Background (dB)')
    axes[2].set_title('Spectral ratio: where sneakers differ from background')
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3, which='both')

    plt.tight_layout()
    plt.show()


def visualize_template(template, sr=DEFAULT_SR, hop_length=DEFAULT_HOP,
                        sneaker_band_hz=SNEAKER_BAND_HZ, thwack_band_hz=THWACK_BAND_HZ):
    """2-panel template plot: magnitude spectrogram + mean frequency profile."""
    freqs   = librosa.fft_frequencies(sr=sr, n_fft=(template.shape[0] - 1) * 2)
    db_tmpl = librosa.amplitude_to_db(template, ref=np.max)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    img = librosa.display.specshow(db_tmpl, sr=sr, hop_length=hop_length,
                                    x_axis='time', y_axis='log', cmap='magma', ax=axes[0])
    axes[0].set_title('Sneaker template (magnitude-averaged, peak-aligned)')
    fig.colorbar(img, ax=axes[0], format='%+2.0f dB')

    profile = template.mean(axis=1)
    axes[1].semilogx(freqs, 20 * np.log10(profile / (profile.max() + 1e-9) + 1e-9))
    axes[1].axvspan(*thwack_band_hz,  alpha=0.10, color='orange', label='thwack 1–6 kHz')
    axes[1].axvspan(*sneaker_band_hz, alpha=0.15, color='green',  label='sneaker 6–11 kHz')
    axes[1].set_xlim(100, sr / 2); axes[1].set_ylim(-60, 5)
    axes[1].set_xlabel('Frequency (Hz)'); axes[1].set_ylabel('dB (normalised)')
    axes[1].set_title('Mean spectral profile'); axes[1].legend(); axes[1].grid(alpha=0.3, which='both')

    plt.tight_layout()
    plt.show()


def export_detections_to_csv(roi_times, output_file):
    """Write ROI tuples to CSV with duration column."""
    d = os.path.dirname(output_file)
    if d:
        os.makedirs(d, exist_ok=True)
    df = pd.DataFrame(roi_times, columns=['start_time', 'end_time'])
    df['duration'] = df['end_time'] - df['start_time']
    df.to_csv(output_file, index=False)
    print(f"Saved {len(roi_times)} detections → {output_file}")


def add_buffer_to_detections(roi_times, buffer_seconds=0.5, video_duration=None):
    """Pad each ROI by buffer_seconds and merge newly overlapping intervals."""
    buffered = [
        (max(0.0, s - buffer_seconds),
         min(video_duration, e + buffer_seconds) if video_duration else e + buffer_seconds)
        for s, e in roi_times
    ]
    return merge_overlapping_intervals(buffered, gap_threshold=0.0)


def print_detection_summary(roi_times, video_duration=None):
    """Print detection stats."""
    total = sum(e - s for s, e in roi_times)
    print(f"Detections  : {len(roi_times)}")
    print(f"Total time  : {total:.1f} s  ({total/60:.2f} min)")
    if roi_times:
        durs = [e - s for s, e in roi_times]
        print(f"Min/Max/Mean: {min(durs):.2f} / {max(durs):.2f} / {np.mean(durs):.2f} s")
    if video_duration:
        print(f"Coverage    : {100*total/video_duration:.1f}%  of {video_duration/60:.1f} min")


# ============================================================================
# VIDEO I/O
# ============================================================================

def crop_video_segment(video_path, start_time, end_time, output_path):
    """Crop a single video segment with audio using ffmpeg directly."""
    try:
        ffmpeg = shutil.which('ffmpeg') or '/opt/anaconda3/envs/E207_Spr26/bin/ffmpeg'
        cmd = [
            ffmpeg, '-y',
            '-ss', str(start_time),
            '-to', str(end_time),
            '-i', video_path,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-avoid_negative_ts', 'make_zero',
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error cropping {output_path}:\n{result.stderr[-300:]}")
            return None
        return output_path
    except Exception as e:
        print(f"Error cropping {output_path}: {e}")
        return None


def crop_video_regions(video_path, roi_times, output_dir=None):
    """Crop multiple ROIs from a video. Returns list of output paths.

    Any existing .mp4 clips in output_dir are moved to output_dir/deprecated/
    before writing new clips.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(video_path), '..', 'output',
                                   'sneakers_detected_clips')
    os.makedirs(output_dir, exist_ok=True)

    existing = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
    if existing:
        dep_dir = os.path.join(os.path.dirname(os.path.abspath(output_dir)), 'deprecated')
        os.makedirs(dep_dir, exist_ok=True)
        for fname in existing:
            shutil.move(os.path.join(output_dir, fname), os.path.join(dep_dir, fname))
        print(f"  Archived {len(existing)} existing clips → {dep_dir}")

    paths = []
    for i, (s, e) in enumerate(roi_times):
        out = os.path.join(output_dir, f'sneaker_clip_{i:03d}.mp4')
        if crop_video_segment(video_path, s, e, out):
            paths.append(out)
    print(f"Cropped {len(paths)} clips → {output_dir}")
    return paths


def concatenate_video_clips(clip_paths, output_path):
    """Concatenate a list of video clips into one file."""
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        from moviepy.editor import concatenate_videoclips
        clips = [VideoFileClip(p) for p in clip_paths]
        final = concatenate_videoclips(clips)
        final.write_videofile(output_path, codec='libx264', audio_codec='aac',
                              verbose=False, logger=None)
        for c in clips:
            c.close()
        print(f"Concatenated {len(clip_paths)} clips → {output_path}")
        return output_path
    except Exception as e:
        print(f"Error concatenating: {e}")
        return None


# ============================================================================
# COMPLETE WORKFLOW
# ============================================================================

def complete_workflow(video_path, annotations_file=None, method='supervised',
                      output_prefix='sneaker', buffer_seconds=0.5,
                      sneaker_band_hz=SNEAKER_BAND_HZ):
    """End-to-end pipeline: load → v2 detect → export CSV → crop clips."""
    y, sr, video_info = load_video_audio(video_path)
    if y is None:
        return {}

    annotations_df = None
    if annotations_file and os.path.exists(annotations_file):
        annotations_df = load_annotations(annotations_file)

    S_mag, freqs, times = extract_stft_spectrogram(y, sr)
    detected, scores, band_ratio = detect_sneaker_frames_v2(S_mag, freqs, times,
                                                             sneaker_band_hz=sneaker_band_hz)
    roi_times = group_frames_to_events(times[detected], min_gap_sec=0.3, min_dur_sec=0.1)
    roi_times = add_buffer_to_detections(roi_times, buffer_seconds=buffer_seconds,
                                          video_duration=video_info['duration'])

    print_detection_summary(roi_times, video_duration=video_info['duration'])
    visualize_detections(y, sr, roi_times)

    base = os.path.join(os.path.dirname(video_path), '..', 'output')
    out_csv = os.path.join(base, f'{output_prefix}_detected_timestamps.csv')
    export_detections_to_csv(roi_times, out_csv)

    clip_paths = crop_video_regions(video_path, roi_times,
                                     output_dir=os.path.join(base, f'{output_prefix}_detected_clips'))

    return {
        'roi_times':   roi_times,
        'output_csv':  out_csv,
        'clip_paths':  clip_paths,
        'band_ratio':  band_ratio,
        'video_info':  video_info,
    }

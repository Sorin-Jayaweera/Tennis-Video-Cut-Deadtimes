"""
Sneaker Noise Detector — high-level class wrapper around sneakers.py.
"""

import numpy as np
from sneakers import (
    extract_stft_spectrogram,
    detect_sneaker_frames,
    detect_sneaker_frames_v2,
    detect_sneaker_hmm,
    group_frames_to_events,
    extract_regions_of_interest,
    add_buffer_to_detections,
    prepare_training_features,
    extract_audio_segment_features,
    simple_gaussian_classifier,
    SNEAKER_BAND_HZ,
)


class SneakerDetector:
    """Wrapper for sneaker noise detection. Holds audio params and trained state."""

    def __init__(self, sr=22050, win_size=2048, hop_length=512,
                 sneaker_band_hz=SNEAKER_BAND_HZ):
        self.sr             = sr
        self.win_size       = win_size
        self.hop_length     = hop_length
        self.sneaker_band   = sneaker_band_hz
        self.X_train        = None
        self.y_train        = None
        self.is_trained     = False

    def detect_unsupervised(self, y, energy_threshold=None):
        """v2 frequency-selective detection — no training needed.
        Returns (roi_times, scores_dict).
        """
        S_mag, freqs, times = extract_stft_spectrogram(y, self.sr, self.win_size, self.hop_length)
        detected, scores, band_ratio = detect_sneaker_frames_v2(
            S_mag, freqs, times,
            sneaker_band_hz=self.sneaker_band,
            energy_threshold=energy_threshold,
        )
        roi_times = group_frames_to_events(times[detected], min_gap_sec=0.3, min_dur_sec=0.1)
        roi_times = add_buffer_to_detections(roi_times, buffer_seconds=0.5,
                                              video_duration=len(y) / self.sr)
        return roi_times, {
            'method':           'unsupervised_v2_band_energy',
            'n_detected_frames': len(detected),
            'n_regions':         len(roi_times),
            'mean_band_ratio':   float(np.mean(band_ratio)),
        }

    def train(self, y, annotations_df):
        """Train Gaussian classifier on annotated segments.
        Returns True on success.
        """
        print("Training classifier...")
        X, y_labels = prepare_training_features(y, self.sr, annotations_df,
                                                 self.win_size, self.hop_length)
        if len(X) == 0:
            print("  No training samples extracted.")
            return False
        print(f"  {len(X)} samples | sneaker: {y_labels.sum()} | features: {X.shape[1]}")
        self.X_train    = X
        self.y_train    = y_labels
        self.is_trained = True
        print("  Training complete.")
        return True

    def detect_supervised(self, y, buffer_seconds=0.5):
        """v2 candidate detection followed by Gaussian classifier scoring.

        Pipeline:
          1. detect_sneaker_frames_v2 → candidate ROIs
          2. extract features for each ROI
          3. score with trained Gaussian; keep ROIs with P(sneaker) > 0.5
        Returns (roi_times, scores_dict).
        """
        if not self.is_trained:
            print("Classifier not trained — call train() first.")
            return [], {}

        print("Running supervised detection...")
        S_mag, freqs, times = extract_stft_spectrogram(y, self.sr, self.win_size, self.hop_length)

        detected, frame_scores, band_ratio = detect_sneaker_frames_v2(
            S_mag, freqs, times, sneaker_band_hz=self.sneaker_band
        )
        candidate_rois = group_frames_to_events(times[detected], min_gap_sec=0.3, min_dur_sec=0.1)
        candidate_rois = add_buffer_to_detections(candidate_rois, buffer_seconds=0.3,
                                                   video_duration=len(y) / self.sr)

        confirmed, probs_list = [], []
        for start, end in candidate_rois:
            feats = extract_audio_segment_features(y, self.sr, start, end,
                                                    self.win_size, self.hop_length)
            if feats is None:
                continue
            _, prob = simple_gaussian_classifier(self.X_train, self.y_train,
                                                  feats.reshape(1, -1))
            probs_list.append(float(prob[0]))
            if prob[0] > 0.5:
                confirmed.append((start, end))

        roi_times = add_buffer_to_detections(confirmed, buffer_seconds=buffer_seconds,
                                              video_duration=len(y) / self.sr)
        print(f"  {len(candidate_rois)} candidates → {len(roi_times)} confirmed")
        return roi_times, {
            'method':              'supervised_gaussian_v2',
            'n_candidates':         len(candidate_rois),
            'n_confirmed':          len(roi_times),
            'mean_sneaker_prob':    float(np.mean(probs_list)) if probs_list else 0.0,
        }

    def detect_hmm(self, y, annotations_df=None, buffer_seconds=0.5,
                   p_stay_background=0.995, p_stay_sneaker=0.90):
        """HMM Viterbi detection — best temporal coherence.

        Uses the two-state HMM from sneakers.detect_sneaker_hmm:
          - Emission parameters estimated from annotations when available
          - Viterbi enforces that sneaker transients cluster together
          - p_stay_sneaker controls minimum event persistence
            (0.90 → ~10 frames ≈ 0.24 s minimum; raise to 0.95 for longer events)
        Returns (roi_times, scores_dict).
        """
        print("Running HMM detection...")
        S_mag, freqs, times = extract_stft_spectrogram(y, self.sr, self.win_size, self.hop_length)
        detected, states, band_ratio = detect_sneaker_hmm(
            S_mag, freqs, times,
            annotations_df=annotations_df,
            sneaker_band_hz=self.sneaker_band,
            p_stay_background=p_stay_background,
            p_stay_sneaker=p_stay_sneaker,
        )
        roi_times = group_frames_to_events(times[detected], min_gap_sec=0.3, min_dur_sec=0.1)
        roi_times = add_buffer_to_detections(roi_times, buffer_seconds=buffer_seconds,
                                              video_duration=len(y) / self.sr)
        n_sneaker = int(np.sum(states == 1))
        print(f"  {n_sneaker} sneaker frames → {len(roi_times)} events")
        return roi_times, {
            'method':           'hmm_viterbi',
            'n_sneaker_frames':  n_sneaker,
            'n_regions':         len(roi_times),
            'mean_band_ratio':   float(np.mean(band_ratio[states == 1])) if n_sneaker else 0.0,
            'states':            states,
            'band_ratio':        band_ratio,
        }

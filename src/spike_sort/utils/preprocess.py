import numpy as np
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
import spikeinterface.extractors as se
from spikeinterface.core import extract_waveforms
from pathlib import Path
import contextlib
import os
from typing import Tuple, Dict, Any

TINY = 1e-8

def detect_spikes(
    recording,
    method:             str = 'locally_exclusive',
    peak_sign:          str = 'neg',
    detect_threshold: float = 5.0,
    exclude_sweep_ms: float = 0.1,
    n_jobs:             int = 1,
    progress_bar:      bool = True
) -> np.ndarray:
    """Detect spike peaks on the given recording.

    Args:
        recording: SpikeInterface recording extractor
        method: Peak detection method
        peak_sign: 'pos' or 'neg'
        detect_threshold: Detection threshold
        exclude_sweep_ms: Refractory window in ms to avoid duplicates
        n_jobs: Parallel jobs for detection
        progress_bar: Show progress bar if True

    Returns:
        Peaks object compatible with SpikeInterface waveform extraction
    """
    peaks = detect_peaks(
        recording,
        method           = method,
        peak_sign        = peak_sign,
        detect_threshold = detect_threshold,
        exclude_sweep_ms = exclude_sweep_ms,
        n_jobs           = n_jobs,
        progress_bar     = progress_bar
    )
    return peaks

def extract_waveform_features(
    recording,
    peaks,
    ms_before           = 0.6,
    ms_after            = 1.4,
    channels_per_spike  = 4,
    max_spikes_per_unit = 1000,
    scale_features      = True,
    verbose             = True,
    n_jobs              = 1,
    chunk_duration      = "1s"
):
    fs = recording.get_sampling_frequency()
    nbefore = int(ms_before * fs / 1000)
    nafter = int(ms_after * fs / 1000)
    n_peaks = len(peaks)
    if n_peaks == 0:
        return None, None, {}
    if n_peaks > max_spikes_per_unit * 200:
        idx = np.random.choice(n_peaks, max_spikes_per_unit * 200, replace=False)
        idx = np.sort(idx)
        peaks = peaks[idx]
        n_peaks = len(peaks)
    times = np.asarray(peaks['sample_index'], dtype=np.int64)
    labels = np.asarray(peaks['channel_index'], dtype=np.int64)
    sorting = se.NumpySorting.from_samples_and_labels([times], [labels], sampling_frequency=fs)
    ts = np.datetime64('now').astype('datetime64[ms]').astype(object).strftime('%Y%m%d_%H%M%S')
    folder = Path(f"results/waveforms_extractor_tmp_{ts}")
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            we = extract_waveforms(
                recording,
                sorting,
                folder=str(folder),
                ms_before=ms_before,
                ms_after=ms_after,
                max_spikes_per_unit=max_spikes_per_unit,
                load_if_exists=None,
            )
    unit_ids = sorting.get_unit_ids()
    features = []
    peak_locations = []
    for unit_id in unit_ids:
        wf = we.get_waveforms(unit_id)
        if wf is None or wf.size == 0:
            continue
        n_spikes_unit = wf.shape[0]
        for si in range(n_spikes_unit):
            w = wf[si]  # (n_samples, n_channels)
            amps = np.min(w, axis=0)
            top_idx = np.argsort(np.abs(amps))[-channels_per_spike:][::-1]
            w_sel = w[:, top_idx]
            v = w_sel.reshape(-1)
            m = v.mean()
            sdev = v.std() + TINY
            v = (v - m) / sdev
            min_amp = float(np.min(w_sel))
            energy = float(np.sum(w_sel * w_sel))
            v = np.concatenate([v.astype(np.float32), np.array([min_amp, energy], dtype=np.float32)])
            features.append(v)
        st = sorting.get_unit_spike_train(unit_id=unit_id)
        ch = int(unit_id) if isinstance(unit_id, (int, np.integer)) else unit_id
        peak_locations.extend([(int(s), ch) for s in st])
    X = np.asarray(features, dtype=np.float32)
    peak_locations = np.asarray(peak_locations, dtype=np.int64)
    if scale_features and X.size > 0:
        mx = X.mean(axis=0, keepdims=True)
        sx = X.std(axis=0, keepdims=True) + TINY
        X = (X - mx) / sx
    meta = {
        'n_spikes': int(X.shape[0]),
        'n_features': int(X.shape[1]) if X.size > 0 else 0,
        'waveform_length': int(nbefore + nafter),
        'channels_per_spike': int(channels_per_spike),
        'ms_before': float(ms_before),
        'ms_after': float(ms_after)
    }
    return X, peak_locations, meta

def preprocess_recording(
    recording,
    method:                str = 'locally_exclusive',
    peak_sign:             str = 'neg',
    detect_threshold:    float = 5.0,
    exclude_sweep_ms:    float = 0.1,
    ms_before:           float = 0.6,
    ms_after:            float = 1.4,
    channels_per_spike:    int = 4,
    max_spikes_per_unit:   int = 1000,
    scale_features:       bool = True,
    n_jobs:                int = 1,
    verbose:              bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Full preprocessing: detect spikes and extract waveform features.

    Args:
        recording: SpikeInterface recording extractor
        method: Peak detection method
        peak_sign: 'pos' or 'neg'
        detect_threshold: Detection threshold
        exclude_sweep_ms: Refractory window in ms to avoid duplicates
        ms_before: Milliseconds before peak for waveform window
        ms_after: Milliseconds after peak for waveform window
        channels_per_spike: Number of channels per spike to include
        max_spikes_per_unit: Cap spikes per unit for extraction
        scale_features: Whether to z-score features per dimension
        n_jobs: Parallel jobs for extraction
        verbose: Log progress if True

    Returns:
        Tuple of (X, peak_locations, metadata)
    """
    peaks = detect_spikes(
        recording,
        method           = method,
        peak_sign        = peak_sign,
        detect_threshold = detect_threshold,
        exclude_sweep_ms = exclude_sweep_ms,
        n_jobs           = n_jobs,
        progress_bar     = verbose
    )
    X, peak_locations, meta = extract_waveform_features(
        recording,
        peaks,
        ms_before           = ms_before,
        ms_after            = ms_after,
        channels_per_spike  = channels_per_spike,
        max_spikes_per_unit = max_spikes_per_unit,
        scale_features      = scale_features,
        verbose             = verbose,
        n_jobs              = n_jobs
    )
    return X, peak_locations, meta
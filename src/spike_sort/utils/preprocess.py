import numpy as np
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
import spikeinterface.extractors as se
from spikeinterface.core import extract_waveforms
import spikeinterface.preprocessing as spre
import spikeinterface.full as si
from pathlib import Path
import contextlib
import os
import io
import shutil
import warnings
from typing import Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass

TINY = 1e-8

SYN_PRE_MOTION = {
    "bandpass_filter": {"freq_min": 300.0, "freq_max": 6000.0},
    "astype": {"dtype": "float32"},
}

REAL_PRE_MOTION = {
    "highpass_filter": {"freq_min": 300.0},
    "phase_shift": {},
    "detect_and_interpolate_bad_channels": {},
    "highpass_spatial_filter": {},
    "astype": {"dtype": "float32"},
}


@dataclass
class PreprocessingResult:
    """Container for preprocessing results with metadata."""
    features: np.ndarray
    peak_locations: np.ndarray
    metadata: Dict[str, Any]
    recording: Any = None
    motion: Any = None
    
    def __post_init__(self):
        """Validate data shapes and types."""
        if self.features.size > 0:
            assert self.features.ndim == 2, "Features must be 2D array"
            assert self.features.shape[0] == self.peak_locations.shape[0], \
                "Features and peak_locations must have same number of samples"


def apply_ibl_preprocessing(
    recording,
    synthetic: bool = False,
    correction_motion: bool = True,
    cache_dir: Optional[Path] = None,
    save_motion: bool = False,
    save_rec: bool = False,
    job_kwargs: Optional[Dict[str, Any]] = None,
    motion_preset: str = "dredge",
) -> Tuple[Any, Optional[Any]]:
    """Apply IBL preprocessing pipeline with motion correction.
    
    Args:
        recording: SpikeInterface recording extractor
        synthetic: If True, use synthetic data pipeline (bandpass only)
        correction_motion: Whether to apply motion correction
        cache_dir: Directory to save intermediate results
        save_motion: Save motion correction data to disk
        save_rec: Save final preprocessed recording to disk
        job_kwargs: Additional arguments for parallel processing
        motion_preset: Motion correction preset ("dredge", "kilosort_like", etc.)
    
    Returns:
        Tuple of (preprocessed_recording, motion_object)
    
    Raises:
        ValueError: If recording is invalid or preprocessing fails
    """
    if recording is None:
        raise ValueError("Recording cannot be None")
    
    job_kwargs = job_kwargs or {}
    ts = np.datetime64('now').astype('datetime64[ms]').astype(object).strftime('%Y%m%d_%H%M%S')
    
    # Setup cache directories with better path handling
    mc_folder = None
    final_rec_folder = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        if save_motion:
            mc_folder = cache_dir / f"motion_info_{ts}"
        if save_rec:
            final_rec_folder = cache_dir / f"final_rec_{ts}"
    
    # Apply preprocessing pipeline
    pipeline = SYN_PRE_MOTION if synthetic else REAL_PRE_MOTION
    print("[Preprocessing] IBL pipeline...")
    
    try:
        preprocessed_rec = si.apply_preprocessing_pipeline(
            recording=recording, 
            pipeline_or_dict=pipeline
        )
    except Exception as e:
        raise ValueError(f"Preprocessing pipeline failed: {str(e)}") from e
    
    motion_corrected_rec = preprocessed_rec
    motion = None
    
    # Apply motion correction if requested
    if correction_motion:
        print(f"[Preprocessing] Motion correction (preset: {motion_preset})...")
        try:
            motion_corrected_rec, motion = spre.correct_motion(
                recording=preprocessed_rec,
                preset=motion_preset,
                folder=mc_folder,
                output_motion=True,
                overwrite=True,
                **job_kwargs,
            )
        except Exception as e:
            warnings.warn(f"Motion correction failed: {str(e)}. Continuing without motion correction.")
            motion_corrected_rec = preprocessed_rec
            motion = None
    
    final_rec = motion_corrected_rec
    
    # Save final recording if requested
    if save_rec and final_rec_folder is not None:
        print(f"[Preprocessing] Saving recording to {final_rec_folder}...")
        try:
            final_rec.save(folder=final_rec_folder, overwrite=True, **job_kwargs)
        except Exception as e:
            warnings.warn(f"Failed to save recording: {str(e)}")
    
    return final_rec, motion


def detect_spikes(
    recording,
    method: str = 'locally_exclusive',
    peak_sign: str = 'neg',
    detect_threshold: float = 5.0,
    exclude_sweep_ms: float = 0.1,
    n_jobs: int = 1,
    progress_bar: bool = True,
    chunk_duration: str = "1s",
) -> np.ndarray:
    """Detect spike peaks on the given recording.

    Args:
        recording: SpikeInterface recording extractor
        method: Peak detection method ('locally_exclusive', 'by_channel', etc.)
        peak_sign: 'pos', 'neg', or 'both'
        detect_threshold: Detection threshold in MAD units
        exclude_sweep_ms: Refractory window in ms to avoid duplicates
        n_jobs: Parallel jobs for detection
        progress_bar: Show progress bar if True
        chunk_duration: Chunk size for processing

    Returns:
        Peaks array compatible with SpikeInterface waveform extraction
        
    Raises:
        ValueError: If parameters are invalid
    """
    if recording is None:
        raise ValueError("Recording cannot be None")
    
    if detect_threshold <= 0:
        raise ValueError("detect_threshold must be positive")
    
    if exclude_sweep_ms < 0:
        raise ValueError("exclude_sweep_ms must be non-negative")
    
    print(f"[Detection] Method: {method}, threshold: {detect_threshold}, sign: {peak_sign}")
    
    try:
        peaks = detect_peaks(
            recording,
            method=method,
            peak_sign=peak_sign,
            detect_threshold=detect_threshold,
            exclude_sweep_ms=exclude_sweep_ms,
            n_jobs=n_jobs,
            progress_bar=progress_bar,
            chunk_duration=chunk_duration,
        )
    except Exception as e:
        raise ValueError(f"Peak detection failed: {str(e)}") from e
    
    print(f"[Detection] Found {len(peaks)} peaks")
    return peaks


def extract_waveform_features(
    recording,
    peaks,
    ms_before: float = 0.6,
    ms_after: float = 1.4,
    channels_per_spike: int = 8,
    max_spikes_per_unit: int = 1000,
    max_spikes_global: Optional[int] = 3000,
    scale_features: bool = True,
    verbose: bool = True,
    n_jobs: int = 1,
    chunk_duration: str = "1s",
    cleanup_temp: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Extract waveform features from detected peaks.
    
    Args:
        recording: SpikeInterface recording extractor
        peaks: Array of detected peaks
        ms_before: Milliseconds before peak for waveform window
        ms_after: Milliseconds after peak for waveform window
        channels_per_spike: Number of channels per spike to include
        max_spikes_per_unit: Cap spikes per unit for extraction
        max_spikes_global: Global cap on total spikes to process
        scale_features: Whether to z-score features per dimension
        verbose: Log progress if True
        n_jobs: Parallel jobs for extraction
        chunk_duration: Chunk size for processing
        cleanup_temp: Remove temporary waveform folder after extraction
    
    Returns:
        Tuple of (features, peak_locations, metadata)
        - features: (n_spikes, n_features) array
        - peak_locations: (n_spikes, 2) array of (sample_index, channel_index)
        - metadata: Dictionary with extraction parameters and statistics
    """
    fs = recording.get_sampling_frequency()
    nbefore = int(ms_before * fs / 1000)
    nafter = int(ms_after * fs / 1000)
    n_peaks = len(peaks)
    
    # Early return for no peaks
    if n_peaks == 0:
        if verbose:
            print("[Features] No peaks detected, returning empty arrays")
        return (
            np.array([], dtype=np.float32).reshape(0, 0),
            np.array([], dtype=np.int64).reshape(0, 2),
            {'n_spikes': 0, 'n_features': 0, 'waveform_length': nbefore + nafter}
        )
    
    # Subsample if needed
    if max_spikes_global and n_peaks > max_spikes_global:
        if verbose:
            print(f"[Features] Subsampling {n_peaks} peaks to {max_spikes_global}")
        idx = np.random.choice(n_peaks, max_spikes_global, replace=False)
        peaks = peaks[np.sort(idx)]
        n_peaks = len(peaks)
    
    # Create sorting object from peaks
    times = np.asarray(peaks['sample_index'], dtype=np.int64)
    labels = np.asarray(peaks['channel_index'], dtype=np.int64)
    sorting = se.NumpySorting.from_times_labels(
        [times], [labels], 
        sampling_frequency=fs
    )
    
    # Create temporary folder for waveform extraction
    ts = np.datetime64('now').astype('datetime64[ms]').astype(object).strftime('%Y%m%d_%H%M%S')
    folder = Path(f"results/waveforms_extractor_tmp_{ts}")
    folder.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract waveforms (suppress output)
    _buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(_buf):
            we = extract_waveforms(
                recording,
                sorting,
                folder=str(folder),
                ms_before=ms_before,
                ms_after=ms_after,
                max_spikes_per_unit=max_spikes_per_unit,
                load_if_exists=None,
                n_jobs=n_jobs,
            )
    except Exception as e:
        if cleanup_temp and folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
        raise ValueError(f"Waveform extraction failed: {str(e)}") from e
    
    # Extract features from waveforms
    unit_ids = sorting.get_unit_ids()
    features = []
    peak_locations = []
    
    for unit_id in unit_ids:
        wf = we.get_waveforms(unit_id)
        if wf is None or wf.size == 0:
            continue
        
        n_spikes_unit = wf.shape[0]
        
        for si in range(n_spikes_unit):
            w = wf[si]  # Shape: (n_samples, n_channels)
            
            # Find channels with largest amplitudes
            amps = np.min(w, axis=0)
            top_idx = np.argsort(np.abs(amps))[-channels_per_spike:][::-1]
            w_sel = w[:, top_idx]
            
            # Flatten and normalize waveform
            v = w_sel.reshape(-1)
            m = v.mean()
            sdev = v.std() + TINY
            v = (v - m) / sdev
            
            # Extract scalar features
            min_amp = float(np.min(w_sel))
            energy = float(np.sum(w_sel * w_sel))
            ptp = float(np.max(w_sel) - np.min(w_sel))
            
            # Calculate spike width at half-amplitude
            ch_min = int(np.argmin(np.min(w_sel, axis=0)))
            w_main = w_sel[:, ch_min]
            i_min = int(np.argmin(w_main))
            half = w_main[i_min] / 2.0
            
            # Find left crossing
            left = i_min
            while left > 0 and w_main[left] < half:
                left -= 1
            
            # Find right crossing
            right = i_min
            nlen = w_main.shape[0]
            while right < nlen - 1 and w_main[right] < half:
                right += 1
            
            width_ms = float((right - left) * 1000.0 / fs)
            
            # Combine waveform features with scalar features
            v = np.concatenate([
                v.astype(np.float32),
                np.array([min_amp, energy, ptp, width_ms], dtype=np.float32)
            ])
            
            features.append(v)
        
        # Store peak locations (sample_index, channel_index)
        st = sorting.get_unit_spike_train(unit_id=unit_id)
        ch = int(unit_id) if isinstance(unit_id, (int, np.integer)) else unit_id
        peak_locations.extend([(int(s), ch) for s in st])
    
    # Convert to arrays
    X = np.asarray(features, dtype=np.float32)
    peak_locations = np.asarray(peak_locations, dtype=[('sample_index', 'int64'), ('channel_index', 'int64')])
    
    # Apply global scaling if requested
    if scale_features and X.size > 0:
        mx = X.mean(axis=0, keepdims=True)
        sx = X.std(axis=0, keepdims=True) + TINY
        X = (X - mx) / sx
    
    # Cleanup temporary folder
    if cleanup_temp:
        try:
            shutil.rmtree(folder, ignore_errors=True)
        except Exception as e:
            warnings.warn(f"Failed to cleanup temporary folder {folder}: {str(e)}")
    
    # Create metadata
    meta = {
        'n_spikes': int(X.shape[0]),
        'n_features': int(X.shape[1]) if X.size > 0 else 0,
        'waveform_length': int(nbefore + nafter),
        'channels_per_spike': int(channels_per_spike),
        'ms_before': float(ms_before),
        'ms_after': float(ms_after),
        'sampling_frequency': float(fs),
        'n_peaks_detected': int(len(peaks)),
        'subsampled': bool(max_spikes_global and n_peaks < len(peaks)),
    }
    
    if verbose:
        print(f"[Features] Extracted {meta['n_spikes']} spikes, {meta['n_features']} features each")
    
    return X, peak_locations, meta


def preprocess_recording(
    recording,
    method: str = 'locally_exclusive',
    peak_sign: str = 'neg',
    detect_threshold: float = 5.0,
    exclude_sweep_ms: float = 0.1,
    ms_before: float = 0.6,
    ms_after: float = 1.4,
    channels_per_spike: int = 4,
    max_spikes_per_unit: int = 1000,
    max_spikes_global: Optional[int] = 3000,
    scale_features: bool = True,
    n_jobs: int = 1,
    verbose: bool = True,
    apply_ibl_pipeline: bool = False,
    synthetic_pipeline: bool = False,
    correct_motion: bool = False,
    motion_preset: str = "dredge",
    cache_dir: Optional[Union[str, Path]] = None,
    save_motion: bool = False,
    save_rec: bool = False,
    cleanup_temp: bool = True,
    job_kwargs: Optional[Dict[str, Any]] = None,
    return_recording: bool = False,
    chunk_duration: str = "1s",
) -> Union[Tuple[np.ndarray, np.ndarray, Dict[str, Any]], PreprocessingResult]:
    """Full preprocessing pipeline: detect spikes and extract waveform features.

    Args:
        recording: SpikeInterface recording extractor
        method: Peak detection method ('locally_exclusive', 'by_channel', etc.)
        peak_sign: 'pos', 'neg', or 'both'
        detect_threshold: Detection threshold in MAD units
        exclude_sweep_ms: Refractory window in ms to avoid duplicates
        ms_before: Milliseconds before peak for waveform window
        ms_after: Milliseconds after peak for waveform window
        channels_per_spike: Number of channels per spike to include in features
        max_spikes_per_unit: Cap spikes per unit for extraction
        max_spikes_global: Global cap on total spikes to process (None for no limit)
        scale_features: Whether to z-score features per dimension
        n_jobs: Parallel jobs for extraction
        verbose: Log progress if True
        apply_ibl_pipeline: Apply IBL preprocessing pipeline
        synthetic_pipeline: Use synthetic data pipeline (bandpass only)
        correct_motion: Apply motion correction (requires apply_ibl_pipeline=True)
        motion_preset: Motion correction preset ("dredge", "kilosort_like", etc.)
        cache_dir: Directory to save intermediate results
        save_motion: Save motion correction data to disk
        save_rec: Save final preprocessed recording to disk
        cleanup_temp: Remove temporary waveform extraction folders
        job_kwargs: Additional arguments for parallel processing
        return_recording: If True, return PreprocessingResult with recording
        chunk_duration: Chunk size for processing

    Returns:
        If return_recording=False (default):
            Tuple of (X, peak_locations, metadata)
        If return_recording=True:
            PreprocessingResult object containing all data and recording
        
        - X: (n_spikes, n_features) feature array
        - peak_locations: (n_spikes, 2) array of (sample_index, channel_index)
        - metadata: Dictionary with extraction parameters and statistics
        
    Raises:
        ValueError: If parameters are invalid or processing fails
    """
    if recording is None:
        raise ValueError("Recording cannot be None")
    
    processed_rec = recording
    motion = None
    
    # Apply IBL preprocessing pipeline if requested
    if apply_ibl_pipeline:
        processed_rec, motion = apply_ibl_preprocessing(
            recording=recording,
            synthetic=synthetic_pipeline,
            correction_motion=correct_motion,
            motion_preset=motion_preset,
            cache_dir=Path(cache_dir) if cache_dir else None,
            save_motion=save_motion,
            save_rec=save_rec,
            job_kwargs=job_kwargs,
        )
    
    # Detect spikes
    if verbose:
        print("[Preprocessing] Starting spike detection...")
    
    peaks = detect_spikes(
        processed_rec,
        method=method,
        peak_sign=peak_sign,
        detect_threshold=detect_threshold,
        exclude_sweep_ms=exclude_sweep_ms,
        n_jobs=n_jobs,
        progress_bar=verbose,
        chunk_duration=chunk_duration,
    )
    
    # Extract features
    if verbose:
        print("[Preprocessing] Extracting waveform features...")
    
    X, peak_locations, meta = extract_waveform_features(
        processed_rec,
        peaks,
        ms_before=ms_before,
        ms_after=ms_after,
        channels_per_spike=channels_per_spike,
        max_spikes_per_unit=max_spikes_per_unit,
        max_spikes_global=max_spikes_global,
        scale_features=scale_features,
        verbose=verbose,
        n_jobs=n_jobs,
        chunk_duration=chunk_duration,
        cleanup_temp=cleanup_temp,
    )
    
    # Add pipeline metadata
    meta['motion_correction'] = bool(apply_ibl_pipeline and correct_motion)
    meta['ibl_pipeline'] = bool(apply_ibl_pipeline)
    meta['synthetic_pipeline'] = bool(synthetic_pipeline)
    meta['motion_preset'] = motion_preset if (apply_ibl_pipeline and correct_motion) else None
    
    if verbose:
        print(f"[Preprocessing] Complete! {meta['n_spikes']} spikes with {meta['n_features']} features")
    
    # Return based on format requested
    if return_recording:
        return PreprocessingResult(
            features=X,
            peak_locations=peak_locations,
            metadata=meta,
            recording=processed_rec,
            motion=motion,
        )
    else:
        return X, peak_locations, meta


# Convenience function for backward compatibility
def get_spike_features(
    recording,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Alias for preprocess_recording for backward compatibility."""
    return preprocess_recording(recording, **kwargs)

"""Spectrogram generation utilities.

This module provides functions to load audio files and generate spectrograms
with configurable parameters (sample rate, FFT size, frequency scale, etc.).
"""

from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import torch
import torchaudio


def load_audio(
    audio_path: Path,
    target_sr: Optional[int] = 48000,
    mono: bool = True,
    backend: str = "librosa",
) -> Tuple[np.ndarray, int]:
    """Load audio file and optionally resample to target sample rate.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate in Hz, or None to keep original sample rate
        mono: Convert to mono if True
        backend: Audio loading backend ('librosa' or 'torchaudio')

    Returns:
        Tuple of (audio_data, sample_rate)
        - audio_data: 1D numpy array of audio samples (float32)
        - sample_rate: Sample rate in Hz (original if target_sr=None)

    Raises:
        FileNotFoundError: If audio file doesn't exist
        RuntimeError: If audio loading fails
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        if backend == "librosa":
            audio, sr = librosa.load(str(audio_path), sr=target_sr, mono=mono)
            return audio.astype(np.float32), sr

        elif backend == "torchaudio":
            waveform, sr = torchaudio.load(str(audio_path))

            # Convert to mono
            if mono and waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample if needed
            if target_sr is not None and sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)
                final_sr = target_sr
            else:
                final_sr = sr

            # Convert to numpy
            audio = waveform.numpy().squeeze()
            return audio.astype(np.float32), final_sr

        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'librosa' or 'torchaudio'")

    except Exception as e:
        raise RuntimeError(f"Failed to load audio from {audio_path}: {e}")


def generate_spectrogram(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    window: str = "hann",
    frequency_scale: str = "linear",
    n_mels: Optional[int] = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """Generate spectrogram from audio signal.

    Args:
        audio: Audio signal (1D numpy array)
        sr: Sample rate in Hz
        n_fft: FFT size (window size)
        hop_length: Hop length in samples (stride between windows)
        window: Window function ('hann', 'hamming', etc.)
        frequency_scale: 'linear' or 'mel'
        n_mels: Number of mel bins (only used if frequency_scale='mel')
        fmin: Minimum frequency for mel scale (Hz)
        fmax: Maximum frequency for mel scale (Hz), None = sr/2 (Nyquist)

    Returns:
        Spectrogram array with shape (freq_bins, time_frames)
        - For linear: freq_bins = n_fft // 2 + 1
        - For mel: freq_bins = n_mels

    Note:
        Returns magnitude spectrogram (NOT power). Use to_db() to convert to dB scale.
    """
    if frequency_scale == "linear":
        # Linear frequency STFT
        S = librosa.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            center=True,
            pad_mode="reflect",
        )
        # Return magnitude
        return np.abs(S)

    elif frequency_scale == "mel":
        # Mel-scale spectrogram
        if fmax is None:
            fmax = sr / 2

        S = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            n_mels=n_mels or 128,
            fmin=fmin,
            fmax=fmax,
            center=True,
            pad_mode="reflect",
        )
        # Return magnitude (melspectrogram returns power by default, take sqrt)
        return np.sqrt(S)

    else:
        raise ValueError(f"Unknown frequency_scale: {frequency_scale}. Use 'linear' or 'mel'")


def to_db(
    S: np.ndarray,
    ref: str = "max",
    ref_value: Optional[float] = None,
    top_db: float = 80.0,
    amin: float = 1e-10,
) -> np.ndarray:
    """Convert magnitude spectrogram to dB scale.

    Args:
        S: Magnitude spectrogram (freq_bins, time_frames)
        ref: Reference for dB calculation:
             - 'max': use maximum value in S
             - 'value': use ref_value
        ref_value: Reference value (only used if ref='value')
        top_db: Threshold the output at top_db below the peak
        amin: Minimum value for numerical stability

    Returns:
        dB-scale spectrogram with same shape as input

    Note:
        Output range is typically [-top_db, 0] when ref='max'
    """
    S_db = librosa.amplitude_to_db(S, ref=np.max if ref == "max" else ref_value, top_db=top_db, amin=amin)
    return S_db


def compute_spectrogram_shape(
    audio_duration: float,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    frequency_scale: str = "linear",
    n_mels: Optional[int] = 128,
) -> Tuple[int, int]:
    """Compute expected spectrogram shape without generating it.

    Args:
        audio_duration: Audio duration in seconds
        sr: Sample rate in Hz
        n_fft: FFT size
        hop_length: Hop length in samples
        frequency_scale: 'linear' or 'mel'
        n_mels: Number of mel bins (only used if frequency_scale='mel')

    Returns:
        Tuple of (freq_bins, time_frames)
    """
    audio_samples = int(audio_duration * sr)

    # Time frames calculation (matches librosa.stft with center=True)
    time_frames = 1 + audio_samples // hop_length

    # Frequency bins
    if frequency_scale == "linear":
        freq_bins = n_fft // 2 + 1
    else:  # mel
        freq_bins = n_mels or 128

    return (freq_bins, time_frames)


def load_and_generate_spectrogram(
    audio_path: Path,
    target_sr: Optional[int] = 48000,
    n_fft: int = 1024,
    hop_length: int = 256,
    window: str = "hann",
    frequency_scale: str = "linear",
    n_mels: Optional[int] = 128,
    to_db_scale: bool = True,
    top_db: float = 80.0,
) -> Tuple[np.ndarray, int]:
    """Load audio and generate spectrogram in one step.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate in Hz, or None to keep original
        n_fft: FFT size
        hop_length: Hop length in samples
        window: Window function
        frequency_scale: 'linear' or 'mel'
        n_mels: Number of mel bins (only used if frequency_scale='mel')
        to_db_scale: Convert to dB scale if True
        top_db: dB threshold (only used if to_db_scale=True)

    Returns:
        Tuple of (spectrogram, sample_rate)
        - spectrogram: Shape (freq_bins, time_frames)
        - sample_rate: Sample rate used (original if target_sr=None)

    Example:
        >>> S_db, sr = load_and_generate_spectrogram(
        ...     "audio.wav",
        ...     target_sr=48000,
        ...     n_fft=1024,
        ...     hop_length=256,
        ...     frequency_scale="linear",
        ...     to_db_scale=True
        ... )
    """
    # Load audio
    audio, sr = load_audio(audio_path, target_sr=target_sr, mono=True)

    # Generate spectrogram
    S = generate_spectrogram(
        audio,
        sr,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        frequency_scale=frequency_scale,
        n_mels=n_mels,
    )

    # Convert to dB if requested
    if to_db_scale:
        S = to_db(S, ref="max", top_db=top_db)

    return S, sr

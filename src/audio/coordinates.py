"""Coordinate transformation utilities for spectrogram bounding boxes.

This module handles the critical transformation from time-frequency annotations
to pixel coordinates to YOLO normalized format.

IMPORTANT: This is the highest-risk component. All coordinate transformations
must be validated visually before training.
"""

from typing import Optional, Tuple

import librosa
import numpy as np


def time_to_pixel(
    time_abs: float,
    segment_start: float,
    hop_length: int,
    sr: int,
    spectrogram_width: int,
) -> int:
    """Convert absolute time to pixel x-coordinate.

    Args:
        time_abs: Absolute annotation time in seconds (from original recording)
        segment_start: Segment start time in seconds (audio_segment_initial_time from CSV)
        hop_length: STFT hop length in samples
        sr: Sample rate in Hz (after resampling)
        spectrogram_width: Width of spectrogram in time frames

    Returns:
        Pixel x-coordinate (0-based, clamped to valid range)

    Note:
        Annotations use absolute time from original recording, but spectrograms
        are generated from 4-second segments. Must convert to relative time first.

    Example:
        >>> time_to_pixel(1.5, 0.0, 256, 48000, 751)  # 1.5s into 4s segment
        281  # Approximately at 1.5s / 4s * 751 frames
    """
    # Convert absolute time to relative time within segment
    time_relative = time_abs - segment_start

    # Clamp to segment bounds (typically [0, 4] seconds)
    time_relative = max(0.0, time_relative)

    # Convert time to frame index
    # librosa.stft with center=True adds padding, so frame calculation is:
    # frame = time_in_samples / hop_length
    frame_index = time_relative * sr / hop_length

    # Clamp to spectrogram width
    pixel_x = int(round(frame_index))
    pixel_x = max(0, min(pixel_x, spectrogram_width - 1))

    return pixel_x


def freq_to_pixel_linear(
    freq_hz: float,
    n_fft: int,
    sr: int,
    spectrogram_height: int,
) -> int:
    """Convert frequency (Hz) to pixel y-coordinate for linear frequency scale.

    Args:
        freq_hz: Frequency in Hz
        n_fft: FFT size
        sr: Sample rate in Hz
        spectrogram_height: Height of spectrogram in frequency bins

    Returns:
        Pixel y-coordinate (0-based, clamped to valid range)

    Note:
        NO Y-AXIS INVERSION here because we flip the spectrogram in visualization/export.
        - Original spectrogram: row 0 = 0 Hz, row N = Nyquist
        - After flip: row 0 = Nyquist (top), row N = 0 Hz (bottom)
        - Direct mapping: freq_bin → pixel_y

    Example:
        >>> freq_to_pixel_linear(1000.0, 1024, 48000, 513)
        # freq_bin = 1000 * 1024 / 48000 ≈ 21
        # pixel_y = 21 (low freq near top after flip will be at bottom)
    """
    # Convert Hz to frequency bin
    # Nyquist = sr / 2
    # freq_bin ranges from 0 to n_fft//2 (which is spectrogram_height - 1)
    freq_bin = freq_hz * n_fft / sr

    # Clamp to valid range
    freq_bin = max(0, min(freq_bin, spectrogram_height - 1))

    # NO INVERSION - freq_bin maps directly to pixel_y
    # The flipping happens in visualization/export with np.flipud()
    pixel_y = int(round(freq_bin))

    # Ensure in valid range
    pixel_y = max(0, min(pixel_y, spectrogram_height - 1))

    return pixel_y


def freq_to_pixel_mel(
    freq_hz: float,
    n_mels: int,
    sr: int,
    spectrogram_height: int,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> int:
    """Convert frequency (Hz) to pixel y-coordinate for mel frequency scale.

    Args:
        freq_hz: Frequency in Hz
        n_mels: Number of mel bins
        sr: Sample rate in Hz
        spectrogram_height: Height of spectrogram in mel bins (should equal n_mels)
        fmin: Minimum frequency for mel scale (Hz)
        fmax: Maximum frequency for mel scale (Hz), None = sr/2

    Returns:
        Pixel y-coordinate (0-based, clamped to valid range)

    Note:
        Mel scale is nonlinear. Need to find nearest mel bin.
        NO Y-AXIS INVERSION - we flip the spectrogram in visualization/export.
    """
    if fmax is None:
        fmax = sr / 2

    # Convert Hz to mel
    mel_value = librosa.hz_to_mel(freq_hz)

    # Get mel bin centers
    mel_bins = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)

    # Find nearest mel bin
    mel_bin_index = np.argmin(np.abs(mel_bins - freq_hz))

    # Clamp to valid range
    mel_bin_index = max(0, min(mel_bin_index, n_mels - 1))

    # NO INVERSION - mel_bin_index maps directly to pixel_y
    pixel_y = mel_bin_index

    # Ensure in valid range
    pixel_y = max(0, min(pixel_y, spectrogram_height - 1))

    return pixel_y


def bbox_to_yolo(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    img_width: int,
    img_height: int,
) -> Optional[Tuple[float, float, float, float]]:
    """Convert pixel bounding box to YOLO normalized format.

    Args:
        x1: Left x-coordinate (pixel)
        y1: Top y-coordinate (pixel)
        x2: Right x-coordinate (pixel)
        y2: Bottom y-coordinate (pixel)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Tuple of (x_center, y_center, width, height) all normalized to [0, 1]
        Returns None if box is degenerate (zero width or height)

    YOLO Format:
        <x_center> <y_center> <width> <height>
        All values normalized by image dimensions to [0, 1]

    Note:
        Clamps coordinates to image bounds before conversion.
    """
    print(f"Bbox: ({x1}, {y1}, {x2}, {y2})")
    print(f"Image size: ({img_width}, {img_height})")

    # Clamp to image bounds
    x1 = max(0, min(x1, img_width - 1))
    x2 = max(0, min(x2, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    y2 = max(0, min(y2, img_height - 1))

    print(f"Clamped bbox: ({x1}, {y1}, {x2}, {y2})")

    # Ensure x1 < x2 and y1 < y2
    if x1 >= x2 or y1 >= y2:
        print("Degenerate box 1st order detected.")
        return None  # Degenerate box

    # Calculate center and dimensions
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1

    # Normalize to [0, 1]
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    # Validate (should always be true after clamping, but check anyway)
    if not (0 <= x_center_norm <= 1 and 0 <= y_center_norm <= 1):
        print("Degenerate box 2nd order detected.")
        return None
    if not (0 < width_norm <= 1 and 0 < height_norm <= 1):
        print("Degenerate box 3rd order detected.")
        return None

    return (x_center_norm, y_center_norm, width_norm, height_norm)


def annotation_to_yolo(
    annotation_time_start: float,
    annotation_time_end: float,
    annotation_freq_min: float,
    annotation_freq_max: float,
    segment_start: float,
    hop_length: int,
    n_fft: int,
    sr: int,
    spectrogram_shape: Tuple[int, int],
    img_size: Tuple[int, int],
    frequency_scale: str = "linear",
    n_mels: Optional[int] = None,
) -> Optional[Tuple[float, float, float, float]]:
    """Convert time-frequency annotation to YOLO format (end-to-end).

    Args:
        annotation_time_start: Annotation start time (seconds, absolute)
        annotation_time_end: Annotation end time (seconds, absolute)
        annotation_freq_min: Minimum frequency (Hz)
        annotation_freq_max: Maximum frequency (Hz)
        segment_start: Segment start time (seconds, typically 0 for 4s segments)
        hop_length: STFT hop length in samples
        n_fft: FFT size
        sr: Sample rate in Hz
        spectrogram_shape: Spectrogram shape as (freq_bins, time_frames)
        img_size: Image size as (width, height) in pixels
        frequency_scale: 'linear' or 'mel'
        n_mels: Number of mel bins (required if frequency_scale='mel')

    Returns:
        YOLO format tuple (x_center, y_center, width, height) normalized to [0, 1]
        Returns None if box is invalid or degenerate

    Example:
        >>> yolo_box = annotation_to_yolo(
        ...     annotation_time_start=1.0,
        ...     annotation_time_end=2.0,
        ...     annotation_freq_min=1000.0,
        ...     annotation_freq_max=3000.0,
        ...     segment_start=0.0,
        ...     hop_length=256,
        ...     n_fft=1024,
        ...     sr=48000,
        ...     spectrogram_shape=(513, 751),
        ...     img_size=(640, 480),
        ...     frequency_scale='linear'
        ... )
        >>> yolo_box  # (0.374, 0.833, 0.248, 0.167) or similar
    """
    freq_bins, time_frames = spectrogram_shape
    img_width, img_height = img_size

    # Convert time to pixel x-coordinates
    x1 = time_to_pixel(annotation_time_start, segment_start, hop_length, sr, time_frames)
    x2 = time_to_pixel(annotation_time_end, segment_start, hop_length, sr, time_frames)

    # Convert frequency to pixel y-coordinates
    if frequency_scale == "linear":
        # Note: freq_min → higher y_pixel (bottom), freq_max → lower y_pixel (top)
        y_top = freq_to_pixel_linear(annotation_freq_max, n_fft, sr, freq_bins)
        y_bottom = freq_to_pixel_linear(annotation_freq_min, n_fft, sr, freq_bins)
    elif frequency_scale == "mel":
        if n_mels is None:
            n_mels = freq_bins  # Assume spectrogram height matches n_mels
        y_top = freq_to_pixel_mel(annotation_freq_max, n_mels, sr, freq_bins)
        y_bottom = freq_to_pixel_mel(annotation_freq_min, n_mels, sr, freq_bins)
    else:
        raise ValueError(f"Unknown frequency_scale: {frequency_scale}")

    # Scale from spectrogram coordinates to image coordinates
    x1_img = int(round(x1 * img_width / time_frames))
    x2_img = int(round(x2 * img_width / time_frames))
    y1_img = int(round(y_bottom * img_height / freq_bins))
    y2_img = int(round(y_top * img_height / freq_bins))

    # Convert to YOLO format
    yolo_box = bbox_to_yolo(x1_img, y1_img, x2_img, y2_img, img_width, img_height)

    return yolo_box


def yolo_to_pixel(
    yolo_box: Tuple[float, float, float, float],
    img_width: int,
    img_height: int,
) -> Tuple[int, int, int, int]:
    """Convert YOLO normalized format back to pixel coordinates.

    Args:
        yolo_box: Tuple of (x_center, y_center, width, height) normalized to [0, 1]
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Tuple of (x1, y1, x2, y2) in pixel coordinates

    Note:
        Useful for visualization and debugging.
    """
    x_center_norm, y_center_norm, width_norm, height_norm = yolo_box

    # Denormalize
    x_center = x_center_norm * img_width
    y_center = y_center_norm * img_height
    width = width_norm * img_width
    height = height_norm * img_height

    # Calculate corners
    x1 = int(round(x_center - width / 2))
    y1 = int(round(y_center - height / 2))
    x2 = int(round(x_center + width / 2))
    y2 = int(round(y_center + height / 2))

    # Clamp to image bounds
    x1 = max(0, min(x1, img_width - 1))
    x2 = max(0, min(x2, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    y2 = max(0, min(y2, img_height - 1))

    return (x1, y1, x2, y2)

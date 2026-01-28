"""Image export utilities for spectrograms.

This module provides functions to convert spectrogram arrays to PNG image files
for use with YOLO object detection.
"""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def normalize_spectrogram(
    S_db: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Normalize spectrogram to [0, 1] range.

    Args:
        S_db: dB-scale spectrogram (freq_bins, time_frames)
        vmin: Minimum value for normalization (default: min of S_db)
        vmax: Maximum value for normalization (default: max of S_db)

    Returns:
        Normalized spectrogram in [0, 1] range
    """
    if vmin is None:
        vmin = S_db.min()
    if vmax is None:
        vmax = S_db.max()

    # Avoid division by zero
    if vmax - vmin < 1e-10:
        return np.zeros_like(S_db)

    S_norm = (S_db - vmin) / (vmax - vmin)
    S_norm = np.clip(S_norm, 0, 1)

    return S_norm


def apply_colormap(
    S_norm: np.ndarray,
    cmap: str = "viridis",
) -> np.ndarray:
    """Apply matplotlib colormap to normalized spectrogram.

    Args:
        S_norm: Normalized spectrogram in [0, 1] range (freq_bins, time_frames)
        cmap: Matplotlib colormap name ('viridis', 'gray', 'hot', 'plasma', etc.)

    Returns:
        RGB image array (height, width, 3) with values in [0, 255] uint8

    Note:
        Output has shape (freq_bins, time_frames, 3) for RGB
    """
    # Get colormap
    cm = plt.get_cmap(cmap)

    # Apply colormap (returns RGBA, shape: freq_bins x time_frames x 4)
    S_rgba = cm(S_norm)

    # Convert to RGB (drop alpha channel) and scale to [0, 255]
    S_rgb = (S_rgba[:, :, :3] * 255).astype(np.uint8)

    return S_rgb


def spectrogram_to_image(
    S_db: np.ndarray,
    output_path: Path,
    target_size: Tuple[int, int] = (640, 480),
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Convert spectrogram to PNG image file.

    Args:
        S_db: dB-scale spectrogram (freq_bins, time_frames)
        output_path: Path to output PNG file
        target_size: Target image size as (width, height) in pixels
        cmap: Matplotlib colormap name
        vmin: Minimum value for normalization (default: min of S_db)
        vmax: Maximum value for normalization (default: max of S_db)

    Note:
        - Spectrogram is oriented with frequency on Y-axis (origin at bottom)
        - Image format has origin at top-left
        - Function handles Y-axis flip automatically
    """
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize to [0, 1]
    S_norm = normalize_spectrogram(S_db, vmin, vmax)

    # Apply colormap
    S_rgb = apply_colormap(S_norm, cmap)

    # Flip vertically (spectrogram has freq=0 at bottom, image has y=0 at top)
    S_rgb_flipped = np.flipud(S_rgb)

    # Convert to PIL Image
    image = Image.fromarray(S_rgb_flipped, mode="RGB")

    # Resize to target size
    # PIL Image.resize expects (width, height)
    image_resized = image.resize(target_size, resample=Image.BILINEAR)

    # Save as PNG
    image_resized.save(output_path, format="PNG")


def spectrogram_to_image_fast(
    S_db: np.ndarray,
    output_path: Path,
    target_size: Tuple[int, int] = (640, 480),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Convert spectrogram to grayscale PNG (faster than colormap version).

    Args:
        S_db: dB-scale spectrogram (freq_bins, time_frames)
        output_path: Path to output PNG file
        target_size: Target image size as (width, height) in pixels
        vmin: Minimum value for normalization
        vmax: Maximum value for normalization

    Note:
        Faster than spectrogram_to_image() but produces grayscale images.
        Use for rapid prototyping or when color is not needed.
    """
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize to [0, 255]
    if vmin is None:
        vmin = S_db.min()
    if vmax is None:
        vmax = S_db.max()

    if vmax - vmin < 1e-10:
        S_uint8 = np.zeros_like(S_db, dtype=np.uint8)
    else:
        S_norm = (S_db - vmin) / (vmax - vmin)
        S_uint8 = (np.clip(S_norm, 0, 1) * 255).astype(np.uint8)

    # Flip vertically
    S_uint8_flipped = np.flipud(S_uint8)

    # Convert to PIL Image
    image = Image.fromarray(S_uint8_flipped, mode="L")  # "L" = grayscale

    # Resize to target size
    image_resized = image.resize(target_size, resample=Image.BILINEAR)

    # Save as PNG
    image_resized.save(output_path, format="PNG")


def get_image_dimensions(S: np.ndarray, target_size: Tuple[int, int]) -> Tuple[int, int]:
    """Get final image dimensions after resizing.

    Args:
        S: Spectrogram array (freq_bins, time_frames)
        target_size: Target size as (width, height)

    Returns:
        Tuple of (width, height) in pixels

    Note:
        This is a simple pass-through since we always resize to target_size.
        Useful for coordinate transformation calculations.
    """
    return target_size


def compute_spectrogram_to_image_scale(
    S_shape: Tuple[int, int],
    target_size: Tuple[int, int],
) -> Tuple[float, float]:
    """Compute scaling factors from spectrogram to image coordinates.

    Args:
        S_shape: Spectrogram shape as (freq_bins, time_frames)
        target_size: Image size as (width, height)

    Returns:
        Tuple of (x_scale, y_scale) factors

    Example:
        >>> S_shape = (513, 751)  # freq_bins x time_frames
        >>> target_size = (640, 480)  # width x height
        >>> x_scale, y_scale = compute_spectrogram_to_image_scale(S_shape, target_size)
        >>> x_scale  # 640 / 751 ≈ 0.852
        >>> y_scale  # 480 / 513 ≈ 0.936
    """
    freq_bins, time_frames = S_shape
    width, height = target_size

    x_scale = width / time_frames
    y_scale = height / freq_bins

    return (x_scale, y_scale)

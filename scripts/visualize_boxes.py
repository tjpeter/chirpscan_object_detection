#!/usr/bin/env python3
"""Visualize bounding boxes overlaid on spectrograms.

This script is CRITICAL for validating coordinate transformations before training.
Use it to manually inspect that bounding boxes align with audio events.

Usage:
    python scripts/visualize_boxes.py \\
        --csv data/ECOSoundSet/annotated_audio_segments.csv \\
        --audio-dir data/ECOSoundSet/Split recordings \\
        --output-dir validation \\
        --num-samples 20 \\
        --config configs/spectrogram_config.yaml
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.audio.coordinates import annotation_to_yolo, yolo_to_pixel
from src.audio.spectrogram import load_and_generate_spectrogram


def load_config(config_path: Path) -> Dict:
    """Load spectrogram configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def visualize_sample(
    audio_path: Path,
    annotations: List[Tuple],
    segment_start: float,
    output_path: Path,
    config: Dict,
    use_original_sr: bool = True,
) -> None:
    """Generate spectrogram with bounding boxes overlaid.

    Args:
        audio_path: Path to audio file
        annotations: List of (time_start, time_end, freq_min, freq_max, label) tuples
        segment_start: Segment start time in seconds (from CSV)
        output_path: Path to save visualization
        config: Spectrogram configuration dict
        use_original_sr: If True, use original sample rate (no resampling) for debugging
    """
    # Get original sample rate first
    import librosa
    original_sr = librosa.get_samplerate(str(audio_path))
    
    # Load audio - use ORIGINAL sample rate for validation
    if use_original_sr:
        target_sr = None  # None = keep original sample rate
    else:
        target_sr = config["audio"]["target_sample_rate"]

    S_db, sr = load_and_generate_spectrogram(
        audio_path,
        target_sr=target_sr,
        n_fft=config["spectrogram"]["n_fft"],
        hop_length=config["spectrogram"]["hop_length"],
        window=config["spectrogram"]["window"],
        frequency_scale=config["spectrogram"]["frequency_scale"],
        n_mels=config["spectrogram"].get("n_mels", 128),
        to_db_scale=True,
        top_db=config["amplitude"]["top_db"],
    )

    print(f"  Sample rate: {sr} Hz (original: {original_sr} Hz)")

    freq_bins, time_frames = S_db.shape
    img_width = config["image"]["width"]
    img_height = config["image"]["height"]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Display spectrogram
    # CRITICAL: Flip vertically to match PNG export (y=0 at top)
    # This makes visualization consistent with final YOLO training images
    #S_db_flipped = np.flipud(S_db)

    # Use white-to-color colormap for better low-dB visibility
    # 'gray_r' = reversed grayscale (white=low, black=high)
    # Or use custom colormap
    cmap = config["image"].get("cmap", "gray_r")

    im = ax.imshow(
        S_db,
        aspect="auto",
        origin="lower",  # Standard image coordinates: y=0 at top
        cmap=cmap,
        extent=[0, img_width, 0, img_height],
    )

    # Add colorbar
    plt.colorbar(im, ax=ax, label="dB")

    # Overlay bounding boxes
    box_info_lines = []
    for time_start, time_end, freq_min, freq_max, label in annotations:
        print(f"time_start: {time_start}, time_end: {time_end}, freq_min: {freq_min}, freq_max: {freq_max}, label: {label}")
        print(f"segment_start: {segment_start}")
        # Convert annotation to YOLO format
        yolo_box = annotation_to_yolo(
            annotation_time_start=time_start,
            annotation_time_end=time_end,
            annotation_freq_min=freq_min,
            annotation_freq_max=freq_max,
            segment_start=segment_start,  # Use actual segment start from CSV
            hop_length=config["spectrogram"]["hop_length"],
            n_fft=config["spectrogram"]["n_fft"],
            sr=sr,
            spectrogram_shape=(freq_bins, time_frames),
            img_size=(img_width, img_height),
            frequency_scale=config["spectrogram"]["frequency_scale"],
            n_mels=config["spectrogram"].get("n_mels", 128),
        )

        if yolo_box is None:
            print(f"    ❌ Skipped: annotation_to_yolo returned None")
            continue  # Skip degenerate boxes

        # Store box info for display
        box_info_lines.append(
            f"{label}: t=[{time_start:.2f}, {time_end:.2f}]s, f=[{freq_min:.0f}, {freq_max:.0f}]Hz"
        )

        # Convert back to pixel coordinates for visualization
        x1, y1, x2, y2 = yolo_to_pixel(yolo_box, img_width, img_height)

        # Draw rectangle
        # Note: Rectangle uses bottom-left corner, so y1 is already correct
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

        # Add label
        ax.text(
            x1,
            y2 + 10,
            label,
            color="red",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    # Calculate time and frequency ranges for axis labels
    duration = time_frames * config["spectrogram"]["hop_length"] / sr
    max_freq = sr / 2  # Nyquist frequency
    
    # Set axis labels with actual units
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")
    
    # Update tick labels to show actual values
    # Time axis: 0.5 second intervals starting from segment_start
    time_step = 0.5  # seconds
    num_time_ticks = int(duration / time_step) + 1
    time_positions = [i * time_step * (img_width / duration) for i in range(num_time_ticks)]
    time_labels = [f"{segment_start + i * time_step:.1f}" for i in range(num_time_ticks)]
    ax.set_xticks(time_positions)
    ax.set_xticklabels(time_labels)
    
    # Frequency axis: 5000 Hz intervals with thousand separator
    # With origin="lower", y=0 is at bottom, so frequency increases upward
    
    if config["spectrogram"]["frequency_scale"] == "mel":
        # For mel scale, frequencies are non-linearly spaced
        # We need to map linear Hz values to mel bin positions
        import librosa
        
        # Create mel filterbank to get frequency boundaries
        n_mels = config["spectrogram"].get("n_mels", 128)
        mel_freqs = librosa.mel_frequencies(n_mels=n_mels + 2, fmin=0.0, fmax=max_freq)
        
        # Find mel bin positions for each frequency tick
        # Use 1000 Hz steps up to 10000 Hz, then 5000 Hz steps
        freq_positions = []
        freq_labels = []
        target_freqs = []
        
        # 0 to 10000 Hz: 1000 Hz steps
        for freq in range(0, min(11000, int(max_freq) + 1), 1000):
            target_freqs.append(freq)
        
        # Above 10000 Hz: 5000 Hz steps
        if max_freq > 10000:
            for freq in range(15000, int(max_freq) + 1, 5000):
                target_freqs.append(freq)
        
        for target_freq in target_freqs:
            if target_freq <= max_freq:
                # Find which mel bin this frequency corresponds to
                mel_bin = np.searchsorted(mel_freqs, target_freq)
                # Convert mel bin to pixel position
                y_pos = (mel_bin / n_mels) * img_height
                freq_positions.append(y_pos)
                freq_labels.append(f"{target_freq:,}")
    else:
        # Linear scale: straightforward mapping with 5000 Hz steps
        freq_step = 5000
        num_freq_ticks = int(max_freq / freq_step) + 1
        freq_positions = [(i * freq_step / max_freq) * img_height for i in range(num_freq_ticks)]
        freq_labels = [f"{i * freq_step:,}" for i in range(num_freq_ticks)]
    
    ax.set_yticks(freq_positions)
    ax.set_yticklabels(freq_labels)
    
    
    if use_original_sr:
        sr_label = f"{sr}Hz (original)"
    else:
        sr_label = f"{sr}Hz (resampled from {original_sr}Hz)"
    ax.set_title(f"Spectrogram: {audio_path.name} | SR: {sr_label}")

    # Add bounding box info text below the plot
    if box_info_lines:
        box_info_text = "Bounding Boxes:\n" + "\n".join(box_info_lines)
        plt.figtext(0.5, 0.01, box_info_text, ha="center", fontsize=8, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8),
                   wrap=True)


    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for box info text
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize bounding boxes on spectrograms for validation"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/ECOSoundSet/annotated_audio_segments.csv"),
        help="Path to annotations CSV file",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("data/ECOSoundSet/Split recordings"),
        help="Base directory containing audio files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation"),
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/spectrogram_config.yaml"),
        help="Path to spectrogram configuration file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection",
    )
    parser.add_argument(
        "--use-original-sr",
        action="store_true",
        default=True,
        help="Use original sample rate (no resampling) for validation",
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        help="Enable resampling to target_sample_rate (overrides --use-original-sr)",
    )

    args = parser.parse_args()

    # Determine whether to use original SR or resample
    use_original_sr = args.use_original_sr and not args.resample

    # Load configuration
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        print("Please create configs/spectrogram_config.yaml first")
        return

    config = load_config(args.config)

    # Load annotations CSV
    print(f"Loading annotations from {args.csv}...")
    df = pd.read_csv(args.csv)

    # Group by audio file
    grouped = df.groupby("audio_segment_file_name")

    # Filter for files with substantial annotations (more than 1 annotation)
    # This ensures we're visualizing files that are actually annotated
    well_annotated_files = []
    for file_name, group in grouped:
        # Only include files with at least 2 annotations
        # (filters out sparsely annotated or unannotated files)
        if len(group) >= 2:
            well_annotated_files.append(file_name)

    print(f"Found {len(well_annotated_files)} well-annotated files "
          f"(out of {len(grouped)} total)")

    # Sample random files from well-annotated set
    random.seed(args.seed)
    sampled_files = random.sample(
        well_annotated_files,
        min(args.num_samples, len(well_annotated_files))
    )

    print(f"\nVisualization mode: {'ORIGINAL sample rate (no resampling)' if use_original_sr else 'RESAMPLED to target SR'}")
    print(f"Visualizing {len(sampled_files)} samples...\n")

    # Track successfully visualized files
    visualized_files = []

    # Process each sampled file
    for file_name in sampled_files:
        group = grouped.get_group(file_name)

        # Find audio file
        audio_path = None
        for license_dir in ["CC BY 4.0", "CC BY-NC 4.0", "CC BY-NC-ND 4.0"]:
            candidate = args.audio_dir / license_dir / file_name
            if candidate.exists():
                audio_path = candidate
                break

        if audio_path is None:
            print(f"Warning: Audio file not found: {file_name}")
            continue

        # Get segment start time (same for all annotations in group)
        segment_start = group.iloc[0]["audio_segment_initial_time"]

        # Extract annotations
        annotations = []
        for _, row in group.iterrows():
            annotations.append(
                (
                    row["annotation_initial_time"],
                    row["annotation_final_time"],
                    row["annotation_min_freq"],
                    row["annotation_max_freq"],
                    row["label"],
                )
            )

        # Generate visualization
        output_path = args.output_dir / f"{file_name.replace('.wav', '.png')}"
        try:
            visualize_sample(
                audio_path,
                annotations,
                segment_start,
                output_path,
                config,
                use_original_sr=use_original_sr,
            )
            visualized_files.append(file_name)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Write comma-separated file names to text file
    if visualized_files:
        file_list_path = args.output_dir / "visualized_files.txt"
        
        # Natural sort to match filesystem order
        import re
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() 
                    for text in re.split(r'(\d+)', s)]
        
        with open(file_list_path, "w") as f:
            f.write(", ".join(sorted(visualized_files, key=natural_sort_key)))
        print(f"\n📝 Saved file list: {file_list_path}")

    print(f"\n✅ Visualization complete! Check {args.output_dir}/ for results")
    print("\nMANUAL VALIDATION CHECKLIST:")
    print("  [ ] Do bounding boxes align with visible sound events?")
    print("  [ ] Are boxes positioned at correct time ranges?")
    print("  [ ] Are boxes positioned at correct frequency ranges?")
    print("  [ ] Do labels match the visible sound patterns?")
    print("\nIf boxes are misaligned, DEBUG src/audio/coordinates.py before proceeding!")


if __name__ == "__main__":
    main()

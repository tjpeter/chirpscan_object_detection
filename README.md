# ChirpScan Object Detection

Production-quality YOLO-based object detection on spectrograms for bioacoustic analysis using the ECOSoundSet dataset.

## Project Status

🚧 **In Development** - Core infrastructure complete (70%), preprocessing and training pipelines in progress (30%).

### Completed ✅
- Data loading and indexing (CSV parsing, multi-label support)
- Class mapping (deterministic 386-class → integer ID mapping)
- Spectrogram generation (linear & mel frequency scales)
- Coordinate transformations (time/freq → pixel → YOLO format)
- Image export (spectrogram → PNG)
- YOLO format conversion (label files, data.yaml generation)
- Visual validation tool (`scripts/visualize_boxes.py`)

### In Progress 🔧
- Main preprocessing script (`prepare_data.py`)
- Configuration files
- Training script
- Evaluation pipeline
- Test suite
- Documentation

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Create spectrogram config
cat > configs/spectrogram_config.yaml << 'EOF'
audio:
  target_sample_rate: 48000
  mono: true
spectrogram:
  n_fft: 1024
  hop_length: 256
  window: hann
  frequency_scale: linear
  n_mels: 128
amplitude:
  scale: db
  ref: max
  top_db: 80
image:
  height: 480
  width: 640
  cmap: gray_r  # Reversed grayscale: white=low dB, black=high dB
EOF

# 3. Validate coordinate transformations (CRITICAL - TWO-STEP PROCESS!)

# STEP 1: Validate at ORIGINAL sample rate (no resampling)
python scripts/visualize_boxes.py --num-samples 20
# Check validation/*.png - boxes should align with audio events

# STEP 2: If Step 1 looks good, validate WITH resampling
python scripts/visualize_boxes.py --num-samples 20 --resample
# Check validation/*.png - boxes should still align
```

## Dataset

**ECOSoundSet**: 117,781 annotations across 41,910 audio segments
- **Classes**: 386 unique labels (insects, birds, environmental sounds)
- **Format**: 4-second WAV segments with time-frequency bounding boxes
- **Splits**: Pre-defined train (75,903) / val (24,459) / test (17,418)
- **Size**: ~3.5 GB (processed spectrograms)

## Installation

### Prerequisites
- Python >= 3.11
- CUDA-capable GPU (recommended for training)
- ~10 GB disk space (dataset + processed images)

### Setup

```bash
# Create conda environment
mamba create -n chirpscan python=3.11
mamba activate chirpscan

# Install
pip install -e ".[dev]"
```

## Usage

See **Quick Start** above for immediate validation. Full workflow documentation pending.

## Architecture

```
src/
├── data/          # ✅ Data loading, class mapping, YOLO format
├── audio/         # ✅ Spectrogram generation, coordinates, export
├── evaluation/    # 🔧 Metrics, error analysis (TODO)
└── utils/         # 🔧 Reproducibility (TODO)

scripts/
├── visualize_boxes.py      # ✅ Visual validation
├── prepare_data.py         # 🔧 Main preprocessing (TODO)
├── train.py                # 🔧 Training (TODO)
├── evaluate.py             # 🔧 Evaluation (TODO)
└── generate_report.py      # 🔧 Reporting (TODO)
```

## Known Limitations

1. **Sample Rate**: 48 kHz → 24 kHz Nyquist limit (ultrasonic content >24 kHz lost)
2. **Class Imbalance**: 386 classes, extreme imbalance (some <10 examples)
3. **Multi-Label**: Up to 10+ overlapping boxes per spectrogram
4. **Coordinate Precision**: ~5ms time, ~47Hz frequency resolution at 48kHz

## Development Status

**Phase 1 (Foundation)**: ✅ Complete
- Repository structure
- Data layer modules
- YOLO format conversion

**Phase 2 (Signal Processing)**: ✅ Complete
- Audio loading & spectrogram generation
- Coordinate transformations (time/freq → pixel → YOLO)
- Visual validation tool

**Phase 3 (Preprocessing)**: 🔧 In Progress (30%)
- Main preprocessing script
- Configuration management

**Phase 4 (Training)**: ⏳ Pending
- Training script
- Reproducibility utilities
- Config files

**Phase 5 (Evaluation & Quality)**: ⏳ Pending
- Evaluation scripts
- Test suite (target: >90% coverage)
- CI/CD pipeline
- Documentation

## Contributing

This is a research project under active development. Core infrastructure is stable.

## License

MIT License

---

**Last Updated**: 2026-01-28
**Next Milestone**: Complete `prepare_data.py` and configuration files

# UAMSNet - Medical Image Segmentation Framework

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/pytorch-latest-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A modern, modular framework for abdominal multi-organ segmentation from CT images, built with Protocol-based architecture and configuration-driven design.

## 🎯 Overview

UAMSNet is designed for accurate automatic organ segmentation in abdominal CT scans. The framework supports multiple state-of-the-art segmentation models and provides a comprehensive training pipeline with modern deep learning best practices.

**Key Features:**
- 🔧 **Modular Architecture**: Protocol-based design with clear separation of concerns
- 📦 **Multiple Models**: UAMSNet, UNet, AttentionUNet, DSCNet
- ⚙️ **Configuration-Driven**: TOML-based experiment management
- 📊 **Rich Metrics**: Dice coefficient, per-class metrics, and more
- 🚀 **Modern Tooling**: UV package manager, Python 3.11, type hints
- 📈 **Training Pipeline**: Early stopping, checkpointing, TensorBoard visualization

## 🏗️ Project Structure

```
UAMSNet/
├── src/
│   ├── framework/          # Core training framework
│   │   ├── contracts/      # Protocol interfaces
│   │   ├── core/           # Trainer, Evaluator, Checkpoint
│   │   ├── registry/       # Component registration
│   │   ├── metrics/        # Evaluation metrics
│   │   └── cli/            # Command-line interface
│   │
│   ├── models/             # Segmentation models
│   │   ├── uamsnet.py      # Multi-scale Boundary Attention
│   │   ├── unet.py         # Standard U-Net
│   │   ├── attention_unet.py
│   │   ├── dscnet.py       # Dynamic Selection Convolution
│   │   └── parts/          # Model components
│   │
│   ├── data_adapters/      # Dataset handling
│   │   └── word_adapter.py
│   │
│   └── tasks/              # Task definitions
│       └── segmentation.py
│
├── configs/                # Configuration files
│   ├── experiment.word.toml
│   └── model.*.toml
│
├── scripts/                # Utility scripts
│   └── preprocess.py
│
├── data/                   # Data directory
├���─ runs/                   # Training outputs
├── main.py                 # Entry point
├── pyproject.toml          # Dependencies
└── .python-version         # Python 3.11
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/suizhaoqinggong/UAMSNet.git
cd UAMSNet

# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### Prepare Data

Process your CT data using the preprocessing script:

```bash
python scripts/preprocess.py --data_dir /path/to/WORD --output_dir data/processed/word
```

This will generate manifest files:
- `data/processed/word/train.csv`
- `data/processed/word/val.csv`
- `data/processed/word/test.csv`

### Training

Train a segmentation model using the CLI:

```bash
# Train UAMSNet
uv run uamsnet train \
  --configs configs/experiment.word.toml \
  --model configs/model.uamsnet.toml

# Train UNet
uv run uamsnet train \
  --configs configs/experiment.word.toml \
  --model configs/model.unet.toml
```

### Evaluation

Evaluate a trained model:

```bash
uv run uamsnet test \
  --configs configs/experiment.word.toml \
  --model configs/model.uamsnet.toml \
  --checkpoint runs/experiment_name/checkpoints/best.ckpt
```

## 📋 Available Models

### 1. UAMSNet (Recommended)
Multi-scale Boundary Attention Network with enhanced organ boundary detection.

```toml
[model]
name = "uamsnet"
[model.params]
n_channels = 3
n_classes = 17
bilinear = true
```

### 2. UNet
Standard U-Net architecture for medical image segmentation.

```toml
[model]
name = "unet"
[model.params]
n_channels = 3
n_classes = 17
bilinear = true
```

### 3. AttentionUNet
U-Net with attention gates for focusing on relevant features.

```toml
[model]
name = "attention_unet"
[model.params]
n_channels = 1
n_classes = 17
```

### 4. DSCNet
Dynamic Selection Convolution Network for adaptive feature extraction.

```toml
[model]
name = "dscnet"
[model.params]
n_channels = 3
n_classes = 17
kernel_size = 9
```

## ⚙️ Configuration

### Experiment Configuration

Edit `configs/experiment.word.toml`:

```toml
[experiment]
name = "word-organ-segmentation"
seed = 42
class_names = ["background", "liver", "spleen", ...]

[data]
adapter = "word"
processed_root = "data/processed/word"
batch_size = 4
num_workers = 4

[task]
name = "segmentation"
num_classes = 17
include_background = false
dice_weight = 0.5
bce_weight = 0.5

[optimizer]
name = "adam"
lr = 0.00001
weight_decay = 1e-8

[trainer]
epochs = 100
patience = 20
device = "auto"
amp = true  # Mixed precision training

[checkpoint]
monitor = "avg_dice"
mode = "max"
```

## 📊 Metrics

The framework provides comprehensive segmentation metrics:

- **DiceMetric**: Average Dice coefficient across all organs
- **PerClassDiceMetric**: Individual Dice score for each organ
- **Accuracy**: Pixel-wise accuracy
- **AUC**: Area under ROC curve

Metrics are automatically logged to TensorBoard and can be monitored in real-time:

```bash
tensorboard --logdir runs/
```

## 🔬 Advanced Features

### Mixed Precision Training

Enable automatic mixed precision for faster training:

```toml
[trainer]
amp = true
```

### Early Stopping

Prevent overfitting with patience-based early stopping:

```toml
[trainer]
patience = 20  # Stop if no improvement for 20 epochs
```

### Gradient Clipping

Stabilize training with gradient clipping:

```toml
[trainer]
grad_clip_norm = 1.0
```

### Reproducibility

Set random seeds for reproducible experiments:

```toml
[experiment]
seed = 42
```

## 📈 Training Pipeline

The framework provides a complete training pipeline:

1. **Data Loading**: Efficient multi-worker data loading with custom collation
2. **Training Loop**: Automatic batch processing with progress tracking
3. **Validation**: Periodic evaluation on validation set
4. **Checkpointing**: Save best models based on monitored metric
5. **Early Stopping**: Automatic training termination when convergence is reached
6. **Logging**: Structured logging with TensorBoard integration
7. **Metrics**: Real-time metric computation and visualization

## 🛠️ Development

### Project Structure

The framework follows Protocol-based design patterns:

- **Contracts**: Define interfaces for components (Model, Task, DataAdapter, Metric)
- **Registry**: Centralized component registration and factory pattern
- **Core**: Training, evaluation, and checkpoint management
- **CLI**: Command-line interface for easy interaction

### Adding New Models

1. Create your model in `src/models/`:

```python
from framework.contracts import Batch, ModelOutput

class MyModel(nn.Module):
    def __init__(self, n_channels: int, n_classes: int):
        super().__init__()
        # Define your architecture

    def forward(self, batch: Batch) -> ModelOutput:
        x = batch["image"]
        # Forward pass
        return logits
```

2. Register in `src/models/catalog.py`:

```python
MODEL_REGISTRY_TABLE = {
    "my_model": MyModel,
    # ...
}
```

3. Create config `configs/model.my_model.toml`

### Adding New Metrics

1. Create metric in `src/framework/metrics/`:

```python
from framework.contracts import Metric, MetricResults

@dataclass
class MyMetric(Metric):
    name: str = "my_metric"
    higher_is_better: bool = True

    def update(self, preds, targets): ...
    def compute(self) -> MetricResults: ...
    def reset(self): ...
```

2. Register in `src/framework/registry/defaults.py`

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{uamsnet2024,
  title={UAMSNet: A Modular Framework for Abdominal Multi-organ Segmentation},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/suizhaoqinggong/UAMSNet}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Framework architecture inspired by [ECG-Research](https://github.com/anthropics/ecg-research)
- WORD dataset for abdominal organ segmentation
- PyTorch team for the excellent deep learning framework

## 📧 Contact

For questions and support, please open an issue on GitHub or contact [suizhaoqinggong@gmail.com](mailto:suizhaoqinggong@gmail.com).

---

**Built with ❤️ for medical image analysis**
# Code Preprocessor

A machine learning pipeline for preprocessing and training models on source code, specifically optimized for Rust codebases.

## Requirements

- Python 3.10 or higher
- CUDA-capable GPU (tested with RTX 4070)
- WSL2 or Linux environment

## Installation

1. Clone the repository:
```bash
git clone https://github.com/argakiig/code-preprocessor.git
cd code-preprocessor
```

2. Install the package and dependencies:
```bash
pip install -e .
```

For development, install with additional tools:
```bash
pip install -e ".[dev]"
```

## Usage

### Basic Usage

Process and train on a Rust codebase:

```bash
code-preprocess --config config.yaml --code-path /path/to/rust/code
```

### Configuration

Create a `config.yaml` file with your settings:

```yaml
# Model settings
model_name: "codellama/CodeLlama-7b-hf"
max_sequence_length: 2048
vocab_size: 32000

# Training settings
batch_size: 1
num_workers: 1
eval_split: 0.1
seed: 42

# Paths
output_dir: "./output"
cache_dir: "./cache"
log_file: "./training.log"

# Logging
log_level: "INFO"
wandb_project: "your-project-name"
```

### Development Tools

The project includes several development tools:

- `black`: Code formatting
- `isort`: Import sorting
- `flake8`: Code linting
- `mypy`: Type checking
- `pytest`: Testing
- `pre-commit`: Git hooks

Set up pre-commit hooks:
```bash
pre-commit install
```

Run tests:
```bash
pytest
```

## Features

- Processes Rust source code for training
- Custom tokenizer trained on Rust code
- Memory-efficient training with LoRA and 4-bit quantization
- Integrated with Weights & Biases for experiment tracking
- Automatic evaluation during training
- Best model checkpoint saving

## Project Structure

```
code-preprocessor/
├── code_preprocessor/
│   ├── __init__.py
│   ├── __main__.py
│   ├── config.py
│   ├── processor.py
│   ├── tokenizers/
│   ├── training/
│   └── utils/
├── tests/
├── .pre-commit-config.yaml
├── pyproject.toml
└── README.md
```

## License

MIT License

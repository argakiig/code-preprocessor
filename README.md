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
# Model configuration
model:
  name: "codellama/CodeLlama-7b-hf"
  vocab_size: 50000
  max_sequence_length: 256

# Training configuration
training:
  batch_size: 2
  num_workers: 8
  epochs: 3
  gradient_accumulation_steps: 1
  eval_split: 0.1
  seed: 42

# Paths
paths:
  output_dir: "./output"
  cache_dir: "~/.cache/huggingface"

# Logging
logging:
  level: "INFO"
  file: "logs/training.log"

# Weights & Biases
wandb:
  project: "your-project-name"

# GPU Configuration
gpu:
  device: "cuda"
  precision: "fp16"
  memory_efficient: true
```

#### Configuration Precedence

The tool supports both YAML configuration files and command-line arguments. When both are provided:

1. Command-line arguments take precedence over the config file settings
2. Any settings not specified in command-line arguments will fall back to the config file values
3. If a setting is not specified in either place, default values will be used

For example:
```bash
# This will use batch_size=4 from CLI, but keep other settings from config.yaml
code-preprocess --config config.yaml --code-path /path/to/code --batch-size 4
```

Available CLI arguments match the configuration file options and can be viewed with:
```bash
code-preprocess --help
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
│   ├── constants.py
│   ├── models/
│   ├── parsers/
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

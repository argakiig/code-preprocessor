"""Constants used throughout the code preprocessor."""

# Model defaults
DEFAULT_VOCAB_SIZE = 50000
DEFAULT_SEQUENCE_LENGTH = 256
DEFAULT_BATCH_SIZE = 2
DEFAULT_NUM_WORKERS = 8
DEFAULT_EPOCHS = 3
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_EVAL_SPLIT = 0.1

# File paths and extensions
RUST_FILE_EXTENSION = ".rs"
DEFAULT_MODEL_NAME = "codellama/CodeLlama-7b-hf"
DEFAULT_OUTPUT_DIR = "~/datasets/rust-model"

# Training
DEFAULT_SEED = 42
DEFAULT_TEST_SAMPLES = 100
DEFAULT_WANDB_PROJECT = "CodeLlama-7b-hf-new"

# Logging
DEFAULT_LOG_LEVEL = "INFO"

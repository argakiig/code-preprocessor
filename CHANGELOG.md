# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed

## [v0.1.0] - 2025-01-02

### Added
- Initial project setup with Python 3.10+ support
- Docker-based CI/CD pipeline with GitHub Actions
- Code quality tools integration:
  - Black for code formatting
  - isort for import sorting
  - flake8 for linting
  - mypy for type checking
  - pytest for testing
  - pre-commit hooks
- Basic project structure with modular design
- Configuration management using YAML
- GPU support with memory efficiency features
- Integration with Weights & Biases for experiment tracking
- Comprehensive test suite with coverage reporting
- Documentation:
  - README with installation and usage instructions
  - Configuration examples
  - Project structure documentation
- Security improvements:
  - Pinned third-party GitHub Actions to commit SHAs
  - Minimal container permissions in CI/CD
  - Dependency management with version constraints

### Changed
- Switched from CodeLlama to DeepSeek Coder model
- Updated CI/CD workflow for better security and efficiency
- Improved test coverage and type annotations

### Fixed
- CI/CD pipeline issues:
  - Build dependencies installation
  - Codecov integration
  - Docker image caching
- Type checking errors in test files

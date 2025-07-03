# Multi-Stream Neural Networks Test Suite

This directory contains comprehensive tests for the Multi-Stream Neural Networks (MSNN) project.

## Test Categories

### API and Model Tests

- **`test_api_methods.py`** - Comprehensive test of all model API methods
  - Tests fit(), evaluate(), predict(), predict_proba() for all models
  - Verifies input shape handling (2D/4D)
  - Validates DataLoader support
  - Ensures consistent API across model types

### CIFAR-100 Integration Tests

- **`test_cifar100_loader.py`** - Tests the CIFAR-100 data loader module
  - Verifies data loading functionality
  - Tests import capabilities
  - Validates data structure and formats

- **`test_cifar100_rgbl_integration.py`** - Integration tests for CIFAR-100 + RGBtoRGBL
  - Tests CIFAR-100 loader with RGBtoRGBL processor
  - Verifies single image and batch processing
  - Performance benchmarking
  - Visual verification

- **`demo_notebook_workflow.py`** - Demonstrates notebook workflow
  - Shows exact usage patterns for the Colab notebook
  - Verifies backward compatibility
  - Demonstrates data processing pipeline

## Running Tests

```bash
# Run individual tests
python tests/test_api_methods.py
python tests/test_cifar100_loader.py
python tests/test_cifar100_rgbl_integration.py
python tests/demo_notebook_workflow.py

# Run all tests (from project root)
python -m pytest tests/ -v
```

## Test Coverage

- ✅ Model API consistency and robustness
- ✅ Input shape handling for different model architectures
- ✅ DataLoader support for all methods
- ✅ CIFAR-100 data loading
- ✅ RGBtoRGBL transformation
- ✅ Integration between components
- ✅ Batch processing
- ✅ Performance validation
- ✅ Visual verification
- ✅ Notebook workflow compatibility

All tests pass and verify the complete MSNN functionality including models, data loading, and processing pipeline.

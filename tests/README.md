# Tests Directory

This directory contains test scripts for the Multi-Stream Neural Networks project.

## Test Files

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
python tests/test_cifar100_loader.py
python tests/test_cifar100_rgbl_integration.py
python tests/demo_notebook_workflow.py

# Run all tests (from project root)
python -m pytest tests/ -v
```

## Test Coverage

- ✅ CIFAR-100 data loading
- ✅ RGBtoRGBL transformation
- ✅ Integration between components
- ✅ Batch processing
- ✅ Performance validation
- ✅ Visual verification
- ✅ Notebook workflow compatibility

All tests pass and verify the complete data loading and processing pipeline.

# Models2 Test Suite

This directory contains unit tests for the `models2` package, following a mirrored directory structure to match the implementation files.

## Directory Structure

The test directory structure mirrors the implementation structure in `src/models2/`:

```
tests/src/models2/
├── __init__.py
├── abstracts/          # Tests for abstract base classes
├── base/               # Tests for base model implementations
├── core/               # Tests for core building blocks (conv, resnet, etc.)
│   ├── test_blocks.py
│   ├── test_conv.py
│   ├── test_resnet.py
│   └── test_resnet_training.py   # Tests for ResNet training methods (compile, fit, predict, evaluate)
└── multi_channel/      # Tests for multi-channel model implementations
```

## Testing Philosophy

1. **Mirrored Structure**: Each implementation file should have a corresponding test file (e.g., `src/models2/core/resnet.py` → `tests/src/models2/core/test_resnet.py`).

2. **Unit Tests**: Each class and function should have appropriate unit tests that verify:
   - Initialization with various parameters
   - Forward pass with expected shapes
   - Specific behaviors or edge cases

3. **Naming Convention**: Test files are prefixed with `test_` followed by the name of the module being tested.

## Running Tests

To run all tests in this directory:

```bash
python -m unittest discover tests/src/models2
```

To run tests for a specific module:

```bash
python -m unittest tests/src/models2/core/test_resnet.py
```

## Test Implementation Guidelines

When implementing tests, consider the following guidelines:

1. **Setup/Teardown**: Use `setUp` and `tearDown` methods for common initialization and cleanup.
2. **Skip Incomplete**: Use `@unittest.skip` for tests of not-yet-implemented features.
3. **Test Edge Cases**: Include tests for boundary conditions and error cases.
4. **Randomness**: Set random seeds for reproducibility when testing with random inputs.
5. **Documentation**: Document the purpose of each test class and test method.

## Dependencies

Tests may require additional dependencies beyond the main project dependencies. These should be documented in a separate test requirements file if necessary.

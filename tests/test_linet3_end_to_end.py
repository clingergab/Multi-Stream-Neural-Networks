"""
COMPREHENSIVE END-TO-END TEST: LINet3 N-Stream Model

Adapted from the original LINet (2-stream) tests to validate LINet3 N-stream refactoring.
Tests the complete workflow with the same data and test scenarios as the old model.

Tests:
1. Model creation with N streams
2. Parameter group creation (N-stream API)
3. Optimizer creation (stream-specific LRs for N streams)
4. Scheduler creation and stepping
5. Compilation with various configurations
6. Training workflow with stream monitoring
7. Stream early stopping
8. Evaluation and prediction
9. All convenience functions
10. Backward compatibility (2-stream with RGB + Depth)
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import new LINet3 model
from src.models.linear_integration.li_net3.li_net import LINet
from src.models.linear_integration.li_net3.blocks import LIBasicBlock, LIBottleneck

# Import training utilities
from src.training.schedulers import setup_scheduler

print("=" * 100)
print("COMPREHENSIVE END-TO-END TEST: LINet3 N-Stream Model")
print("=" * 100)

# ============================================================================
# Test Suite Configuration
# ============================================================================
DEVICE = 'cpu'
NUM_CLASSES = 10
IMAGE_SIZE = 32
BATCH_SIZE = 4
NUM_TRAIN_SAMPLES = 20
NUM_VAL_SAMPLES = 12

test_results = []

def log_test(test_name, passed, details=""):
    """Log test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    test_results.append((test_name, passed))
    print(f"\n{status}: {test_name}")
    if details:
        print(f"  → {details}")

# ============================================================================
# Helper Classes and Functions
# ============================================================================

class NStreamDataset(Dataset):
    """Dataset that returns N-stream data in the format LINet3 expects."""

    def __init__(self, stream_data_list, labels):
        """
        Args:
            stream_data_list: List of tensors, one per stream [num_samples, channels, H, W]
            labels: Target labels [num_samples]
        """
        self.stream_data = stream_data_list
        self.labels = labels
        self.num_samples = labels.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return dict with 'streams' and 'labels' keys
        return {
            'streams': [stream_data[idx] for stream_data in self.stream_data],
            'labels': self.labels[idx]
        }


def create_dummy_2stream_data(num_samples=20):
    """Create 2-stream dummy data (RGB + Depth) - same as old tests."""
    rgb = torch.randn(num_samples, 3, IMAGE_SIZE, IMAGE_SIZE)
    depth = torch.randn(num_samples, 1, IMAGE_SIZE, IMAGE_SIZE)
    labels = torch.randint(0, NUM_CLASSES, (num_samples,))
    return [rgb, depth], labels


def create_dummy_3stream_data(num_samples=20):
    """Create 3-stream dummy data (RGB + Depth + HHA)."""
    rgb = torch.randn(num_samples, 3, IMAGE_SIZE, IMAGE_SIZE)
    depth = torch.randn(num_samples, 1, IMAGE_SIZE, IMAGE_SIZE)
    hha = torch.randn(num_samples, 3, IMAGE_SIZE, IMAGE_SIZE)
    labels = torch.randint(0, NUM_CLASSES, (num_samples,))
    return [rgb, depth, hha], labels


def create_2stream_loaders():
    """Create 2-stream data loaders (RGB + Depth)."""
    train_streams, train_labels = create_dummy_2stream_data(NUM_TRAIN_SAMPLES)
    val_streams, val_labels = create_dummy_2stream_data(NUM_VAL_SAMPLES)

    train_dataset = NStreamDataset(train_streams, train_labels)
    val_dataset = NStreamDataset(val_streams, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


def create_3stream_loaders():
    """Create 3-stream data loaders (RGB + Depth + HHA)."""
    train_streams, train_labels = create_dummy_3stream_data(NUM_TRAIN_SAMPLES)
    val_streams, val_labels = create_dummy_3stream_data(NUM_VAL_SAMPLES)

    train_dataset = NStreamDataset(train_streams, train_labels)
    val_dataset = NStreamDataset(val_streams, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


# ============================================================================
# TEST SECTION 1: Model Creation (2-stream backward compatibility)
# ============================================================================
print("\n" + "=" * 100)
print("TEST SECTION 1: Model Creation (2-stream backward compatibility)")
print("=" * 100)

# Test 1.1: LINet3 with 2 streams (RGB + Depth) - backward compatible
try:
    model_2stream = LINet(
        block=LIBasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=NUM_CLASSES,
        stream_input_channels=[3, 1],  # RGB + Depth (same as old LINet)
        zero_init_residual=False,
        device=DEVICE,
        dropout_p=0.3
    )
    assert model_2stream.num_streams == 2, f"Expected 2 streams, got {model_2stream.num_streams}"
    assert model_2stream.stream_input_channels == [3, 1], "Stream channels mismatch"
    log_test("LINet3 2-stream creation (RGB + Depth)", True,
             f"Created successfully with {model_2stream.num_streams} streams")
except Exception as e:
    log_test("LINet3 2-stream creation (RGB + Depth)", False, str(e))
    raise

# Test 1.2: LINet3 with 3 streams (RGB + Depth + HHA)
try:
    model_3stream = LINet(
        block=LIBasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=NUM_CLASSES,
        stream_input_channels=[3, 1, 3],  # RGB + Depth + HHA
        zero_init_residual=False,
        device=DEVICE
    )
    assert model_3stream.num_streams == 3, f"Expected 3 streams, got {model_3stream.num_streams}"
    log_test("LINet3 3-stream creation (RGB + Depth + HHA)", True,
             f"Created successfully with {model_3stream.num_streams} streams")
except Exception as e:
    log_test("LINet3 3-stream creation (RGB + Depth + HHA)", False, str(e))
    raise

# Test 1.3: Default 2-stream creation
try:
    model_default = LINet(
        block=LIBasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=NUM_CLASSES,
        # No stream_input_channels - should default to [3, 1]
        device=DEVICE
    )
    assert model_default.num_streams == 2, "Default should create 2 streams"
    assert model_default.stream_input_channels == [3, 1], "Default should be [3, 1]"
    log_test("Default 2-stream creation", True, "Defaults to [3, 1] as expected")
except Exception as e:
    log_test("Default 2-stream creation", False, str(e))
    raise

# ============================================================================
# TEST SECTION 2: Parameter Groups (N-stream API)
# ============================================================================
print("\n" + "=" * 100)
print("TEST SECTION 2: Parameter Groups (N-stream API)")
print("=" * 100)

# Test 2.1: get_stream_parameter_groups() with list (2-stream)
try:
    param_groups_2s = model_2stream.get_stream_parameter_groups(
        stream_lrs=[2e-4, 7e-4],  # List for 2 streams
        stream_weight_decays=[1e-4, 2e-4],
        shared_lr=5e-4,
        shared_weight_decay=1.5e-4
    )
    assert len(param_groups_2s) == 3, f"Expected 3 groups (stream0, stream1, shared), got {len(param_groups_2s)}"
    assert param_groups_2s[0]['lr'] == 2e-4, "Stream0 LR mismatch"
    assert param_groups_2s[1]['lr'] == 7e-4, "Stream1 LR mismatch"
    assert param_groups_2s[2]['lr'] == 5e-4, "Shared LR mismatch"
    log_test("get_stream_parameter_groups() with list (2-stream)", True,
             f"{len(param_groups_2s)} groups with correct LRs")
except Exception as e:
    log_test("get_stream_parameter_groups() with list (2-stream)", False, str(e))
    raise

# Test 2.2: get_stream_parameter_groups() with list (3-stream)
try:
    param_groups_3s = model_3stream.get_stream_parameter_groups(
        stream_lrs=[2e-4, 7e-4, 5e-4],
        stream_weight_decays=[1e-4, 2e-4, 1.5e-4],
        shared_lr=4e-4
    )
    assert len(param_groups_3s) == 4, f"Expected 4 groups (3 streams + shared), got {len(param_groups_3s)}"
    assert param_groups_3s[0]['lr'] == 2e-4, "Stream_0 LR mismatch"
    assert param_groups_3s[1]['lr'] == 7e-4, "Stream_1 LR mismatch"
    assert param_groups_3s[2]['lr'] == 5e-4, "Stream_2 LR mismatch"
    assert param_groups_3s[3]['lr'] == 4e-4, "Shared LR mismatch"
    log_test("get_stream_parameter_groups() with list (3-stream)", True,
             f"{len(param_groups_3s)} groups created correctly")
except Exception as e:
    log_test("get_stream_parameter_groups() with list (3-stream)", False, str(e))
    raise

# Test 2.3: get_stream_parameter_groups() with single float (same LR for all)
try:
    param_groups_single = model_3stream.get_stream_parameter_groups(
        stream_lrs=1e-3,  # Single float for all streams
        shared_lr=5e-4
    )
    assert len(param_groups_single) == 4, "Expected 4 groups"
    # All stream groups should have same LR
    assert all(pg['lr'] == 1e-3 for pg in param_groups_single[:3]), "All streams should have 1e-3"
    assert param_groups_single[3]['lr'] == 5e-4, "Shared LR mismatch"
    log_test("get_stream_parameter_groups() with single float", True,
             "All streams got same LR (1e-3)")
except Exception as e:
    log_test("get_stream_parameter_groups() with single float", False, str(e))
    raise

# Test 2.4: Verify all parameters accounted for
try:
    total_params = sum(len(pg['params']) for pg in param_groups_2s)
    model_params = len(list(model_2stream.parameters()))
    assert total_params == model_params, f"Param count mismatch: {total_params} vs {model_params}"
    log_test("Parameter group completeness", True,
             f"All {model_params} parameters accounted for")
except Exception as e:
    log_test("Parameter group completeness", False, str(e))
    raise

# ============================================================================
# TEST SECTION 3: Optimizer Creation
# ============================================================================
print("\n" + "=" * 100)
print("TEST SECTION 3: Optimizer Creation")
print("=" * 100)

# Test 3.1: Simple optimizer (single LR)
try:
    simple_opt = torch.optim.AdamW(model_2stream.parameters(), lr=1e-3, weight_decay=1e-4)
    assert len(simple_opt.param_groups) == 1, "Simple optimizer should have 1 param group"
    assert simple_opt.param_groups[0]['lr'] == 1e-3, "LR mismatch"
    log_test("Simple optimizer creation", True, "Single LR optimizer created")
except Exception as e:
    log_test("Simple optimizer creation", False, str(e))
    raise

# Test 3.2: Stream-specific optimizer (2-stream)
try:
    stream_param_groups = model_2stream.get_stream_parameter_groups(
        stream_lrs=[2e-4, 7e-4],
        shared_lr=5e-4
    )
    stream_opt = torch.optim.AdamW(stream_param_groups)
    assert len(stream_opt.param_groups) == 3, "Should have 3 param groups"
    log_test("Stream-specific optimizer (2-stream)", True,
             f"{len(stream_opt.param_groups)} parameter groups")
except Exception as e:
    log_test("Stream-specific optimizer (2-stream)", False, str(e))
    raise

# Test 3.3: Stream-specific optimizer (3-stream)
try:
    stream_param_groups_3s = model_3stream.get_stream_parameter_groups(
        stream_lrs=[1e-4, 3e-4, 2e-4],
        shared_lr=2.5e-4
    )
    stream_opt_3s = torch.optim.AdamW(stream_param_groups_3s)
    assert len(stream_opt_3s.param_groups) == 4, "Should have 4 param groups"
    log_test("Stream-specific optimizer (3-stream)", True,
             f"{len(stream_opt_3s.param_groups)} parameter groups")
except Exception as e:
    log_test("Stream-specific optimizer (3-stream)", False, str(e))
    raise

# ============================================================================
# TEST SECTION 4: Scheduler Creation
# ============================================================================
print("\n" + "=" * 100)
print("TEST SECTION 4: Scheduler Creation")
print("=" * 100)

# Test 4.1: Decaying cosine scheduler
try:
    test_opt = torch.optim.AdamW(model_2stream.parameters(), lr=1e-3)
    scheduler_dc = setup_scheduler(
        test_opt, 'decaying_cosine',
        epochs=10, train_loader_len=5,
        t_max=5, eta_min=1e-5
    )
    assert scheduler_dc is not None, "Scheduler not created"
    log_test("Decaying cosine scheduler creation", True, "Scheduler created successfully")
except Exception as e:
    log_test("Decaying cosine scheduler creation", False, str(e))
    raise

# Test 4.2: Cosine annealing scheduler
try:
    test_opt_2 = torch.optim.AdamW(model_3stream.parameters(), lr=1e-3)
    scheduler_cos = setup_scheduler(
        test_opt_2, 'cosine',
        epochs=10, train_loader_len=5,
        t_max=10
    )
    assert scheduler_cos is not None, "Scheduler not created"
    log_test("Cosine annealing scheduler creation", True, "Scheduler created successfully")
except Exception as e:
    log_test("Cosine annealing scheduler creation", False, str(e))
    raise

# ============================================================================
# TEST SECTION 5: Model Compilation
# ============================================================================
print("\n" + "=" * 100)
print("TEST SECTION 5: Model Compilation")
print("=" * 100)

# Test 5.1: Compile with optimizer and scheduler
try:
    compile_model = LINet(
        block=LIBasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=NUM_CLASSES,
        stream_input_channels=[3, 1],
        device=DEVICE
    )
    compile_opt = torch.optim.AdamW(compile_model.parameters(), lr=1e-3)
    compile_sched = setup_scheduler(compile_opt, 'decaying_cosine', epochs=10,
                                     train_loader_len=5, t_max=5, eta_min=6e-5)
    compile_model.compile(optimizer=compile_opt, scheduler=compile_sched, loss='cross_entropy')

    assert compile_model.is_compiled, "is_compiled flag not set"
    assert compile_model.optimizer is compile_opt, "Optimizer not stored"
    assert compile_model.scheduler is compile_sched, "Scheduler not stored"
    assert compile_model.criterion is not None, "Criterion not created"
    log_test("compile() with optimizer and scheduler", True, "All components stored correctly")
except Exception as e:
    log_test("compile() with optimizer and scheduler", False, str(e))
    raise

# Test 5.2: Compile without scheduler
try:
    compile_model_2 = LINet(
        block=LIBasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=NUM_CLASSES,
        stream_input_channels=[3, 1],
        device=DEVICE
    )
    compile_opt_2 = torch.optim.SGD(compile_model_2.parameters(), lr=0.01)
    compile_model_2.compile(optimizer=compile_opt_2, scheduler=None, loss='cross_entropy')

    assert compile_model_2.scheduler is None, "Scheduler should be None"
    assert compile_model_2.is_compiled, "is_compiled flag not set"
    log_test("compile() without scheduler", True, "Works with scheduler=None")
except Exception as e:
    log_test("compile() without scheduler", False, str(e))
    raise

# ============================================================================
# TEST SECTION 6: Training Workflow (2-stream)
# ============================================================================
print("\n" + "=" * 100)
print("TEST SECTION 6: Training Workflow (2-stream)")
print("=" * 100)

# Create 2-stream data loaders
train_loader_2s, val_loader_2s = create_2stream_loaders()

# Test 6.1: Training 2-stream model with stream monitoring
try:
    train_model_2s = LINet(
        block=LIBasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=NUM_CLASSES,
        stream_input_channels=[3, 1],  # RGB + Depth
        device=DEVICE,
        dropout_p=0.3
    )
    train_param_groups_2s = train_model_2s.get_stream_parameter_groups(
        stream_lrs=[2e-4, 7e-4],
        shared_lr=5e-4
    )
    train_opt_2s = torch.optim.AdamW(train_param_groups_2s)
    train_sched_2s = setup_scheduler(train_opt_2s, 'decaying_cosine', epochs=3,
                                      train_loader_len=len(train_loader_2s), t_max=2, eta_min=1e-5)
    train_model_2s.compile(optimizer=train_opt_2s, scheduler=train_sched_2s, loss='cross_entropy')

    history_2s = train_model_2s.fit(
        train_loader=train_loader_2s,
        val_loader=val_loader_2s,
        epochs=3,
        verbose=False,
        stream_monitoring=True
    )

    assert 'train_loss' in history_2s, "train_loss not in history"
    assert 'val_loss' in history_2s, "val_loss not in history"
    assert len(history_2s['train_loss']) == 3, f"Expected 3 epochs, got {len(history_2s['train_loss'])}"
    log_test("Training 2-stream model with stream monitoring", True,
             f"Completed 3 epochs, final train_loss={history_2s['train_loss'][-1]:.4f}")
except Exception as e:
    log_test("Training 2-stream model with stream monitoring", False, str(e))
    import traceback
    traceback.print_exc()
    raise

# Test 6.2: Training without stream monitoring
try:
    simple_train_model = LINet(
        block=LIBasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=NUM_CLASSES,
        stream_input_channels=[3, 1],
        device=DEVICE
    )
    simple_opt = torch.optim.AdamW(simple_train_model.parameters(), lr=1e-3)
    simple_train_model.compile(optimizer=simple_opt, loss='cross_entropy')

    history_simple = simple_train_model.fit(
        train_loader=train_loader_2s,
        val_loader=val_loader_2s,
        epochs=3,
        verbose=False,
        stream_monitoring=False  # Disabled
    )

    assert len(history_simple['train_loss']) == 3, "Expected 3 epochs"
    log_test("Training without stream monitoring", True, "Training works without monitoring")
except Exception as e:
    log_test("Training without stream monitoring", False, str(e))
    raise

# ============================================================================
# TEST SECTION 7: Training Workflow (3-stream)
# ============================================================================
print("\n" + "=" * 100)
print("TEST SECTION 7: Training Workflow (3-stream)")
print("=" * 100)

# Create 3-stream data loaders
train_loader_3s, val_loader_3s = create_3stream_loaders()

# Test 7.1: Training 3-stream model
try:
    train_model_3s = LINet(
        block=LIBasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=NUM_CLASSES,
        stream_input_channels=[3, 1, 3],  # RGB + Depth + HHA
        device=DEVICE
    )
    train_param_groups_3s = train_model_3s.get_stream_parameter_groups(
        stream_lrs=[1e-4, 3e-4, 2e-4],
        shared_lr=2e-4
    )
    train_opt_3s = torch.optim.AdamW(train_param_groups_3s)
    train_sched_3s = setup_scheduler(train_opt_3s, 'cosine', epochs=3,
                                      train_loader_len=len(train_loader_3s), t_max=3)
    train_model_3s.compile(optimizer=train_opt_3s, scheduler=train_sched_3s, loss='cross_entropy')

    history_3s = train_model_3s.fit(
        train_loader=train_loader_3s,
        val_loader=val_loader_3s,
        epochs=3,
        verbose=False,
        stream_monitoring=True
    )

    assert len(history_3s['train_loss']) == 3, "Expected 3 epochs"
    log_test("Training 3-stream model", True,
             f"Completed 3 epochs, final val_acc={history_3s['val_accuracy'][-1]:.4f}")
except Exception as e:
    log_test("Training 3-stream model", False, str(e))
    import traceback
    traceback.print_exc()
    raise

# ============================================================================
# TEST SECTION 8: Prediction and Evaluation
# ============================================================================
print("\n" + "=" * 100)
print("TEST SECTION 8: Prediction and Evaluation")
print("=" * 100)

# Test 8.1: predict() method
try:
    predictions = train_model_2s.predict(val_loader_2s)
    assert predictions.shape[0] == NUM_VAL_SAMPLES, f"Expected {NUM_VAL_SAMPLES} predictions, got {predictions.shape[0]}"
    log_test("predict() method", True, f"Predictions shape: {predictions.shape}")
except Exception as e:
    log_test("predict() method", False, str(e))
    raise

# Test 8.2: predict_proba() method
try:
    probabilities = train_model_2s.predict_proba(val_loader_2s)
    assert probabilities.shape == (NUM_VAL_SAMPLES, NUM_CLASSES), \
        f"Expected shape ({NUM_VAL_SAMPLES}, {NUM_CLASSES}), got {probabilities.shape}"
    log_test("predict_proba() method", True, f"Probabilities shape: {probabilities.shape}")
except Exception as e:
    log_test("predict_proba() method", False, str(e))
    raise

# Test 8.3: evaluate() method
try:
    metrics = train_model_2s.evaluate(val_loader_2s, stream_monitoring=True)
    assert 'loss' in metrics, "Metrics should contain 'loss'"
    assert 'accuracy' in metrics, "Metrics should contain 'accuracy'"
    log_test("evaluate() method", True,
             f"loss={metrics['loss']:.4f}, accuracy={metrics['accuracy']:.4f}")
except Exception as e:
    log_test("evaluate() method", False, str(e))
    raise

# ============================================================================
# TEST SECTION 9: Early Stopping
# ============================================================================
print("\n" + "=" * 100)
print("TEST SECTION 9: Early Stopping")
print("=" * 100)

# Test 9.1: Main early stopping
try:
    es_model = LINet(
        block=LIBasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=NUM_CLASSES,
        stream_input_channels=[3, 1],
        device=DEVICE
    )
    es_opt = torch.optim.AdamW(es_model.parameters(), lr=1e-3)
    es_model.compile(optimizer=es_opt, loss='cross_entropy')

    history_es = es_model.fit(
        train_loader=train_loader_2s,
        val_loader=val_loader_2s,
        epochs=20,  # High number
        verbose=False,
        early_stopping=True,
        patience=3,
        monitor='val_loss'
    )

    # Early stopping should trigger before 20 epochs
    assert len(history_es['train_loss']) < 20, \
        f"Early stopping should trigger, but ran {len(history_es['train_loss'])} epochs"
    log_test("Main early stopping", True,
             f"Stopped at epoch {len(history_es['train_loss'])}/20")
except Exception as e:
    log_test("Main early stopping", False, str(e))
    raise

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "=" * 100)
print("TEST SUMMARY")
print("=" * 100)

passed = sum(1 for _, result in test_results if result)
total = len(test_results)
pass_rate = (passed / total) * 100 if total > 0 else 0

print(f"\nTotal tests: {total}")
print(f"Passed: {passed}")
print(f"Failed: {total - passed}")
print(f"Pass rate: {pass_rate:.1f}%")

if passed == total:
    print("\n" + "=" * 100)
    print("✅ ALL TESTS PASSED - LINet3 N-stream model is fully functional!")
    print("=" * 100)
else:
    print("\n" + "=" * 100)
    print(f"❌ {total - passed} TEST(S) FAILED")
    print("=" * 100)
    failed_tests = [name for name, result in test_results if not result]
    for test_name in failed_tests:
        print(f"  - {test_name}")
    sys.exit(1)

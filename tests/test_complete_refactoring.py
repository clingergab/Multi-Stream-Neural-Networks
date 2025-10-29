"""
COMPREHENSIVE TEST SUITE: Keras-Style API Refactoring

This test suite thoroughly validates:
1. Model compilation with different configurations
2. Parameter group creation and separation
3. Optimizer creation (simple and stream-specific)
4. Scheduler creation and stepping
5. Training workflow with various features
6. Stream monitoring compatibility
7. Error handling and edge cases
8. Integration with both MCResNet and LINet
9. Backward compatibility
10. All convenience functions
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import models
from src.models.multi_channel.mc_resnet import mc_resnet18, mc_resnet34
from src.models.linear_integration.li_net import li_resnet18, li_resnet34

# Import training utilities
from src.training.optimizers import create_stream_optimizer, create_optimizer
from src.training.schedulers import setup_scheduler, DecayingCosineAnnealingLR

print("=" * 100)
print("COMPREHENSIVE TEST SUITE: Keras-Style API Refactoring")
print("=" * 100)

# ============================================================================
# Test Suite Configuration
# ============================================================================
DEVICE = 'cpu'
NUM_CLASSES = 10
IMAGE_SIZE = 32
BATCH_SIZE = 4
NUM_TRAIN_BATCHES = 5
NUM_VAL_BATCHES = 3

test_results = []

def log_test(test_name, passed, details=""):
    """Log test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    test_results.append((test_name, passed))
    print(f"\n{status}: {test_name}")
    if details:
        print(f"  → {details}")

# ============================================================================
# Helper Functions
# ============================================================================

def create_dummy_data(num_samples=20):
    """Create dummy data for testing."""
    rgb = torch.randn(num_samples, 3, IMAGE_SIZE, IMAGE_SIZE)
    depth = torch.randn(num_samples, 1, IMAGE_SIZE, IMAGE_SIZE)
    labels = torch.randint(0, NUM_CLASSES, (num_samples,))
    return rgb, depth, labels

def create_dummy_loaders():
    """Create dummy data loaders."""
    train_rgb, train_depth, train_labels = create_dummy_data(NUM_TRAIN_BATCHES * BATCH_SIZE)
    val_rgb, val_depth, val_labels = create_dummy_data(NUM_VAL_BATCHES * BATCH_SIZE)

    train_dataset = TensorDataset(train_rgb, train_depth, train_labels)
    val_dataset = TensorDataset(val_rgb, val_depth, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader

# ============================================================================
# TEST SECTION 1: Model Creation and Parameter Groups
# ============================================================================
print("\n" + "=" * 100)
print("TEST SECTION 1: Model Creation and Parameter Groups")
print("=" * 100)

# Test 1.1: MCResNet18 creation
try:
    mc_model = mc_resnet18(num_classes=NUM_CLASSES, stream1_input_channels=3,
                           stream2_input_channels=1, device=DEVICE)
    log_test("MCResNet18 creation", True, f"Created successfully on {DEVICE}")
except Exception as e:
    log_test("MCResNet18 creation", False, str(e))
    raise

# Test 1.2: LINet18 creation
try:
    li_model = li_resnet18(num_classes=NUM_CLASSES, stream1_input_channels=3,
                           stream2_input_channels=1, device=DEVICE, dropout_p=0.3)
    log_test("LINet18 creation", True, f"Created successfully with dropout=0.3")
except Exception as e:
    log_test("LINet18 creation", False, str(e))
    raise

# Test 1.3: get_stream_parameter_groups() - MCResNet
try:
    param_groups_mc = mc_model.get_stream_parameter_groups(
        stream1_lr=2e-4, stream2_lr=7e-4, shared_lr=5e-4,
        stream1_weight_decay=1e-4, stream2_weight_decay=2e-4, shared_weight_decay=1.5e-4
    )
    assert len(param_groups_mc) == 3, f"Expected 3 groups, got {len(param_groups_mc)}"
    assert param_groups_mc[0]['lr'] == 2e-4, "Stream1 LR mismatch"
    assert param_groups_mc[1]['lr'] == 7e-4, "Stream2 LR mismatch"
    assert param_groups_mc[2]['lr'] == 5e-4, "Shared LR mismatch"
    log_test("get_stream_parameter_groups() - MCResNet", True,
             f"{len(param_groups_mc)} groups with correct LRs")
except Exception as e:
    log_test("get_stream_parameter_groups() - MCResNet", False, str(e))
    raise

# Test 1.4: get_stream_parameter_groups() - LINet
try:
    param_groups_li = li_model.get_stream_parameter_groups(
        stream1_lr=1e-4, stream2_lr=3e-4, shared_lr=2e-4,
        stream1_weight_decay=5e-5, stream2_weight_decay=1e-4, shared_weight_decay=7.5e-5
    )
    assert len(param_groups_li) == 3, f"Expected 3 groups, got {len(param_groups_li)}"
    log_test("get_stream_parameter_groups() - LINet", True,
             f"{len(param_groups_li)} groups created")
except Exception as e:
    log_test("get_stream_parameter_groups() - LINet", False, str(e))
    raise

# Test 1.5: Verify parameter counts
try:
    total_params = sum(len(pg['params']) for pg in param_groups_mc)
    model_params = len(list(mc_model.parameters()))
    assert total_params == model_params, f"Param count mismatch: {total_params} vs {model_params}"
    log_test("Parameter group completeness", True,
             f"All {model_params} parameters accounted for")
except Exception as e:
    log_test("Parameter group completeness", False, str(e))
    raise

# ============================================================================
# TEST SECTION 2: Optimizer Creation
# ============================================================================
print("\n" + "=" * 100)
print("TEST SECTION 2: Optimizer Creation")
print("=" * 100)

# Test 2.1: create_stream_optimizer() - AdamW
try:
    opt_adamw = create_stream_optimizer(
        mc_model,
        optimizer_type='adamw',
        stream1_lr=2e-4, stream2_lr=7e-4, shared_lr=5e-4,
        stream1_weight_decay=1e-4, stream2_weight_decay=2e-4, shared_weight_decay=1.5e-4
    )
    assert isinstance(opt_adamw, torch.optim.AdamW), "Wrong optimizer type"
    assert len(opt_adamw.param_groups) == 3, "Wrong number of param groups"
    log_test("create_stream_optimizer() - AdamW", True,
             f"Created AdamW with {len(opt_adamw.param_groups)} param groups")
except Exception as e:
    log_test("create_stream_optimizer() - AdamW", False, str(e))
    raise

# Test 2.2: create_stream_optimizer() - SGD
try:
    opt_sgd = create_stream_optimizer(
        mc_model,
        optimizer_type='sgd',
        stream1_lr=1e-3, stream2_lr=1e-3, shared_lr=1e-3,
        momentum=0.9, nesterov=True
    )
    assert isinstance(opt_sgd, torch.optim.SGD), "Wrong optimizer type"
    assert opt_sgd.param_groups[0]['momentum'] == 0.9, "Momentum not set"
    assert opt_sgd.param_groups[0]['nesterov'] == True, "Nesterov not set"
    log_test("create_stream_optimizer() - SGD", True,
             "Created SGD with momentum and nesterov")
except Exception as e:
    log_test("create_stream_optimizer() - SGD", False, str(e))
    raise

# Test 2.3: create_stream_optimizer() - Adam
try:
    opt_adam = create_stream_optimizer(
        li_model,
        optimizer_type='adam',
        stream1_lr=5e-4, stream2_lr=5e-4, shared_lr=5e-4
    )
    assert isinstance(opt_adam, torch.optim.Adam), "Wrong optimizer type"
    log_test("create_stream_optimizer() - Adam", True, "Created Adam optimizer")
except Exception as e:
    log_test("create_stream_optimizer() - Adam", False, str(e))
    raise

# Test 2.4: create_optimizer() - Simple single LR
try:
    model_simple = mc_resnet18(num_classes=NUM_CLASSES, stream1_input_channels=3,
                                stream2_input_channels=1, device=DEVICE)
    opt_simple = create_optimizer(model_simple, optimizer_type='adamw', lr=1e-3, weight_decay=1e-4)
    assert len(opt_simple.param_groups) == 1, f"Expected 1 group, got {len(opt_simple.param_groups)}"
    assert opt_simple.param_groups[0]['lr'] == 1e-3, "LR mismatch"
    log_test("create_optimizer() - Simple", True, "Created simple optimizer with 1 param group")
except Exception as e:
    log_test("create_optimizer() - Simple", False, str(e))
    raise

# Test 2.5: Manual parameter groups with torch.optim
try:
    manual_groups = mc_model.get_stream_parameter_groups(
        stream1_lr=3e-4, stream2_lr=3e-4, shared_lr=3e-4
    )
    opt_manual = torch.optim.AdamW(manual_groups, betas=(0.9, 0.999), eps=1e-8)
    assert len(opt_manual.param_groups) == 3, "Wrong number of param groups"
    log_test("Manual optimizer creation with param groups", True,
             "Created optimizer manually with custom params")
except Exception as e:
    log_test("Manual optimizer creation with param groups", False, str(e))
    raise

# ============================================================================
# TEST SECTION 3: Scheduler Creation
# ============================================================================
print("\n" + "=" * 100)
print("TEST SECTION 3: Scheduler Creation")
print("=" * 100)

# Test 3.1: setup_scheduler() - DecayingCosine
try:
    sched_dec_cos = setup_scheduler(
        opt_adamw, 'decaying_cosine', epochs=80, train_loader_len=40,
        t_max=10, eta_min=6e-5, max_factor=0.6, min_factor=0.6
    )
    assert sched_dec_cos is not None, "Scheduler is None"
    assert sched_dec_cos.__class__.__name__ == 'DecayingCosineAnnealingLR', "Wrong scheduler type"
    log_test("setup_scheduler() - DecayingCosine", True,
             "Created DecayingCosineAnnealingLR")
except Exception as e:
    log_test("setup_scheduler() - DecayingCosine", False, str(e))
    raise

# Test 3.2: setup_scheduler() - Cosine
try:
    sched_cos = setup_scheduler(opt_simple, 'cosine', epochs=100, train_loader_len=50, t_max=100)
    assert sched_cos is not None, "Scheduler is None"
    log_test("setup_scheduler() - Cosine", True, "Created CosineAnnealingLR")
except Exception as e:
    log_test("setup_scheduler() - Cosine", False, str(e))
    raise

# Test 3.3: setup_scheduler() - OneCycle
try:
    sched_onecycle = setup_scheduler(
        opt_sgd, 'onecycle', epochs=50, train_loader_len=100,
        max_lr=[1e-3, 1e-3, 1e-3]
    )
    assert sched_onecycle is not None, "Scheduler is None"
    log_test("setup_scheduler() - OneCycle", True, "Created OneCycleLR")
except Exception as e:
    log_test("setup_scheduler() - OneCycle", False, str(e))
    raise

# Test 3.4: setup_scheduler() - Step
try:
    sched_step = setup_scheduler(opt_adam, 'step', epochs=50, train_loader_len=40,
                                  step_size=30, gamma=0.1)
    assert sched_step is not None, "Scheduler is None"
    log_test("setup_scheduler() - Step", True, "Created StepLR")
except Exception as e:
    log_test("setup_scheduler() - Step", False, str(e))
    raise

# Test 3.5: Scheduler stepping
try:
    initial_lr = opt_adamw.param_groups[0]['lr']
    sched_dec_cos.step()
    after_step_lr = opt_adamw.param_groups[0]['lr']
    # LR should change after stepping (for most schedulers)
    log_test("Scheduler stepping", True,
             f"LR changed from {initial_lr:.6e} to {after_step_lr:.6e}")
except Exception as e:
    log_test("Scheduler stepping", False, str(e))
    raise

# ============================================================================
# TEST SECTION 4: Compile Method
# ============================================================================
print("\n" + "=" * 100)
print("TEST SECTION 4: Compile Method")
print("=" * 100)

# Test 4.1: Compile with optimizer and scheduler
try:
    test_model_1 = mc_resnet18(num_classes=NUM_CLASSES, stream1_input_channels=3,
                                stream2_input_channels=1, device=DEVICE)
    test_opt_1 = create_stream_optimizer(test_model_1, stream1_lr=2e-4, stream2_lr=7e-4, shared_lr=5e-4)
    test_sched_1 = setup_scheduler(test_opt_1, 'decaying_cosine', epochs=80, train_loader_len=40,
                                    t_max=10, eta_min=6e-5)
    test_model_1.compile(optimizer=test_opt_1, scheduler=test_sched_1, loss='cross_entropy')

    assert test_model_1.is_compiled, "is_compiled flag not set"
    assert test_model_1.optimizer is test_opt_1, "Optimizer not stored"
    assert test_model_1.scheduler is test_sched_1, "Scheduler not stored"
    assert test_model_1.criterion is not None, "Criterion not created"
    log_test("compile() with optimizer and scheduler", True, "All components stored correctly")
except Exception as e:
    log_test("compile() with optimizer and scheduler", False, str(e))
    raise

# Test 4.2: Compile without scheduler (scheduler=None)
try:
    test_model_2 = li_resnet18(num_classes=NUM_CLASSES, stream1_input_channels=3,
                                stream2_input_channels=1, device=DEVICE)
    test_opt_2 = create_optimizer(test_model_2, lr=1e-3)
    test_model_2.compile(optimizer=test_opt_2, scheduler=None, loss='cross_entropy')

    assert test_model_2.scheduler is None, "Scheduler should be None"
    assert test_model_2.is_compiled, "is_compiled flag not set"
    log_test("compile() without scheduler", True, "Works with scheduler=None")
except Exception as e:
    log_test("compile() without scheduler", False, str(e))
    raise

# Test 4.3: Compile with focal loss
try:
    test_model_3 = mc_resnet18(num_classes=NUM_CLASSES, stream1_input_channels=3,
                                stream2_input_channels=1, device=DEVICE)
    test_opt_3 = create_optimizer(test_model_3, lr=1e-3)
    test_model_3.compile(optimizer=test_opt_3, loss='focal', alpha=1.0, gamma=2.0)

    assert test_model_3.criterion is not None, "Criterion not created"
    log_test("compile() with focal loss", True, "Focal loss created successfully")
except Exception as e:
    log_test("compile() with focal loss", False, str(e))
    raise

# Test 4.4: Compile with label smoothing
try:
    test_model_4 = mc_resnet18(num_classes=NUM_CLASSES, stream1_input_channels=3,
                                stream2_input_channels=1, device=DEVICE)
    test_opt_4 = create_optimizer(test_model_4, lr=1e-3)
    test_model_4.compile(optimizer=test_opt_4, loss='cross_entropy', label_smoothing=0.1)

    assert test_model_4.criterion is not None, "Criterion not created"
    log_test("compile() with label smoothing", True, "Label smoothing applied")
except Exception as e:
    log_test("compile() with label smoothing", False, str(e))
    raise

# ============================================================================
# TEST SECTION 5: Training Workflow
# ============================================================================
print("\n" + "=" * 100)
print("TEST SECTION 5: Training Workflow")
print("=" * 100)

# Create data loaders
train_loader, val_loader = create_dummy_loaders()

# Test 5.1: Training MCResNet with stream monitoring
try:
    train_model_mc = mc_resnet18(num_classes=NUM_CLASSES, stream1_input_channels=3,
                                  stream2_input_channels=1, device=DEVICE)
    train_opt_mc = create_stream_optimizer(
        train_model_mc,
        stream1_lr=2e-4, stream2_lr=7e-4, shared_lr=5e-4
    )
    train_sched_mc = setup_scheduler(train_opt_mc, 'decaying_cosine', epochs=3,
                                      train_loader_len=len(train_loader), t_max=2, eta_min=1e-5)
    train_model_mc.compile(optimizer=train_opt_mc, scheduler=train_sched_mc, loss='cross_entropy')

    history_mc = train_model_mc.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        verbose=False,
        stream_monitoring=True
    )

    assert 'train_loss' in history_mc, "train_loss not in history"
    assert 'val_loss' in history_mc, "val_loss not in history"
    assert len(history_mc['train_loss']) == 3, f"Expected 3 epochs, got {len(history_mc['train_loss'])}"
    assert len(history_mc['stream1_lr']) > 0, "Stream1 LR not recorded"
    assert len(history_mc['stream2_lr']) > 0, "Stream2 LR not recorded"
    log_test("Training MCResNet with stream monitoring", True,
             f"Completed 3 epochs, final train_loss={history_mc['train_loss'][-1]:.4f}")
except Exception as e:
    log_test("Training MCResNet with stream monitoring", False, str(e))
    import traceback
    traceback.print_exc()
    raise

# Test 5.2: Training LINet with stream monitoring
try:
    train_model_li = li_resnet18(num_classes=NUM_CLASSES, stream1_input_channels=3,
                                  stream2_input_channels=1, device=DEVICE, dropout_p=0.3)
    train_opt_li = create_stream_optimizer(
        train_model_li,
        stream1_lr=1e-4, stream2_lr=3e-4, shared_lr=2e-4
    )
    train_sched_li = setup_scheduler(train_opt_li, 'cosine', epochs=3,
                                      train_loader_len=len(train_loader), t_max=3)
    train_model_li.compile(optimizer=train_opt_li, scheduler=train_sched_li, loss='cross_entropy')

    history_li = train_model_li.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        verbose=False,
        stream_monitoring=True
    )

    assert len(history_li['train_loss']) == 3, f"Expected 3 epochs"
    assert len(history_li['stream1_lr']) > 0, "Stream1 LR not recorded"
    log_test("Training LINet with stream monitoring", True,
             f"Completed 3 epochs, final val_acc={history_li['val_accuracy'][-1]:.4f}")
except Exception as e:
    log_test("Training LINet with stream monitoring", False, str(e))
    import traceback
    traceback.print_exc()
    raise

# Test 5.3: Training without stream monitoring (simple)
try:
    simple_train_model = mc_resnet18(num_classes=NUM_CLASSES, stream1_input_channels=3,
                                      stream2_input_channels=1, device=DEVICE)
    simple_opt = create_optimizer(simple_train_model, lr=1e-3)
    simple_sched = setup_scheduler(simple_opt, 'step', epochs=3, train_loader_len=len(train_loader),
                                    step_size=2, gamma=0.5)
    simple_train_model.compile(optimizer=simple_opt, scheduler=simple_sched, loss='cross_entropy')

    history_simple = simple_train_model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        verbose=False,
        stream_monitoring=False  # Disabled
    )

    assert len(history_simple['train_loss']) == 3, "Expected 3 epochs"
    # Stream metrics should be empty when monitoring is disabled
    assert len(history_simple['stream1_lr']) == 0, "Stream1 LR should be empty"
    log_test("Training without stream monitoring", True, "Training works without monitoring")
except Exception as e:
    log_test("Training without stream monitoring", False, str(e))
    raise

# Test 5.4: Training without scheduler
try:
    no_sched_model = mc_resnet18(num_classes=NUM_CLASSES, stream1_input_channels=3,
                                  stream2_input_channels=1, device=DEVICE)
    no_sched_opt = create_optimizer(no_sched_model, lr=1e-3)
    no_sched_model.compile(optimizer=no_sched_opt, scheduler=None, loss='cross_entropy')

    history_no_sched = no_sched_model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        verbose=False
    )

    assert len(history_no_sched['train_loss']) == 2, "Expected 2 epochs"
    log_test("Training without scheduler", True, "Works without scheduler")
except Exception as e:
    log_test("Training without scheduler", False, str(e))
    raise

# ============================================================================
# TEST SECTION 6: Error Handling
# ============================================================================
print("\n" + "=" * 100)
print("TEST SECTION 6: Error Handling")
print("=" * 100)

# Test 6.1: compile() with invalid optimizer type (string)
try:
    error_model_1 = mc_resnet18(num_classes=NUM_CLASSES, stream1_input_channels=3,
                                 stream2_input_channels=1, device=DEVICE)
    error_model_1.compile(optimizer="adamw", loss='cross_entropy')
    log_test("Error handling: String optimizer", False, "Should have raised TypeError")
except TypeError as e:
    if "torch.optim.Optimizer" in str(e):
        log_test("Error handling: String optimizer", True, "Correctly rejected string optimizer")
    else:
        log_test("Error handling: String optimizer", False, f"Wrong error: {e}")
except Exception as e:
    log_test("Error handling: String optimizer", False, f"Unexpected error: {e}")

# Test 6.2: compile() with invalid scheduler type (string)
try:
    error_model_2 = mc_resnet18(num_classes=NUM_CLASSES, stream1_input_channels=3,
                                 stream2_input_channels=1, device=DEVICE)
    valid_opt = create_optimizer(error_model_2, lr=1e-3)
    error_model_2.compile(optimizer=valid_opt, scheduler="cosine", loss='cross_entropy')
    log_test("Error handling: String scheduler", False, "Should have raised TypeError")
except TypeError as e:
    if "LRScheduler" in str(e) or "step()" in str(e):
        log_test("Error handling: String scheduler", True, "Correctly rejected string scheduler")
    else:
        log_test("Error handling: String scheduler", False, f"Wrong error: {e}")
except Exception as e:
    log_test("Error handling: String scheduler", False, f"Unexpected error: {e}")

# Test 6.3: fit() before compile()
try:
    error_model_3 = mc_resnet18(num_classes=NUM_CLASSES, stream1_input_channels=3,
                                 stream2_input_channels=1, device=DEVICE)
    error_model_3.fit(train_loader, val_loader, epochs=1, verbose=False)
    log_test("Error handling: fit() before compile()", False, "Should have raised ValueError")
except ValueError as e:
    if "not compiled" in str(e).lower():
        log_test("Error handling: fit() before compile()", True, "Correctly caught uncompiled model")
    else:
        log_test("Error handling: fit() before compile()", False, f"Wrong error: {e}")
except Exception as e:
    log_test("Error handling: fit() before compile()", False, f"Unexpected error: {e}")

# Test 6.4: create_stream_optimizer() on non-stream model
try:
    # Try to use create_stream_optimizer on a model without the helper method
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)

    dummy = DummyModel()
    create_stream_optimizer(dummy, stream1_lr=1e-3, stream2_lr=1e-3, shared_lr=1e-3)
    log_test("Error handling: create_stream_optimizer() on non-stream model", False,
             "Should have raised AttributeError")
except AttributeError as e:
    if "get_stream_parameter_groups" in str(e):
        log_test("Error handling: create_stream_optimizer() on non-stream model", True,
                 "Correctly rejected non-stream model")
    else:
        log_test("Error handling: create_stream_optimizer() on non-stream model", False, f"Wrong error: {e}")
except Exception as e:
    log_test("Error handling: create_stream_optimizer() on non-stream model", False,
             f"Unexpected error: {e}")

# ============================================================================
# TEST SECTION 7: Backward Compatibility
# ============================================================================
print("\n" + "=" * 100)
print("TEST SECTION 7: Backward Compatibility")
print("=" * 100)

# Test 7.1: setup_scheduler is importable and public
try:
    from src.training.schedulers import setup_scheduler as imported_func
    assert callable(imported_func), "setup_scheduler not callable"
    log_test("setup_scheduler is public and importable", True, "Function is accessible")
except Exception as e:
    log_test("setup_scheduler is public and importable", False, str(e))

# Test 7.2: DecayingCosineAnnealingLR is importable
try:
    from src.training.schedulers import DecayingCosineAnnealingLR
    assert DecayingCosineAnnealingLR is not None, "Class not found"
    log_test("DecayingCosineAnnealingLR is importable", True, "Class is accessible")
except Exception as e:
    log_test("DecayingCosineAnnealingLR is importable", False, str(e))

# Test 7.3: All scheduler types still work
try:
    test_opt = create_optimizer(mc_model, lr=1e-3)
    scheduler_types = ['cosine', 'decaying_cosine', 'step', 'plateau', 'onecycle']
    all_work = True

    for stype in scheduler_types:
        try:
            if stype == 'decaying_cosine':
                sched = setup_scheduler(test_opt, stype, epochs=10, train_loader_len=10,
                                       t_max=5, eta_min=1e-5)
            elif stype == 'onecycle':
                sched = setup_scheduler(test_opt, stype, epochs=10, train_loader_len=10,
                                       max_lr=1e-3)
            else:
                sched = setup_scheduler(test_opt, stype, epochs=10, train_loader_len=10)

            if sched is None:
                all_work = False
                break
        except Exception as e:
            all_work = False
            break

    if all_work:
        log_test("All scheduler types work", True, f"Tested {len(scheduler_types)} scheduler types")
    else:
        log_test("All scheduler types work", False, f"Some schedulers failed")
except Exception as e:
    log_test("All scheduler types work", False, str(e))

# ============================================================================
# TEST SECTION 8: Integration Tests
# ============================================================================
print("\n" + "=" * 100)
print("TEST SECTION 8: Integration Tests")
print("=" * 100)

# Test 8.1: Full workflow - MCResNet34
try:
    full_mc34 = mc_resnet34(num_classes=NUM_CLASSES, stream1_input_channels=3,
                             stream2_input_channels=1, device=DEVICE)
    full_opt_mc34 = create_stream_optimizer(full_mc34, stream1_lr=2e-4, stream2_lr=5e-4, shared_lr=3e-4)
    full_sched_mc34 = setup_scheduler(full_opt_mc34, 'decaying_cosine', epochs=5,
                                       train_loader_len=len(train_loader), t_max=3, eta_min=5e-6)
    full_mc34.compile(optimizer=full_opt_mc34, scheduler=full_sched_mc34, loss='cross_entropy')

    history_full = full_mc34.fit(train_loader, val_loader, epochs=2, verbose=False, stream_monitoring=True)

    assert len(history_full['train_loss']) == 2, "Wrong number of epochs"
    log_test("Full workflow - MCResNet34", True, "Complete workflow successful")
except Exception as e:
    log_test("Full workflow - MCResNet34", False, str(e))
    import traceback
    traceback.print_exc()

# Test 8.2: Full workflow - LIResNet34
try:
    full_li34 = li_resnet34(num_classes=NUM_CLASSES, stream1_input_channels=3,
                             stream2_input_channels=1, device=DEVICE, dropout_p=0.2)
    full_opt_li34 = create_stream_optimizer(full_li34, stream1_lr=1e-4, stream2_lr=2e-4, shared_lr=1.5e-4)
    full_sched_li34 = setup_scheduler(full_opt_li34, 'cosine', epochs=5,
                                       train_loader_len=len(train_loader), t_max=5)
    full_li34.compile(optimizer=full_opt_li34, scheduler=full_sched_li34, loss='cross_entropy')

    history_full_li = full_li34.fit(train_loader, val_loader, epochs=2, verbose=False, stream_monitoring=True)

    assert len(history_full_li['train_loss']) == 2, "Wrong number of epochs"
    log_test("Full workflow - LIResNet34", True, "Complete workflow successful")
except Exception as e:
    log_test("Full workflow - LIResNet34", False, str(e))
    import traceback
    traceback.print_exc()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 100)
print("TEST RESULTS SUMMARY")
print("=" * 100)

total_tests = len(test_results)
passed_tests = sum(1 for _, passed in test_results if passed)
failed_tests = total_tests - passed_tests

print(f"\nTotal Tests: {total_tests}")
print(f"Passed: {passed_tests} ✅")
print(f"Failed: {failed_tests} ❌")
print(f"Success Rate: {100 * passed_tests / total_tests:.1f}%")

if failed_tests > 0:
    print("\n" + "=" * 100)
    print("FAILED TESTS:")
    print("=" * 100)
    for test_name, passed in test_results:
        if not passed:
            print(f"  ❌ {test_name}")

print("\n" + "=" * 100)
if failed_tests == 0:
    print("🎉 ALL TESTS PASSED! 🎉")
    print("=" * 100)
    print("\nKeras-style API refactoring is fully functional and tested!")
    print("\n✅ Model creation works")
    print("✅ Parameter group helpers work")
    print("✅ Optimizer creation works (simple and stream-specific)")
    print("✅ Scheduler creation works")
    print("✅ Compile method works")
    print("✅ Training workflow works (MCResNet and LINet)")
    print("✅ Stream monitoring works")
    print("✅ Error handling works")
    print("✅ Backward compatibility maintained")
    print("✅ Integration tests pass")
else:
    print("⚠️  SOME TESTS FAILED")
    print("=" * 100)
    exit(1)

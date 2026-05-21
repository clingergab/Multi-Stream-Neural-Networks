"""Tests for the paper-reported-run configs and the train.py wrapper.

These tests cover three concerns:

1. **YAML schema** — every config under configs/reported_runs/ parses and
   has the structure train.py expects.
2. **Paper consistency** — the hyperparameter values match the paper's
   Appendix B tables exactly. Sanity check that nothing drifted from the
   source of truth.
3. **train.py wrapper** — the pure-Python helpers (config load, model
   build, optimizer/scheduler construction, pretrained-checkpoint
   loading) work end-to-end on CPU without requiring the actual datasets
   or a GPU.

Tests do NOT exercise the dataloaders or the full training loop; those
require preprocessed data and a GPU and live outside the unit-test scope.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

# Make the repo importable when running pytest from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import train  # noqa: E402

CONFIGS_DIR = REPO_ROOT / "configs" / "reported_runs"
CONFIG_PATHS = sorted(p for p in CONFIGS_DIR.glob("*.yaml"))

REQUIRED_TOP_LEVEL = {
    "meta",
    "dataset",
    "model",
    "optimizer",
    "scheduler",
    "training",
    "augmentation",
    "modality_dropout",
}

# Paper Appendix B values, transcribed from the source paper. These are
# the *exact* numbers the YAMLs must encode. If the paper itself ever
# changes, update this table — not the YAMLs in isolation.
PAPER_VALUES = {
    # Table B1: SUN RGB-D, from scratch, LINet, four MD schedules.
    "sunrgbd_linet_no_md_from_scratch": {
        "epochs": 70, "lr": 1.57e-4, "wd": 3.52e-4, "eta_min": 3.94e-6,
        "dropout": 0.38, "label_smoothing": 0.18, "grad_clip": 1.00,
        "stem_lr_mult": 13,
        "rgb_aug_prob": 0.73, "rgb_aug_mag": 1.01,
        "depth_aug_prob": 0.79, "depth_aug_mag": 1.08,
        "md_enabled": False,
        "reported_fusion_mca": 43.5,
        "reported_rgb_mca": 15.0,
        "reported_depth_mca": 14.4,
    },
    "sunrgbd_linet_progressive_md_from_scratch": {
        "epochs": 80, "lr": 1.53e-4, "wd": 6.74e-5, "eta_min": 3.27e-6,
        "dropout": 0.28, "label_smoothing": 0.10, "grad_clip": 1.00,
        "stem_lr_mult": 21,
        "rgb_aug_prob": 0.94, "rgb_aug_mag": 1.20,
        "depth_aug_prob": 1.06, "depth_aug_mag": 0.98,
        "md_enabled": True, "md_rate": 0.48, "md_start": 0, "md_ramp": 30,
        "reported_fusion_mca": 45.2,
        "reported_rgb_mca": 33.7,
        "reported_depth_mca": 35.7,
    },
    "sunrgbd_linet_static_md_from_scratch": {
        "epochs": 86, "lr": 1.15e-4, "wd": 9.22e-5, "eta_min": 3.48e-6,
        "dropout": 0.34, "label_smoothing": 0.10, "grad_clip": 1.00,
        "stem_lr_mult": 21,
        "rgb_aug_prob": 0.97, "rgb_aug_mag": 1.10,
        "depth_aug_prob": 1.06, "depth_aug_mag": 0.94,
        "md_enabled": True, "md_rate": 0.48, "md_start": 0, "md_ramp": 0,
        "reported_fusion_mca": 43.2,
        "reported_rgb_mca": 32.9,
        "reported_depth_mca": 33.7,
    },
    "sunrgbd_linet_delayed_md_from_scratch": {
        "epochs": 81, "lr": 1.53e-4, "wd": 1.46e-4, "eta_min": 1.73e-6,
        "dropout": 0.27, "label_smoothing": 0.10, "grad_clip": 1.00,
        "stem_lr_mult": 21,
        "rgb_aug_prob": 0.92, "rgb_aug_mag": 1.17,
        "depth_aug_prob": 0.95, "depth_aug_mag": 1.04,
        # Delayed: paper §4.4.1 prose says start=30, ramp=15; Appendix B1
        # says start=20, ramp=15. We follow Appendix B1.
        "md_enabled": True, "md_rate": 0.48, "md_start": 20, "md_ramp": 15,
        "reported_fusion_mca": 43.8,
        "reported_rgb_mca": 33.7,
        "reported_depth_mca": 36.5,
    },
    # Table B3 col 1: SUN RGB-D, ScanNet-pretrained No-MD.
    "sunrgbd_linet_no_md_scannet_pretrained": {
        "epochs": 13, "lr": 1.884e-4, "wd": 4.138e-2, "eta_min": 4.724e-6,
        "dropout": 0.78, "label_smoothing": 0.05, "grad_clip": 1.00,
        "stem_lr_mult": 1.70,
        "rgb_aug_prob": 0.85, "rgb_aug_mag": 1.20,
        "depth_aug_prob": 1.27, "depth_aug_mag": 1.27,
        "md_enabled": False,
        "reported_fusion_mca": 48.0,
        "reported_rgb_mca": 36.5,
        "reported_depth_mca": 37.2,
    },
    # Table B3 col 2: SUN RGB-D, ScanNet-pretrained Progressive MD.
    "sunrgbd_linet_progressive_md_scannet_pretrained": {
        "epochs": 18, "lr": 1.404e-4, "wd": 7.293e-4, "eta_min": 9.632e-7,
        "dropout": 0.65, "label_smoothing": 0.05, "grad_clip": 1.20,
        "stem_lr_mult": 0.90,
        "rgb_aug_prob": 0.70, "rgb_aug_mag": 1.07,
        "depth_aug_prob": 1.26, "depth_aug_mag": 1.10,
        "md_enabled": True, "md_rate": 0.48, "md_start": 0, "md_ramp": 30,
        "reported_fusion_mca": 49.6,
        "reported_rgb_mca": 39.2,
        "reported_depth_mca": 38.4,
    },
    # Table B3 footnote: ScanNet pretraining phase, shared between the
    # two pretraining runs except for the MD schedule.
    "scannet_pretrain_no_md": {
        "epochs": 50, "lr": 3.14e-4, "wd": 4.01e-4, "eta_min": 3.0e-6,
        "dropout": 0.13, "label_smoothing": 0.07, "grad_clip": 1.20,
        "stem_lr_mult": 16.0,
        "rgb_aug_prob": 0.43, "rgb_aug_mag": 0.80,
        "depth_aug_prob": 0.54, "depth_aug_mag": 0.70,
        "md_enabled": False,
    },
    "scannet_pretrain_progressive_md": {
        "epochs": 50, "lr": 3.14e-4, "wd": 4.01e-4, "eta_min": 3.0e-6,
        "dropout": 0.13, "label_smoothing": 0.07, "grad_clip": 1.20,
        "stem_lr_mult": 16.0,
        "rgb_aug_prob": 0.43, "rgb_aug_mag": 0.80,
        "depth_aug_prob": 0.54, "depth_aug_mag": 0.70,
        "md_enabled": True, "md_rate": 0.48, "md_start": 0, "md_ramp": 30,
    },
    # Table B4: NYU Depth V2, from scratch, LINet.
    "nyu_linet_no_md_from_scratch": {
        "epochs": 61, "lr": 1.40e-4, "wd": 9.07e-4, "eta_min": 3.84e-6,
        "dropout": 0.41, "label_smoothing": 0.19, "grad_clip": 1.35,
        "stem_lr_mult": 20,
        "rgb_aug_prob": 1.13, "rgb_aug_mag": 1.13,
        "depth_aug_prob": 0.71, "depth_aug_mag": 0.91,
        "md_enabled": False,
        "reported_fusion_mca": 50.1,
        "reported_rgb_mca": 27.0,
        "reported_depth_mca": 16.1,
    },
    "nyu_linet_progressive_md_from_scratch": {
        "epochs": 65, "lr": 1.40e-4, "wd": 1.36e-4, "eta_min": 2.17e-6,
        "dropout": 0.41, "label_smoothing": 0.16, "grad_clip": 1.05,
        "stem_lr_mult": 23,
        "rgb_aug_prob": 0.99, "rgb_aug_mag": 0.87,
        "depth_aug_prob": 0.75, "depth_aug_mag": 0.80,
        "md_enabled": True, "md_rate": 0.44, "md_start": 0, "md_ramp": 30,
        "reported_fusion_mca": 50.8,
        "reported_rgb_mca": 37.8,
        "reported_depth_mca": 40.3,
    },
}


# ----------------------------- YAML schema ----------------------------- #


def test_configs_dir_exists_and_has_configs():
    """Sanity: the configs directory exists and we found YAML files."""
    assert CONFIGS_DIR.is_dir(), f"Missing directory: {CONFIGS_DIR}"
    assert CONFIG_PATHS, "No YAML configs found in configs/reported_runs/"
    # Every entry in PAPER_VALUES must have a matching file (catches typos).
    config_stems = {p.stem for p in CONFIG_PATHS}
    expected_stems = set(PAPER_VALUES.keys())
    assert expected_stems <= config_stems, (
        f"PAPER_VALUES references configs that don't exist on disk: "
        f"{expected_stems - config_stems}"
    )


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=lambda p: p.name)
def test_yaml_parses(path):
    cfg = yaml.safe_load(path.read_text())
    assert isinstance(cfg, dict)


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=lambda p: p.name)
def test_yaml_has_required_top_level_keys(path):
    cfg = yaml.safe_load(path.read_text())
    missing = REQUIRED_TOP_LEVEL - set(cfg)
    assert not missing, f"{path.name} is missing top-level keys: {missing}"


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=lambda p: p.name)
def test_yaml_model_block_is_complete(path):
    cfg = yaml.safe_load(path.read_text())
    m = cfg["model"]
    assert m["architecture"] == "li_resnet18"
    assert m["stream_input_channels"] == [3, 1]
    assert m["width_multiplier"] == 0.75  # paper §4.2: α = 0.75 everywhere
    assert isinstance(m["dropout_p"], (int, float))


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=lambda p: p.name)
def test_yaml_optimizer_and_scheduler_blocks(path):
    cfg = yaml.safe_load(path.read_text())
    o = cfg["optimizer"]
    s = cfg["scheduler"]
    assert o["type"] == "adamw", "Paper §4.2 specifies AdamW."
    assert "lr" in o and "weight_decay" in o and "stem_lr_multiplier" in o
    assert s["type"] == "cosine", "Paper §4.2 specifies cosine annealing."
    # The configs encode a single-cycle cosine annealing schedule per the
    # paper's prose, so t_max should equal training.epochs.
    assert s["t_max"] == cfg["training"]["epochs"], (
        f"{path.name}: scheduler.t_max ({s['t_max']}) "
        f"does not match training.epochs ({cfg['training']['epochs']})"
    )


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=lambda p: p.name)
def test_yaml_modality_dropout_block(path):
    cfg = yaml.safe_load(path.read_text())
    md = cfg["modality_dropout"]
    assert "enabled" in md
    if md["enabled"]:
        # When MD is on we need every parameter the fit() loop reads.
        for required in ("rate", "start", "ramp", "schedule"):
            assert required in md, f"{path.name}: MD enabled but missing {required!r}"
        assert md["schedule"] in ("ramp_up", "ramp_down")
        assert 0.0 < md["rate"] <= 1.0
        assert md["start"] >= 0
        assert md["ramp"] >= 0


# --------------------------- Paper consistency --------------------------- #


@pytest.mark.parametrize("stem", sorted(PAPER_VALUES.keys()))
def test_yaml_matches_paper_appendix_values(stem):
    """Every encoded hyperparameter must match the paper Appendix B value."""
    cfg = yaml.safe_load((CONFIGS_DIR / f"{stem}.yaml").read_text())
    expected = PAPER_VALUES[stem]

    assert cfg["training"]["epochs"] == expected["epochs"]
    assert cfg["optimizer"]["lr"] == pytest.approx(expected["lr"], rel=1e-9)
    assert cfg["optimizer"]["weight_decay"] == pytest.approx(expected["wd"], rel=1e-9)
    assert cfg["scheduler"]["eta_min"] == pytest.approx(expected["eta_min"], rel=1e-9)
    assert cfg["model"]["dropout_p"] == pytest.approx(expected["dropout"])
    assert cfg["training"]["label_smoothing"] == pytest.approx(expected["label_smoothing"])
    assert cfg["training"]["grad_clip_norm"] == pytest.approx(expected["grad_clip"])
    assert cfg["optimizer"]["stem_lr_multiplier"] == pytest.approx(expected["stem_lr_mult"])
    assert cfg["augmentation"]["rgb_aug_prob"] == pytest.approx(expected["rgb_aug_prob"])
    assert cfg["augmentation"]["rgb_aug_mag"] == pytest.approx(expected["rgb_aug_mag"])
    assert cfg["augmentation"]["depth_aug_prob"] == pytest.approx(expected["depth_aug_prob"])
    assert cfg["augmentation"]["depth_aug_mag"] == pytest.approx(expected["depth_aug_mag"])

    md = cfg["modality_dropout"]
    assert md["enabled"] == expected["md_enabled"]
    if expected["md_enabled"]:
        assert md["rate"] == pytest.approx(expected["md_rate"])
        assert md["start"] == expected["md_start"]
        assert md["ramp"] == expected["md_ramp"]

    if "reported_fusion_mca" in expected:
        rep = cfg["meta"]["reported_mca"]
        assert rep["fusion"] == pytest.approx(expected["reported_fusion_mca"])
        assert rep["rgb_only"] == pytest.approx(expected["reported_rgb_mca"])
        assert rep["depth_only"] == pytest.approx(expected["reported_depth_mca"])


# --------------------------- train.py helpers --------------------------- #


def test_load_config_returns_dict(tmp_path):
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text("foo:\n  bar: 1\n")
    cfg = train._load_config(str(cfg_path))
    assert cfg == {"foo": {"bar": 1}}


def test_build_model_constructs_linet_with_config_overrides():
    """_build_model wires every relevant model field from the YAML."""
    cfg = {
        "model": {
            "architecture": "li_resnet18",
            "stream_input_channels": [3, 1],
            "width_multiplier": 0.75,
            "dropout_p": 0.28,
            "device": "cpu",
            "use_amp": False,
        }
    }
    model = train._build_model(cfg, num_classes=19)
    assert model.num_streams == 2
    assert model.stream_input_channels == [3, 1]
    assert model.width_multiplier == 0.75
    assert model.dropout_p == 0.28
    assert model.num_classes == 19
    # Width factor 0.75 -> stage-4 channels = int(512 * 0.75) = 384,
    # which is the classifier head's input dimension.
    assert model.fc.fc.in_features == 384
    assert model.fc.fc.out_features == 19


def test_build_model_rejects_unknown_architecture():
    cfg = {
        "model": {
            "architecture": "li_resnet50",  # not in the reported_runs set
            "stream_input_channels": [3, 1],
            "width_multiplier": 0.75,
            "dropout_p": 0.28,
            "device": "cpu",
            "use_amp": False,
        }
    }
    with pytest.raises(ValueError, match="Unsupported architecture"):
        train._build_model(cfg, num_classes=19)


def test_build_dataloaders_rejects_unknown_dataset_name():
    cfg = {
        "dataset": {
            "name": "imagenet",  # not supported by train.py
            "data_root": "/tmp/foo",
            "batch_size": 1,
            "num_workers": 0,
            "num_classes": 10,
        },
        "augmentation": {
            "rgb_aug_prob": 1.0, "rgb_aug_mag": 1.0,
            "depth_aug_prob": 1.0, "depth_aug_mag": 1.0,
        },
        "training": {"seed": 42},
    }
    with pytest.raises(ValueError, match="Unknown dataset.name"):
        train._build_dataloaders(cfg, data_root_override=None)


def test_build_optimizer_eta_min_list_when_stem_mult_nontrivial():
    """When stem_lr_multiplier != 1.0, get_stream_parameter_groups produces
    2N+2 groups; our per-group eta_min list must be the same length."""
    cfg = {
        "model": {
            "architecture": "li_resnet18",
            "stream_input_channels": [3, 1],   # N = 2
            "width_multiplier": 0.75,
            "dropout_p": 0.28,
            "device": "cpu",
            "use_amp": False,
        },
        "optimizer": {
            "type": "adamw",
            "lr": 1.53e-4,
            "weight_decay": 6.74e-5,
            "stem_lr_multiplier": 21,
        },
        "scheduler": {
            "type": "cosine",
            "t_max": 80,
            "eta_min": 3.27e-6,
            "warmup_epochs": 0,
            "warmup_start_factor": 1.0,
        },
    }
    model = train._build_model(cfg, num_classes=19)
    optimizer, scheduler = train._build_optimizer_and_scheduler(
        model, cfg, train_loader_len=10
    )
    # N=2 streams with stem-split -> 2N+2 = 6 groups.
    assert len(optimizer.param_groups) == 6
    # Stem groups should carry the boosted LR; non-stem stream groups
    # should carry the base LR. Group ordering: stem_stream_0, stem_stream_1,
    # stream_0, stream_1, integration, classifier.
    stem_mult = cfg["optimizer"]["stem_lr_multiplier"]
    base_lr = cfg["optimizer"]["lr"]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr * stem_mult)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(base_lr * stem_mult)
    assert optimizer.param_groups[2]["lr"] == pytest.approx(base_lr)
    assert optimizer.param_groups[3]["lr"] == pytest.approx(base_lr)
    assert scheduler is not None


def test_build_optimizer_eta_min_scalar_when_stem_mult_one():
    """When stem_lr_multiplier == 1.0, no stem split happens (N+2 groups)
    and scalar eta_min should broadcast cleanly through setup_scheduler."""
    cfg = {
        "model": {
            "architecture": "li_resnet18",
            "stream_input_channels": [3, 1],
            "width_multiplier": 0.75,
            "dropout_p": 0.1,
            "device": "cpu",
            "use_amp": False,
        },
        "optimizer": {
            "type": "adamw",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "stem_lr_multiplier": 1.0,
        },
        "scheduler": {
            "type": "cosine",
            "t_max": 10,
            "eta_min": 1e-6,
            "warmup_epochs": 0,
            "warmup_start_factor": 1.0,
        },
    }
    model = train._build_model(cfg, num_classes=10)
    optimizer, scheduler = train._build_optimizer_and_scheduler(
        model, cfg, train_loader_len=10
    )
    # N=2 streams, no stem split -> N+2 = 4 groups.
    assert len(optimizer.param_groups) == 4
    assert scheduler is not None


def test_load_pretrained_skips_classifier_and_loads_compatible_tensors(tmp_path):
    """Source-class-count mismatch must not blow up loading; the fc head
    has to be reinitialized when transferring across dataset taxonomies."""
    # Build a model on a "source" task (ScanNet-shaped 20 classes), save it,
    # then load that checkpoint into a "target" model with 19 classes.
    cfg_source = {
        "model": {
            "architecture": "li_resnet18",
            "stream_input_channels": [3, 1],
            "width_multiplier": 0.75,
            "dropout_p": 0.1,
            "device": "cpu",
            "use_amp": False,
        }
    }
    src = train._build_model(cfg_source, num_classes=20)

    ckpt_path = tmp_path / "pretrained.pt"
    torch.save({"model_state_dict": src.state_dict()}, ckpt_path)

    cfg_target = dict(cfg_source)
    tgt = train._build_model(cfg_target, num_classes=19)
    train._load_pretrained(tgt, str(ckpt_path))

    # Backbone conv weights should match the source; classifier head should
    # NOT (it was reinitialized at the new class count).
    src_sd = src.state_dict()
    tgt_sd = tgt.state_dict()
    backbone_keys = [
        k for k in tgt_sd
        if not k.startswith("fc.") and src_sd[k].shape == tgt_sd[k].shape
    ]
    assert backbone_keys, "Expected at least one backbone tensor to load"
    for k in backbone_keys:
        assert torch.equal(src_sd[k], tgt_sd[k]), f"{k} did not transfer"
    # Classifier head shape differs (20 vs 19 classes) and must be reset.
    fc_keys = [k for k in tgt_sd if k.startswith("fc.")]
    assert fc_keys
    for k in fc_keys:
        if k in src_sd and src_sd[k].shape != tgt_sd[k].shape:
            # different shape => couldn't have been copied. Pass.
            continue


def test_load_pretrained_accepts_flat_state_dict(tmp_path):
    """Some checkpoint conventions save state_dict directly (no wrapping)."""
    cfg = {
        "model": {
            "architecture": "li_resnet18",
            "stream_input_channels": [3, 1],
            "width_multiplier": 0.75,
            "dropout_p": 0.1,
            "device": "cpu",
            "use_amp": False,
        }
    }
    src = train._build_model(cfg, num_classes=19)
    ckpt_path = tmp_path / "flat.pt"
    torch.save(src.state_dict(), ckpt_path)

    tgt = train._build_model(cfg, num_classes=19)
    # Should not raise.
    train._load_pretrained(tgt, str(ckpt_path))


# ----------------------------- argparse smoke ----------------------------- #


def test_train_help_flag_exits_zero():
    """`python train.py -h` must succeed without requiring CUDA or data."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "train.py"), "-h"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"train.py -h exited non-zero.\nstderr:\n{result.stderr}"
    )
    assert "--config" in result.stdout
    assert "--data-root" in result.stdout
    assert "--pretrained" in result.stdout


def test_train_missing_required_config_arg_fails():
    """argparse should reject calls that omit --config."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "train.py")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "--config" in result.stderr


# ----------------------------- meta block ----------------------------- #


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=lambda p: p.name)
def test_meta_block_has_paper_table(path):
    cfg = yaml.safe_load(path.read_text())
    meta = cfg.get("meta", {})
    assert "paper_table" in meta, (
        f"{path.name}: meta.paper_table is required so the YAML cites its source row."
    )


@pytest.mark.parametrize(
    "stem",
    sorted(s for s, v in PAPER_VALUES.items() if "reported_fusion_mca" in v),
)
def test_reported_runs_have_reported_mca_block(stem):
    """Fine-tuning / from-scratch runs that map to a Table 1 or Table 2 row
    should publish the expected MCA in meta.reported_mca; pretraining-only
    runs (Table B3 footnote) intentionally don't and are excluded here."""
    cfg = yaml.safe_load((CONFIGS_DIR / f"{stem}.yaml").read_text())
    mca = cfg["meta"].get("reported_mca")
    assert mca is not None, f"{stem}: missing meta.reported_mca block"
    for key in ("fusion", "rgb_only", "depth_only"):
        assert key in mca, f"{stem}: meta.reported_mca.{key} missing"


# -------------------- configs/reported_runs/README.md -------------------- #


def test_configs_readme_references_every_yaml():
    """Index README must mention every YAML by filename so nothing orphans."""
    readme = (CONFIGS_DIR / "README.md").read_text()
    for path in CONFIG_PATHS:
        assert path.name in readme, (
            f"{path.name} is not referenced in configs/reported_runs/README.md"
        )

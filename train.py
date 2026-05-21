"""Config-driven training entry point for LINet.

Reads a YAML config from `configs/reported_runs/` and reproduces the
training recipe for a single paper-reported run. This is a thin wrapper
around the same `model.compile() -> model.fit()` pipeline the notebooks
under `notebooks/` already use; nothing is reimplemented from scratch.

Usage:
    python3 train.py --config configs/reported_runs/<name>.yaml \\
                     --data-root <preprocessed-dataset-path>

    # ScanNet-pretrained fine-tune (two-phase recipe):
    python3 train.py --config configs/reported_runs/sunrgbd_linet_progressive_md_scannet_pretrained.yaml \\
                     --data-root <sunrgbd-path> \\
                     --pretrained <scannet-checkpoint.pt>

GPU is required (paper reports T4 for SUN/NYU final runs, A100 for ScanNet
pretraining). The configs hold the exact HPO-optimized hyperparameters from
paper Appendix B; reproducing a row's number requires running the matching
config against the corresponding preprocessed dataset.

See configs/reported_runs/README.md for the per-row index.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

from src.data_utils.sunrgbd_dataset import get_sunrgbd_dataloaders
from src.data_utils.nyu_depth_v2_dataset import get_nyu_depth_v2_dataloaders
from src.data_utils.scannet_pretrain_dataset import get_scannet_pretrain_dataloaders
from src.models.linear_integration.li_net3 import li_resnet18
from src.training.optimizers import create_stream_optimizer
from src.training.schedulers import setup_scheduler


def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _build_dataloaders(cfg: dict, data_root_override: Optional[str]) -> tuple:
    """Build (train_loader, val_loader, num_classes) for the configured dataset.

    For SUN RGB-D and NYU the third returned value is the test loader (used
    as the validation set during training, matching the notebooks' practice
    of training on the official train split and evaluating on the official
    test split). For ScanNet pretraining the val loader is the dataset's
    own val split.
    """
    dcfg = cfg["dataset"]
    aug = cfg["augmentation"]
    data_root = data_root_override or dcfg["data_root"]
    name = dcfg["name"]
    seed = cfg["training"].get("seed", 42)

    if name == "sunrgbd_19":
        train_loader, val_loader, test_loader = get_sunrgbd_dataloaders(
            data_root=data_root,
            batch_size=dcfg["batch_size"],
            num_workers=dcfg["num_workers"],
            seed=seed,
            stratified=dcfg.get("stratified", True),
            normalize=dcfg.get("normalize", True),
            use_hha=dcfg.get("use_hha", False),
            rgb_aug_prob=aug["rgb_aug_prob"],
            rgb_aug_mag=aug["rgb_aug_mag"],
            depth_aug_prob=aug["depth_aug_prob"],
            depth_aug_mag=aug["depth_aug_mag"],
        )
        # No val/ directory in the official SUN RGB-D 19-class split, so the
        # test loader doubles as the val loader during training (this matches
        # how the notebooks evaluate). early_stopping is disabled by default
        # in every reported_runs/ config so this does not introduce test-set
        # peeking via best-checkpoint selection.
        eval_loader = val_loader if val_loader is not None else test_loader
        return train_loader, eval_loader, dcfg["num_classes"]

    if name == "nyu_depth_v2":
        train_loader, val_loader, test_loader = get_nyu_depth_v2_dataloaders(
            data_root=data_root,
            batch_size=dcfg["batch_size"],
            num_workers=dcfg["num_workers"],
            seed=seed,
            stratified=dcfg.get("stratified", True),
            normalize=dcfg.get("normalize", True),
            rgb_aug_prob=aug["rgb_aug_prob"],
            rgb_aug_mag=aug["rgb_aug_mag"],
            depth_aug_prob=aug["depth_aug_prob"],
            depth_aug_mag=aug["depth_aug_mag"],
        )
        eval_loader = val_loader if val_loader is not None else test_loader
        return train_loader, eval_loader, dcfg["num_classes"]

    if name == "scannet_100k":
        train_loader, val_loader, num_classes = get_scannet_pretrain_dataloaders(
            data_root=data_root,
            batch_size=dcfg["batch_size"],
            num_workers=dcfg["num_workers"],
            seed=seed,
            normalize=dcfg.get("normalize", True),
            use_hha=dcfg.get("use_hha", False),
            rgb_aug_prob=aug["rgb_aug_prob"],
            rgb_aug_mag=aug["rgb_aug_mag"],
            depth_aug_prob=aug["depth_aug_prob"],
            depth_aug_mag=aug["depth_aug_mag"],
        )
        return train_loader, val_loader, num_classes

    raise ValueError(
        f"Unknown dataset.name={name!r}. Supported: 'sunrgbd_19', "
        f"'nyu_depth_v2', 'scannet_100k'."
    )


def _build_model(cfg: dict, num_classes: int) -> torch.nn.Module:
    mcfg = cfg["model"]
    arch = mcfg.get("architecture", "li_resnet18")
    if arch != "li_resnet18":
        raise ValueError(
            f"Unsupported architecture {arch!r}. Reported runs all use 'li_resnet18'."
        )
    return li_resnet18(
        num_classes=num_classes,
        stream_input_channels=mcfg["stream_input_channels"],
        width_multiplier=mcfg["width_multiplier"],
        dropout_p=mcfg["dropout_p"],
        device=mcfg.get("device", "cuda"),
        use_amp=mcfg.get("use_amp", True),
    )


def _load_pretrained(model: torch.nn.Module, ckpt_path: str) -> None:
    """Load a pretrained checkpoint, skipping the classifier head.

    The classifier head (`model.fc`) is reinitialized because the source
    (ScanNet) and target (SUN RGB-D / NYU) class counts differ.
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state.get("model_state_dict", state)
    own = model.state_dict()
    filtered = {
        k: v for k, v in sd.items()
        if k in own and own[k].shape == v.shape and not k.startswith("fc.")
    }
    missing = [k for k in own if k not in filtered]
    model.load_state_dict(filtered, strict=False)
    print(f"[pretrained] loaded {len(filtered)} tensors from {ckpt_path}")
    print(f"[pretrained] reinitialized (not loaded): {len(missing)} tensors "
          f"(includes classifier head)")


def _build_optimizer_and_scheduler(model: torch.nn.Module, cfg: dict, train_loader_len: int):
    ocfg = cfg["optimizer"]
    scfg = cfg["scheduler"]
    mcfg = cfg["model"]

    num_streams = len(mcfg["stream_input_channels"])
    stream_lrs = [ocfg["lr"]] * num_streams
    stream_wds = [ocfg["weight_decay"]] * num_streams

    optimizer = create_stream_optimizer(
        model,
        optimizer_type=ocfg.get("type", "adamw"),
        stream_lrs=stream_lrs,
        stream_weight_decays=stream_wds,
        shared_lr=ocfg["lr"],
        integration_weight_decay=ocfg["weight_decay"],
        stem_lr_multiplier=ocfg.get("stem_lr_multiplier", 1.0),
    )

    # Per-group eta_min, accounting for stem-LR-multiplier group expansion
    # (see setup_scheduler docstring and the notebook).
    stem_mult = ocfg.get("stem_lr_multiplier", 1.0)
    eta_min = scfg["eta_min"]
    if stem_mult != 1.0:
        eta_min_per_group = (
            [eta_min * stem_mult] * num_streams
            + [eta_min] * (num_streams + 2)
        )
    else:
        eta_min_per_group = eta_min

    scheduler = setup_scheduler(
        optimizer,
        scheduler_type=scfg.get("type", "cosine"),
        train_loader_len=train_loader_len,
        t_max=scfg["t_max"],
        eta_min=eta_min_per_group,
        warmup_epochs=scfg.get("warmup_epochs", 0),
        warmup_start_factor=scfg.get("warmup_start_factor", 1.0),
    )
    return optimizer, scheduler


def _maybe_write_run_metadata(out_dir: Path, cfg: dict, config_path: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config_snapshot.yaml", "w") as f:
        yaml.safe_dump({"_source_config": config_path, **cfg}, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(
        description="Train LINet from a paper-reported-run YAML config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to a YAML config under configs/reported_runs/.",
    )
    parser.add_argument(
        "--data-root", default=None,
        help="Override dataset.data_root from the YAML "
             "(useful when the preprocessed data lives outside the config's default path).",
    )
    parser.add_argument(
        "--pretrained", default=None,
        help="Path to a checkpoint .pt to load before training (classifier "
             "head is reinitialized to match the target class count).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Where to write checkpoints, history JSON, and config snapshot. "
             "Defaults to checkpoints/<config-stem>_<timestamp>/.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(
            "[error] CUDA is not available. The reported_runs/ configs assume "
            "a GPU (T4 / A100 in the paper). Aborting.",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = _load_config(args.config)
    config_stem = Path(args.config).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir or f"checkpoints/{config_stem}_{timestamp}")
    _maybe_write_run_metadata(out_dir, cfg, args.config)

    print(f"=== {config_stem} ===")
    meta = cfg.get("meta", {})
    if meta:
        print(f"Paper provenance: {meta.get('paper_table', '(unspecified)')}")
        if "reported_mca" in meta:
            mca = meta["reported_mca"]
            print(f"Reported MCA (Fusion / RGB-only / Depth-only): "
                  f"{mca.get('fusion')} / {mca.get('rgb_only')} / {mca.get('depth_only')}")
    print(f"Output directory: {out_dir}")
    print()

    # Build dataset
    train_loader, eval_loader, dataset_num_classes = _build_dataloaders(cfg, args.data_root)

    # Resolve num_classes (config may leave it null for ScanNet)
    num_classes = cfg["model"].get("num_classes") or dataset_num_classes
    if num_classes is None:
        raise ValueError(
            "num_classes is unset both in the config and on the dataset; "
            "cannot build the classifier head."
        )

    # Build model
    model = _build_model(cfg, num_classes)
    if args.pretrained:
        _load_pretrained(model, args.pretrained)

    # Optimizer + scheduler
    optimizer, scheduler = _build_optimizer_and_scheduler(model, cfg, len(train_loader))

    # Compile
    aug = cfg["augmentation"]
    model.compile(
        optimizer=optimizer,
        scheduler=scheduler,
        loss="cross_entropy",
        label_smoothing=cfg["training"]["label_smoothing"],
        gpu_augmentation=False,
        rgb_aug_prob=aug["rgb_aug_prob"],
        rgb_aug_mag=aug["rgb_aug_mag"],
        depth_aug_prob=aug["depth_aug_prob"],
        depth_aug_mag=aug["depth_aug_mag"],
    )

    # Fit
    tcfg = cfg["training"]
    mdcfg = cfg.get("modality_dropout", {"enabled": False})

    history = model.fit(
        train_loader=train_loader,
        val_loader=eval_loader,
        epochs=tcfg["epochs"],
        verbose=True,
        save_path=str(out_dir / "best_model.pt"),
        early_stopping=tcfg.get("early_stopping", False),
        restore_best_weights=tcfg.get("restore_best_weights", False),
        monitor=tcfg.get("monitor", "val_mca"),
        grad_clip_norm=tcfg["grad_clip_norm"],
        modality_dropout=mdcfg.get("enabled", False),
        modality_dropout_start=mdcfg.get("start", 0),
        modality_dropout_ramp=mdcfg.get("ramp", 0),
        modality_dropout_rate=mdcfg.get("rate", 0.0),
        modality_dropout_schedule=mdcfg.get("schedule", "ramp_up"),
    )

    # Persist history + final weights so the run is auditable end-to-end
    final_ckpt = out_dir / "final_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "config": cfg,
            "history_keys": list(history.keys()),
        },
        final_ckpt,
    )
    history_jsonable: dict[str, Any] = {
        k: [float(x) for x in v] if isinstance(v, list) and v and isinstance(v[0], (int, float)) else v
        for k, v in history.items()
    }
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history_jsonable, f, indent=2, default=str)

    print()
    print(f"Done. Final checkpoint: {final_ckpt}")
    print(f"Training history JSON: {out_dir / 'training_history.json'}")


if __name__ == "__main__":
    main()

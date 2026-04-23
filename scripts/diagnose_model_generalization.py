"""Quantify per-class generalization and feature-level divergence for a trained
LINet3 checkpoint across: (a) full training pool, (b) reconstructed HPO val,
(c) official test.

Outputs:
  - overall acc/MCA/loss per split
  - per-class accuracy CSV (aligned across all three splits on the same weights)
  - confusion matrix (official test) as CSV + PNG
  - per-layer linear MMD for penultimate features (train<->val, train<->test, val<->test)
  - per-class penultimate MMD train_c <-> test_c (which classes diverge most)
  - histogram of predicted-class probabilities for correct vs wrong (test)

Usage:
    python3 scripts/diagnose_model_generalization.py \
        --checkpoint /path/to/best_model.pt \
        --data-root data/sunrgbd_19_traintest \
        --seed 152 \
        --out-dir reports/diagnostics
"""

import argparse
import csv
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset

from src.data_utils.sunrgbd_dataset import SUNRGBDDataset, _load_norm_stats
from src.models.linear_integration.li_net3 import li_resnet18


def reconstruct_hpo_val_indices(data_root, seed=152, n_splits=5):
    with open(os.path.join(data_root, "train", "scene_groups.json")) as f:
        scene_groups = json.load(f)
    with open(os.path.join(data_root, "train", "labels.txt")) as f:
        labels = [int(x) for x in f.read().strip().splitlines()]
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf.split(range(len(labels)), labels, scene_groups))
    return sorted(train_idx.tolist()), sorted(val_idx.tolist()), labels


def build_dataset_noaug(data_root, split):
    """Eval-mode dataset: no augmentation, CenterCrop, in-dataloader normalization."""
    ds = SUNRGBDDataset(data_root=data_root, split=split, normalize=True)
    ds.split = "val"  # forces __getitem__ to skip augmentation
    return ds


def build_model(num_classes, dropout_p, width_multiplier, device):
    model = li_resnet18(
        num_classes=num_classes,
        stream_input_channels=[3, 1],
        dropout_p=dropout_p,
        width_multiplier=width_multiplier,
        device=device,
        use_amp=True,
    )
    return model


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt
    else:
        raise RuntimeError(f"Unrecognized checkpoint format at {ckpt_path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [load] missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"  [load] unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")


@torch.no_grad()
def eval_collect(model, loader, device, num_classes, hook_module):
    """Run the model on loader and collect predictions, labels, and penultimate features
    captured via a forward hook on `hook_module` (expects a tensor input).

    Returns dict with arrays: preds, labels, probs (for predicted class), features.
    Also returns summary metrics (loss, acc, mca).
    """
    model.eval()

    captured = {"features": []}

    def _hook(module, inputs, output):
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        if x.ndim > 2:
            x = torch.flatten(x, 1)
        captured["features"].append(x.detach().float().cpu())

    handle = hook_module.register_forward_hook(_hook)

    all_preds, all_labels, all_probs, all_logits_max = [], [], [], []
    total_loss, total_n = 0.0, 0
    criterion = model.criterion if getattr(model, "criterion", None) is not None else torch.nn.CrossEntropyLoss()

    for batch in loader:
        *streams, labels = batch
        streams = [s.to(device, non_blocking=True) for s in streams]
        labels = labels.to(device, non_blocking=True)
        # Use AMP-safe path; gpu_augmentation is expected to be OFF (normalize=True in dataset)
        if getattr(model, "gpu_augmentation", False) and getattr(model, "gpu_aug", None) is not None:
            streams[0], streams[1] = model.gpu_aug(streams[0], streams[1])
        with torch.amp.autocast(device_type=device.type, enabled=getattr(model, "use_amp", False)):
            logits = model(streams)
            loss = criterion(logits, labels)
        probs = torch.softmax(logits.float(), dim=1)
        preds = probs.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.gather(1, preds.unsqueeze(1)).squeeze(1).cpu().numpy())
        all_logits_max.append(logits.max(dim=1).values.cpu().numpy())

        total_loss += loss.item() * labels.size(0)
        total_n += labels.size(0)

    handle.remove()

    preds = np.concatenate(all_preds)
    labels_np = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)
    features = torch.cat(captured["features"], dim=0).numpy()

    # Overall metrics
    acc = float((preds == labels_np).mean())
    per_class_acc = np.zeros(num_classes, dtype=np.float64)
    per_class_n = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        mask = labels_np == c
        per_class_n[c] = int(mask.sum())
        per_class_acc[c] = float((preds[mask] == c).mean()) if mask.any() else float("nan")
    mca = float(np.nanmean(per_class_acc))
    loss = total_loss / max(1, total_n)

    return {
        "preds": preds,
        "labels": labels_np,
        "probs": probs,
        "features": features,
        "acc": acc,
        "mca": mca,
        "loss": loss,
        "per_class_acc": per_class_acc,
        "per_class_n": per_class_n,
    }


def linear_mmd_sq(x, y):
    """Squared MMD with linear kernel = ||mean(x) - mean(y)||^2."""
    return float(np.sum((x.mean(axis=0) - y.mean(axis=0)) ** 2))


def rbf_mmd_sq(x, y, gamma=None, max_samples=1024):
    """Biased squared MMD with RBF kernel. Subsampled for speed."""
    rng = np.random.default_rng(0)
    xi = rng.choice(len(x), size=min(max_samples, len(x)), replace=False)
    yi = rng.choice(len(y), size=min(max_samples, len(y)), replace=False)
    x = x[xi]
    y = y[yi]
    if gamma is None:
        # median heuristic on combined set
        z = np.concatenate([x, y], axis=0)
        # sub-sample pairwise distances to keep it cheap
        m = min(512, len(z))
        zi = rng.choice(len(z), size=m, replace=False)
        d2 = np.sum((z[zi][:, None, :] - z[zi][None, :, :]) ** 2, axis=-1)
        med = np.median(d2[d2 > 0])
        gamma = 1.0 / med if med > 0 else 1.0

    def k(a, b):
        d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return np.exp(-gamma * d2)

    kxx = k(x, x).mean()
    kyy = k(y, y).mean()
    kxy = k(x, y).mean()
    return float(kxx + kyy - 2 * kxy), float(gamma)


def confusion_matrix(preds, labels, num_classes):
    m = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        m[t, p] += 1
    return m


def plot_per_class_acc(per_class_accs, per_class_ns, names, out_path, title=""):
    splits = list(per_class_accs.keys())
    x = np.arange(len(names))
    width = 0.8 / len(splits)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    for i, s in enumerate(splits):
        ax1.bar(x + (i - (len(splits) - 1) / 2) * width,
                per_class_accs[s], width, label=s)
    ax1.set_ylabel("Per-class accuracy")
    ax1.set_ylim(0, 1)
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # sample counts on test to contextualize
    test_n = per_class_ns.get("official_test", np.zeros(len(names)))
    train_n = per_class_ns.get("hpo_train", np.zeros(len(names)))
    ax2.bar(x - width / 2, train_n, width, label="hpo_train n", color="tab:blue", alpha=0.7)
    ax2.bar(x + width / 2, test_n, width, label="test n", color="tab:green", alpha=0.7)
    ax2.set_ylabel("samples")
    ax2.legend()
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confusion(cm, names, out_path):
    with np.errstate(invalid="ignore"):
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Test confusion (row-normalized)")
    for i in range(len(names)):
        for j in range(len(names)):
            v = cm[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black",
                        fontsize=7)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="data/sunrgbd_19_traintest")
    p.add_argument("--seed", type=int, default=152)
    p.add_argument("--dropout-p", type=float, default=0.69)
    p.add_argument("--width-multiplier", type=float, default=0.75)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out-dir", default="reports/diagnostics")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(os.path.join(args.data_root, "class_names.txt")) as f:
        names = [line.strip() for line in f if line.strip()]
    num_classes = len(names)

    # Reconstruct HPO split on train pool
    hpo_train_idx, hpo_val_idx, _ = reconstruct_hpo_val_indices(args.data_root, seed=args.seed)
    print(f"HPO split: train={len(hpo_train_idx)}, val={len(hpo_val_idx)}")

    # Datasets (all eval-mode, no aug)
    train_ds = build_dataset_noaug(args.data_root, split="train")
    test_ds = build_dataset_noaug(args.data_root, split="test")
    hpo_train_subset = Subset(train_ds, hpo_train_idx)
    hpo_val_subset = Subset(train_ds, hpo_val_idx)

    def make_loader(ds):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    loaders = {
        "hpo_train": make_loader(hpo_train_subset),
        "hpo_val": make_loader(hpo_val_subset),
        "official_test": make_loader(test_ds),
    }

    # Model
    model = build_model(num_classes, args.dropout_p, args.width_multiplier, device)
    # compile minimally (sets criterion, etc.) but no optimizer update
    norm_stats = _load_norm_stats(args.data_root)  # loaded but not used when normalize is in-dataset
    _ = norm_stats
    model.compile(
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
        scheduler=None,
        loss="cross_entropy",
        label_smoothing=0.0,
        gpu_augmentation=False,
    )
    load_checkpoint(model, args.checkpoint, device)
    model.to(device)

    # Hook the input of fc to capture the penultimate (integrated) feature vector
    hook_module = model.fc

    # Evaluate on each split
    results = {}
    for name, loader in loaders.items():
        print(f"\nEvaluating on {name}...")
        results[name] = eval_collect(model, loader, device, num_classes, hook_module)
        print(f"  loss={results[name]['loss']:.4f}  acc={results[name]['acc']*100:.2f}%"
              f"  mca={results[name]['mca']*100:.2f}%  n={len(results[name]['labels'])}")

    # Per-class accuracy CSV
    pc_csv = os.path.join(args.out_dir, "per_class_accuracy.csv")
    with open(pc_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["class_idx", "class_name"]
        for s in loaders:
            header += [f"{s}_acc", f"{s}_n"]
        header += ["train_test_n_ratio", "acc_gap_val_minus_test"]
        w.writerow(header)
        for c in range(num_classes):
            row = [c, names[c]]
            for s in loaders:
                row += [f"{results[s]['per_class_acc'][c]:.4f}",
                        int(results[s]['per_class_n'][c])]
            tn = results["hpo_train"]["per_class_n"][c]
            ten = results["official_test"]["per_class_n"][c]
            ratio = (tn / ten) if ten > 0 else float("inf")
            gap = results["hpo_val"]["per_class_acc"][c] - results["official_test"]["per_class_acc"][c]
            row += [f"{ratio:.3f}", f"{gap:+.4f}"]
            w.writerow(row)
    print(f"\nWrote {pc_csv}")

    # Confusion matrix on test
    cm = confusion_matrix(results["official_test"]["preds"],
                          results["official_test"]["labels"], num_classes)
    cm_csv = os.path.join(args.out_dir, "test_confusion_matrix.csv")
    with open(cm_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + [f"pred_{n}" for n in names])
        for i, row in enumerate(cm):
            w.writerow([f"true_{names[i]}"] + row.tolist())
    plot_confusion(cm, names, os.path.join(args.out_dir, "test_confusion.png"))

    # Per-class accuracy plot
    plot_per_class_acc(
        {s: results[s]["per_class_acc"] for s in loaders},
        {s: results[s]["per_class_n"] for s in loaders},
        names,
        os.path.join(args.out_dir, "per_class_accuracy.png"),
        title=f"Per-class accuracy: {os.path.basename(args.checkpoint)}",
    )

    # Penultimate MMD (linear + RBF)
    feats = {s: results[s]["features"] for s in loaders}
    mmd_summary = {}
    pairs = [("hpo_train", "hpo_val"), ("hpo_train", "official_test"),
             ("hpo_val", "official_test")]
    for a, b in pairs:
        lin = linear_mmd_sq(feats[a], feats[b])
        rbf, gamma = rbf_mmd_sq(feats[a], feats[b])
        mmd_summary[f"{a}__{b}"] = {"linear_mmd_sq": lin, "rbf_mmd_sq": rbf, "rbf_gamma": gamma}
        print(f"MMD {a} vs {b}: linear^2={lin:.4f}  RBF^2={rbf:.4f}  (gamma={gamma:.3e})")

    # Per-class penultimate MMD, train vs test
    per_class_mmd_linear = np.zeros(num_classes)
    per_class_mmd_rbf = np.zeros(num_classes)
    per_class_train_n = np.zeros(num_classes, dtype=np.int64)
    per_class_test_n = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        tr_mask = results["hpo_train"]["labels"] == c
        te_mask = results["official_test"]["labels"] == c
        ntr, nte = int(tr_mask.sum()), int(te_mask.sum())
        per_class_train_n[c] = ntr
        per_class_test_n[c] = nte
        if ntr < 5 or nte < 5:
            per_class_mmd_linear[c] = float("nan")
            per_class_mmd_rbf[c] = float("nan")
            continue
        xa = feats["hpo_train"][tr_mask]
        xb = feats["official_test"][te_mask]
        per_class_mmd_linear[c] = linear_mmd_sq(xa, xb)
        per_class_mmd_rbf[c], _ = rbf_mmd_sq(xa, xb, max_samples=256)

    pcmmd_csv = os.path.join(args.out_dir, "per_class_penultimate_mmd.csv")
    with open(pcmmd_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_idx", "class_name", "hpo_train_n", "test_n",
                    "linear_mmd_sq", "rbf_mmd_sq", "test_acc", "hpo_val_acc"])
        for c in range(num_classes):
            w.writerow([
                c, names[c], int(per_class_train_n[c]), int(per_class_test_n[c]),
                f"{per_class_mmd_linear[c]:.6f}", f"{per_class_mmd_rbf[c]:.6f}",
                f"{results['official_test']['per_class_acc'][c]:.4f}",
                f"{results['hpo_val']['per_class_acc'][c]:.4f}",
            ])
    print(f"Wrote {pcmmd_csv}")

    # Summary JSON
    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "seed": args.seed,
        "per_split_metrics": {
            s: {
                "n": int(len(results[s]["labels"])),
                "loss": results[s]["loss"],
                "accuracy": results[s]["acc"],
                "mca": results[s]["mca"],
            } for s in loaders
        },
        "penultimate_mmd": mmd_summary,
        "per_class_penultimate_mmd_csv": os.path.relpath(pcmmd_csv),
        "per_class_accuracy_csv": os.path.relpath(pc_csv),
        "class_names": names,
    }
    with open(os.path.join(args.out_dir, "model_generalization_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Console: print most-divergent classes sorted by RBF MMD train<->test
    order = np.argsort(-np.nan_to_num(per_class_mmd_rbf, nan=-1.0))
    print("\n=== Classes ranked by train<->test penultimate RBF MMD (higher = more divergent) ===")
    print(f"  {'class':<18}{'train_n':>8}{'test_n':>8}{'rbf_mmd':>10}"
          f"{'test_acc':>10}{'val_acc':>10}{'acc_gap':>10}")
    for c in order:
        print(f"  {names[c]:<18}{int(per_class_train_n[c]):>8}{int(per_class_test_n[c]):>8}"
              f"{per_class_mmd_rbf[c]:>10.3e}"
              f"{results['official_test']['per_class_acc'][c]*100:>9.1f}%"
              f"{results['hpo_val']['per_class_acc'][c]*100:>9.1f}%"
              f"{(results['hpo_val']['per_class_acc'][c]-results['official_test']['per_class_acc'][c])*100:>+9.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()

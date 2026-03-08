"""
Dataset-level stream analysis tools for LINet3 multi-stream neural networks.

Provides stream redundancy analysis, per-class stream dominance,
layer-wise train vs test activation divergence, BN stats reset experiments,
and integration weight evolution visualization.

All analyzers are N-stream compatible (no hardcoded stream names).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from collections import OrderedDict
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _default_stream_labels(num_streams: int, stream_labels: Optional[dict[int, str]] = None) -> dict[int, str]:
    labels = {i: f"Stream {i}" for i in range(num_streams)}
    if stream_labels is not None:
        labels.update(stream_labels)
    return labels


def _save_or_show(fig: plt.Figure, save_path: Optional[str] = None) -> None:
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 2F: Stream Redundancy Analysis
# ---------------------------------------------------------------------------

class StreamRedundancyAnalyzer:
    """Compute pairwise cosine similarity between stream feature maps at each layer.

    Uses centered cosine similarity (subtracts channel mean) to avoid inflation
    from shared near-zero background regions.
    """

    def __init__(self, model: nn.Module, stream_labels: Optional[dict[int, str]] = None):
        self.model = model
        self.device = _get_device(model)
        self.num_streams = model.num_streams
        self.labels = _default_stream_labels(self.num_streams, stream_labels)
        self._captured = {}
        self._hooks = []

    def _register_hooks(self, layer_names: list[str]) -> None:
        """Register forward hooks on specified layers."""
        modules = dict(self.model.named_modules())
        for lname in layer_names:
            if lname not in modules:
                continue
            module = modules[lname]

            def make_hook(name):
                def hook_fn(mod, inp, out):
                    if isinstance(out, tuple) and len(out) == 2:
                        self._captured[name] = [s.detach().cpu() for s in out[0]]
                return hook_fn

            h = module.register_forward_hook(make_hook(lname))
            self._hooks.append(h)

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._captured.clear()

    @torch.no_grad()
    def analyze(
        self,
        dataloader: DataLoader,
        layer_names: Optional[list[str]] = None,
        n: int = 64,
        save_path: Optional[str] = None,
    ) -> dict[str, np.ndarray]:
        """Compute per-layer pairwise centered cosine similarity between streams.

        Args:
            dataloader: DataLoader yielding (stream1, ..., streamN, labels).
            layer_names: layers to analyze (default: layer1-4).
            n: number of samples to average over.
            save_path: optional save path for plot.

        Returns:
            dict mapping layer_name -> [num_streams, num_streams] similarity matrix.
        """
        if layer_names is None:
            layer_names = ["layer1", "layer2", "layer3", "layer4"]

        self.model.eval()
        self._register_hooks(layer_names)

        # Accumulate flattened stream features for centered cosine similarity.
        # Use streaming vector sums to avoid storing all features.
        # For centered cosine between vectors x, y across N samples:
        #   centered_dot = sum_n(x_n · y_n) - (1/N) * sum_n(x_n) · sum_n(y_n)
        #   var_x = sum_n(x_n · x_n) - (1/N) * sum_n(x_n) · sum_n(x_n)
        #   cos = centered_dot / sqrt(var_x * var_y)
        # We accumulate: dot[i,j] (scalar), sum_vec[i] (vector D), sq_norm[i] (scalar), count.
        accum = {}  # Initialized lazily once we know feature dim

        total = 0
        for batch_data in dataloader:
            *stream_batches, targets = batch_data
            stream_batches = [s.to(self.device) for s in stream_batches]

            self.model(stream_batches)

            for lname in layer_names:
                if lname not in self._captured:
                    continue
                streams = self._captured[lname]
                # Flatten spatial dims: [B, C, H, W] -> [B, C*H*W]
                flat = [s.flatten(1).numpy() for s in streams]
                B = flat[0].shape[0]
                D = flat[0].shape[1]

                # Lazy init
                if lname not in accum:
                    accum[lname] = {
                        "dot": np.zeros((self.num_streams, self.num_streams)),
                        "sum_vec": [np.zeros(D) for _ in range(self.num_streams)],
                        "sq_norm": np.zeros(self.num_streams),
                        "count": 0,
                    }

                for b in range(B):
                    if total + b >= n:
                        break
                    accum[lname]["count"] += 1
                    for si in range(self.num_streams):
                        xi = flat[si][b]
                        accum[lname]["sum_vec"][si] += xi
                        accum[lname]["sq_norm"][si] += (xi * xi).sum()
                        for sj in range(si, self.num_streams):
                            xj = flat[sj][b]
                            accum[lname]["dot"][si, sj] += (xi * xj).sum()

            total += stream_batches[0].shape[0]
            self._captured.clear()
            if total >= n:
                break

        self._remove_hooks()

        # Compute centered cosine similarity
        results = {}
        for lname in layer_names:
            if lname not in accum:
                continue
            a = accum[lname]
            N = a["count"]
            if N == 0:
                continue
            sim = np.zeros((self.num_streams, self.num_streams))
            for si in range(self.num_streams):
                for sj in range(si, self.num_streams):
                    # centered_dot = sum_n(xi·xj) - (1/N) * sum_n(xi) · sum_n(xj)
                    dot_ij = a["dot"][si, sj]
                    cross_mean = np.dot(a["sum_vec"][si], a["sum_vec"][sj]) / N
                    centered_dot = dot_ij - cross_mean

                    var_i = a["sq_norm"][si] - np.dot(a["sum_vec"][si], a["sum_vec"][si]) / N
                    var_j = a["sq_norm"][sj] - np.dot(a["sum_vec"][sj], a["sum_vec"][sj]) / N

                    denom = np.sqrt(max(var_i, 1e-8) * max(var_j, 1e-8))
                    cos_sim = float(centered_dot / denom)
                    sim[si, sj] = cos_sim
                    sim[sj, si] = cos_sim
            results[lname] = sim

        if results:
            self._plot(results, save_path)

        return results

    def _plot(self, results: dict[str, np.ndarray], save_path: Optional[str]) -> None:
        """Plot similarity matrices and similarity vs depth."""
        n_layers = len(results)
        fig, axes = plt.subplots(1, n_layers + 1, figsize=((n_layers + 1) * 3, 3))

        stream_names = [self.labels[i] for i in range(self.num_streams)]

        # Heatmaps
        for i, (lname, sim) in enumerate(results.items()):
            ax = axes[i]
            im = ax.imshow(sim, vmin=-1, vmax=1, cmap="RdBu_r")
            ax.set_xticks(range(self.num_streams))
            ax.set_xticklabels(stream_names, fontsize=7, rotation=45)
            ax.set_yticks(range(self.num_streams))
            ax.set_yticklabels(stream_names, fontsize=7)
            ax.set_title(lname, fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046)

        # Similarity vs depth (off-diagonal pairs)
        ax = axes[-1]
        layers = list(results.keys())
        for si in range(self.num_streams):
            for sj in range(si + 1, self.num_streams):
                vals = [results[l][si, sj] for l in layers]
                ax.plot(range(len(layers)), vals,
                        label=f"{self.labels[si]}-{self.labels[sj]}", marker="o")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, fontsize=8)
        ax.set_ylabel("Centered Cosine Similarity")
        ax.set_title("Similarity vs Depth")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        fig.suptitle("Stream Redundancy Analysis", fontsize=12)
        plt.tight_layout()
        _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 2G: Per-Class Stream Dominance
# ---------------------------------------------------------------------------

class PerClassDominanceAnalyzer:
    """Aggregate stream contribution dominance maps by ground-truth class.

    Uses LIConv2d._capture_contributions to decompose per-stream contributions,
    then aggregates mean contribution ratios per class across the dataset.
    """

    def __init__(self, model: nn.Module, stream_labels: Optional[dict[int, str]] = None):
        self.model = model
        self.device = _get_device(model)
        self.num_streams = model.num_streams
        self.labels = _default_stream_labels(self.num_streams, stream_labels)

    @torch.no_grad()
    def analyze(
        self,
        dataloader: DataLoader,
        layer: str = "layer4",
        class_names: Optional[dict[int, str]] = None,
        save_path: Optional[str] = None,
    ) -> dict[int, np.ndarray]:
        """Compute per-class mean contribution ratios for each stream.

        Args:
            dataloader: DataLoader yielding (stream1, ..., streamN, labels).
            layer: target layer for contribution decomposition.
            class_names: optional {class_idx: name} mapping.
            save_path: optional save path.

        Returns:
            dict mapping class_idx -> [num_streams] array of mean contribution ratios.
        """
        self.model.eval()

        # Enable capture on target layer
        for name, module in self.model.named_modules():
            if name.startswith(layer) and hasattr(module, "_capture_contributions"):
                module._capture_contributions = True

        # Per-class accumulators: class_idx -> [num_streams] running sum, count
        class_sums = {}
        class_counts = {}

        for batch_data in dataloader:
            *stream_batches, targets = batch_data
            stream_batches = [s.to(self.device) for s in stream_batches]
            targets = targets.numpy()

            self.model(stream_batches)

            # Collect contributions from last conv in layer
            contribs = []
            for name, module in self.model.named_modules():
                if name.startswith(layer) and hasattr(module, "_last_contributions") and module._last_contributions is not None:
                    contribs.append(module._last_contributions)

            if not contribs:
                continue

            last = contribs[-1]
            batch_size = last["stream_contributions"][0].shape[0]

            for b in range(batch_size):
                cls = int(targets[b])
                # Compute per-stream total magnitude for this sample
                magnitudes = np.array([
                    last["stream_contributions"][si][b].norm().item()
                    for si in range(self.num_streams)
                ])
                total = magnitudes.sum() + 1e-8
                ratios = magnitudes / total

                if cls not in class_sums:
                    class_sums[cls] = np.zeros(self.num_streams)
                    class_counts[cls] = 0
                class_sums[cls] += ratios
                class_counts[cls] += 1

        # Disable capture
        for module in self.model.modules():
            if hasattr(module, "_capture_contributions"):
                module._capture_contributions = False
                module._last_contributions = None

        # Compute means
        results = {}
        for cls in sorted(class_sums.keys()):
            results[cls] = class_sums[cls] / class_counts[cls]

        if results:
            self._plot(results, class_names, save_path)

        return results

    def _plot(self, results: dict[int, np.ndarray],
              class_names: Optional[dict[int, str]],
              save_path: Optional[str]) -> None:
        """Plot per-class stream dominance as stacked bar chart."""
        classes = sorted(results.keys())
        n_classes = len(classes)
        data = np.array([results[c] for c in classes])  # [n_classes, num_streams]

        fig, ax = plt.subplots(figsize=(max(8, n_classes * 0.6), 5))

        x = np.arange(n_classes)
        bottom = np.zeros(n_classes)

        colors = plt.cm.tab10(np.linspace(0, 1, self.num_streams))
        for si in range(self.num_streams):
            ax.bar(x, data[:, si], bottom=bottom, color=colors[si], label=self.labels[si])
            bottom += data[:, si]

        if class_names:
            tick_labels = [class_names.get(c, str(c)) for c in classes]
        else:
            tick_labels = [str(c) for c in classes]

        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean Contribution Ratio")
        ax.set_title("Per-Class Stream Dominance")
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 2I: Layer-wise Train vs Test Activation Statistics
# ---------------------------------------------------------------------------

class ActivationDivergenceAnalyzer:
    """Compare train vs test activation distributions per layer.

    Uses MMD (Maximum Mean Discrepancy) as the divergence metric.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.device = _get_device(model)

    @torch.no_grad()
    def analyze(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        layer_names: Optional[list[str]] = None,
        n: int = 128,
        save_path: Optional[str] = None,
    ) -> dict[str, dict]:
        """Compare train vs test activation statistics per layer.

        Args:
            train_loader: training DataLoader.
            test_loader: test DataLoader.
            layer_names: layers to analyze (default: layer1-4).
            n: samples per split.
            save_path: optional save path.

        Returns:
            dict mapping layer_name -> {train_mean, train_std, test_mean, test_std, mmd}.
        """
        if layer_names is None:
            layer_names = ["layer1", "layer2", "layer3", "layer4"]

        self.model.eval()

        train_stats = self._collect_stats(train_loader, layer_names, n)
        test_stats = self._collect_stats(test_loader, layer_names, n)

        results = {}
        for lname in layer_names:
            if lname not in train_stats or lname not in test_stats:
                continue
            ts = train_stats[lname]
            es = test_stats[lname]

            # MMD with linear kernel (mean embedding difference)
            mmd = float(np.linalg.norm(ts["mean_features"] - es["mean_features"]))

            results[lname] = {
                "train_mean_activation": float(ts["mean_act"]),
                "train_std_activation": float(ts["std_act"]),
                "test_mean_activation": float(es["mean_act"]),
                "test_std_activation": float(es["std_act"]),
                "mmd": mmd,
            }

        if results:
            self._plot(results, save_path)

        return results

    def _collect_stats(self, dataloader: DataLoader, layer_names: list[str], n: int) -> dict:
        """Collect activation statistics for specified layers."""
        captured = {}
        hooks = []
        modules = dict(self.model.named_modules())

        for lname in layer_names:
            if lname not in modules:
                continue

            def make_hook(name):
                def hook_fn(mod, inp, out):
                    if isinstance(out, tuple):
                        act = out[1]  # integrated
                    else:
                        act = out
                    captured[name] = act.detach().cpu()
                return hook_fn

            h = modules[lname].register_forward_hook(make_hook(lname))
            hooks.append(h)

        # Accumulate statistics
        accum = {lname: {"acts": [], "features": []} for lname in layer_names}
        total = 0

        for batch_data in dataloader:
            *stream_batches, targets = batch_data
            stream_batches = [s.to(self.device) for s in stream_batches]

            self.model(stream_batches)

            for lname in layer_names:
                if lname in captured:
                    act = captured[lname]
                    # Mean activation magnitude per sample
                    accum[lname]["acts"].append(act.mean(dim=(1, 2, 3)).numpy())
                    # GAP features for MMD
                    accum[lname]["features"].append(
                        F.adaptive_avg_pool2d(act, 1).flatten(1).numpy()
                    )

            captured.clear()
            total += stream_batches[0].shape[0]
            if total >= n:
                break

        for h in hooks:
            h.remove()

        result = {}
        for lname in layer_names:
            if accum[lname]["acts"]:
                acts = np.concatenate(accum[lname]["acts"])[:n]
                feats = np.concatenate(accum[lname]["features"])[:n]
                result[lname] = {
                    "mean_act": acts.mean(),
                    "std_act": acts.std(),
                    "mean_features": feats.mean(axis=0),
                }

        return result

    def _plot(self, results: dict, save_path: Optional[str]) -> None:
        """Plot train vs test activation comparison and MMD."""
        layers = list(results.keys())
        n = len(layers)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Mean activation comparison
        train_means = [results[l]["train_mean_activation"] for l in layers]
        test_means = [results[l]["test_mean_activation"] for l in layers]
        train_stds = [results[l]["train_std_activation"] for l in layers]
        test_stds = [results[l]["test_std_activation"] for l in layers]

        x = np.arange(n)
        w = 0.35
        axes[0].bar(x - w / 2, train_means, w, yerr=train_stds, label="Train", alpha=0.8)
        axes[0].bar(x + w / 2, test_means, w, yerr=test_stds, label="Test", alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(layers, fontsize=9)
        axes[0].set_ylabel("Mean Activation")
        axes[0].set_title("Train vs Test Activations")
        axes[0].legend()

        # MMD
        mmds = [results[l]["mmd"] for l in layers]
        axes[1].bar(x, mmds, color="coral")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(layers, fontsize=9)
        axes[1].set_ylabel("MMD (linear kernel)")
        axes[1].set_title("Activation Divergence (MMD)")

        plt.tight_layout()
        _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 2I (continued): BN Stats Reset Experiment
# ---------------------------------------------------------------------------

def reset_bn_stats(
    model: nn.Module,
    data_loader: DataLoader,
    num_batches: Optional[int] = None,
) -> nn.Module:
    """Recompute BatchNorm running statistics on the provided data.

    DIAGNOSTIC TOOL ONLY. When called with test data, this is an oracle
    experiment (test-set adaptation) that is NOT a deployable fix. It helps
    diagnose whether BN statistics drift is the source of a generalization gap.

    Usage:
        # Oracle experiment (diagnostic only):
        model_copy = reset_bn_stats(copy.deepcopy(model), test_loader)
        test_acc_oracle = evaluate(model_copy, test_loader)
        # If accuracy jumps significantly, BN stats drift is the gap source.

        # Control test (should not change accuracy):
        model_copy = reset_bn_stats(copy.deepcopy(model), train_loader)
        test_acc_control = evaluate(model_copy, test_loader)
        # If this changes accuracy too, the function has a bug.

    Args:
        model: model to update (will be modified in-place).
        data_loader: data to compute BN stats from.
        num_batches: optional limit on batches to use.

    Returns:
        The model with updated BN running stats.
    """
    model.train()  # Set BN layers to training mode (updates running stats)

    # Freeze all parameters (we only want BN stats to update)
    for param in model.parameters():
        param.requires_grad_(False)

    # Reset running stats for both standard BN and LINet's custom _LIBatchNorm modules.
    # _LIBatchNorm inherits from _LINormBase (not nn.BatchNorm*), so we detect it
    # via the stream_num_features attribute unique to LI norm layers.
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.reset_running_stats()
        elif hasattr(module, 'stream_num_features') and hasattr(module, 'reset_running_stats'):
            module.reset_running_stats()

    device = _get_device(model)

    with torch.no_grad():
        for i, batch_data in enumerate(data_loader):
            if num_batches is not None and i >= num_batches:
                break
            *stream_batches, targets = batch_data
            stream_batches = [s.to(device) for s in stream_batches]
            model(stream_batches)

    model.eval()

    # Restore requires_grad
    for param in model.parameters():
        param.requires_grad_(True)

    return model


# ---------------------------------------------------------------------------
# 2J: Integration Weight Evolution Visualization
# ---------------------------------------------------------------------------

class IntegrationWeightEvolutionVisualizer:
    """Visualize how integration weights evolve during training.

    Reads data collected by fit() with track_integration_weights=True.
    Does NOT collect data itself -- purely a visualization tool.
    """

    def __init__(self, stream_labels: Optional[dict[int, str]] = None):
        self.stream_labels = stream_labels

    def plot_norm_evolution(
        self,
        history: dict,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot per-stream integration weight norms over epochs.

        Args:
            history: model training history dict containing 'integration_weight_norms'.
            save_path: optional save path.
        """
        norms_history = history.get("integration_weight_norms", [])
        if not norms_history:
            print("No integration weight norm data in history.")
            return

        # Organize by parameter name
        all_keys = set()
        for epoch_norms in norms_history:
            all_keys.update(epoch_norms.keys())

        # Group by layer (strip stream index)
        layer_groups = OrderedDict()
        for key in sorted(all_keys):
            parts = key.rsplit(".", 1)
            if len(parts) == 2:
                layer_prefix = parts[0]
                stream_idx = int(parts[1])
            else:
                layer_prefix = key
                stream_idx = 0

            if layer_prefix not in layer_groups:
                layer_groups[layer_prefix] = {}
            layer_groups[layer_prefix][stream_idx] = key

        # Determine num_streams from data
        num_streams = max(max(streams.keys()) for streams in layer_groups.values()) + 1
        labels = _default_stream_labels(num_streams, self.stream_labels)

        # Plot per-layer norm evolution
        n_layers = len(layer_groups)
        ncols = min(4, n_layers)
        nrows = (n_layers + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), squeeze=False)

        epochs = range(len(norms_history))

        for idx, (layer_prefix, streams) in enumerate(layer_groups.items()):
            ax = axes[idx // ncols, idx % ncols]
            for si, key in sorted(streams.items()):
                vals = [nh.get(key, 0.0) for nh in norms_history]
                ax.plot(epochs, vals, label=labels.get(si, f"S{si}"), marker=".", markersize=3)
            ax.set_title(layer_prefix.replace(".", "\n"), fontsize=7)
            ax.set_xlabel("Epoch", fontsize=8)
            ax.set_ylabel("L2 Norm", fontsize=8)
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for idx in range(n_layers, nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        fig.suptitle("Integration Weight Norm Evolution", fontsize=12)
        plt.tight_layout()
        _save_or_show(fig, save_path)

    def plot_snapshot_heatmaps(
        self,
        snapshot_dir: str,
        epochs: Optional[list[int]] = None,
        layer_filter: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Visualize full weight snapshot evolution from disk.

        Args:
            snapshot_dir: directory containing epoch_N_integration_weights.pt files.
            epochs: specific epochs to plot (default: all found).
            layer_filter: only show layers containing this string.
            save_path: optional save path.
        """
        import os
        import glob

        pattern = os.path.join(snapshot_dir, "epoch_*_integration_weights.pt")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"No integration weight snapshots found in {snapshot_dir}")
            return

        # Parse epoch numbers and filter
        file_epochs = []
        for f in files:
            base = os.path.basename(f)
            epoch_str = base.split("_")[1]
            file_epochs.append((int(epoch_str), f))

        if epochs is not None:
            file_epochs = [(e, f) for e, f in file_epochs if e in epochs]

        if not file_epochs:
            print("No matching snapshots found.")
            return

        # Load first snapshot to determine structure
        first_snapshot = torch.load(file_epochs[0][1], map_location="cpu", weights_only=True)
        param_names = sorted(first_snapshot.keys())
        if layer_filter:
            param_names = [p for p in param_names if layer_filter in p]

        if not param_names:
            print(f"No parameters match filter '{layer_filter}'.")
            return

        n_epochs = len(file_epochs)
        n_params = len(param_names)
        fig, axes = plt.subplots(n_params, n_epochs, figsize=(n_epochs * 3, n_params * 2.5), squeeze=False)

        for ei, (epoch, fpath) in enumerate(file_epochs):
            snapshot = torch.load(fpath, map_location="cpu", weights_only=True)
            for pi, pname in enumerate(param_names):
                ax = axes[pi, ei]
                if pname in snapshot:
                    w = snapshot[pname].numpy().squeeze()
                    if w.ndim == 1:
                        w = w.reshape(-1, 1)
                    im = ax.imshow(w, aspect="auto", cmap="RdBu_r")
                    plt.colorbar(im, ax=ax, fraction=0.046)
                ax.set_title(f"E{epoch}", fontsize=8)
                if ei == 0:
                    ax.set_ylabel(pname.replace(".", "\n"), fontsize=6)
                ax.tick_params(labelsize=6)

        fig.suptitle("Integration Weight Evolution (Snapshots)", fontsize=12)
        plt.tight_layout()
        _save_or_show(fig, save_path)

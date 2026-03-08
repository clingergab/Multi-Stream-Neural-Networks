"""
Stream visualization tools for LINet3 multi-stream neural networks.

Provides feature map visualization, stream contribution decomposition,
stream-decomposed Grad-CAM, integration weight visualization, and helpers
for misclassification analysis and sample comparison.

All visualizers are N-stream compatible (no hardcoded stream names).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union
from collections import OrderedDict
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device(model: nn.Module) -> torch.device:
    """Get the device of a model."""
    return next(model.parameters()).device


def _default_stream_labels(num_streams: int, stream_labels: Optional[dict[int, str]] = None) -> dict[int, str]:
    """Build stream label mapping, filling in defaults for missing indices."""
    labels = {i: f"Stream {i}" for i in range(num_streams)}
    if stream_labels is not None:
        labels.update(stream_labels)
    return labels


def _percentile_normalize(tensor: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Clip to percentile range and normalize to [0, 1]."""
    vmin = np.percentile(tensor, lo)
    vmax = np.percentile(tensor, hi)
    if vmax - vmin < 1e-8:
        return np.zeros_like(tensor)
    return np.clip((tensor - vmin) / (vmax - vmin), 0, 1)


def _topk_channels(activation: torch.Tensor, k: int, mode: str = "l2") -> list[int]:
    """Select top-k channel indices by L2 norm (default) or variance.

    Args:
        activation: [C, H, W] tensor (single image) or [C,] norms.
        k: number of channels to select.
        mode: "l2" or "variance" (variance only works for batch stats).
    """
    if activation.dim() == 3:
        C = activation.shape[0]
        if mode == "l2":
            scores = activation.flatten(1).norm(dim=1)
        else:
            scores = activation.flatten(1).var(dim=1)
    else:
        scores = activation
    k = min(k, len(scores))
    return scores.topk(k).indices.tolist()


def _save_or_show(fig: plt.Figure, save_path: Optional[str] = None) -> None:
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 2A: Feature Map Visualizer
# ---------------------------------------------------------------------------

class FeatureMapVisualizer:
    """Visualize feature maps at any layer of LINet3.

    Three viewing modes:
      1. Full model (all streams + integrated): normal forward pass.
      2. Single-stream isolated: uses model._forward_stream_pathway().
      3. Ablation: blanks a stream and runs full forward.
    """

    def __init__(self, model: nn.Module, stream_labels: Optional[dict[int, str]] = None):
        self.model = model
        self.device = _get_device(model)
        self.num_streams = model.num_streams
        self.labels = _default_stream_labels(self.num_streams, stream_labels)
        self._captured = {}
        self._hooks = []

    # -- hook management -----------------------------------------------------

    def _register_hook(self, layer_name: str) -> None:
        """Register a forward hook on the named layer."""
        layer = dict(self.model.named_modules())[layer_name]

        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) == 2:
                stream_outputs, integrated = output
                self._captured["streams"] = [s.detach().cpu() for s in stream_outputs]
                self._captured["integrated"] = integrated.detach().cpu()
            else:
                self._captured["raw"] = output.detach().cpu() if isinstance(output, torch.Tensor) else output

        handle = layer.register_forward_hook(hook_fn)
        self._hooks.append(handle)

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._captured.clear()

    # -- public API ----------------------------------------------------------

    @torch.no_grad()
    def visualize(
        self,
        stream_inputs: list[torch.Tensor],
        layer: str = "layer4",
        mode: str = "full",
        stream_idx: Optional[int] = None,
        top_k: int = 8,
        channel_mode: str = "l2",
        save_path: Optional[str] = None,
    ) -> None:
        """Visualize feature maps for a single image.

        Args:
            stream_inputs: list of [1, C, H, W] input tensors (one per stream).
            layer: layer name (e.g. 'layer4', 'layer2').
            mode: 'full' | 'stream' | 'ablation'.
            stream_idx: required for mode='stream' or 'ablation'.
            top_k: number of channels to show.
            channel_mode: 'l2' (default) or 'variance' (only for batch).
            save_path: optional path to save figure.
        """
        self.model.eval()

        if channel_mode == "variance":
            print("WARNING: Variance channel selection is unavailable for single images. Falling back to L2 norm.")
            channel_mode = "l2"

        inputs = [s.to(self.device) for s in stream_inputs]

        if mode == "full":
            self._register_hook(layer)
            self.model(inputs)
            self._plot_full(top_k, channel_mode, layer, save_path)
        elif mode == "stream":
            assert stream_idx is not None, "stream_idx required for mode='stream'"
            stream_x = self._forward_stream_to_layer(stream_idx, inputs[stream_idx], layer)
            self._plot_single("streams", stream_x, top_k, channel_mode,
                              f"{self.labels[stream_idx]} isolated @ {layer}", save_path)
        elif mode == "ablation":
            assert stream_idx is not None, "stream_idx required for mode='ablation'"
            blanked = {stream_idx: torch.ones(inputs[0].shape[0], dtype=torch.bool, device=self.device)}
            self._register_hook(layer)
            self.model(inputs, blanked_mask=blanked)
            title = f"Ablation (blanked {self.labels[stream_idx]}) @ {layer}"
            if "integrated" in self._captured:
                self._plot_single("integrated", self._captured["integrated"], top_k, channel_mode, title, save_path)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self._remove_hooks()

    @torch.no_grad()
    def visualize_batch(
        self,
        dataloader: DataLoader,
        layer: str = "layer4",
        n: int = 32,
        class_label: Optional[int] = None,
        top_k: int = 8,
        channel_mode: str = "l2",
        save_path: Optional[str] = None,
    ) -> None:
        """Visualize average feature maps over n samples (streaming mean).

        Args:
            dataloader: DataLoader yielding (stream1, ..., streamN, labels).
            layer: layer name.
            n: number of samples to average.
            class_label: optional class to filter.
            top_k: channels to show.
            channel_mode: 'l2' or 'variance'.
            save_path: optional save path.
        """
        self.model.eval()
        self._register_hook(layer)

        # Streaming statistics (Welford for variance, running mean)
        count = 0
        stream_means = None
        integrated_mean = None
        # Welford accumulators (for variance channel selection)
        stream_m2 = None
        integrated_m2 = None

        for batch_data in dataloader:
            *stream_batches, targets = batch_data
            stream_batches = [s.to(self.device) for s in stream_batches]
            targets = targets.to(self.device)

            if class_label is not None:
                mask = targets == class_label
                if not mask.any():
                    continue
                stream_batches = [s[mask] for s in stream_batches]

            self.model(stream_batches)

            if "streams" not in self._captured:
                break

            for sample_i in range(self._captured["integrated"].shape[0]):
                count += 1
                if count > n:
                    break

                # Update streaming stats for integrated
                int_val = self._captured["integrated"][sample_i]
                if integrated_mean is None:
                    integrated_mean = torch.zeros_like(int_val)
                    integrated_m2 = torch.zeros_like(int_val)
                delta = int_val - integrated_mean
                integrated_mean += delta / count
                integrated_m2 += delta * (int_val - integrated_mean)

                # Update streaming stats per stream
                if stream_means is None:
                    stream_means = [torch.zeros_like(self._captured["streams"][i][sample_i]) for i in range(self.num_streams)]
                    stream_m2 = [torch.zeros_like(s) for s in stream_means]
                for si in range(self.num_streams):
                    sv = self._captured["streams"][si][sample_i]
                    d = sv - stream_means[si]
                    stream_means[si] += d / count
                    stream_m2[si] += d * (sv - stream_means[si])

            if count >= n:
                break

        self._remove_hooks()

        if count == 0:
            print("No samples matched criteria.")
            return

        # Build channel scores for selection
        if channel_mode == "variance" and count > 1:
            int_var = (integrated_m2 / (count - 1)).mean(dim=(-2, -1))  # per-channel variance
            stream_vars = [(m / (count - 1)).mean(dim=(-2, -1)) for m in stream_m2]
        else:
            int_var = None
            stream_vars = None

        # Plot
        ncols = top_k
        nrows = self.num_streams + 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
        fig.suptitle(f"Batch-averaged feature maps @ {layer} (n={count})", fontsize=12)

        for si in range(self.num_streams):
            act = stream_means[si]
            scores = stream_vars[si] if stream_vars is not None else None
            ch_mode = "variance" if scores is not None else "l2"
            channels = _topk_channels(act if scores is None else scores, top_k, "l2")
            for ci, ch in enumerate(channels):
                ax = axes[si, ci] if nrows > 1 else axes[ci]
                ax.imshow(_percentile_normalize(act[ch].numpy()), cmap="viridis")
                ax.set_title(f"ch {ch}", fontsize=8)
                ax.axis("off")
            axes[si, 0].set_ylabel(self.labels[si], fontsize=9) if nrows > 1 else None

        # Integrated
        channels = _topk_channels(integrated_mean if int_var is None else int_var, top_k, "l2")
        for ci, ch in enumerate(channels):
            ax = axes[-1, ci] if nrows > 1 else axes[ci]
            ax.imshow(_percentile_normalize(integrated_mean[ch].numpy()), cmap="viridis")
            ax.set_title(f"ch {ch}", fontsize=8)
            ax.axis("off")
        axes[-1, 0].set_ylabel("Integrated", fontsize=9) if nrows > 1 else None

        plt.tight_layout()
        _save_or_show(fig, save_path)

    # -- internal helpers ----------------------------------------------------

    def _forward_stream_to_layer(self, stream_idx: int, stream_input: torch.Tensor, layer: str) -> torch.Tensor:
        """Forward a single stream and capture its activations at the specified layer."""
        captured = {}

        def hook_fn(module, input, output):
            captured["out"] = output.detach().cpu() if isinstance(output, torch.Tensor) else output

        target_layer = dict(self.model.named_modules()).get(layer)
        if target_layer is None:
            raise ValueError(f"Layer '{layer}' not found in model.")

        # Use forward_stream on the target layer
        stream_x = self.model.conv1.forward_stream(stream_idx, stream_input)
        stream_x = self.model.bn1.forward_stream(stream_idx, stream_x)
        stream_x = F.relu(stream_x, inplace=False)
        stream_x = self.model.maxpool.forward_stream(stream_idx, stream_x)

        layer_map = {"layer1": self.model.layer1, "layer2": self.model.layer2,
                     "layer3": self.model.layer3, "layer4": self.model.layer4}
        for lname, lmod in layer_map.items():
            stream_x = lmod.forward_stream(stream_idx, stream_x)
            if lname == layer:
                return stream_x.detach().cpu()

        raise ValueError(f"Layer '{layer}' not found in layer1-4.")

    def _plot_full(self, top_k: int, channel_mode: str, layer: str, save_path: Optional[str]) -> None:
        """Plot feature maps from a full-model forward pass."""
        nrows = self.num_streams + 1
        fig, axes = plt.subplots(nrows, top_k, figsize=(top_k * 2, nrows * 2))
        fig.suptitle(f"Feature maps @ {layer}", fontsize=12)

        for si in range(self.num_streams):
            act = self._captured["streams"][si][0]  # first sample
            channels = _topk_channels(act, top_k, channel_mode)
            for ci, ch in enumerate(channels):
                ax = axes[si, ci]
                ax.imshow(_percentile_normalize(act[ch].numpy()), cmap="viridis")
                ax.set_title(f"ch {ch}", fontsize=8)
                ax.axis("off")
            axes[si, 0].set_ylabel(self.labels[si], fontsize=9)

        int_act = self._captured["integrated"][0]
        channels = _topk_channels(int_act, top_k, channel_mode)
        for ci, ch in enumerate(channels):
            ax = axes[-1, ci]
            ax.imshow(_percentile_normalize(int_act[ch].numpy()), cmap="viridis")
            ax.set_title(f"ch {ch}", fontsize=8)
            ax.axis("off")
        axes[-1, 0].set_ylabel("Integrated", fontsize=9)

        plt.tight_layout()
        _save_or_show(fig, save_path)

    def _plot_single(self, key: str, activation: torch.Tensor, top_k: int,
                     channel_mode: str, title: str, save_path: Optional[str]) -> None:
        """Plot feature maps from a single activation tensor [B, C, H, W] or [C, H, W]."""
        if activation.dim() == 4:
            activation = activation[0]
        channels = _topk_channels(activation, top_k, channel_mode)
        fig, axes = plt.subplots(1, top_k, figsize=(top_k * 2, 2))
        fig.suptitle(title, fontsize=12)
        if top_k == 1:
            axes = [axes]
        for ci, ch in enumerate(channels):
            axes[ci].imshow(_percentile_normalize(activation[ch].numpy()), cmap="viridis")
            axes[ci].set_title(f"ch {ch}", fontsize=8)
            axes[ci].axis("off")
        plt.tight_layout()
        _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 2B: Stream Contribution Decomposition Visualizer
# ---------------------------------------------------------------------------

class StreamContributionVisualizer:
    """Visualize per-stream contributions to integrated pathway neurons.

    Uses LIConv2d._capture_contributions flag to decompose the integrated
    output into per-stream contribution tensors.
    """

    def __init__(self, model: nn.Module, stream_labels: Optional[dict[int, str]] = None):
        self.model = model
        self.device = _get_device(model)
        self.num_streams = model.num_streams
        self.labels = _default_stream_labels(self.num_streams, stream_labels)

    def _enable_capture(self, layer_name: str) -> None:
        """Enable contribution capture on LIConv2d modules within the target layer."""
        for name, module in self.model.named_modules():
            if name.startswith(layer_name) and hasattr(module, "_capture_contributions"):
                module._capture_contributions = True

    def _disable_capture(self) -> None:
        """Disable contribution capture on all LIConv2d modules."""
        for module in self.model.modules():
            if hasattr(module, "_capture_contributions"):
                module._capture_contributions = False
                module._last_contributions = None

    def _collect_contributions(self, layer_name: str) -> list[dict]:
        """Collect captured contributions from all LIConv2d modules in the layer."""
        results = []
        for name, module in self.model.named_modules():
            if name.startswith(layer_name) and hasattr(module, "_last_contributions") and module._last_contributions is not None:
                results.append({"name": name, **module._last_contributions})
        return results

    @torch.no_grad()
    def visualize(
        self,
        stream_inputs: list[torch.Tensor],
        layer: str = "layer4",
        aggregation: str = "l2",
        save_path: Optional[str] = None,
    ) -> Optional[dict]:
        """Visualize per-stream contribution maps for a single image.

        Args:
            stream_inputs: list of [1, C, H, W] tensors.
            layer: target layer name.
            aggregation: 'l2', 'mean', or 'max' across channels.
            save_path: optional save path.

        Returns:
            dict with 'stream_contributions' (list of [H, W] numpy arrays),
            'integrated_from_prev' ([H, W] or scalar), and 'dominance_map'.
        """
        self.model.eval()
        inputs = [s.to(self.device) for s in stream_inputs]

        self._enable_capture(layer)
        self.model(inputs)
        contribs = self._collect_contributions(layer)
        self._disable_capture()

        if not contribs:
            print(f"No contributions captured at layer '{layer}'.")
            return None

        # Use last conv in the layer (deepest)
        last = contribs[-1]
        return self._plot_contributions(last, aggregation, layer, save_path)

    @torch.no_grad()
    def visualize_batch(
        self,
        dataloader: DataLoader,
        layer: str = "layer4",
        n: int = 32,
        class_label: Optional[int] = None,
        aggregation: str = "l2",
        save_path: Optional[str] = None,
    ) -> Optional[dict]:
        """Visualize batch-averaged stream contributions (streaming mean)."""
        self.model.eval()
        self._enable_capture(layer)

        count = 0
        running_contribs = None  # list of [C, H, W] running means per stream

        for batch_data in dataloader:
            *stream_batches, targets = batch_data
            stream_batches = [s.to(self.device) for s in stream_batches]

            if class_label is not None:
                targets = targets.to(self.device)
                mask = targets == class_label
                if not mask.any():
                    continue
                stream_batches = [s[mask] for s in stream_batches]

            self.model(stream_batches)
            contribs = self._collect_contributions(layer)
            if not contribs:
                continue

            last = contribs[-1]
            for sample_i in range(last["stream_contributions"][0].shape[0]):
                count += 1
                if count > n:
                    break

                if running_contribs is None:
                    running_contribs = [sc[sample_i].clone() for sc in last["stream_contributions"]]
                else:
                    for si in range(self.num_streams):
                        delta = last["stream_contributions"][si][sample_i] - running_contribs[si]
                        running_contribs[si] += delta / count

            if count >= n:
                break

        self._disable_capture()

        if count == 0:
            print("No samples matched criteria.")
            return None

        # Build a fake single-sample contrib dict for plotting
        fake_last = {
            "stream_contributions": [rc.unsqueeze(0) for rc in running_contribs],
            "integrated_from_prev": 0,
            "name": f"{layer} (batch avg, n={count})",
        }
        return self._plot_contributions(fake_last, aggregation, layer, save_path)

    def sanity_check(self, stream_inputs: list[torch.Tensor], layer: str = "layer4") -> float:
        """Verify sum(contributions) + bias ≈ integrated_output. Returns max abs error."""
        self.model.eval()
        inputs = [s.to(self.device) for s in stream_inputs]

        self._enable_capture(layer)
        self.model(inputs)
        contribs = self._collect_contributions(layer)
        self._disable_capture()

        if not contribs:
            print("No contributions captured.")
            return float("inf")

        last = contribs[-1]
        reconstructed = sum(last["stream_contributions"])
        prev = last["integrated_from_prev"]
        if isinstance(prev, torch.Tensor):
            reconstructed = reconstructed + prev
        bias = last.get("integrated_bias")
        if bias is not None:
            reconstructed = reconstructed + bias.view(1, -1, 1, 1)

        # Get actual integrated output from the same conv
        for name, module in self.model.named_modules():
            if name == last["name"] and hasattr(module, "forward"):
                break

        # Re-run to get actual output
        self._enable_capture(layer)
        self.model(inputs)

        # Get the actual output via hook
        actual = None
        def capture_hook(mod, inp, out):
            nonlocal actual
            if isinstance(out, tuple):
                actual = out[1].detach().cpu()  # integrated
            else:
                actual = out.detach().cpu()

        target = dict(self.model.named_modules()).get(last["name"])
        if target is not None:
            h = target.register_forward_hook(capture_hook)
            self.model(inputs)
            h.remove()

        self._disable_capture()

        if actual is not None:
            error = (reconstructed - actual).abs().max().item()
            print(f"Sanity check: max abs error = {error:.2e}")
            return error
        print("Could not verify (target module not found).")
        return float("inf")

    # -- internal plotting ---------------------------------------------------

    def _aggregate_channels(self, tensor: torch.Tensor, method: str) -> np.ndarray:
        """Aggregate [B, C, H, W] or [C, H, W] across channels to [H, W]."""
        if tensor.dim() == 4:
            tensor = tensor[0]
        if method == "l2":
            return tensor.norm(dim=0).numpy()
        elif method == "mean":
            return tensor.mean(dim=0).numpy()
        elif method == "max":
            return tensor.max(dim=0)[0].numpy()
        raise ValueError(f"Unknown aggregation: {method}")

    def _plot_contributions(self, contrib_dict: dict, aggregation: str,
                            layer: str, save_path: Optional[str]) -> dict:
        """Plot per-stream contribution maps and dominance map."""
        stream_maps = []
        for si in range(self.num_streams):
            hmap = self._aggregate_channels(contrib_dict["stream_contributions"][si], aggregation)
            stream_maps.append(hmap)

        # Dominance map
        stacked = np.stack(stream_maps, axis=0)  # [N, H, W]
        winner = np.argmax(stacked, axis=0)  # [H, W]

        # Entropy map
        eps = 1e-8
        total = stacked.sum(axis=0, keepdims=True) + eps
        probs = stacked / total
        entropy = -(probs * np.log(probs + eps)).sum(axis=0)

        # Plot
        ncols = self.num_streams + 2  # streams + winner + entropy
        fig, axes = plt.subplots(1, ncols, figsize=(ncols * 3, 3))
        fig.suptitle(f"Stream contributions @ {layer} ({aggregation})", fontsize=12)

        for si in range(self.num_streams):
            axes[si].imshow(_percentile_normalize(stream_maps[si]), cmap="hot")
            axes[si].set_title(self.labels[si], fontsize=9)
            axes[si].axis("off")

        # Winner-takes-all with distinct colors
        cmap = plt.cm.get_cmap("tab10", self.num_streams)
        axes[-2].imshow(winner, cmap=cmap, vmin=0, vmax=self.num_streams - 1)
        axes[-2].set_title("Dominance", fontsize=9)
        axes[-2].axis("off")

        axes[-1].imshow(entropy, cmap="coolwarm")
        axes[-1].set_title("Entropy", fontsize=9)
        axes[-1].axis("off")

        # Barycentric coloring for 3 streams
        if self.num_streams == 3:
            fig2, ax2 = plt.subplots(1, 1, figsize=(4, 4))
            normalized = stacked / (total + eps)
            rgb_map = np.stack([
                _percentile_normalize(normalized[0]),
                _percentile_normalize(normalized[1]),
                _percentile_normalize(normalized[2]),
            ], axis=-1)
            ax2.imshow(rgb_map)
            ax2.set_title("Barycentric (R=S0, G=S1, B=S2)", fontsize=10)
            ax2.axis("off")
            plt.tight_layout()
            if save_path:
                bary_path = save_path.rsplit(".", 1)
                bary_path = f"{bary_path[0]}_barycentric.{bary_path[1]}" if len(bary_path) == 2 else save_path + "_barycentric"
                fig2.savefig(bary_path, dpi=150, bbox_inches="tight")
                plt.close(fig2)
            else:
                plt.show()

        plt.tight_layout()
        _save_or_show(fig, save_path)

        return {
            "stream_contributions": stream_maps,
            "dominance_map": winner,
            "entropy_map": entropy,
        }


# ---------------------------------------------------------------------------
# 2C: Stream-Decomposed Grad-CAM
# ---------------------------------------------------------------------------

class StreamGradCAM:
    """Grad-CAM extended for multi-stream LINet3.

    Three modes:
      - integrated: Standard Grad-CAM on integrated activations.
      - stream: Per-stream isolated Grad-CAM via auxiliary classifiers.
      - decomposed: Stream contributions weighted by integrated Grad-CAM.
    """

    def __init__(self, model: nn.Module, stream_labels: Optional[dict[int, str]] = None):
        self.model = model
        self.device = _get_device(model)
        self.num_streams = model.num_streams
        self.labels = _default_stream_labels(self.num_streams, stream_labels)
        self._contrib_viz = StreamContributionVisualizer(model, stream_labels)

    def _check_aux_classifier(self, stream_idx: int, dataloader: DataLoader) -> float:
        """Quick accuracy check on first batch for auxiliary classifier guard."""
        self.model.eval()
        with torch.no_grad():
            for batch_data in dataloader:
                *stream_batches, targets = batch_data
                stream_batches = [s.to(self.device) for s in stream_batches]
                targets = targets.to(self.device)

                features = self.model._forward_stream_pathway(stream_idx, stream_batches[stream_idx])
                logits = self.model.fc_streams[stream_idx](features)
                preds = logits.argmax(dim=1)
                acc = (preds == targets).float().mean().item()
                return acc
        return 0.0

    def visualize(
        self,
        stream_inputs: list[torch.Tensor],
        layer: str = "layer4",
        mode: str = "integrated",
        target_class: Optional[int] = None,
        stream_idx: Optional[int] = None,
        dataloader: Optional[DataLoader] = None,
        overlay_input: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Compute and visualize Grad-CAM.

        Args:
            stream_inputs: list of [1, C, H, W] tensors.
            layer: target layer for Grad-CAM.
            mode: 'integrated', 'stream', or 'decomposed'.
            target_class: class to compute Grad-CAM for (None = predicted class).
            stream_idx: required for mode='stream'.
            dataloader: optional, for auxiliary classifier accuracy check.
            overlay_input: optional [H, W, 3] image for overlay.
            save_path: optional save path.

        Returns:
            Grad-CAM heatmap as [H, W] numpy array.
        """
        self.model.eval()

        if mode == "integrated":
            return self._gradcam_integrated(stream_inputs, layer, target_class, overlay_input, save_path)
        elif mode == "stream":
            assert stream_idx is not None, "stream_idx required for mode='stream'"
            if dataloader is not None:
                acc = self._check_aux_classifier(stream_idx, dataloader)
                num_classes = self.model.num_classes
                if acc < 2.0 / num_classes:
                    print(f"WARNING: Auxiliary classifier for {self.labels[stream_idx]} appears "
                          f"untrained (acc={acc:.1%}). Per-stream Grad-CAM results may not be meaningful.")
            return self._gradcam_stream(stream_inputs, stream_idx, layer, target_class, overlay_input, save_path)
        elif mode == "decomposed":
            return self._gradcam_decomposed(stream_inputs, layer, target_class, overlay_input, save_path)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _gradcam_integrated(self, stream_inputs, layer, target_class, overlay_input, save_path):
        """Standard Grad-CAM on integrated stream."""
        inputs = [s.to(self.device) for s in stream_inputs]
        activations = {}
        gradients = {}

        # Hook on the target layer
        target_module = dict(self.model.named_modules())[layer]

        def fwd_hook(mod, inp, out):
            if isinstance(out, tuple):
                activations["val"] = out[1].detach()  # integrated
            else:
                activations["val"] = out.detach()

        def bwd_hook(mod, grad_in, grad_out):
            if isinstance(grad_out, tuple):
                # For LISequential returning (list[Tensor], Tensor),
                # grad_out[1] is the gradient for the integrated output
                gradients["val"] = grad_out[1].detach() if grad_out[1] is not None else None
            else:
                gradients["val"] = grad_out.detach()

        fwd_h = target_module.register_forward_hook(fwd_hook)
        bwd_h = target_module.register_full_backward_hook(bwd_hook)

        with torch.enable_grad():
            for s in inputs:
                s.requires_grad_(False)
            output = self.model(inputs)
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            score = output[0, target_class]
            self.model.zero_grad()
            score.backward()

        fwd_h.remove()
        bwd_h.remove()

        heatmap = self._compute_heatmap(activations.get("val"), gradients.get("val"))
        self._plot_gradcam(heatmap, f"Integrated Grad-CAM @ {layer} (class {target_class})",
                           overlay_input, save_path)
        return heatmap

    def _gradcam_stream(self, stream_inputs, stream_idx, layer, target_class, overlay_input, save_path):
        """Per-stream isolated Grad-CAM via auxiliary classifier."""
        stream_input = stream_inputs[stream_idx].to(self.device)

        # Forward through stream pathway, capturing activations at the target layer
        activations = {}
        gradients = {}

        # We need to hook into the layer's forward_stream path
        # Run manually to capture at the right point
        with torch.enable_grad():
            stream_x = self.model.conv1.forward_stream(stream_idx, stream_input)
            stream_x = self.model.bn1.forward_stream(stream_idx, stream_x)
            stream_x = F.relu(stream_x, inplace=False)
            stream_x = self.model.maxpool.forward_stream(stream_idx, stream_x)

            layer_map = OrderedDict([
                ("layer1", self.model.layer1), ("layer2", self.model.layer2),
                ("layer3", self.model.layer3), ("layer4", self.model.layer4),
            ])

            for lname, lmod in layer_map.items():
                stream_x = lmod.forward_stream(stream_idx, stream_x)
                if lname == layer:
                    stream_x.retain_grad()
                    activations["val"] = stream_x

            pooled = self.model.avgpool(stream_x)
            features = torch.flatten(pooled, 1)
            logits = self.model.fc_streams[stream_idx](features)

            if target_class is None:
                target_class = logits.argmax(dim=1).item()
            score = logits[0, target_class]
            self.model.zero_grad()
            score.backward()

        grad = activations["val"].grad
        heatmap = self._compute_heatmap(activations["val"].detach(), grad)
        self._plot_gradcam(heatmap,
                           f"{self.labels[stream_idx]} Grad-CAM @ {layer} (class {target_class})",
                           overlay_input, save_path)
        return heatmap

    def _gradcam_decomposed(self, stream_inputs, layer, target_class, overlay_input, save_path):
        """Decomposed Grad-CAM: stream contributions weighted by integrated Grad-CAM."""
        # Get integrated Grad-CAM
        int_heatmap = self._gradcam_integrated(stream_inputs, layer, target_class, None, None)

        # Get stream contributions
        contrib_result = self._contrib_viz.visualize(stream_inputs, layer, save_path=None)
        if contrib_result is None:
            return None

        # Weight each stream contribution by integrated Grad-CAM
        # Resize Grad-CAM to match contribution spatial dims
        stream_maps = contrib_result["stream_contributions"]
        h, w = stream_maps[0].shape
        int_resized = np.array(
            F.interpolate(
                torch.tensor(int_heatmap).unsqueeze(0).unsqueeze(0).float(),
                size=(h, w), mode="bilinear", align_corners=False
            ).squeeze().numpy()
        )

        weighted = []
        for si, smap in enumerate(stream_maps):
            weighted.append(smap * int_resized)

        # Plot
        ncols = self.num_streams + 1  # per-stream + integrated
        fig, axes = plt.subplots(1, ncols, figsize=(ncols * 3, 3))
        fig.suptitle(f"Decomposed Grad-CAM @ {layer} (class {target_class})", fontsize=12)

        for si in range(self.num_streams):
            axes[si].imshow(_percentile_normalize(weighted[si]), cmap="jet")
            axes[si].set_title(self.labels[si], fontsize=9)
            axes[si].axis("off")

        axes[-1].imshow(_percentile_normalize(int_heatmap), cmap="jet")
        axes[-1].set_title("Integrated", fontsize=9)
        axes[-1].axis("off")

        plt.tight_layout()
        _save_or_show(fig, save_path)
        return int_heatmap

    def _compute_heatmap(self, activations: Optional[torch.Tensor],
                         gradients: Optional[torch.Tensor]) -> np.ndarray:
        """Compute Grad-CAM heatmap from activations and gradients."""
        if activations is None or gradients is None:
            return np.zeros((1, 1))

        if activations.dim() == 4:
            activations = activations[0]
        if gradients.dim() == 4:
            gradients = gradients[0]

        # GAP over spatial dims for weights
        weights = gradients.mean(dim=(-2, -1))  # [C]
        # Weighted combination
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * activations).sum(dim=0)  # [H, W]
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    def _plot_gradcam(self, heatmap: np.ndarray, title: str,
                      overlay_input: Optional[torch.Tensor], save_path: Optional[str]) -> None:
        """Plot Grad-CAM heatmap, optionally overlaid on input image."""
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        if overlay_input is not None:
            if isinstance(overlay_input, torch.Tensor):
                overlay_input = overlay_input.cpu().numpy()
            if overlay_input.ndim == 3 and overlay_input.shape[0] in (1, 3):
                overlay_input = np.transpose(overlay_input, (1, 2, 0))
            if overlay_input.max() <= 1.0:
                overlay_input = (overlay_input * 255).astype(np.uint8)

            h, w = overlay_input.shape[:2]
            heatmap_resized = np.array(
                F.interpolate(
                    torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).float(),
                    size=(h, w), mode="bilinear", align_corners=False
                ).squeeze().numpy()
            )

            ax.imshow(overlay_input)
            im = ax.imshow(heatmap_resized, cmap="jet", alpha=0.5)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            im = ax.imshow(heatmap, cmap="jet")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(title, fontsize=10)
        ax.axis("off")
        plt.tight_layout()
        _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 2D: Integration Weight Visualizer
# ---------------------------------------------------------------------------

class IntegrationWeightVisualizer:
    """Visualize learned integration_from_streams weights and their properties."""

    def __init__(self, model: nn.Module, stream_labels: Optional[dict[int, str]] = None):
        self.model = model
        self.num_streams = model.num_streams
        self.labels = _default_stream_labels(self.num_streams, stream_labels)

    def visualize_weights(self, save_path: Optional[str] = None) -> None:
        """Plot integration weights as heatmaps per layer and stream."""
        weights_by_layer = self._collect_weights()

        n_layers = len(weights_by_layer)
        fig, axes = plt.subplots(n_layers, self.num_streams, figsize=(self.num_streams * 4, n_layers * 3))
        if n_layers == 1:
            axes = [axes]
        fig.suptitle("Integration Weights (out_channels x in_channels)", fontsize=13)

        for li, (layer_name, stream_weights) in enumerate(weights_by_layer.items()):
            for si, w in enumerate(stream_weights):
                ax = axes[li][si] if self.num_streams > 1 else axes[li]
                im = ax.imshow(w, aspect="auto", cmap="RdBu_r")
                ax.set_title(f"{layer_name}\n{self.labels[si]}", fontsize=8)
                ax.set_xlabel("in_ch")
                ax.set_ylabel("out_ch")
                plt.colorbar(im, ax=ax, fraction=0.046)

        plt.tight_layout()
        _save_or_show(fig, save_path)

    def visualize_cross_stream(self, save_path: Optional[str] = None) -> None:
        """Compare relative integration weight magnitudes across streams per layer."""
        weights_by_layer = self._collect_weights()

        layers = list(weights_by_layer.keys())
        norms = np.zeros((len(layers), self.num_streams))
        for li, (_, stream_weights) in enumerate(weights_by_layer.items()):
            for si, w in enumerate(stream_weights):
                norms[li, si] = np.linalg.norm(w)

        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(layers))
        width = 0.8 / self.num_streams
        for si in range(self.num_streams):
            ax.bar(x + si * width, norms[:, si], width, label=self.labels[si])
        ax.set_xticks(x + width * (self.num_streams - 1) / 2)
        ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Weight L2 Norm")
        ax.set_title("Integration Weight Magnitude by Layer and Stream")
        ax.legend()
        plt.tight_layout()
        _save_or_show(fig, save_path)

    def compute_effective_rank(self) -> dict[str, list[float]]:
        """Compute effective rank of integration weights via SVD.

        Returns dict mapping layer names to list of effective ranks per stream.
        """
        weights_by_layer = self._collect_weights()
        result = {}
        for layer_name, stream_weights in weights_by_layer.items():
            ranks = []
            for w in stream_weights:
                # w is [out_c, in_c] -- compute SVD
                s = np.linalg.svd(w, compute_uv=False)
                # Effective rank: exp(entropy of normalized singular values)
                s_norm = s / (s.sum() + 1e-8)
                s_norm = s_norm[s_norm > 1e-10]
                entropy = -(s_norm * np.log(s_norm)).sum()
                ranks.append(float(np.exp(entropy)))
            result[layer_name] = ranks
        return result

    def _collect_weights(self) -> OrderedDict:
        """Collect integration weights organized by layer."""
        result = OrderedDict()
        for name, param in self.model.named_parameters():
            if "integration_from_streams" not in name:
                continue
            # Parse layer name: e.g. "layer1.0.conv1.integration_from_streams.0"
            parts = name.split(".")
            # Find the layer prefix (everything before integration_from_streams)
            idx = parts.index("integration_from_streams")
            layer_name = ".".join(parts[:idx])
            stream_idx = int(parts[idx + 1])

            if layer_name not in result:
                result[layer_name] = [None] * self.num_streams
            # Reshape from [out_c, in_c, 1, 1] to [out_c, in_c]
            w = param.data.cpu().numpy().squeeze()
            if w.ndim == 1:
                w = w.reshape(-1, 1)
            result[layer_name][stream_idx] = w

        return result


# ---------------------------------------------------------------------------
# 2E: Misclassification Analysis Helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def find_misclassified(
    model: nn.Module,
    dataloader: DataLoader,
    n: int = 10,
) -> list[dict]:
    """Find n misclassified samples with metadata.

    Returns list of dicts with:
        stream_inputs: list of [1, C, H, W] CPU tensors
        true_label: int
        predicted_label: int
        confidence: float
    """
    model.eval()
    device = _get_device(model)
    results = []

    for batch_data in dataloader:
        *stream_batches, targets = batch_data
        stream_batches = [s.to(device) for s in stream_batches]
        targets = targets.to(device)

        logits = model(stream_batches)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        wrong = preds != targets

        for i in wrong.nonzero(as_tuple=True)[0]:
            idx = i.item()
            results.append({
                "stream_inputs": [s[idx:idx+1].cpu() for s in stream_batches],
                "true_label": targets[idx].item(),
                "predicted_label": preds[idx].item(),
                "confidence": probs[idx, preds[idx]].item(),
            })
            if len(results) >= n:
                return results

    return results


# ---------------------------------------------------------------------------
# 2H: Compare Samples Helper
# ---------------------------------------------------------------------------

def compare_samples(
    model: nn.Module,
    correct_sample: dict,
    misclassified_sample: dict,
    layer: str = "layer4",
    stream_labels: Optional[dict[int, str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Side-by-side Grad-CAM + contribution comparison of two samples.

    Args:
        model: LINet3 model.
        correct_sample: dict from find_misclassified or similar, with 'stream_inputs' and 'true_label'.
        misclassified_sample: same format.
        layer: target layer.
        stream_labels: optional stream name mapping.
        save_path: optional save path.
    """
    gradcam = StreamGradCAM(model, stream_labels)
    contrib = StreamContributionVisualizer(model, stream_labels)
    num_streams = model.num_streams
    labels = _default_stream_labels(num_streams, stream_labels)

    # 2 rows (correct, misclassified) x (num_streams + 2) cols (gradcam + per-stream contribs + dominance)
    ncols = num_streams + 2
    fig, axes = plt.subplots(2, ncols, figsize=(ncols * 3, 6))

    samples = [
        ("Correct", correct_sample),
        ("Misclassified", misclassified_sample),
    ]

    for row, (label, sample) in enumerate(samples):
        inputs = sample["stream_inputs"]
        true_label = sample.get("true_label", "?")
        pred_label = sample.get("predicted_label", true_label)

        # Integrated Grad-CAM
        heatmap = gradcam._gradcam_integrated(inputs, layer, None, None, None)
        axes[row, 0].imshow(heatmap, cmap="jet")
        axes[row, 0].set_title(f"Grad-CAM\ntrue={true_label} pred={pred_label}", fontsize=8)
        axes[row, 0].axis("off")

        # Stream contributions
        contrib_result = contrib.visualize(inputs, layer, save_path=None)
        if contrib_result is not None:
            for si in range(num_streams):
                axes[row, si + 1].imshow(
                    _percentile_normalize(contrib_result["stream_contributions"][si]), cmap="hot")
                axes[row, si + 1].set_title(labels[si], fontsize=8)
                axes[row, si + 1].axis("off")

            axes[row, -1].imshow(contrib_result["dominance_map"],
                                 cmap=plt.cm.get_cmap("tab10", num_streams),
                                 vmin=0, vmax=num_streams - 1)
            axes[row, -1].set_title("Dominance", fontsize=8)
            axes[row, -1].axis("off")

        axes[row, 0].set_ylabel(label, fontsize=10)

    fig.suptitle(f"Sample Comparison @ {layer}", fontsize=12)
    plt.tight_layout()
    _save_or_show(fig, save_path)

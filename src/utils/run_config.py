"""Single source of truth for the ``run_config.json`` checkpoint sidecar.

Every training run that writes a checkpoint should also call
``write_run_config`` to drop a ``run_config.json`` next to the .pt file. The
sidecar gives ``load_pretrained_backbone`` enough metadata to print a
self-documenting info line at load time AND to detect two qualitatively
different staleness signals:

  - **Tier 1 (soft INFO)** — drop-list hash mismatch only. Validated-scenes
    set has changed since training; HHA values are still geometrically
    correct, just trained against slightly different scenes. Retrain when
    convenient.
  - **Tier 2 (loud WARNING)** — ``axis_alignment_inverted`` mismatch. The
    HHA values in training data had axes wrong relative to current ground
    truth. Do not use this checkpoint for transfer learning to the current
    data.

Both stay informational; nothing here gates the load (the user retains
autonomy). The schema is fixed in code so the four notebooks that write
sidecars (3 ScanNet HHA + the existing baseline if updated) can't drift.
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Literal, Optional

DepthModality = Literal["raw_depth", "hha"]

SCHEMA_VERSION = "1"


def _sha256_file(path: str, *, chunk: int = 1 << 20) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def _git_short_sha() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 — best-effort metadata
        pass
    return None


def _now_iso_utc() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_sha256(obj: Any) -> str:
    """Stable hash of an arbitrary JSON-serializable object."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_run_config(
    checkpoint_dir: str,
    *,
    depth_modality: DepthModality,
    model_class: str,
    stream_input_channels: list[int],
    dataset_tarball_path: Optional[str] = None,
    hha_norm_stats: Optional[dict] = None,
    axis_alignment_inverted: Optional[bool] = None,
    training_config: Optional[dict] = None,
    drop_list_path: Optional[str] = None,
    num_classes: int = 0,
    extra: Optional[dict] = None,
) -> Path:
    """Write run_config.json to ``checkpoint_dir/run_config.json``.

    Returns the path written.

    All sha256 computations are best-effort: if a file isn't present
    (running outside Colab, missing Drive mount, etc.) the corresponding
    field is set to ``None`` and the load-time info line says
    'unavailable' rather than crashing.
    """
    config: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "depth_modality": depth_modality,
        "model_class": model_class,
        "stream_input_channels": list(stream_input_channels),
        "axis_alignment_inverted": axis_alignment_inverted,
        "timestamp": _now_iso_utc(),
        "git_commit": _git_short_sha(),
        "dataset_tarball_path": dataset_tarball_path,
        "dataset_tarball_sha256": (
            _sha256_file(dataset_tarball_path) if dataset_tarball_path else None
        ),
        "drop_list_path": drop_list_path,
        "drop_list_sha256": (
            _sha256_file(drop_list_path) if drop_list_path else None
        ),
        "hha_norm_stats": hha_norm_stats if depth_modality == "hha" else None,
        "training_config": training_config,
        "training_config_hash": (
            _canonical_json_sha256(training_config) if training_config is not None else None
        ),
        "num_classes": num_classes,
        "extra": extra or {},
    }

    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "run_config.json"
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    return out_path


def read_run_config(checkpoint_path: str) -> Optional[dict]:
    """Return the run_config.json sidecar from the checkpoint's parent dir.

    Returns ``None`` if the sidecar is absent (e.g. older checkpoints
    trained before this schema existed). Callers should treat ``None`` as
    'no metadata available' rather than an error.
    """
    parent = Path(checkpoint_path).resolve().parent
    sidecar = parent / "run_config.json"
    if not sidecar.is_file():
        return None
    try:
        with open(sidecar, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def format_load_info_line(
    sidecar: dict,
    *,
    current_drop_list_path: Optional[str] = None,
) -> str:
    """Compose the single info line printed by ``load_pretrained_backbone``.

    Includes two staleness signals when ``current_drop_list_path`` resolves:
        - Tier 1: drop-list-hash mismatch (soft INFO)
        - Tier 2: axis_alignment_inverted mismatch (loud WARNING)

    Both stay informational — no exceptions raised here.
    """
    parts: list[str] = []
    if sidecar.get("depth_modality"):
        parts.append(f"modality={sidecar['depth_modality']}")
    if sidecar.get("model_class"):
        parts.append(f"model={sidecar['model_class']}")
    if sidecar.get("stream_input_channels") is not None:
        parts.append(f"streams={sidecar['stream_input_channels']}")
    ds = sidecar.get("dataset_tarball_sha256")
    if ds:
        parts.append(f"dataset_sha256={ds[:8]}")
    dl = sidecar.get("drop_list_sha256")
    if dl:
        parts.append(f"drop_list_sha256={dl[:8]}")
    ts = sidecar.get("timestamp")
    if ts:
        parts.append(f"trained={ts[:19]}")
    gc = sidecar.get("git_commit")
    if gc:
        parts.append(f"git={gc}")

    base = "Loading checkpoint: " + ", ".join(parts) if parts else (
        "Loading checkpoint (no run_config.json metadata available)"
    )

    # Drop-list staleness comparison.
    if current_drop_list_path is None:
        return base
    if not os.path.isfile(current_drop_list_path):
        return base + "  (drop_list comparison unavailable -- file not found)"
    current_hash = _sha256_file(current_drop_list_path)
    sidecar_hash = sidecar.get("drop_list_sha256")
    if sidecar_hash is None or current_hash is None:
        return base

    # Tier 2 -- loud warning if axis_alignment_inverted differs.
    sidecar_inv = sidecar.get("axis_alignment_inverted")
    current_inv = None
    try:
        with open(current_drop_list_path, "r") as f:
            current_drop = json.load(f)
        current_inv = current_drop.get("axis_alignment_inverted")
    except (OSError, json.JSONDecodeError):
        pass
    if (sidecar_inv is not None and current_inv is not None
            and sidecar_inv != current_inv):
        banner = (
            "\n" + "=" * 70 + "\n"
            "  WARNING: axis_alignment_inverted MISMATCH between checkpoint and current Drive drop list.\n"
            f"    checkpoint: axis_alignment_inverted = {sidecar_inv}\n"
            f"    current:    axis_alignment_inverted = {current_inv}\n"
            "  The HHA training data for this checkpoint had axes wrong relative\n"
            "  to current ground truth. Depth-stem and integration weights are\n"
            "  calibrated to a flipped world frame.  DO NOT use this checkpoint\n"
            "  for transfer learning to data preprocessed with the current drop list.\n"
            + "=" * 70
        )
        return base + banner

    # Tier 1 -- soft info if only drop-list hash differs.
    if sidecar_hash != current_hash:
        return base + (
            "  INFO: drop_list differs from Drive (validated-scenes set updated since training; retrain when convenient)"
        )

    return base


def format_arch_mismatch_warning(
    sidecar: dict,
    *,
    current_model_class: str,
    current_stream_input_channels: list[int],
) -> Optional[str]:
    """If model_class or stream_input_channels disagree with the target
    model, return a multi-line WARNING banner. Otherwise return None.

    Loading mismatched-architecture weights is a silent-corruption failure
    mode — early conv keys may happen to match and the load will appear
    successful while integration weights expect a different stream-feature
    distribution. Same severity as Tier 2 axis-flip; informational, not a
    hard gate.
    """
    sidecar_mc = sidecar.get("model_class")
    sidecar_sic = sidecar.get("stream_input_channels")
    mc_mismatch = sidecar_mc is not None and sidecar_mc != current_model_class
    sic_mismatch = (
        sidecar_sic is not None
        and list(sidecar_sic) != list(current_stream_input_channels)
    )
    if not (mc_mismatch or sic_mismatch):
        return None
    lines = ["", "=" * 70,
             "  WARNING: pretrained-checkpoint architecture mismatch."]
    if mc_mismatch:
        lines.append(f"    checkpoint model_class={sidecar_mc!r}, current model_class={current_model_class!r}")
    if sic_mismatch:
        lines.append(
            f"    checkpoint stream_input_channels={sidecar_sic}, "
            f"current stream_input_channels={current_stream_input_channels}"
        )
    lines.append("  Loading mismatched-architecture weights is silent corruption:")
    lines.append("  early conv keys may happen to match while integration weights")
    lines.append("  expect a different feature distribution. Verify deliberately.")
    lines.append("=" * 70)
    return "\n".join(lines)

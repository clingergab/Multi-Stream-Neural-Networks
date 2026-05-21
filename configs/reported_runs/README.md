# Reported-run configurations

One YAML per LINet result reported in the paper. Each file holds the exact
optimum hyperparameters returned by the HPO campaign (paper Appendix A/B);
re-running `train.py` against a config reproduces the paper's training
recipe for that row up to dataloader-shuffling and CUDA non-determinism.

```bash
python3 train.py --config configs/reported_runs/<name>.yaml \
                 --data-root <preprocessed-dataset-path>
```

## SUN RGB-D, from scratch (Table 1; Appendix Table B1)

| Config | Schedule | Reported Fusion / RGB-only / Depth-only MCA |
|---|---|---|
| [`sunrgbd_linet_no_md_from_scratch.yaml`](sunrgbd_linet_no_md_from_scratch.yaml) | No modality dropout | 43.5 / 15.0 / 14.4 |
| [`sunrgbd_linet_progressive_md_from_scratch.yaml`](sunrgbd_linet_progressive_md_from_scratch.yaml) | **Progressive MD (paper headline)** | **45.2 / 33.7 / 35.7** |
| [`sunrgbd_linet_static_md_from_scratch.yaml`](sunrgbd_linet_static_md_from_scratch.yaml) | Static MD (constant from epoch 0) | 43.2 / 32.9 / 33.7 |
| [`sunrgbd_linet_delayed_md_from_scratch.yaml`](sunrgbd_linet_delayed_md_from_scratch.yaml) | Delayed MD (start ep 20, ramp 15) | 43.8 / 33.7 / 36.5 |

The Progressive-MD variant at p_max = 0.8 reported in Table 3 (44.1% Fusion
MCA) uses [`sunrgbd_linet_progressive_md_from_scratch.yaml`](sunrgbd_linet_progressive_md_from_scratch.yaml) verbatim with only
`modality_dropout.rate` changed to `0.8`.

## SUN RGB-D, ScanNet-pretrained (Table 1; Appendix Table B3)

ScanNet pretraining is a two-phase recipe: pretrain on the 100K-frame
ScanNet subset, then fine-tune on SUN RGB-D.

| Config | Phase | Reported Fusion / RGB-only / Depth-only MCA |
|---|---|---|
| [`scannet_pretrain_no_md.yaml`](scannet_pretrain_no_md.yaml) | Pretrain (No-MD) | 99.8 / 94.6 / 27.4 (ScanNet val) |
| [`scannet_pretrain_progressive_md.yaml`](scannet_pretrain_progressive_md.yaml) | Pretrain (Progressive MD) | 99.8 / 98.9 / 97.3 (ScanNet val) |
| [`sunrgbd_linet_no_md_scannet_pretrained.yaml`](sunrgbd_linet_no_md_scannet_pretrained.yaml) | Fine-tune from ScanNet (No-MD) | 48.0 / 36.5 / 37.2 (SUN test) |
| [`sunrgbd_linet_progressive_md_scannet_pretrained.yaml`](sunrgbd_linet_progressive_md_scannet_pretrained.yaml) | **Fine-tune from ScanNet (Progressive MD; paper headline)** | **49.6 / 39.2 / 38.4 (SUN test)** |

## NYU Depth V2, from scratch (Table 2; Appendix Table B4)

| Config | Schedule | Reported Fusion / RGB-only / Depth-only MCA |
|---|---|---|
| [`nyu_linet_no_md_from_scratch.yaml`](nyu_linet_no_md_from_scratch.yaml) | No modality dropout | 50.1 / 27.0 / 16.1 |
| [`nyu_linet_progressive_md_from_scratch.yaml`](nyu_linet_progressive_md_from_scratch.yaml) | **Progressive MD** | **50.8 / 37.8 / 40.3** |

## Notes on reproducibility

- **HPO scope**: each configuration in the paper was searched independently
  (Ray Tune + TPE, 500 trials, stratified 20% validation holdout). The
  values listed in each YAML are the per-run optimum, not a shared default.
- **Augmentation dials**: the `aug_prob` / `aug_mag` fields are the two
  per-modality global control dials defined in Appendix A, not literal
  per-augmentation probabilities. Values above 1.0 simply turn the
  corresponding dial past its nominal setting.
- **Epoch budget**: for SUN RGB-D each from-scratch run uses a fixed epoch
  budget equal to the best epoch found on an 80/20 train/val split of the
  SUN training set; the train/test run then retrains for exactly that many
  epochs with no validation or best-weight restoration, all other settings
  (including `t_max`) held constant.
- **Cosine schedule**: paper §4.2 specifies a single cosine cycle to
  `η_min`. We set `scheduler.t_max = training.epochs` so the cosine
  schedule decays once over the full training budget (no warm restart, no
  warmup), matching the paper's prose. The notebooks sometimes use shorter
  `t_max` for warm-restart variants during HPO; that is a notebook-side
  choice that is not part of the reported runs.
- **Hardware**: SUN RGB-D and NYU final runs were on a single NVIDIA T4;
  ScanNet pretraining was on a single NVIDIA A100. HPO trials were
  distributed across A100s.
- **Mixed precision**: AMP is enabled for every run (paper §4.2).
- **Known paper-internal inconsistency on Delayed MD**: §4.4.1 prose
  describes the schedule as "p = 0 for 30 epochs, then a 15-epoch ramp",
  while Appendix Table B1 lists "start ep 20, ramp 15". The YAML follows
  Appendix B1 since the appendix is the explicit reproducibility source;
  flagged here for transparency, no silent correction.

## Single-stream and discrete-fusion baselines

The early-fusion / late-fusion / RGB-only / Depth-only baselines in
paper Tables 1, 2, B2, B5 use a different model class (the MCResNet
baseline in `src/models/multi_channel/`, not LINet). They are not LINet
configurations and so don't have entries here. Their HPO optima are
documented in Appendix Tables B2 and B5 of the paper; reproducing them
requires the corresponding baseline notebooks under `notebooks/`
(e.g., `colab_LINet3_SUN_training_baselines.ipynb`).

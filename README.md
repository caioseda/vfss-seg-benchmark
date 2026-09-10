# VFSS Segmentation Benchmark

This repository is dedicated to benchmarking diverse models for the task of Semantic Segmentation in Videofluoroscopic Swallowing Studies (VFSS). The goal is to provide a comprehensive evaluation of various models, including both traditional and state-of-the-art approaches, to identify the most effective techniques for this specific application.

## Dataset
The datasets used for benchmarking consists of both publicly available and private datasets of VFSS images, annotated for semantic segmentation tasks.

## Models Evaluated
- U-Net
- U-Net++
- TransUNet
- SwinUNet

## Evaluation Metrics
The models are evaluated using the following metrics:
- Mean Intersection over Union (Mean IoU)
- Dice Score
- Average Symmetric Surface Distance (ASSD)
- 95th-percentile Hausdorff Distance (HD95)

Metrics are reported per class as well as aggregated. Which foreground classes get their own line is
set by `lit_module.params.report_class_ids` (e.g. `{1: C2, 3: C4}`); the aggregate averages **only**
those classes. This matters for `multiclass_c2_c4`, where the model has 4 outputs but class 2 (C3)
never appears in any mask -- averaging over all foreground classes would silently drag every number
down by a third.

ASSD and HD95 cost roughly 1000x what IoU/Dice cost (~170 ms per 8x256x256 batch on GPU, ~18 s on
CPU), so by default they are computed only at test time (`surface_metric_stages`).

## Semi-supervised segmentation

`src/models/ssl/` holds the semi-supervised baselines, all sharing a two-stream batch (labeled +
unlabeled) via `src/data/samplers.py:TwoStreamBatchSampler` and the label regime in
`src/data/vfss_frame_dataset.py:apply_label_regime`, which hides a fraction of the training labels
without dropping the frames:

| Wrapper | Method |
|---|---|
| `SupervisedLitWrapper` | supervised baseline at a given annotation budget |
| `MeanTeacherLitWrapper` | Mean Teacher (EMA teacher + consistency) |
| `FixMatchLitWrapper` | FixMatch (confidence-thresholded pseudo-labels) |
| `DiffRectLitWrapper` | phase-2 slot, raises `NotImplementedError` |

Loss composition and hyperparameters follow [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) (MIT);
vendored code lives under `src/third_party/` with provenance headers, and each wrapper's docstring
records where it deliberately diverges from the reference.

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

Standard library `unittest`, no extra dependency, CPU-only, ~8s. Three tiers, deliberately distinct:

| File | What it pins | Why a smoke test would not catch it |
|---|---|---|
| `tests/test_ssl_methods.py` | Algorithm invariants: hidden labels are never supervised, the consistency term carries gradient, the teacher is an EMA and not an optimised parameter, pseudo-labels come from the class axis, the ramp/warm-up schedule | Each of these bugs runs fine and produces a plausible loss curve |
| `tests/test_reference_equivalence.py` | Numerical equality with [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) on identical inputs (DiceLoss, `0.5*(CE+Dice)`, the EMA rule, the consistency MSE) | A port can be self-consistent and still not be the published method |
| `tests/test_learning.py` | The loop actually optimises: every method memorises a 3-frame batch to Dice 1.0, losses stay finite, the teacher lags then closes the gap | Wrong loss sign or a mis-wired optimizer still "runs" |

The suite is itself checked by mutation testing: 18 bugs are injected one at a time -- supervise the
hidden labels, invert the EMA, feed the teacher the whole batch, argmax the wrong axis, take the
pseudo-label from the strong view, drop the confidence threshold, flatten the consistency ramp,
invert the Dice sign -- and the suite must fail for each. **17 of 18 are caught.** The one that
survives is provably an equivalent mutant, not a bug: `sum(target*target)` vs `sum(target)` inside
`_dice_loss`, where `target` is one-hot, so `x*x == x` and the two expressions are identical.

Three of those 18 were caught only *after* mutation testing exposed that nothing covered them --
notably the consistency ramp, where the original test asserted the weight was monotonic and capped,
both of which a constant weight also satisfies. A test that cannot fail is not a test.

## Experiment X1

`notebooks/hyphotesis_testing/experiment_1_does_ssl_helps.ipynb` asks what the ~45k **never
annotated** video frames add, and how much of the result depends on how many of the ~1k annotated
frames carry supervision. It sweeps 25/50/75/100% annotation budgets across the methods above on a
single fixed fold, and reports Dice/IoU/ASSD/HD95 per class (C2, C4). Configs live in
`configs/experiment/x1/`.

The design departs deliberately from the semi-supervised literature. Public benchmarks (ACDC, LA,
Pancreas-CT) are fully annotated and *simulate* a low-label regime by hiding labels, so at 100% no
unlabeled data is left and every method collapses onto its supervised baseline. Here annotation is
the scarce resource, so the unlabeled stream comes from a real pool of never-annotated frames
(`unlabeled_pool_size`, built by `build_unlabeled_frame_pool` from train-split videos only) and is
present at **every** budget -- which makes the 100% cell the interesting one rather than a
degenerate one, and keeps batch composition (`labeled_batch_size` of `batch_size`) identical across
budgets.

Training diagnostics are logged against the ground truth the label regime hides
(`expose_hidden_targets=True`): `train/pseudo_accuracy`, `train/pseudo_dice`,
`train/pseudo_label_coverage` for FixMatch and `train/teacher_*` for Mean Teacher, alongside the
separated loss components and the consistency ramp. No loss ever reads those tensors.

X1 samples the unlabeled rows of a batch with the `same_video` policy: each one comes from the same
video as one of that batch's labeled frames, at any position -- the dataset is video, not a
collection of independent slices.

## Experiment X1B

`notebooks/hyphotesis_testing/experiment_1b_unlabeled_sampling_policy.ipynb` isolates that last
choice, which the literature never had to make: its benchmarks have no notion of "same video" or
"neighbouring frame", so their two-stream samplers draw globally by default rather than by decision.
X1B compares `global`, `same_video`, and temporal windows of W = 3, 5 and 10 frames
(`train_batch_sampler.params.unlabeled_policy` / `temporal_window`, implemented in
`src/data/samplers.py`), asking whether temporal proximity between a labeled frame and the unlabeled
frames it is trained against changes how well semi-supervision works.

It reuses the X1 configs rather than copying them, so the control is structural, and it runs at
100% of the labels on purpose: below that, the video-conditioned policies also restrict *which*
videos can contribute unlabeled frames (39 of 165 at 25%), and the comparison would measure coverage
and temporal proximity at once. It also uses an uncapped, unstrided pool -- with X1's default pool a
W=3 window has no candidates ~9% of the time. `TwoStreamBatchSampler.draw_report()` reports how
often each policy was actually served, and every X1B run records it.

## Setup

Create the conda environment from `environment.yml` and activate it:

```bash
conda env create -f environment.yml
conda activate vfss
```

Datasets are read from local paths configured in each YAML config (see `dataset_path` / `data_root` under `data.params.shared_dataset_params`). Update those paths to point at your local copy of the dataset before running an experiment.

## Running an experiment

Experiments are defined by YAML config files under `configs/experiment/`, organized by model (`unet`, `unetplusplus`, `transunet`, `swinunet`). Run a single experiment with:

```bash
python scripts/train.py --config configs/experiment/unet/vfss-inca-unet.yaml
```

If `--config` is omitted, it defaults to `configs/experiment/unet/vfss-inca-unet.yaml`.

To quickly smoke-test a config (runs a handful of batches instead of a full training run):

```bash
python scripts/train.py --config configs/experiment/unet/vfss-inca-unet.yaml --fast-dev-run
# or with an explicit number of batches
python scripts/train.py --config configs/experiment/unet/vfss-inca-unet.yaml --fast-dev-run 5
```

Each experiment writes logs, checkpoints, and metrics to `logs/<experiment_name>/<timestamp>/`.

## Running a benchmark

A benchmark config (under `configs/benchmark/`) lists multiple experiment configs to run sequentially and aggregates their results:

```bash
python scripts/train.py --config configs/benchmark/vfss-baseline.yaml
```

Results are written to `logs/benchmarks/<benchmark_name>/<run_id>-<timestamp>/`, including a console log and a summary of metrics across all experiments.

## Config structure

Both experiment and benchmark configs are YAML files loaded with OmegaConf, each requiring a `type` field (`experiment` or `benchmark`). Configs use a `target`/`params` pattern to instantiate Python classes (model, optimizer, lr_scheduler, trainer, data module, datasets, callbacks). See existing configs under `configs/` for examples of each model/dataset combination.

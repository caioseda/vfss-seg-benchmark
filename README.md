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

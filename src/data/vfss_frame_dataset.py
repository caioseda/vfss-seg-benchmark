import os
import logging
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision import transforms

from typing import Sequence, Union, Optional

logger = logging.getLogger(__name__)

TARGET_TYPES = ("mask", "points")
TARGET_VARIANTS = ("raw", "binary_c2_c4", "multiclass_c2_c4", "multiclass_all")
SPLIT_NAMES = ("train", "val", "test")
DEFAULT_SPLIT_RATIOS = (0.7, 0.10, 0.2)


def load_video_frame_dataframe(dataset_path: Union[str, Path], video_frame_table_filename: str = 'inca-video-frame-dataset.csv') -> pd.DataFrame:
    '''Load the pool of video frames/labels from the dataset CSV.'''
    dataset_path = Path(dataset_path)
    df = pd.read_csv(dataset_path / video_frame_table_filename)
    df['frame_id'] = df['frame_id'].astype(int)
    df['video_id'] = df['video_id'].astype(int)
    # Every row of this table is an annotated frame. The distinction only becomes meaningful once
    # `build_unlabeled_frame_pool` appends frames that have no ground truth at all -- which is not
    # the same thing as a frame whose label `apply_label_regime` hides.
    df['has_target'] = True
    assert df.shape[0] > 0, "The video frame DataFrame is empty. Please check the provided dataset_path and video_frame_table_filename."
    return df


def _target_path_axis(target_path: str, axis: int) -> str:
    '''Read one axis (type/variant/labeler) out of a `targets/<type>/<variant>/<labeler>/<file>` path.'''
    parts = Path(target_path).parts
    if len(parts) < 5 or parts[0] != "targets":
        raise ValueError(
            f"Unexpected target_path format: '{target_path}'. "
            "Expected 'targets/<modalidade>/<variante>/<rotulador>/<video_frame>.<ext>'."
        )
    return parts[axis]


def with_target_type(df: pd.DataFrame, target_type: str) -> pd.DataFrame:
    '''Return a copy of `df` pointing `target_path` at a different modality (`mask` / `points`).'''
    if target_type not in TARGET_TYPES:
        raise ValueError(f"target_type must be one of {TARGET_TYPES}, got '{target_type}'")

    df = df.copy()
    variants = df['target_path'].apply(lambda p: _target_path_axis(p, 2))
    suffixes = df['target_path'].apply(lambda p: Path(p).suffix)
    df['target_path'] = [
        f"targets/{target_type}/{variant}/{labeler}/{video_frame}{suffix}"
        for variant, labeler, video_frame, suffix in zip(variants, df['selected_labeler'], df['video_frame'], suffixes)
    ]
    return df


def with_target_variant(df: pd.DataFrame, target_variant: str, valid_variants: Optional[Sequence[str]] = None) -> pd.DataFrame:
    '''Return a copy of `df` pointing `target_path` at a different mask variant (e.g. `binary_c2_c4`).

    `valid_variants` restricts which variant names are accepted (datasets may expose different
    variants on disk, e.g. UNM also has `binary_c2_c3_c4`/`multiclass_c2_c3_c4`). Pass None to skip
    validation (the path is built regardless; a missing variant surfaces as a FileNotFoundError when loaded).
    '''
    if valid_variants is not None and target_variant not in valid_variants:
        raise ValueError(f"target_variant must be one of {valid_variants}, got '{target_variant}'")

    df = df.copy()
    types = df['target_path'].apply(lambda p: _target_path_axis(p, 1))
    suffixes = df['target_path'].apply(lambda p: Path(p).suffix)
    df['target_path'] = [
        f"targets/{t}/{target_variant}/{labeler}/{video_frame}{suffix}"
        for t, labeler, video_frame, suffix in zip(types, df['selected_labeler'], df['video_frame'], suffixes)
    ]
    df['target_variant'] = target_variant
    return df


def split_by_group(
    df: pd.DataFrame,
    group_column: str,
    seed: int = 42,
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    split_names: Sequence[str] = SPLIT_NAMES,
) -> pd.DataFrame:
    '''
    Assign each row of `df` to a split ('train' / 'val' / 'test' by default) based on
    `group_column`, so that every frame belonging to the same group (e.g. patient or video)
    lands in the same split.

    Groups are shuffled with `seed` and then greedily assigned to splits following `ratios`,
    targeting the requested proportion of *frames* (not groups) per split.

    Returns a copy of `df` with a `split` column added/overwritten.
    '''
    ratios = tuple(ratios)
    if len(ratios) != len(split_names):
        raise ValueError("`ratios` and `split_names` must have the same length.")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"`ratios` must sum to 1.0, got {ratios}.")

    frames_per_group = df.groupby(group_column).size()
    rng = np.random.RandomState(seed)
    shuffled_groups = rng.permutation(frames_per_group.index.to_numpy())

    total_frames = int(frames_per_group.sum())
    target_counts = [r * total_frames for r in ratios]

    group_to_split = {}
    split_idx = 0
    running_count = 0
    for group_id in shuffled_groups:
        while split_idx < len(split_names) - 1 and running_count >= target_counts[split_idx]:
            split_idx += 1
            running_count = 0
        group_to_split[group_id] = split_names[split_idx]
        running_count += frames_per_group[group_id]

    df = df.copy()
    df['split'] = df[group_column].map(group_to_split)
    return df


def split_by_patient(
    df: pd.DataFrame,
    seed: int = 42,
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    split_names: Sequence[str] = SPLIT_NAMES,
) -> pd.DataFrame:
    '''Convenience wrapper around `split_by_group` grouping by `paciente_id` (a patient may have more than one video/session).'''
    return split_by_group(df, 'paciente_id', seed=seed, ratios=ratios, split_names=split_names)


def apply_label_regime(
    df: pd.DataFrame,
    group_column: str = "paciente_id",
    label_fraction: float = 1.0,
    seed: int = 42,
    split_column: str = "split",
    label_regime_splits: Sequence[str] = ("train",),
) -> pd.DataFrame:
    '''
    Add an `is_labeled` boolean column to `df`, revealing the ground-truth label of only
    `label_fraction` of the frames belonging to `label_regime_splits` (by default just 'train'),
    grouped by `group_column` (by default `paciente_id`, so a patient's frames are never split
    between labeled and unlabeled). Rows outside `label_regime_splits` (e.g. val/test) always get
    `is_labeled=True`.

    Regimes are nested for a fixed `seed`: `split_by_group` shuffles the groups exactly once from
    `seed` alone (never from `ratios`), then walks that fixed order accumulating frame counts until
    each split's threshold is reached. For `f2 > f1`, the `f2` threshold is reached later (or at the
    same point) along that same fixed walk, so the 'labeled' prefix at `f1` is always a prefix of the
    'labeled' prefix at `f2` — i.e. `labeled_patients(f1) <= labeled_patients(f2)` whenever `f1 <= f2`
    and `seed` is held fixed. This lets label regimes (100%/50%/20%/10%, ...) be directly comparable:
    every patient revealed at a smaller fraction stays revealed at every larger fraction.

    No rows are ever dropped by this function — frames whose label is hidden remain in `df`,
    just flagged `is_labeled=False`, so example counts stay identical across label fractions
    (required for semi-supervised training, where hidden-label frames are used as unlabeled data).
    '''
    if not (0.0 <= label_fraction <= 1.0):
        raise ValueError(f"label_fraction must be in [0.0, 1.0], got {label_fraction}")

    df = df.copy()
    df['is_labeled'] = True

    applicable_mask = df[split_column].isin(label_regime_splits)
    if applicable_mask.any():
        subset = df.loc[applicable_mask]
        regime_df = split_by_group(
            subset, group_column, seed=seed,
            ratios=(label_fraction, 1.0 - label_fraction),
            split_names=("labeled", "unlabeled"),
        )
        df.loc[applicable_mask, "is_labeled"] = (regime_df["split"] == "labeled").to_numpy()

    return df


def build_unlabeled_frame_pool(
    dataset_path: Union[str, Path],
    annotated_df: pd.DataFrame,
    splits: Sequence[str] = ("train",),
    pool_size: Optional[int] = None,
    frame_stride: int = 1,
    seed: int = 42,
    frame_extension: str = ".png",
) -> pd.DataFrame:
    '''
    Build a DataFrame of video frames that were **never annotated**, to be used as the unlabeled
    stream of semi-supervised training.

    This is the difference between this dataset and the public benchmarks the semi-supervised
    literature is built on. There, a fully annotated dataset is *simulated* into a low-label regime
    by hiding labels, so "100% of the labels" means there is no unlabeled data left and every
    semi-supervised method collapses onto its supervised baseline. Here the annotations are the
    scarce resource (~1k annotated frames against ~46k frames on disk) and the unlabeled data is
    real and abundant, so it belongs in *every* regime -- including 100% -- and the question the
    experiment can answer becomes the useful one: what do the unannotated frames add on top of all
    the annotation we have?

    Frames are taken only from videos whose `split` is in `splits` (by default just `train`), so no
    frame of a validation or test patient ever enters training -- the unlabeled pool is a leakage
    surface exactly like the labeled data is.

    Args:
        dataset_path: dataset root; `image_path` values are resolved relative to it.
        annotated_df: the annotated-frame table, already carrying a `split` column (i.e. after
            `split_by_group`). Its rows supply the video-level metadata (patient, session, ...)
            that pool frames inherit from the video they come from.
        splits: splits whose videos may contribute unlabeled frames.
        pool_size: cap on the number of frames returned, sampled uniformly without replacement using
            `seed`. None keeps every eligible frame.
        frame_stride: keep every `frame_stride`-th frame of each video. Consecutive VFSS frames are
            nearly identical, so a stride buys diversity per unit of compute far more cheaply than a
            larger `pool_size` does.
        seed: seed of the sub-sampling, so a pool is reproducible and nested across runs.

    Returns:
        A DataFrame with the same columns as `annotated_df`, with `is_labeled=False`,
        `has_target=False` and `target_path=NA` (these frames have no ground truth at all --
        unlike a frame whose label is merely hidden by `apply_label_regime`).
    '''
    dataset_path = Path(dataset_path)
    if "split" not in annotated_df.columns:
        raise ValueError("`annotated_df` must already carry a 'split' column (call `split_by_group` first).")
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")

    eligible = annotated_df[annotated_df["split"].isin(tuple(splits))]
    if eligible.empty:
        raise ValueError(f"No annotated rows found for splits={tuple(splits)}; cannot locate the videos to pool from.")

    annotated_keys = set(zip(eligible["video_id"].astype(int), eligible["frame_id"].astype(int)))

    blocks = []
    for video_id, video_rows in eligible.groupby("video_id", sort=True):
        template = video_rows.iloc[0]
        frames_dir = Path(os.path.dirname(template.image_path))
        absolute_dir = dataset_path / frames_dir
        if not absolute_dir.is_dir():
            logger.warning(f"Frame directory not found for video {video_id}: {absolute_dir}. Skipping it.")
            continue

        frame_ids = sorted(
            int(path.stem) for path in absolute_dir.glob(f"*{frame_extension}") if path.stem.isdigit()
        )
        frame_ids = [f for f in frame_ids if (int(video_id), f) not in annotated_keys]
        if frame_stride > 1:
            frame_ids = frame_ids[::frame_stride]
        if not frame_ids:
            continue

        block = video_rows.iloc[[0] * len(frame_ids)].copy()
        block["frame_id"] = frame_ids
        block["video_frame"] = [f"v{int(video_id)}_f{frame_id}" for frame_id in frame_ids]
        block["image_path"] = [str(frames_dir / f"{frame_id}{frame_extension}") for frame_id in frame_ids]
        blocks.append(block)

    if not blocks:
        raise ValueError(
            f"The unlabeled pool is empty: no unannotated {frame_extension} frames were found under "
            f"{dataset_path} for splits={tuple(splits)}."
        )

    pool = pd.concat(blocks, ignore_index=True)
    for column in ("target_path", "target_dir"):
        if column in pool.columns:
            pool[column] = pd.NA
    pool["is_labeled"] = False
    pool["has_target"] = False

    if pool_size is not None and pool_size < len(pool):
        rng = np.random.RandomState(seed)
        selected = np.sort(rng.choice(len(pool), size=int(pool_size), replace=False))
        pool = pool.iloc[selected].reset_index(drop=True)

    logger.info(
        f"Unlabeled pool: {len(pool)} frames from {pool.video_id.nunique()} videos "
        f"(splits={tuple(splits)}, frame_stride={frame_stride}, pool_size={pool_size}, seed={seed})."
    )
    return pool


class VFSSFrameDatasetBase(Dataset):
    '''
    Base class owning everything that is not specific to loading a window of frames:
    CSV loading, target type/variant selection, train/val/test splitting, label-regime
    masking, path resolution and generic single-frame image/mask loading/preprocessing.

    Concrete subclasses (e.g. `VFSSWindowImageDataset`) add their own `__getitem__` and any
    frame-loading strategy on top of this, reusing `_lookup_row`/`_base_metadata` and the
    protected `_load_image`/`_load_mask_from_path`/`_preprocess_image`/`_preprocess_mask` helpers.
    '''

    def __init__(self,
                 dataset_path: Union[str, Path],
                 video_frame_table_filename='inca-video-frame-dataset.csv',
                 size=256,
                 return_metadata=True,
                 image_transform=None,
                 target_transform: list = None,
                 repeat_channels=False,
                 image_interpolation="bilinear",
                 mask_interpolation="nearest",
                 target_type: str = "mask",
                 target_variant: str = "raw",
                 target_variants: Sequence[str] = TARGET_VARIANTS,
                 split=None,
                 split_mode: str = "legado",
                 split_seed: int = 42,
                 split_ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
                 split_group_column: str = "paciente_id",
                 label_fraction: float = 1.0,
                 label_seed: int = 42,
                 label_group_column: str = "paciente_id",
                 label_regime_splits: Sequence[str] = ("train",),
                 use_unlabeled_pool: Optional[bool] = None,
                 unlabeled_pool_size: Optional[int] = None,
                 unlabeled_pool_frame_stride: int = 1,
                 unlabeled_pool_seed: int = 42,
                 unlabeled_pool_splits: Sequence[str] = ("train",),
                 expose_hidden_targets: bool = False):
        '''
        Args:
            dataset_path (str | Path): Diretório raiz do dataset.
            video_frame_table_filename (str): Nome do arquivo CSV contendo o DataFrame com as informações dos frames de vídeo e seus alvos.
            size (int): Dimensão de saída
            image_transform (callable, optional): Transformação a ser aplicada às imagens.
            target_transform (callable, optional): Transformação a ser aplicada à mascara.
            repeat_channels (bool): Se True, repete os canais das imagens em escala de cinza para criar imagens RGB.
            target_type (str): Modalidade do alvo ('mask' ou 'points').
            target_variant (str): Variante da máscara/pontos a usar (deve estar em `target_variants`).
            target_variants (Sequence[str]): Variantes válidas para este dataset (datasets diferentes podem expor variantes diferentes em disco).
            split (str, optional): O split do dataset a ser carregado ('train', 'val' ou 'test'). Se None, carrega todo o dataset.
            split_mode (str): 'legado' usa a coluna `split_legado` do CSV (divisão histórica, apenas retrocompatibilidade).
                'seed' ignora `split_legado` e divide o dataset em tempo de execução por `split_group_column`, usando `split_seed`/`split_ratios`.
            split_seed (int): Seed usada para gerar a divisão quando `split_mode='seed'`.
            split_ratios (Sequence[float]): Proporções (train, val, test) usadas quando `split_mode='seed'`. Devem somar 1.0.
            split_group_column (str): Coluna usada para agrupar frames que não podem ficar em splits diferentes
                (por padrão `paciente_id`; datasets sem esse identificador podem usar `video_id`).
            label_fraction (float): Fração (por `label_group_column`) dos frames de `label_regime_splits` que mantêm
                o rótulo revelado; o restante é marcado `is_labeled=False` mas continua no dataset (não é descartado).
                Regimes são aninhados para um `label_seed` fixo: ver `apply_label_regime`.
            label_seed (int): Seed usada para computar o regime de rótulo. Deve ser mantida fixa entre diferentes
                `label_fraction` para garantir o aninhamento dos conjuntos revelados.
            label_group_column (str): Coluna usada para agrupar frames ao aplicar o regime de rótulo (por padrão `paciente_id`).
            label_regime_splits (Sequence[str]): Splits sujeitos ao mascaramento de rótulo (por padrão só `('train',)`);
                qualquer split fora dessa lista (ex: val/test) sempre mantém `is_labeled=True`.
            use_unlabeled_pool (bool, optional): Liga/desliga o pool explicitamente. None (padrão)
                infere de `unlabeled_pool_size is not None`. Passe True com `unlabeled_pool_size=None`
                para um pool **sem teto** (todos os frames elegíveis) — sem esta flag, "sem teto" e
                "sem pool" seriam a mesma coisa.
            unlabeled_pool_size (int, optional): Teto do número de frames
                **nunca anotados** (ver `build_unlabeled_frame_pool`), amostrados dos vídeos de
                `unlabeled_pool_splits`. Diferentemente dos frames com rótulo escondido por
                `label_fraction`, esses frames existem em qualquer orçamento de anotação — inclusive
                em `label_fraction=1.0` — e são o fluxo não supervisionado real do dataset.
                None significa **sem teto** quando `use_unlabeled_pool=True`, e desliga o pool
                quando `use_unlabeled_pool` é None (comportamento histórico: só frames anotados).
            unlabeled_pool_frame_stride (int): Mantém 1 a cada N frames de cada vídeo no pool. Frames
                consecutivos de VFSS são quase idênticos; o stride compra diversidade mais barato que
                um pool maior.
            unlabeled_pool_seed (int): Seed da amostragem do pool.
            unlabeled_pool_splits (Sequence[str]): Splits cujos vídeos podem contribuir frames não
                anotados (por padrão só `train`, para não vazar pacientes de val/test).
            expose_hidden_targets (bool): Se True, cada amostra também carrega `hidden_segmentation`
                e `metadata['has_hidden_target']` — a ground truth que o regime de rótulo escondeu.
                Serve **apenas para diagnóstico** (medir a qualidade do pseudo-rótulo durante o
                treino); nenhuma loss deve lê-la. Frames do pool não têm anotação alguma e vêm com
                `has_hidden_target=False`.
        '''

        self.dataset_path = Path(dataset_path)
        self.video_frame_df = load_video_frame_dataframe(self.dataset_path, video_frame_table_filename)

        if target_type not in TARGET_TYPES:
            raise ValueError(f"target_type must be one of {TARGET_TYPES}, got '{target_type}'")
        if target_variant not in target_variants:
            raise ValueError(f"target_variant must be one of {target_variants}, got '{target_variant}'")
        self.target_type = target_type
        self.target_variant = target_variant
        self.target_variants = tuple(target_variants)
        self._is_multiclass_target = target_variant.startswith("multiclass")
        self.video_frame_df = with_target_type(self.video_frame_df, target_type)
        self.video_frame_df = with_target_variant(self.video_frame_df, target_variant)

        if split_mode not in ("legado", "seed"):
            raise ValueError(f"split_mode must be 'legado' or 'seed', got '{split_mode}'")
        self.split_mode = split_mode
        self.split = split
        self.split_group_column = split_group_column

        if split_mode == "legado":
            self.video_frame_df['split'] = self.video_frame_df['split_legado']
        else:
            self.split_seed = split_seed
            self.split_ratios = tuple(split_ratios)
            self.video_frame_df = split_by_group(
                self.video_frame_df, split_group_column, seed=split_seed, ratios=split_ratios
            )
            logger.info(
                f"Computed new '{split_group_column}'-level split (seed={split_seed}, ratios={self.split_ratios}) for reproducibility."
            )

        self.label_fraction = label_fraction
        self.label_seed = label_seed
        self.label_group_column = label_group_column
        self.label_regime_splits = tuple(label_regime_splits)
        self.video_frame_df = apply_label_regime(
            self.video_frame_df,
            group_column=label_group_column,
            label_fraction=label_fraction,
            seed=label_seed,
            split_column="split",
            label_regime_splits=self.label_regime_splits,
        )

        # The unlabeled pool is appended *after* the label regime: `label_fraction` is a fraction of
        # the annotated frames, and these frames were never annotated, so they are outside its
        # accounting entirely. That is what keeps an unsupervised stream available at every budget,
        # `label_fraction=1.0` included.
        self.unlabeled_pool_size = unlabeled_pool_size
        self.unlabeled_pool_frame_stride = unlabeled_pool_frame_stride
        self.unlabeled_pool_seed = unlabeled_pool_seed
        self.unlabeled_pool_splits = tuple(unlabeled_pool_splits)
        # `unlabeled_pool_size` is a *cap*, and None means "no cap" -- which is not the same as "no
        # pool". Without this second knob the two would collide and an uncapped pool would silently
        # become no pool at all.
        self.use_unlabeled_pool = (
            unlabeled_pool_size is not None if use_unlabeled_pool is None else bool(use_unlabeled_pool)
        )
        if self.use_unlabeled_pool:
            pool_df = build_unlabeled_frame_pool(
                self.dataset_path,
                self.video_frame_df,
                splits=self.unlabeled_pool_splits,
                pool_size=unlabeled_pool_size,
                frame_stride=unlabeled_pool_frame_stride,
                seed=unlabeled_pool_seed,
            )
            self.video_frame_df = pd.concat([self.video_frame_df, pool_df], ignore_index=True)

        self.expose_hidden_targets = expose_hidden_targets

        if split is not None:
            if split not in SPLIT_NAMES:
                raise ValueError(f"split must be one of {SPLIT_NAMES}, got '{split}'")
            self.video_frame_df = self.video_frame_df[self.video_frame_df['split'] == split]
            self.video_frame_df.reset_index(drop=True, inplace=True)

        assert self.video_frame_df.shape[0] > 0, (
            f"No samples found for split='{split}' (split_mode='{split_mode}'). "
            "Please check the provided dataset_path, split and split_mode."
        )

        self._total_frames_cache = {}
        self.image_size = (size, size)
        self.return_metadata = return_metadata
        self.repeat_channels = repeat_channels
        self.image_interpolation = self._parse_interpolation(image_interpolation)
        self.mask_interpolation = self._parse_interpolation(mask_interpolation)

        if self._is_multiclass_target and self.mask_interpolation != T.InterpolationMode.NEAREST:
            raise ValueError(
                f"mask_interpolation must be 'nearest' for multiclass target_variant='{target_variant}' "
                f"(got '{mask_interpolation}') -- any other interpolation blends class indices into "
                "meaningless intermediate values."
            )

        if image_transform:
            self.image_transform = image_transform
        else:
            self.image_transform = T.Resize(
                self.image_size, interpolation=self.image_interpolation
            )

        if target_transform:
            self.target_transform = target_transform
        else:
            self.target_transform = T.Resize(
                self.image_size, interpolation=self.mask_interpolation
            )

    def __len__(self):
        return self.video_frame_df.shape[0]

    @staticmethod
    def _parse_interpolation(name):
        table = {
            "nearest": T.InterpolationMode.NEAREST,
            "bilinear": T.InterpolationMode.BILINEAR,
            "bicubic": T.InterpolationMode.BICUBIC,
            "lanczos": T.InterpolationMode.LANCZOS,
        }
        key = str(name).lower()
        if key not in table:
            raise ValueError(f"Unsupported interpolation: {name}")
        return table[key]

    def _load_mask_from_path(self, path: str):
        ''' Load mask image from the given path. '''

        path = self._resolve_path(path)

        mask = Image.open(path).convert("L")
        mask = T.PILToTensor()(mask)

        return mask

    def _load_image(self, video_id: int, frame_id: int, path: str = None, color="greyscale"):
        '''
        Load image frame from path

        Args:
            video_id (int): The ID of the video to load the frame from.
            frame_id (int): The ID of the frame to load.
            path (str): The file path to load the image from.
            color (str): The color mode to load the image in ('greyscale' or 'rgb').

        Return:
            image (Tensor): The loaded image as a tensor.
        '''

        assert color in ['greyscale', 'rgb'], "Color mode must be either 'greyscale' or 'rgb'"
        assert path is not None or (
                video_id is not None
                and frame_id is not None
            ), "Either path or both video_id and frame_id must be provided."

        # Check if path is provided, if not, resolve path using video_id and frame_id
        if path is None:
            video_df = self.video_frame_df[(self.video_frame_df.video_id == video_id)]
            video_frame_row = video_df[video_df.frame_id == frame_id]

            # If the specific frame is not found, we can use the path of labeled frame in the same video to construct the path for the desired frame.
            if video_frame_row.empty:
                labeled_frame_path = video_df.iloc[0].image_path
                image_folder = os.path.dirname(labeled_frame_path)
                path = os.path.join(image_folder, f"{frame_id}.png")
            else:
                path = video_frame_row.iloc[0].image_path

        path = self._resolve_path(path)

        if color == 'greyscale':
            image_color = 'L'
        elif color == 'rgb':
            image_color = 'RGB'

        path = self._resolve_path(path)
        image = Image.open(path).convert(image_color)
        image = T.PILToTensor()(image)

        return image

    def _resolve_path(self, path: str):
        path = self.dataset_path / path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
        return Path(path).expanduser().resolve()

    def get_total_frames_in_video(self, video_id: Union[str, int], filetype='.png'):
        ''' Get the total number of frames in a video based on the video_id '''
        # Memoised: this is called once per `__getitem__` and each call is an `os.listdir` of a
        # directory holding every frame of the video. Harmless at ~1k annotated samples, a real cost
        # once the unlabeled pool multiplies the dataset by ~50.
        cache_key = (int(video_id), filetype)
        if cache_key in self._total_frames_cache:
            return self._total_frames_cache[cache_key]

        video_frame_row = self.video_frame_df[self.video_frame_df.video_id == video_id].iloc[0]
        image_folder_path = os.path.dirname(video_frame_row.image_path)
        image_folder_path = self._resolve_path(image_folder_path)

        frame_files = [f for f in os.listdir(image_folder_path) if f.endswith(filetype)]
        total_frames = len(frame_files)
        self._total_frames_cache[cache_key] = total_frames
        return total_frames

    def _preprocess_image(self, image: torch.Tensor):
        '''Preprocess the image tensor (e.g., normalization)'''

        # logger.debug(f"Original image dimension: {image.shape} ({image.dtype}). Range: [{image.min()}, {image.max()}]")
        image = image.float()

        if image.max() > 1.0:
            image = image / 255.0

        image_min = image.amin(dim=[-1, -2], keepdim=True)
        image_max = image.amax(dim=[-1, -2], keepdim=True)

        # Normalize to [0, 1]
        image = (image - image_min) / (image_max - image_min + 1e-8)
        image = (image * 2.0) - 1.0  # Scale to [-1, 1]

        if self.image_transform:
            image = self.image_transform(image)

        logger.debug(f"Preprocessed image dimension: {image.shape}. Range: [{image.min()}, {image.max()}]")
        return image

    def _preprocess_mask(self, mask: torch.Tensor):
        '''Preprocess the mask tensor (resize + binarize for binary/raw targets, or resize while
        preserving class indices for multiclass targets -- see `self._is_multiclass_target`).'''
        logger.debug(f"Original mask dimension: {mask.shape} ({mask.dtype}). Range: [{mask.min()}, {mask.max()}]")

        if self._is_multiclass_target:
            mask = mask.long()

            if self.target_transform:
                mask = self.target_transform(mask)

            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)

            logger.debug(f"Preprocessed mask dimension: {mask.shape} ({mask.dtype}). Range: [{mask.min()}, {mask.max()}]")
            return mask.long()

        if mask.max() > 1.0:
            mask = mask.float() / 255.0

        if self.target_transform:
            mask = self.target_transform(mask)

        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)

        mask = (mask > 0).long()

        logger.debug(f"Preprocessed mask dimension: {mask.shape} ({mask.dtype}). Range: [{mask.min()}, {mask.max()}]")
        return mask

    def _lookup_row(self, idx=None, video_id=None, frame_id=None, video_frame=None) -> pd.Series:
        '''Uniquely resolve a row of `video_frame_df` from an index or a (video_id, frame_id) or video_frame identifier.'''

        if idx is None and (video_id is None or frame_id is None) and video_frame is None:
            raise ValueError("Either idx or (video_id and frame_id) or video_frame must be provided.")

        if idx is not None:
            row = self.video_frame_df[self.video_frame_df.index == idx]
        elif video_id is not None and frame_id is not None:
            row = self.video_frame_df[(self.video_frame_df.video_id == video_id) & (self.video_frame_df.frame_id == frame_id)]
        elif video_frame is not None:
            row = self.video_frame_df[self.video_frame_df.video_frame == video_frame]

        assert row is not None, "No matching row found for the given identifiers."
        assert row.shape[0] == 1, f"Multiple rows found for the given identifiers. Please ensure they uniquely identify a single row.\nidx={idx}\nrow={row}\nvideo_frame_df={self.video_frame_df}"

        return row.iloc[0]

    def _base_metadata(self, row: pd.Series) -> dict:
        '''Build the metadata fields common to every dataset variant built on this base class.'''
        metadata = {
            'frame_id': int(row.frame_id),
            'video_id': int(row.video_id),
            'video_frame': row.video_frame,
            'selected_labeler': row.selected_labeler,
            'target_type': self.target_type,
            'target_variant': row.target_variant,
            'split': row.split,
            'is_labeled': bool(row.is_labeled),
            # `is_labeled=False, has_target=True`  -> annotated frame, label hidden by the regime.
            # `is_labeled=False, has_target=False` -> frame from the unlabeled pool, never annotated.
            'has_target': bool(row.get('has_target', True)),
        }
        for optional_column in ('paciente_id', 'momento', 'procedimento'):
            if optional_column in self.video_frame_df.columns:
                metadata[optional_column] = row[optional_column]
        return metadata

    def __getitem__(self, idx=None, video_id=None, frame_id=None, video_frame=None):
        raise NotImplementedError("Subclasses of VFSSFrameDatasetBase must implement __getitem__.")


class VFSSWindowImageDataset(VFSSFrameDatasetBase):
    '''
    Retorna uma janela de frames (não rotulados, exceto o central) ao redor de um frame central,
    junto com a máscara do frame central (quando `is_labeled=True` para esse frame — ver
    `apply_label_regime`/`label_fraction` na classe base).
    '''

    def __init__(self, window_size: int = 1, stride: int = 1, **kwargs):
        '''
        Args:
            window_size (int): O número total de frames a serem carregados em cada janela (deve ser ímpar).
            stride (int): O passo entre os frames na janela.
            **kwargs: Demais parâmetros repassados para `VFSSFrameDatasetBase`.
        '''
        super().__init__(**kwargs)
        self.window_size = window_size
        self.stride = stride

    def get_valid_window(self, frame_id, total_frames, window_size=3, stride=1, boundary_mode='repeat'):
        if window_size % 2 == 0:
            raise ValueError("Window size must be odd.")

        half_window = window_size // 2
        valid_widow_frame_ids = {'begin': [], 'end': []}
        for boundary in ['begin', 'end']:
            last_valid_frame_id = frame_id

            for i_window in range(1, half_window + 1):
                # Determine the direction to check based on the boundary type.
                #   For 'begin', we check frames before the current frame (negative direction)
                #   For 'end', we check frames after the current frame (positive direction).
                direction = -1 if boundary == 'begin' else 1
                check_frame_id = frame_id + direction * (i_window * stride)

                is_frame_id_valid = check_frame_id > 0 and check_frame_id <= total_frames
                if is_frame_id_valid:
                    valid_widow_frame_ids[boundary].append(check_frame_id)
                    last_valid_frame_id = check_frame_id
                else:
                    if boundary_mode == 'repeat':
                        valid_widow_frame_ids[boundary].append(last_valid_frame_id)
                    elif boundary_mode == 'constant':
                        # Use None to indicate out-of-bound frames
                        valid_widow_frame_ids[boundary].append(None)
                    else:
                        raise ValueError("Boundary mode must be either 'repeat' or 'constant'")

        window = valid_widow_frame_ids['begin'][::-1] + [frame_id] + valid_widow_frame_ids['end']
        return window

    def __load_window_images_from_path(self, video_id: Union[str, int], frame_id: Union[str, int], window_size: int, stride: int, color="greyscale", boundary_mode='repeat', repeat_channels=False):
        '''
        Load a window of image frames from a video based on the given video_id and frame_id.

        Args:
            video_id (str | int): The ID of the video to load frames from.
            frame_id (str | int): The ID of the central frame in the window to load.
            window_size (int): The total number of frames to load in the window (must be odd).
            stride (int): The stride between frames in the window.
            color (str): The color mode to load the images in ('greyscale' or 'rgb').
            boundary_mode (str): The mode to handle out-of-bound frames ('repeat' or 'constant').
                Handling out-of-bound frames:
                - 'repeat': Repeat the last valid frame when the window exceeds the video boundaries.
                - 'constant': Use a constant value (e.g., zero) for out-of-bound frames.

                Repeat mode example:
                For a video with 100 frames, window_size=5 and stride=1:
                - If frame_id=1 (beginning of the video)
                    Window frame IDs: [-2, -1, 0, 1, 2]
                    Resolved frame IDs with 'repeat': [1, 1, 1, 1, 2]
                - If frame_id=100 (end of the video)
                    Window frame IDs: [98, 99, 100, 101, 102]
                    Resolved frame IDs with 'repeat': [98, 99, 100, 100, 100]

        Returns:
            images (Tensor): A tensor of size (window_size, C, H, W) containing the loaded image frames in the specified color mode.
        '''
        total_frames = self.get_total_frames_in_video(video_id)
        valid_window_frame_ids = self.get_valid_window(frame_id, total_frames, window_size, stride, boundary_mode)
        images = []
        for i, valid_frame_id in enumerate(valid_window_frame_ids):
            if valid_frame_id is not None:
                image = self._load_image(video_id, valid_frame_id, color=color)
            else:
                n_channels = 3 if color == 'rgb' else 1
                image = torch.zeros((n_channels, self.image_size[0], self.image_size[1]))
            images.append(image)

        self.__current_image_original_dim = (images[0].shape[-1], images[0].shape[-2])
        images = torch.stack(images, dim=0)

        if repeat_channels and color == 'greyscale':
            images = images.repeat(1, 3, 1, 1)

        if window_size == 1:
            images = images.squeeze(0)

        return images, valid_window_frame_ids

    def __getitem__(self, idx=None, video_id=None, frame_id=None, video_frame=None):
        '''
        Get a sample from the dataset based on the provided identifiers.
        The sample includes a window of image frames and the corresponding ground truth mask, along with metadata

        Args:
            idx (int, optional): The index of the sample to retrieve. If provided, it takes precedence over video_id and frame_id.
            video_id (str | int, optional): The ID of the video to retrieve the sample from. Must be provided if idx is not provided.
            frame_id (str | int, optional): The ID of the frame to retrieve the sample from. Must be provided if idx is not provided.
            video_frame (str, optional): The combined identifier in the format "video_{video_id}_frame_{frame_id}". Must be provided if idx is not provided.

        Returns:
            A dict containing:
            - 'image' (Tensor): A tensor of size (window_size, C, H, W) containing the preprocessed image frames in the specified color mode.
            - 'segmentation' (Tensor): The preprocessed ground truth mask for the central frame, or a placeholder of
              zeros (same shape/dtype) when the central frame's label is hidden under the current label regime
              (`metadata['is_labeled'] is False` — see `VFSSFrameDatasetBase.label_fraction`).
            - 'metadata' (dict, optional): A dictionary containing metadata about the sample, including:
                - 'frame_id': The ID of the central frame in the window.
                - 'video_id': The ID of the video the sample belongs to.
                - 'video_frame': The combined identifier for the video and frame.
                - 'window_size': The size of the window used to load image frames.
                - 'stride': The stride used to load image frames in the window.
                - 'window_frame_ids': The list of frame IDs included in the loaded window of images.
                - 'is_labeled': Whether the central frame's ground truth is revealed under the current label regime.
                - 'original_dim': The original dimensions of the loaded images before preprocessing.
        '''

        row = self._lookup_row(idx=idx, video_id=video_id, frame_id=frame_id, video_frame=video_frame)

        frame_id = int(row.frame_id)
        video_id = int(row.video_id)
        frames, frame_ids = self.__load_window_images_from_path(
            video_id,
            frame_id,
            self.window_size,
            self.stride,
            color="greyscale",
            boundary_mode='repeat',
            repeat_channels=self.repeat_channels
        )
        frames = self._preprocess_image(frames)

        has_target = bool(row.get('has_target', True))
        if row.is_labeled:
            gt_mask = self._load_mask_from_path(row.target_path)
            gt_mask = self._preprocess_mask(gt_mask)
        else:
            gt_mask = torch.zeros(self.image_size, dtype=torch.long)

        metadata = self._base_metadata(row)
        metadata.update({
            'window_size': self.window_size,
            'stride': self.stride,
            'window_frame_ids': frame_ids,
            'original_dim': self.__current_image_original_dim,
        })

        returns = {}
        returns['image'] = frames
        returns['segmentation'] = gt_mask
        returns['metadata'] = metadata

        if self.expose_hidden_targets:
            # Diagnostics only -- see `expose_hidden_targets`. The key is always present (collation
            # needs a consistent schema); `has_hidden_target` says whether it means anything.
            if has_target and not row.is_labeled:
                hidden = self._preprocess_mask(self._load_mask_from_path(row.target_path))
            elif has_target:
                hidden = gt_mask
            else:
                hidden = torch.zeros(self.image_size, dtype=torch.long)
            returns['hidden_segmentation'] = hidden
            metadata['has_hidden_target'] = has_target

        return returns


class VFSSIncaTrain(VFSSWindowImageDataset):
    def __init__(self, **kwargs):
        super().__init__(
            split='train',
            **kwargs
        )


class VFSSIncaVal(VFSSWindowImageDataset):
    def __init__(self, **kwargs):
        super().__init__(
            split='val',
            **kwargs
        )

class VFSSIncaTest(VFSSWindowImageDataset):
    def __init__(self, **kwargs):
        super().__init__(
            split='test',
            **kwargs
        )

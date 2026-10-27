'''
The article protocol split: one fixed, persisted, patient-level partition shared by every method.

Why this is not `split_by_group`. That function (used by every X1 config through `split_mode: seed`)
recomputes the split on every run and targets a proportion of *frames*, so with seed 42 the
"10%" validation set ends up with 25 patients and the "20%" test set with 20. The article protocol
asks for proportions of *patients*, for five fixed folds inside the development set, and for the
assignment to live on disk so every experiment provably sees the same patients.

Why stratify. INCA is extremely unbalanced per patient: 15 of 114 patients have 3 to 12 videos and
hold 564 of the 961 annotated frames, while the other 99 have a single video of ~4 frames. An
unstratified draw can put none or four of the heavy patients into an 11-patient validation set
(40 to 207 frames over 200 simulated seeds). Splitting each stratum separately keeps the patient
counts exact while spreading the heavy patients evenly over partitions and folds.

The table has one row per patient:
    paciente_id, partition ('dev' | 'val' | 'test'), fold (0..n_folds-1 for 'dev', empty otherwise),
    stratum ('multi_video' | 'single_video'), n_videos, n_frames
'''

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PARTITIONS = ("dev", "val", "test")
STRATA = ("multi_video", "single_video")
SPLIT_TABLE_COLUMNS = ("paciente_id", "partition", "fold", "stratum", "n_videos", "n_frames")
# (dev, val, test), as proportions of patients.
PROTOCOL_RATIOS = (0.7, 0.1, 0.2)


def _round_half_up(value: float) -> int:
    '''`round()` would round 1.5 to 2 but 2.5 to 2 (banker's rounding); the split must not depend on that.'''
    return int(np.floor(value + 0.5))


def resolve_project_path(path: Union[str, Path]) -> Path:
    '''A relative path is tried against the cwd first, then the repo root -- notebooks run from their own folder.'''
    path = Path(path).expanduser()
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


def make_protocol_split(
    df: pd.DataFrame,
    seed: int = 42,
    ratios: Sequence[float] = PROTOCOL_RATIOS,
    n_folds: int = 5,
    group_column: str = "paciente_id",
    video_column: str = "video_id",
) -> pd.DataFrame:
    '''
    Build the per-patient protocol table from the annotated-frame table `df`.

    The rule, in order:
      1. Patient totals: `n_test = round_half_up(test_ratio * N)`, `n_val = round_half_up(val_ratio * N)`,
         and the rest is `dev`.
      2. The `multi_video` stratum gets `round_half_up(ratio * n_multi)` test and val patients; the
         `single_video` stratum takes whatever completes the totals.
      3. Inside each stratum, patients (sorted by id) are permuted with `RandomState(seed)` and taken
         in the order test, val, dev. The multi stratum is permuted first, so it alone determines
         the first draw.
      4. Folds: `multi_video` dev patients are sorted by frame count (descending, ties in permuted
         order) and dealt in a serpentine 0..k-1, k-1..0, so each fold gets one large and one small
         patient; `single_video` dev patients then go, in permuted order, to whichever fold has the
         fewest patients so far (ties: lowest fold), which keeps fold sizes within one patient.

    Args:
        df: one row per annotated frame, carrying `group_column` and `video_column`.
        ratios: `(dev, val, test)` proportions of patients; must sum to 1.
    '''
    ratios = tuple(float(r) for r in ratios)
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"`ratios` must be (dev, val, test) summing to 1.0, got {ratios}.")
    if n_folds < 2:
        raise ValueError(f"`n_folds` must be at least 2, got {n_folds}.")
    _, val_ratio, test_ratio = ratios

    patients = (
        df.groupby(group_column)
        .agg(n_videos=(video_column, "nunique"), n_frames=(video_column, "size"))
        .reset_index()
        .rename(columns={group_column: "paciente_id"})
        .sort_values("paciente_id", kind="mergesort")
        .reset_index(drop=True)
    )
    patients["stratum"] = np.where(patients.n_videos > 1, "multi_video", "single_video")

    n_total = len(patients)
    totals = {"test": _round_half_up(test_ratio * n_total), "val": _round_half_up(val_ratio * n_total)}
    multi = patients[patients.stratum == "multi_video"]
    multi_counts = {"test": _round_half_up(test_ratio * len(multi)),
                    "val": _round_half_up(val_ratio * len(multi))}
    counts = {
        "multi_video": multi_counts,
        "single_video": {name: totals[name] - multi_counts[name] for name in ("test", "val")},
    }

    rng = np.random.RandomState(seed)
    partition: Dict = {}
    fold: Dict = {}
    for stratum in STRATA:
        members = patients[patients.stratum == stratum]
        n_test, n_val = counts[stratum]["test"], counts[stratum]["val"]
        if min(n_test, n_val) < 0 or n_test + n_val > len(members):
            raise ValueError(
                f"Stratum '{stratum}' has {len(members)} patients, too few for {n_test} test + "
                f"{n_val} val. Change `ratios` or the dataset."
            )
        order = rng.permutation(members.paciente_id.to_numpy())
        for patient in order[:n_test]:
            partition[patient] = "test"
        for patient in order[n_test:n_test + n_val]:
            partition[patient] = "val"
        dev = list(order[n_test + n_val:])
        for patient in dev:
            partition[patient] = "dev"

        if stratum == "multi_video":
            frames = members.set_index("paciente_id").n_frames
            # A stable sort over the permuted order: ties keep the random order, not the id order.
            dev = sorted(dev, key=lambda p: -int(frames[p]))
            serpentine = list(range(n_folds)) + list(reversed(range(n_folds)))
            for index, patient in enumerate(dev):
                fold[patient] = serpentine[index % len(serpentine)]
        else:
            fold_sizes = [sum(1 for f in fold.values() if f == k) for k in range(n_folds)]
            for patient in dev:
                target = int(np.argmin(fold_sizes))   # first minimum: ties go to the lowest fold
                fold[patient] = target
                fold_sizes[target] += 1

    table = patients.assign(
        partition=patients.paciente_id.map(partition),
        fold=patients.paciente_id.map(fold).astype("Int64"),
    )
    return table[list(SPLIT_TABLE_COLUMNS)]


def summarize_protocol_split(table: pd.DataFrame) -> pd.DataFrame:
    '''Patients / multi-video patients / videos / frames per partition and per dev fold.'''
    rows = []
    for name in PARTITIONS:
        part = table[table.partition == name]
        rows.append({"partition": name, "fold": pd.NA, **_counts(part)})
    dev = table[table.partition == "dev"]
    for fold_id in sorted(dev.fold.dropna().unique()):
        rows.append({"partition": "dev", "fold": int(fold_id), **_counts(dev[dev.fold == fold_id])})
    return pd.DataFrame(rows)


def _counts(part: pd.DataFrame) -> Dict[str, int]:
    return {
        "patients": int(len(part)),
        "multi_video_patients": int((part.stratum == "multi_video").sum()),
        "videos": int(part.n_videos.sum()),
        "frames": int(part.n_frames.sum()),
    }


def load_protocol_split(path: Union[str, Path]) -> pd.DataFrame:
    '''
    Read and validate a protocol table. `paciente_id` is read as a string: INCA ids are ints and UNM
    ids are strings, and the comparison against the frame table is always done on strings.
    '''
    path = resolve_project_path(path)
    table = pd.read_csv(path, dtype={"paciente_id": str})
    missing = set(SPLIT_TABLE_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}.")
    table["fold"] = table["fold"].astype("Int64")

    if table.paciente_id.duplicated().any():
        raise ValueError(f"{path}: duplicated patients {table.paciente_id[table.paciente_id.duplicated()].tolist()}.")
    unknown = set(table.partition) - set(PARTITIONS)
    if unknown:
        raise ValueError(f"{path}: unknown partitions {sorted(unknown)}.")
    dev = table.partition == "dev"
    if table.loc[dev, "fold"].isna().any() or table.loc[~dev, "fold"].notna().any():
        raise ValueError(f"{path}: every 'dev' patient needs a fold, and only 'dev' patients may have one.")
    return table


def assign_protocol_split(
    df: pd.DataFrame,
    table: pd.DataFrame,
    train_folds: Optional[Sequence[int]] = None,
    cv_fold: Optional[int] = None,
    group_column: str = "paciente_id",
    split_column: str = "split",
) -> pd.DataFrame:
    '''
    Write the per-run `split` column of the frame table from the protocol table.

        train <- dev patients whose fold is in `train_folds` (default: every fold except `cv_fold`)
        val   <- the fixed validation partition
        test  <- dev patients of `cv_fold`; nobody when `cv_fold` is None
        NaN   <- everyone else, the whole INCA test partition included

    NaN is the same mechanism `split_mode: legado` already relies on for its unassigned frames:
    those rows match no split, and the label regime and the unlabeled pool only look at 'train'.
    '''
    key = df[group_column].astype(str)
    in_table = set(table.paciente_id)
    in_frames = set(key)
    if in_frames - in_table:
        raise ValueError(
            f"Patients missing from the protocol split: {sorted(in_frames - in_table)[:10]}. "
            "The frame table changed after the split was generated."
        )
    if in_table - in_frames:
        raise ValueError(
            f"The protocol split lists patients absent from the frame table: "
            f"{sorted(in_table - in_frames)[:10]}. The frame table changed after the split was generated."
        )

    folds = sorted(int(f) for f in table.fold.dropna().unique())
    if cv_fold is not None and int(cv_fold) not in folds:
        raise ValueError(f"cv_fold={cv_fold} is not a fold of this split ({folds}).")
    if train_folds is None:
        train_folds = [f for f in folds if cv_fold is None or f != int(cv_fold)]
    train_folds = [int(f) for f in train_folds]
    if not train_folds:
        raise ValueError("`train_folds` is empty: nothing to train on.")
    unknown = set(train_folds) - set(folds)
    if unknown:
        raise ValueError(f"train_folds {sorted(unknown)} are not folds of this split ({folds}).")
    if cv_fold is not None and int(cv_fold) in train_folds:
        raise ValueError(f"cv_fold={cv_fold} is also in train_folds={train_folds}: the evaluated fold would be trained on.")

    split_of: Dict[str, str] = {}
    for row in table.itertuples(index=False):
        if row.partition == "val":
            split_of[row.paciente_id] = "val"
        elif row.partition == "dev" and int(row.fold) in train_folds:
            split_of[row.paciente_id] = "train"
        elif row.partition == "dev" and cv_fold is not None and int(row.fold) == int(cv_fold):
            split_of[row.paciente_id] = "test"

    df = df.copy()
    df[split_column] = key.map(split_of)
    return df

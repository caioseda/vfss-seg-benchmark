'''
Prepare the ACDC dataset for the semi-supervised reproduction notebook.

Why this exists
---------------
The DiffRect and SSL4MIS results this repository is checked against are reported on a *specific*
preprocessing of ACDC and a *specific* split. Reproducing the numbers means reproducing both, not
just "using ACDC":

  - **Preprocessing**, transcribed from SSL4MIS's `code/dataloaders/acdc_data_processing.py` (which
    DiffRect vendors unchanged as `dataloaders/acdc_data_processing.py`): read the annotated
    end-diastole / end-systole volumes, min-max normalise each *volume* to [0, 1] as float32, and
    store it slice by slice alongside its integer label map.
  - **Split**, taken verbatim from SSL4MIS's `data/ACDC/{train,val,test,train_slices}.list`:
    140 / 20 / 40 volumes and 1312 training slices. These files are the split; regenerating them
    with our own seed would produce a different experiment that happens to use the same dataset.

The preprocessed archive SSL4MIS distributes lives on Baidu Disk, which cannot be fetched from a
script. So this downloads the *raw* ACDC training set from an ungated Hugging Face mirror
(`MedOtter/ACDC`, which preserves the original `training/patientXXX/` layout) and applies the
transcribed preprocessing, reading the split lists straight from SSL4MIS.

Deviation from the reference, container format only
---------------------------------------------------
The reference writes one HDF5 file per slice. This writes one compressed `.npz` per volume, for two
reasons: it removes an h5py dependency, and an HDF5 handle opened in `Dataset.__init__` and read
from forked DataLoader workers is a well-known source of silent corruption. The *content* -- volume
level min-max normalisation, float32 images, uint8 labels, slice indexing -- is unchanged, which is
what the reproduction depends on.

Licence
-------
ACDC is distributed under CC BY-NC-SA 4.0 and requires citing Bernard et al., IEEE TMI 2018
("Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis:
Is the Problem Solved?"). Research use only.

Usage
-----
    python scripts/prepare_acdc.py --root /data_ssd/caioseda/data/ACDC
    python scripts/prepare_acdc.py --root ... --skip-download    # raw already on disk
'''

import argparse
import pathlib
import re
import sys
from typing import Dict, List, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HF_REPO_ID = "MedOtter/ACDC"
SSL4MIS_LISTS_URL = "https://raw.githubusercontent.com/HiLab-git/SSL4MIS/master/data/ACDC"
SPLIT_LISTS = ("train", "val", "test", "train_slices")

# What the SSL4MIS lists must contain if the download and the split have not drifted.
EXPECTED_COUNTS = {"train": 140, "val": 20, "test": 40, "train_slices": 1312}
EXPECTED_PATIENTS = 100
EXPECTED_VOLUMES = 200


def download_raw(root: pathlib.Path, repo_id: str = HF_REPO_ID) -> pathlib.Path:
    '''Fetch the raw ACDC training set from the Hugging Face mirror. Returns the `training/` dir.'''
    from huggingface_hub import snapshot_download

    print(f"Downloading {repo_id} (raw ACDC, ~2 GB) ...")
    local = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(root / "raw"),
        allow_patterns=["training/**"],
    )
    training_dir = pathlib.Path(local) / "training"
    _verify_download(training_dir, repo_id)
    return training_dir


def _verify_download(training_dir: pathlib.Path, repo_id: str = HF_REPO_ID) -> None:
    '''
    Check that every annotated frame downloaded as an (image, ground truth) *pair*.

    Observed in practice: `snapshot_download` returned success having silently skipped two files
    (`patient027_frame01.nii.gz` and `patient087_frame01_gt.nii.gz`), leaving 199 images and 199
    masks that did not pair up. Both were present in the remote repository. Counting files would
    not have caught it -- the totals matched -- so the check has to be on the pairing.

    ACDC's training set is 100 patients x 2 annotated frames (ED and ES) = 200 pairs. A run built on
    198 of them is not the published split, so this fails rather than warns.
    '''
    images = {p.with_name(p.name.replace(".nii.gz", "")).name
              for p in training_dir.glob("patient*/patient*_frame*.nii.gz")
              if not p.name.endswith("_gt.nii.gz")}
    masks = {p.name.replace("_gt.nii.gz", "") for p in training_dir.glob("patient*/*_gt.nii.gz")}

    missing_images = sorted(masks - images)
    missing_masks = sorted(images - masks)
    problems = []
    if missing_images:
        problems.append(f"{len(missing_images)} mask(s) with no image: {missing_images[:5]}")
    if missing_masks:
        problems.append(f"{len(missing_masks)} image(s) with no mask: {missing_masks[:5]}")
    paired = images & masks
    if len(paired) != EXPECTED_VOLUMES:
        problems.append(f"{len(paired)} complete pairs, expected {EXPECTED_VOLUMES}")

    if problems:
        raise RuntimeError(
            f"The download from {repo_id} is incomplete:\n  - "
            + "\n  - ".join(problems)
            + f"\n\nRe-run without --skip-download to fill the gaps, or fetch the individual "
              f"files with huggingface_hub.hf_hub_download(repo_id='{repo_id}', "
              f"repo_type='dataset', filename='training/<patient>/<file>')."
        )
    print(f"Download verified: {len(paired)} image/mask pairs.")


def fetch_split_lists(root: pathlib.Path) -> Dict[str, List[str]]:
    '''
    Download SSL4MIS's split lists, which *are* the experiment's split.

    Cached under `<root>/lists/` so a second run is offline.
    '''
    import urllib.request

    lists_dir = root / "lists"
    lists_dir.mkdir(parents=True, exist_ok=True)

    splits: Dict[str, List[str]] = {}
    for name in SPLIT_LISTS:
        path = lists_dir / f"{name}.list"
        if not path.exists():
            url = f"{SSL4MIS_LISTS_URL}/{name}.list"
            print(f"Fetching {url}")
            with urllib.request.urlopen(url) as response:
                path.write_bytes(response.read())
        entries = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        splits[name] = entries

        expected = EXPECTED_COUNTS[name]
        if len(entries) != expected:
            raise RuntimeError(
                f"{name}.list has {len(entries)} entries, expected {expected}. The upstream split "
                f"has changed; the reproduction targets no longer apply. Delete {path} to refetch."
            )
    return splits


def _canonical_cases(patient: pathlib.Path) -> List[Tuple[str, pathlib.Path, pathlib.Path]]:
    '''
    Map a patient's two annotated phases onto SSL4MIS's case names.

    SSL4MIS **renumbers** the annotated frames: raw ACDC names them by their index in the cine loop
    (patient001 has `frame01` and `frame12`; patient099 has `frame01` and `frame09`), but every
    entry in `train.list` / `val.list` / `test.list` ends in `_frame01` or `_frame02` -- exactly 100
    of each. So `frame01` is end-diastole and `frame02` is end-systole, whatever the original
    indices were.

    Getting this backwards would not fail loudly: both phases of a patient always land in the same
    split (verified: 0 of 100 patients are split across train/val/test), so the *split* would still
    be right. What would silently change is `train_slices.list`, whose first N rows are the labeled
    subset for the 1%/5%/10% regimes -- the low-label cells would train on the wrong phase.

    The phases are read from each patient's `Info.cfg` rather than inferred from sort order. On this
    dataset the two agree (ED < ES for all 100 patients, and the annotated frames are exactly
    {ED, ES}), but `Info.cfg` is the authoritative statement and the check is free.
    '''
    config = {}
    for line in (patient / "Info.cfg").read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            config[key.strip()] = value.strip()

    phases = [("frame01", int(config["ED"])), ("frame02", int(config["ES"]))]
    annotated = sorted(int(re.search(r"_frame(\d+)\.nii\.gz$", path.name).group(1))
                       for path in patient.glob("*_frame*.nii.gz")
                       if not path.name.endswith("_gt.nii.gz"))
    if annotated != sorted(index for _, index in phases):
        raise RuntimeError(
            f"{patient.name}: annotated frames {annotated} do not match Info.cfg's "
            f"ED/ES {sorted(index for _, index in phases)}."
        )

    cases = []
    for canonical, index in phases:
        raw = f"{patient.name}_frame{index:02d}"
        image_path = patient / f"{raw}.nii.gz"
        gt_path = patient / f"{raw}_gt.nii.gz"
        if not image_path.exists() or not gt_path.exists():
            raise FileNotFoundError(
                f"{patient.name}: expected {raw}.nii.gz and {raw}_gt.nii.gz for phase {canonical}."
            )
        cases.append((f"{patient.name}_{canonical}", image_path, gt_path))
    return cases


def convert(training_dir: pathlib.Path, out_dir: pathlib.Path, force: bool = False) -> int:
    '''
    Raw `.nii.gz` -> one `.npz` per annotated volume.

    Transcribed from `acdc_data_processing.py`: only volumes that have a `_gt` file are kept (the
    two annotated cardiac phases per patient), each image is min-max normalised over the whole
    volume and cast to float32, and the label map is stored as-is.
    '''
    import numpy as np
    import nibabel as nib

    if not training_dir.is_dir():
        raise FileNotFoundError(
            f"{training_dir} does not exist. Expected the original ACDC layout "
            f"(training/patient001/patient001_frame01.nii.gz). Run without --skip-download, or "
            f"point --root at a directory whose `raw/training/` holds it."
        )

    patients = sorted(p for p in training_dir.iterdir() if p.is_dir() and p.name.startswith("patient"))
    if len(patients) != EXPECTED_PATIENTS:
        raise RuntimeError(
            f"found {len(patients)} patient directories under {training_dir}, expected "
            f"{EXPECTED_PATIENTS}. The mirror's layout has changed."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for patient in patients:
        for case, image_path, gt_path in _canonical_cases(patient):
            target = out_dir / f"{case}.npz"
            if target.exists() and not force:
                written += 1
                continue

            # nibabel yields [X, Y, Z]; the reference reads with SimpleITK, which yields [Z, Y, X].
            # Transposed here so the slice axis is first, matching the reference's `image[slice_ind]`.
            image = np.asanyarray(nib.load(str(image_path)).dataobj).transpose(2, 1, 0)
            label = np.asanyarray(nib.load(str(gt_path)).dataobj).transpose(2, 1, 0)
            if image.shape != label.shape:
                raise RuntimeError(f"{case}: image {image.shape} and label {label.shape} disagree.")

            image = (image - image.min()) / (image.max() - image.min())
            np.savez_compressed(target, image=image.astype(np.float32), label=label.astype(np.uint8))
            written += 1

    if written != EXPECTED_VOLUMES:
        raise RuntimeError(f"converted {written} volumes, expected {EXPECTED_VOLUMES}.")
    return written


def verify(root: pathlib.Path, splits: Dict[str, List[str]]) -> None:
    '''Every case named in a split list must exist on disk, and the slice counts must line up.'''
    import numpy as np

    volumes = root / "volumes"
    missing = [case for name in ("train", "val", "test") for case in splits[name]
               if not (volumes / f"{case}.npz").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} cases from the split lists are missing, e.g. {missing[:3]}")

    total_slices = 0
    for case in splits["train"]:
        with np.load(volumes / f"{case}.npz") as data:
            total_slices += data["image"].shape[0]
    if total_slices != EXPECTED_COUNTS["train_slices"]:
        raise RuntimeError(
            f"the 140 training volumes hold {total_slices} slices, but train_slices.list has "
            f"{EXPECTED_COUNTS['train_slices']}. The preprocessing does not match the reference's."
        )
    print(f"Verified: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])} "
          f"train/val/test volumes, {total_slices} training slices.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=pathlib.Path, required=True,
                        help="destination; holds raw/, volumes/ and lists/")
    parser.add_argument("--skip-download", action="store_true",
                        help="the raw ACDC is already under <root>/raw/training/")
    parser.add_argument("--force", action="store_true", help="reconvert volumes that already exist")
    args = parser.parse_args()

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    training_dir = root / "raw" / "training"
    if args.skip_download:
        # An already-present tree gets the same integrity check as a fresh download: "the files are
        # on disk" and "the files are complete" are different claims.
        _verify_download(training_dir)
    else:
        training_dir = download_raw(root)

    splits = fetch_split_lists(root)
    count = convert(training_dir, root / "volumes", force=args.force)
    print(f"Converted {count} annotated volumes into {root / 'volumes'}")
    verify(root, splits)
    print("\nACDC is distributed under CC BY-NC-SA 4.0. Cite Bernard et al., IEEE TMI 2018.")


if __name__ == "__main__":
    main()

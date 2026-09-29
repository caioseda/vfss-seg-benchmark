'''
Derive a binary (fundo-vs-estrutura) mask variant from `multiclass_all`.

Why this exists
----------------
`targets/mask/binary_c2_c4` and `targets/mask/binary_c2_c3_c4` already exist on disk, precomputed
elsewhere. Both are exactly `isin(multiclass_all, <ids>)` rendered as a `{0, 255}` uint8 mask --
verified pixel-for-pixel against all 971 `multiclass_all` frames before writing this script (0
mismatches). A new fused vertebra set is therefore just a different `<ids>` list applied to the same
formula; no new annotation or external pipeline is needed.

This script produces `targets/mask/binary_c2_c3_c4_c5_c6`, fusing C2, C3, C4, C5 and C6 (multiclass
labels 1-5) into the foreground. C7/C8 (labels 6/7, rare: 199/18 of 961 annotated frames) are left as
background, matching how C3 is left out of `binary_c2_c4`.

Usage
-----
    python scripts/build_binary_mask_variant.py \\
        --dataset-path /data_ssd/caioseda/data/dataset_inca_all_frames/ \\
        --labels 1 2 3 4 5 \\
        --out-variant binary_c2_c3_c4_c5_c6
'''

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

SOURCE_VARIANT = "multiclass_all"


def build(dataset_path: Path, labels: list, out_variant: str, overwrite: bool) -> int:
    source_dir = dataset_path / "targets" / "mask" / SOURCE_VARIANT
    out_dir = dataset_path / "targets" / "mask" / out_variant
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source variant not found: {source_dir}")

    written = 0
    for labeler_dir in sorted(source_dir.iterdir()):
        if not labeler_dir.is_dir():
            continue
        dest_labeler_dir = out_dir / labeler_dir.name
        dest_labeler_dir.mkdir(parents=True, exist_ok=True)
        for src_file in sorted(labeler_dir.glob("*")):
            dest_file = dest_labeler_dir / src_file.name
            if dest_file.exists() and not overwrite:
                continue
            mask = np.array(Image.open(src_file))
            binary = (np.isin(mask, labels) * 255).astype(np.uint8)
            Image.fromarray(binary, mode="L").save(dest_file)
            written += 1
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--labels", type=int, nargs="+", required=True,
                         help="multiclass_all pixel values to fuse into the foreground")
    parser.add_argument("--out-variant", required=True,
                         help="name of the new variant directory under targets/mask/")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    written = build(args.dataset_path, args.labels, args.out_variant, args.overwrite)
    print(f"{written} mask(s) written to "
          f"{args.dataset_path / 'targets' / 'mask' / args.out_variant}")


if __name__ == "__main__":
    main()

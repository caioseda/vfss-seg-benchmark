from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class VFSSUnmBase(Dataset):
    """UNM dataset with split folders: <split>/{data,target}."""

    def __init__(
        self,
        split_dir,
        size=256,
        num_classes=2,
        image_interpolation="bilinear",
        mask_interpolation="nearest",
        **kwargs,
    ):
        super().__init__()
        self.split_dir = Path(split_dir).expanduser().resolve()
        self.size = self._normalize_size(size)
        self.num_classes = num_classes

        self.image_resize = T.Resize(
            self.size, interpolation=self._parse_interpolation(image_interpolation)
        )
        self.mask_resize = T.Resize(
            self.size, interpolation=self._parse_interpolation(mask_interpolation)
        )

        self.data_dir = self.split_dir / "data"
        self.target_dir = self.split_dir / "target"

        if not self.data_dir.exists() or not self.target_dir.exists():
            raise FileNotFoundError(
                f"Expected split directories at {self.data_dir} and {self.target_dir}"
            )

        data_map = {p.stem: p for p in sorted(self.data_dir.glob("*.png"))}
        target_map = {p.stem: p for p in sorted(self.target_dir.glob("*.tif"))}
        common_ids = sorted(set(data_map) & set(target_map))

        if not common_ids:
            raise ValueError(
                f"No paired samples found in {self.data_dir} and {self.target_dir}"
            )

        missing_data = sorted(set(target_map) - set(data_map))
        missing_target = sorted(set(data_map) - set(target_map))
        if missing_data or missing_target:
            raise ValueError(
                "Data/target mismatch for split "
                f"{self.split_dir}: missing_data={missing_data[:5]}, "
                f"missing_target={missing_target[:5]}"
            )

        self.data_paths = [str(data_map[sample_id]) for sample_id in common_ids]
        self.labels = dict(
            file_path_=[str(target_map[sample_id]) for sample_id in common_ids]
        )
        self._length = len(self.data_paths)

        print(f"[Dataset]: UNM {self.split_dir.name} with {self._length} samples initialized.")

    def __len__(self):
        return self._length

    @staticmethod
    def _normalize_size(size):
        if isinstance(size, int):
            if size <= 0:
                raise ValueError("size must be > 0")
            return (size, size)
        if isinstance(size, (tuple, list)) and len(size) == 2:
            h, w = int(size[0]), int(size[1])
            if h <= 0 or w <= 0:
                raise ValueError("size values must be > 0")
            return (h, w)
        raise ValueError("size must be int or tuple/list of 2 ints")

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

    def _validate_image_segmentation_size(self, image, segmentation):
        image_size = image.size
        segmentation_size = segmentation.size

        assert image_size == segmentation_size, "Image and segmentation size mismatch!"
        assert image_size == (
            self.size[1],
            self.size[0],
        ), "Image size does not match the specified size!"
        assert segmentation_size == (
            self.size[1],
            self.size[0],
        ), "Segmentation size does not match the specified size!"

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)

        image = Image.open(self.data_paths[i]).convert("RGB")
        segmentation = Image.open(example["file_path_"]).convert("L")

        image = self.image_resize(image)
        segmentation = self.mask_resize(segmentation)

        self._validate_image_segmentation_size(image, segmentation)

        segmentation = np.array(segmentation).astype(np.uint8)
        if self.num_classes == 2:
            segmentation = (segmentation > 0).astype(np.int64)
        else:
            raise NotImplementedError("Only support binary segmentation now.")
        example["segmentation"] = torch.from_numpy(segmentation)

        image = np.array(image).astype(np.float32) / 255.0
        image = (image * 2.0) - 1.0
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        example["image"] = image

        assert (
            torch.max(example["segmentation"]) <= 1
            and torch.min(example["segmentation"]) >= 0
        )
        assert (
            torch.max(example["image"]) <= 1.0 and torch.min(example["image"]) >= -1.0
        )
        return example


class VFSSUnmTrain(VFSSUnmBase):
    def __init__(self, data_root="../dados_unm", **kwargs):
        super().__init__(Path(data_root) / "train", **kwargs)


class VFSSUnmVal(VFSSUnmBase):
    def __init__(self, data_root="../dados_unm", **kwargs):
        super().__init__(Path(data_root) / "val", **kwargs)


class VFSSUnmTest(VFSSUnmBase):
    def __init__(self, data_root="../dados_unm", **kwargs):
        super().__init__(Path(data_root) / "test", **kwargs)

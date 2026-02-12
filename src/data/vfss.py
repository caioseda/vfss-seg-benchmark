import numpy as np
import torch
from torchvision import transforms

from vfss_data_split.datasets.vfss_dataset import VFSSImageDataset
import vfss_data_split.data_extraction.video_frame as video_frame

class VFSSIncaDatasetBase(VFSSImageDataset):
    """VFSS Dataset Base for ldm.data"""

    def __init__(self, data_root, size=256, num_classes=2, **kwargs):
        self.video_frame_df = video_frame.load_video_frame_metadata_from_csv(data_root)
        super().__init__(video_frame_df=self.video_frame_df, **kwargs)
        self.size = size
        self.num_classes = num_classes

        self.data_paths = self.video_frame_df.image_path.tolist()
        self._length = len(self.data_paths)
        self.labels = dict(file_path_=self.video_frame_df.target_path.tolist())

        print(
            f"[Dataset]: VFSS with {self.video_frame_df.shape[0]} samples initialized."
        )

    def _validate_image_segmentation_size(self, image, segmentation):
        image_size = image.size
        segmentation_size = segmentation.size

        assert image_size == segmentation_size, "Image and segmentation size mismatch!"
        if self.size is not None:
            assert self.size > 0, "Size must be positive!"
            assert image_size == (
                self.size,
                self.size,
            ), "Image size does not match the specified size!"
            assert segmentation_size == (
                self.size,
                self.size,
            ), "Segmentation size does not match the specified size!"

    def __getitem__(self, i):
        # Build example dict
        example = dict((k, self.labels[k][i]) for k in self.labels)

        # Read image and segmentation from VFSSImageDataset
        image, segmentation = super().__getitem__(i)
        image = transforms.ToPILImage()(image).convert("RGB")
        # Use single-channel mask to avoid extra channels
        segmentation = transforms.ToPILImage()(segmentation).convert("L")

        # Validate sizes
        self._validate_image_segmentation_size(image, segmentation)

        # process segmentation
        segmentation = np.array(segmentation).astype(np.uint8)
        if self.num_classes == 2:
            # Treat any non-zero pixel as foreground for binary masks

            segmentation = (segmentation > 0).astype(np.int64)
        else:
            raise NotImplementedError("Only support binary segmentation now.")

        # example["segmentation"] = (segmentation * 2) - 1
        example["segmentation"] = torch.from_numpy(segmentation)

        # process image
        image = np.array(image).astype(np.float32) / 255.0
        image = (image * 2.0) - 1.0  # range from -1 to 1

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()

        example["image"] = image
        example["class_id"] = torch.tensor([-1])  # doesn't matter for binary seg

        assert (
            torch.max(example["segmentation"]) <= 1
            and torch.min(example["segmentation"]) >= 0
        )
        assert (
            torch.max(example["image"]) <= 1.0 and torch.min(example["image"]) >= -1.0
        )
        return example

class VFSSIncaTrain(VFSSIncaDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(
            f"../dados_inca/metadados/video_frame_metadata_train.csv",
            target="mask",
            from_images=True,
            return_single_target=True,
            return_metadata=False,
            **kwargs,
        )

class VFSSIncaVal(VFSSIncaDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(
            f'../dados_inca/metadados/video_frame_metadata_val.csv',
            target='mask',
            from_images=True,
            return_single_target=True, 
            return_metadata=False,
            **kwargs
        )
    
class VFSSIncaTest(VFSSIncaDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(
            f'../dados_inca/metadados/video_frame_metadata_test.csv',
            target='mask',
            from_images=True,
            return_single_target=True,
            return_metadata=False,
            **kwargs,
        )


class VFSSTest(VFSSDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(
            f"../dados_inca/metadados/video_frame_metadata_test.csv",
            target="mask",
            from_images=True,
            return_single_target=True,
            return_metadata=False,
            **kwargs,
        )


if __name__ == "__main__":
    print("Testing VFSSIncaDatasetBase...")

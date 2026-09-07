from .vfss_frame_dataset import DEFAULT_SPLIT_RATIOS, VFSSWindowImageDataset

UNM_TARGET_VARIANTS = (
    "raw",
    "binary_c2_c4",
    "binary_c2_c3_c4",
    "multiclass_c2_c4",
    "multiclass_c2_c3_c4",
    "multiclass_all",
)


class VFSSUnmWindowImageDataset(VFSSWindowImageDataset):
    '''VFSSWindowImageDataset configured for the UNM pool (`unm-video-frame-dataset.csv`).'''

    def __init__(self,
                 dataset_path,
                 video_frame_table_filename='unm-video-frame-dataset.csv',
                 target_variants=UNM_TARGET_VARIANTS,
                 split_ratios=DEFAULT_SPLIT_RATIOS,
                 **kwargs):
        super().__init__(
            dataset_path=dataset_path,
            video_frame_table_filename=video_frame_table_filename,
            target_variants=target_variants,
            split_ratios=split_ratios,
            **kwargs,
        )


class VFSSUnmTrain(VFSSUnmWindowImageDataset):
    def __init__(self, **kwargs):
        super().__init__(split='train', **kwargs)


class VFSSUnmVal(VFSSUnmWindowImageDataset):
    def __init__(self, **kwargs):
        super().__init__(split='val', **kwargs)


class VFSSUnmTest(VFSSUnmWindowImageDataset):
    def __init__(self, **kwargs):
        super().__init__(split='test', **kwargs)

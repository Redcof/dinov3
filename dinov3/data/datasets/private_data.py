# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import os
from enum import Enum
from typing import Any, Callable, Optional, Union

from PIL import Image

from .decoders import Decoder, DenseTargetDecoder, ImageDataDecoder
from .extended import VisionDataset


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"

    @property
    def data_fname(self) -> str:
        _DATA_FNAMES = {
            _Split.TRAIN: "train.txt",
            _Split.VAL: "val.txt",
        }
        return _DATA_FNAMES[self]


class PRIVATE_DATA(VisionDataset):
    Split = Union[_Split]
    Labels = Union[Image.Image]

    def __init__(
        self,
        *,
        split: "PRIVATE_DATA.Split",
        root: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_decoder: Decoder = ImageDataDecoder,
        target_decoder: Decoder = DenseTargetDecoder,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            # image_decoder=image_decoder,
            # target_decoder=target_decoder,
        )
        self.image_paths = []
        self.labels = []
        with open(os.path.join(root, split.data_fname)) as f:
            lines = f.readlines()            
        
            for line in lines:
                image_relpath, label = line.split(',')                
                self.image_paths.append(image_relpath)
                self.labels.append(label)

    def get_image_data(self, index: int) -> bytes:
        
        image_full_path = self.image_paths[index]
        image_data = Image.open(image_full_path).convert(mode="RGB")
        return image_data

    def get_target(self, index: int) -> Any:
        label = self.labels[index]
        return 0 # always returning 0 as we have only one class

    def __getitem__(self, index):
        image = self.get_image_data(index)
        target = self.get_target(index)
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
    
    def __len__(self) -> int:
        return len(self.image_paths)

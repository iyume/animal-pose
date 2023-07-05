from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from keypoints import Annotation, annotations, images


class AnimalPose(Dataset[Tuple[Tensor, Tensor]]):
    """Returns image(3 chan) with its keypoints(20,3).

    每个点由 3 个数字组成，前两个数字是 (Col,Row)，第三个数字是 0/1 表示点是否存在
    """

    def __init__(self, *, dataset_dir: str = ".") -> None:
        self.num_test = len(images) // 5
        self.image_dir = Path(dataset_dir) / "images"

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        ann = annotations[index]
        keypoints = [i[:2] for i in ann.keypoints]
        img_path = str(self.image_dir / images[str(ann.image_id)])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (500, 500))
        assert img.ndim == 3
        return self.transform(img), torch.tensor(keypoints, dtype=torch.float)

    @staticmethod
    def transform(im: np.ndarray) -> Tensor:
        return F.to_tensor(im)

    def get_testdata(self) -> Tuple[Tensor, List[Annotation]]:
        """Get last 20% dataset for test."""
        images = []
        anns = []
        for i in range(len(images) - self.num_test, len(images)):
            image, ann = self[i]
            images.append(image)
            anns.append(ann)
        return torch.stack(images), anns

    def __len__(self) -> int:
        return len(images) - self.num_test

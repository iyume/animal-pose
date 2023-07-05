from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import json
import torchvision.transforms.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from keypoints import Annotation, annotations, images


class AnimalPose(Dataset[Tuple[Tensor, Tensor]]):
    """Returns image(3 chan) with its keypoints(20,3).

    每个点由 3 个数字组成，前两个数字是 (Col,Row)，第三个数字是 0/1 表示点是否存在

    label 范围 0-20，0为背景
        *变更: label是20个点形成的长度40的序列
    """

    def __init__(self, *, dataset_dir: str = ".") -> None:
        self.num_test = len(annotations) // 5
        self.image_dir = Path(dataset_dir) / "images"

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        ann = annotations[index]
        img_path = str(self.image_dir / images[str(ann.image_id)])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (400, 400))
        assert img.ndim == 3
        # label = np.zeros(img.shape[:2], dtype=np.int8)
        # for i, p in enumerate(ann.keypoints):
        #     if p[2] == 1:
        #         label[p[:2]] = i + 1
        # return self.transform(img), torch.from_numpy(label).to(torch.long)
        label = torch.tensor(
            [p[:2] for p in ann.keypoints], dtype=torch.float
        ).flatten()
        return self.transform(img), label

    @staticmethod
    def transform(im: np.ndarray) -> Tensor:
        return F.to_tensor(im)

    def get_testdata(self) -> Tuple[Tensor, List[Annotation]]:
        """Get last 20% dataset for test."""
        images = []
        anns = []
        for i in range(len(annotations) - self.num_test, len(annotations)):
            image, ann = self[i]
            images.append(image)
            anns.append(ann)
        return torch.stack(images), anns

    def __len__(self) -> int:
        return len(annotations) - self.num_test


class CowDataset(Dataset[Tuple[Tensor, Tensor]]):
    """Returns image and flatten 5 points."""

    def __init__(self, dataset_dir="./cow dataset") -> None:
        self.dataset_dir = Path(dataset_dir)
        assert self.dataset_dir.exists()
        self.images: list[str] = []
        self.annotations: list[str] = []
        for img_path in self.dataset_dir.iterdir():
            if img_path.suffix.lower() in (".jpg", ".png"):
                json_path = img_path.with_suffix(".json")
                self.images.append(str(img_path))
                assert json_path.exists(), f"json for {img_path} not exists"
                self.annotations.append(str(json_path))

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img = cv2.imread(self.images[index])
        img = cv2.resize(img, (400, 400))
        assert img.ndim == 3
        annotation = json.loads(open(self.annotations[index]).read())
        points = []
        for i in annotation["shapes"]:
            assert len(i["points"]) == 1
            assert len(i["points"][0]) == 2
            points.append([int(i["points"][0][0]), int(i["points"][0][1])])
        assert len(points) == 6
        return self.transform(img), torch.tensor(points).flatten().to(torch.long)

    @staticmethod
    def transform(im: np.ndarray) -> Tensor:
        return F.to_tensor(im)

    def __len__(self) -> int:
        return len(self.images)

from typing import Tuple, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from datasets import CowDataset

from model import Net


pth_file = "ckpt/model_v0.1_epoch600.pth"

net = Net(3, 12)
net.load_state_dict(torch.load(pth_file, "cpu")["model_state_dict"])
net.eval()


def draw_points(
    img: np.ndarray, points: np.ndarray, size: Optional[Tuple[int, int]] = None
) -> None:
    assert img.shape[2] == 3
    if size is None:
        size = img.shape[:2]
    # calc ratio and apply to resulting points
    ratH = size[0] / 400
    ratW = size[1] / 400
    for p in points:
        if (not 0 < p[0] < 400) or (not 0 < p[1] < 400):
            print(f"incorrect result point: {p}")
            continue
        x = int(p[0] * ratW)
        y = int(p[1] * ratH)
        cv2.circle(img, (x, y), radius=4, color=(0, 0, 255), thickness=-1)


def evaluate(img_path: str):
    raw_test_img = cv2.imread(img_path)
    assert raw_test_img.ndim == 3
    size = raw_test_img.shape[:2]

    raw_test_img_resied = cv2.resize(raw_test_img, (400, 400))
    test_img = F.to_tensor(raw_test_img_resied).unsqueeze_(0)

    out = net(test_img)
    out = out.detach().numpy().reshape(6, 2)

    out_vis = raw_test_img.copy()
    draw_points(out_vis, out, size)
    cv2.imwrite("output.png", out_vis)

    out_vis = raw_test_img_resied.copy()
    draw_points(out_vis, out, (400, 400))
    cv2.imwrite("output-shrinked.png", out_vis)


def test_train_img_vis():
    dataset = CowDataset()
    _, label, filepath = dataset[0]
    test_img = cv2.imread(filepath)
    test_img = cv2.resize(test_img, (400, 400))
    # test_img = (test_img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    # test_img = np.array(test_img)
    draw_points(test_img, label.numpy().reshape(6, 2))
    cv2.imwrite("test_train_img_vis.png", test_img)


if __name__ == "__main__":
    # evaluate("cow dataset/542696674_1a7a164508.jpg")
    test_train_img_vis()

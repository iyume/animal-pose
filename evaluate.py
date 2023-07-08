from typing import Tuple

import cv2
import numpy as np
from pathlib import Path
import torch
import json
import torchvision.transforms.functional as F
from datasets import CowDataset

from model import Net


pth_file = "ckpt/model_v0.1_epoch700.pth"

net = Net(3, 12)
net.load_state_dict(torch.load(pth_file, "cpu")["model_state_dict"])
net.eval()


def draw_points(img: np.ndarray, points: np.ndarray) -> None:
    assert img.shape[2] == 3
    # calc ratio and apply to resulting points
    for p in points:
        if (not 0 < p[0] < img.shape[1]) or (not 0 < p[1] < img.shape[0]):
            print(f"incorrect result point: {p}")
            continue
        cv2.circle(img, tuple(p), radius=4, color=(0, 0, 255), thickness=-1)


def fix_points(points: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    ratH = size[0] / 400
    ratW = size[1] / 400
    new_points = []
    for p in points:
        x = int(p[0] * ratW)
        y = int(p[1] * ratH)
        new_points.append([x, y])
    return np.array(new_points)


def evaluate(img_path: str, save_path: str):
    raw_test_img = cv2.imread(img_path)
    assert raw_test_img.ndim == 3

    raw_test_img_resized = cv2.resize(raw_test_img, (400, 400))
    test_img = F.to_tensor(raw_test_img_resized).unsqueeze_(0)

    out = net(test_img)
    points = out.detach().numpy().reshape(6, 2)

    out_vis = raw_test_img.copy()
    draw_points(out_vis, fix_points(points, raw_test_img.shape[:2]))
    cv2.imwrite(save_path, out_vis)

    # out_vis = raw_test_img_resized.copy()
    # draw_points(out_vis, points)
    # cv2.imwrite(save_path, out_vis)


def evaluate_train_images(dataset_dir: str):
    outpath = Path("train-images-output")
    outpath.mkdir(exist_ok=True)
    for imgpath in Path(dataset_dir).iterdir():
        if imgpath.suffix.lower() in (".jpg", ".png"):
            evaluate(str(imgpath), str(outpath / imgpath.name))


def visualize_dataset(dataset_dir: str):
    outpath = Path("visualize-cow-dataset")
    outpath.mkdir(exist_ok=True)
    for imgpath in Path(dataset_dir).iterdir():
        if imgpath.suffix.lower() not in (".jpg", ".png"):
            continue
        img = cv2.imread(str(imgpath))
        annotation = json.loads(open(imgpath.with_suffix(".json")).read())
        points: list[list[int]] = []
        for i in annotation["shapes"]:
            assert len(i["points"]) == 1
            assert len(i["points"][0]) == 2
            points.append([int(i["points"][0][0]), int(i["points"][0][1])])
        assert len(points) == 6
        draw_points(img, np.array(points))
        cv2.imwrite(str(outpath / imgpath.name), img)


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
    evaluate("cow dataset/542696674_1a7a164508.jpg", "output.png")
    # evaluate("example/Snipaste 20230705223659.png")
    # test_train_img_vis()

    # evaluate_train_images("cow dataset")
    # visualize_dataset("cow dataset/1-brown-cow-louise-magno.jpg")

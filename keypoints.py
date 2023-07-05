from __future__ import annotations

from pydantic import BaseModel


class KeyPoints(BaseModel):
    images: dict[str, str]
    """image_id to file path."""
    annotations: list["Annotation"]


class Annotation(BaseModel):
    image_id: int
    bbox: tuple[int, int, int, int]
    keypoints: list[list[int]]
    """Two eyes, Throat, Nose, Withers, Two Earbases, Tailbase, Four Elbows, Four Knees, Four Paws."""
    num_keypoints: int
    category_id: int


obj = KeyPoints.model_validate_json(open("keypoints.json").read())

images = obj.images
annotations = obj.annotations


def check_annotations():
    for ann in annotations:
        assert len(ann.keypoints) == 20


check_annotations()

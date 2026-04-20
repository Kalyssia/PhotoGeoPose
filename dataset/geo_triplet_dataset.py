from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from utils import angle_diff_deg, haversine_m

@dataclass
class Annotation:
    id: int
    lat: float
    lon: float
    angle: float
    sequence_id: str


class AnnotationStore:
    def __init__(self, annotations_path: str | Path, image_dir: str | Path):
        self.annotations_path = Path(annotations_path)
        self.image_dir = Path(image_dir)
        self.items = self._load_annotations()

    def _load_annotations(self) -> list[Annotation]:
        with self.annotations_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        items = []
        for x in raw:
            image_path = self.image_dir / f"{x['id']}.jpg"
            if image_path.exists():
                items.append(
                    Annotation(
                        id=int(x["id"]),
                        lat=float(x["lat"]),
                        lon=float(x["lon"]),
                        angle=float(x["angle"]),
                        sequence_id=str(x["sequence_id"]),
                    )
                )
        return items

    def __len__(self):
        return len(self.items)

    def get_image_path(self, index: int) -> Path:
        return self.image_dir / f"{self.items[index].id}.jpg"

    def load_image(self, index: int):
        return Image.open(self.get_image_path(index)).convert("RGB")



class GeoTripletDataset(Dataset):
    """Dataset for training with triplet loss (anchor, positive, negative).

    This dataset now relies on ``AnnotationStore`` to load and filter
    annotations, instead of reading the JSON file directly.
    """

    def __init__(
        self,
        annotations_path: str | Path,
        image_dir: str | Path,
        transform=None,
        pos_dist: float = 20.0,
        neg_dist: float = 80.0,
        max_angle: float = 360.0,
        store: AnnotationStore | None = None,
    ) -> None:
        # Use the provided AnnotationStore or build one from paths
        self.store = store or AnnotationStore(annotations_path, image_dir)

        self.items: list[Annotation] = self.store.items
        self.transform = transform
        self.pos_dist = pos_dist
        self.neg_dist = neg_dist
        self.max_angle = max_angle

    def __len__(self):
        return len(self.items)

    def load_img(self, item: Annotation):
        path = self.store.image_dir / f"{item.id}.jpg"
        img = Image.open(path).convert("RGB")
        # applies transforms if provided
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        anchor = self.items[idx]

        positives: list[Annotation] = []
        negatives: list[Annotation] = []

        # Find positives and negatives based on distance and angle criteria
        for cand in self.items:
            if cand.id == anchor.id:
                continue

            dist = haversine_m(anchor.lat, anchor.lon, cand.lat, cand.lon)
            angle_delta = angle_diff_deg(anchor.angle, cand.angle)

            if dist <= self.pos_dist and angle_delta <= self.max_angle:
                positives.append(cand)
            elif dist >= self.neg_dist:
                negatives.append(cand)

        # If no valid positives or negatives, recursively try another anchor
        if len(positives) == 0 or len(negatives) == 0:
            return self.__getitem__(random.randint(0, len(self.items)-1))

        pos = random.choice(positives)
        neg = random.choice(negatives)

        return self.load_img(anchor), self.load_img(pos), self.load_img(neg)
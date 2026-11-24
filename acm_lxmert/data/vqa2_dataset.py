"""VQA v2 datasets for Faster R-CNN obj36 LMDB features and offline CLIP RN50x4 grids."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from acm_lxmert.models.lxmert_backbone import official_modules, official_tokenizer

_ENVIRONMENTS: dict[str, lmdb.Environment] = {}
_ENV_PID: int | None = None


def validate_region_arrays(features: np.ndarray, boxes: np.ndarray) -> None:
    if features.shape != (36, 2048) or boxes.shape != (36, 4):
        raise ValueError("Official obj36 feature shape mismatch")
    if not np.isfinite(features).all() or not np.isfinite(boxes).all():
        raise ValueError("Non-finite official features")
    if not (boxes < 1 + 1e-5).all() or not (-boxes < 0 + 1e-5).all():
        raise ValueError("Official obj36 boxes outside normalized coordinate bounds")


def open_lmdb_readonly(path: str | Path) -> lmdb.Environment:
    global _ENV_PID
    if _ENV_PID != os.getpid():
        for env in _ENVIRONMENTS.values():
            env.close()
        _ENVIRONMENTS.clear()
        _ENV_PID = os.getpid()
    resolved = str(Path(path).resolve())
    if resolved not in _ENVIRONMENTS:
        _ENVIRONMENTS[resolved] = lmdb.open(
            resolved,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=2048,
        )
    return _ENVIRONMENTS[resolved]


class VQA2RegionDataset(Dataset):
    """VQA v2 dataset loading official LXMERT JSON splits and cached obj36 LMDB features."""

    def __init__(
        self,
        vqa_data_root: str | Path,
        split_names: list[str],
        train_lmdb: str | Path,
        val_lmdb: str | Path,
        test_lmdb: str | Path | None = None,
        max_tokens: int = 20,
        bert_model: str = "bert-base-uncased",
    ):
        root = Path(vqa_data_root)
        self.answers = json.loads((root / "trainval_label2ans.json").read_text(encoding="utf-8"))
        self.answer2id = json.loads((root / "trainval_ans2label.json").read_text(encoding="utf-8"))
        if len(self.answers) != 3129:
            raise ValueError(f"Expected 3129 VQA v2 answers, got {len(self.answers)}")
        self.items = []
        for name in split_names:
            self.items.extend(json.loads((root / f"{name}.json").read_text(encoding="utf-8")))
        if len({int(row["question_id"]) for row in self.items}) != len(self.items):
            raise ValueError("Duplicate question IDs across splits")
        self.features = {"train": str(train_lmdb), "val": str(val_lmdb)}
        if test_lmdb is not None:
            self.features["test"] = str(test_lmdb)
        self.max_tokens = int(max_tokens)
        self.tokenizer = official_tokenizer(bert_model)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        row = self.items[index]
        image_id = row["img_id"]
        if "_train2014_" in image_id:
            split = "train"
        elif "_val2014_" in image_id:
            split = "val"
        elif "_test2015_" in image_id:
            split = "test"
        else:
            raise ValueError(f"Unrecognized COCO image id: {image_id}")

        key = str(int(image_id.rsplit("_", 1)[-1]))
        with open_lmdb_readonly(self.features[split]).begin(write=False) as txn:
            features = np.load(
                io.BytesIO(txn.get(f"visual::visual/{key}.npy".encode())),
                allow_pickle=False,
            )
            boxes = np.load(
                io.BytesIO(txn.get(f"spatial::spatial/{key}.npy".encode())),
                allow_pickle=False,
            )[:, :4]
        validate_region_arrays(features, boxes)

        official_modules()
        from lxrt.entry import convert_sents_to_features

        tokens = convert_sents_to_features([row["sent"]], self.max_tokens, self.tokenizer)[0]
        target = torch.zeros(3129)
        for answer, score in row.get("label", {}).items():
            if answer in self.answer2id:
                target[self.answer2id[answer]] = float(score)
        return {
            "question_id": int(row["question_id"]),
            "visual_features": torch.from_numpy(features.astype(np.float32, copy=True)),
            "spatials": torch.from_numpy(boxes.astype(np.float32, copy=True)),
            "question_token_ids": torch.tensor(tokens.input_ids),
            "question_token_mask": torch.tensor(tokens.input_mask),
            "target": target,
        }


class WithVQA2ClipGrid(Dataset):
    """Wrap VQA2RegionDataset with memory-mapped float16 [49, 2560] CLIP RN50x4 grids."""

    def __init__(
        self,
        base: VQA2RegionDataset,
        clip_root: str | Path,
        splits: tuple[str, ...] = ("train2014", "val2014"),
    ):
        self.base = base
        self.items = base.items
        self.answers = base.answers
        self.rows: dict[str, dict[int, int]] = {}
        self.files: dict[str, str] = {}
        self.maps: dict[str, np.ndarray] = {}
        root = Path(clip_root)
        for split in splits:
            index_path = root / f"{split}_index.json"
            if index_path.is_file():
                index = json.loads(index_path.read_text(encoding="utf-8"))
                self.rows[split] = {int(image): row for row, image in enumerate(index["image_ids"])}
                self.files[split] = str(root / f"{split}_grid.f16.npy")
        self.where = [self.locate(item["img_id"]) for item in self.items]

    def locate(self, image_id: str) -> tuple[str, int]:
        if "_train2014_" in image_id:
            split = "train2014"
        elif "_val2014_" in image_id:
            split = "val2014"
        elif "_test2015_" in image_id:
            split = "test2015"
        else:
            raise ValueError(f"Unrecognized image id: {image_id}")
        return split, self.rows[split][int(image_id.rsplit("_", 1)[-1])]

    def load_grid(self, split: str, position: int) -> torch.Tensor:
        if split not in self.maps:
            self.maps[split] = np.load(self.files[split], mmap_mode="r")
        value = np.array(self.maps[split][position])
        if value.shape != (49, 2560) or not np.isfinite(value).all():
            raise ValueError(f"Invalid CLIP grid feature at {split}[{position}]")
        return torch.from_numpy(value)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        row = self.base[index]
        split, position = self.where[index]
        row["clip_grid"] = self.load_grid(split, position)
        return row


class VQA2ClipSwapDataset(Dataset):
    """VQA v2 dataset pairing each sample with a donor-image CLIP grid from a fixed derangement."""

    def __init__(self, wrapped: WithVQA2ClipGrid, donor_map: dict[str, str]):
        self.wrapped = wrapped
        self.items = wrapped.items
        self.answers = wrapped.answers
        self.donor_map = donor_map
        images = {str(item["img_id"]) for item in self.items}
        if set(donor_map) != images or set(donor_map.values()) != images:
            raise ValueError("Donor mapping does not cover evaluation images bijectively")
        self.donor_where = {image: wrapped.locate(donor) for image, donor in donor_map.items()}

    def __len__(self) -> int:
        return len(self.wrapped)

    def __getitem__(self, index: int) -> dict:
        row = self.wrapped[index]
        image = str(self.items[index]["img_id"])
        split, position = self.donor_where[image]
        row["swap_clip_grid"] = self.wrapped.load_grid(split, position)
        return row

"""GQA datasets for Faster R-CNN obj36 LMDB features and offline CLIP RN50x4 grids."""

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

_ENVS: dict[str, lmdb.Environment] = {}
_ENV_PID: int | None = None


def open_gqa_lmdb(path: str | Path) -> lmdb.Environment:
    global _ENV_PID
    if _ENV_PID != os.getpid():
        for env in _ENVS.values():
            env.close()
        _ENVS.clear()
        _ENV_PID = os.getpid()
    resolved = str(Path(path).resolve())
    if resolved not in _ENVS:
        _ENVS[resolved] = lmdb.open(
            resolved,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=2048,
        )
    return _ENVS[resolved]


def lmdb_keys(image_id: str) -> tuple[bytes, bytes]:
    image_id = str(image_id)
    return (
        f"visual::visual/{image_id}.npy".encode(),
        f"spatial::spatial/{image_id}.npy".encode(),
    )


class GQADataset(Dataset):
    """GQA dataset using official LXMERT GQA splits (1,842 answers) and obj36 LMDB caches."""

    def __init__(
        self,
        gqa_root: str | Path,
        split_names: list[str],
        vg_lmdb: str | Path,
        testdev_lmdb: str | Path,
        max_tokens: int = 20,
        bert_model: str = "bert-base-uncased",
    ):
        self.root = Path(gqa_root)
        self.answers = json.loads((self.root / "trainval_label2ans.json").read_text(encoding="utf-8"))
        self.answer2id = json.loads((self.root / "trainval_ans2label.json").read_text(encoding="utf-8"))
        if len(self.answers) != 1842:
            raise ValueError(f"GQA answer vocabulary must contain 1842 labels, got {len(self.answers)}")
        if any(self.answer2id.get(answer) != index for index, answer in enumerate(self.answers)):
            raise ValueError("GQA answer mappings are not exact inverses")

        self.items = []
        for split in split_names:
            rows = json.loads((self.root / f"{split}.json").read_text(encoding="utf-8"))
            for row in rows:
                entry = dict(row)
                entry["_split"] = split
                self.items.append(entry)
        qids = [str(row["question_id"]) for row in self.items]
        if len(qids) != len(set(qids)):
            raise ValueError(f"Duplicate question IDs across {split_names!r}")

        self.vg_lmdb = str(vg_lmdb)
        self.testdev_lmdb = str(testdev_lmdb)
        self.max_tokens = int(max_tokens)
        self.tokenizer = official_tokenizer(bert_model)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        row = self.items[index]
        image_id = str(row["img_id"])
        database = self.testdev_lmdb if row["_split"].startswith("testdev") else self.vg_lmdb
        visual_key, spatial_key = lmdb_keys(image_id)
        with open_gqa_lmdb(database).begin(write=False) as txn:
            visual_blob, spatial_blob = txn.get(visual_key), txn.get(spatial_key)
        if visual_blob is None or spatial_blob is None:
            raise KeyError(f"Missing obj36 features for GQA image {image_id}")
        features = np.load(io.BytesIO(visual_blob), allow_pickle=False)
        boxes = np.load(io.BytesIO(spatial_blob), allow_pickle=False)
        if features.shape != (36, 2048) or boxes.shape != (36, 4):
            raise ValueError(f"Bad obj36 tensors for {image_id}: {features.shape}, {boxes.shape}")

        official_modules()
        from lxrt.entry import convert_sents_to_features

        tokens = convert_sents_to_features([row["sent"]], self.max_tokens, self.tokenizer)[0]
        target = torch.zeros(1842)
        for answer, score in row.get("label", {}).items():
            if answer in self.answer2id:
                target[self.answer2id[answer]] = float(score)
        return {
            "question_id": str(row["question_id"]),
            "image_id": image_id,
            "visual_features": torch.from_numpy(features.astype(np.float32, copy=True)),
            "spatials": torch.from_numpy(boxes.astype(np.float32, copy=True)),
            "question_token_ids": torch.tensor(tokens.input_ids),
            "question_token_mask": torch.tensor(tokens.input_mask),
            "target": target,
        }


class WithGQAClipGrid(Dataset):
    """Wrap GQADataset with memory-mapped float16 [49, 2560] CLIP RN50x4 grids."""

    def __init__(self, base: GQADataset, clip_root: str | Path):
        self.base = base
        self.items = base.items
        self.answers = base.answers
        root = Path(clip_root)
        index = json.loads((root / "gqa_index.json").read_text(encoding="utf-8"))
        if (index.get("grid"), index.get("dim"), index.get("dtype")) != (7, 2560, "float16"):
            raise ValueError("CLIP index is not RN50x4 grid7 float16")
        self.rows = {str(image_id): row for row, image_id in enumerate(index["image_ids"])}
        if len(self.rows) != len(index["image_ids"]):
            raise ValueError("Duplicate image IDs in CLIP index")
        missing = sorted({str(item["img_id"]) for item in self.items} - set(self.rows))
        if missing:
            raise KeyError(f"CLIP index misses {len(missing)} images; first={missing[:10]!r}")
        self.locations = [self.rows[str(item["img_id"])] for item in self.items]
        self.grid_path = root / "gqa_grid.f16.npy"
        self.grid: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        row = self.base[index]
        if self.grid is None:
            self.grid = np.load(self.grid_path, mmap_mode="r")
            if self.grid.shape != (len(self.rows), 49, 2560):
                raise ValueError(f"CLIP array/index row mismatch: {self.grid.shape}")
        row["clip_grid"] = torch.from_numpy(
            np.array(self.grid[self.locations[index]], dtype=np.float16)
        )
        return row

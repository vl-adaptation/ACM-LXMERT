"""Shared training, optimization, loss, and deterministic derangement utilities."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from acm_lxmert.models.lxmert_backbone import official_modules

INPUT_KEYS = ("visual_features", "spatials", "question_token_ids", "question_token_mask")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def full_vocab_soft_target_bce(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Sum-over-answers BCE loss matching official LXMERT scaling (mean * num_answers)."""
    return F.binary_cross_entropy_with_logits(scores, targets.float(), reduction="mean") * scores.size(1)


def make_bert_adam(
    model: torch.nn.Module,
    lr: float,
    total_steps: int,
    warmup: float = 0.1,
    weight_decay: float = 0.01,
    epsilon: float = 1e-6,
    betas: tuple[float, float] = (0.9, 0.999),
):
    """Instantiate official LXMERT BertAdam optimizer."""
    official_modules()
    from lxrt.optimization import BertAdam

    return BertAdam(
        list(model.parameters()),
        lr=lr,
        warmup=warmup,
        t_total=total_steps,
        e=epsilon,
        weight_decay=weight_decay,
        b1=betas[0],
        b2=betas[1],
    )


def make_epoch_loader(
    dataset,
    batch_size: int,
    seed: int,
    epoch: int,
    num_workers: int = 4,
) -> DataLoader:
    """Deterministic per-epoch shuffling identical across Plain, RG-LXMERT, and ACM-LXMERT."""
    generator = torch.Generator().manual_seed(seed + epoch - 1)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    usable = (len(dataset) // batch_size) * batch_size
    indices = indices[:usable]
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=indices,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed + epoch + 10000),
    )


def batch_to_device(raw: dict, device: torch.device) -> dict:
    batch = {key: raw[key].to(device, non_blocking=True) for key in INPUT_KEYS}
    batch["targets"] = raw["target"].to(device, non_blocking=True)
    if "clip_grid" in raw:
        batch["clip_grid"] = raw["clip_grid"].to(device, non_blocking=True)
    return batch


def model_inputs(batch: dict) -> dict:
    return {key: value for key, value in batch.items() if key != "targets"}


def build_derangement(items: list[dict], seed: int = 9595) -> tuple[dict[str, str], dict]:
    """Construct a fixed-point-free one-to-one image permutation for feature swapping."""
    images = sorted({str(item["img_id"]) for item in items})
    source = np.asarray(images, dtype=object)
    rng = np.random.default_rng(seed)
    attempts = 0
    while True:
        attempts += 1
        donor = source[rng.permutation(len(source))]
        if bool(np.all(source != donor)):
            break
    mapping = dict(zip(images, donor.tolist()))
    if set(mapping) != set(mapping.values()) or any(a == b for a, b in mapping.items()):
        raise RuntimeError("Generated mapping is not a valid derangement")
    canonical = json.dumps(mapping, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    metadata = {
        "seed": seed,
        "unique_images": len(images),
        "attempts": attempts,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
    return mapping, metadata


def atomic_json(path: Path, payload) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_save(path: Path, payload) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

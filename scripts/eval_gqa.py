#!/usr/bin/env python3
"""Evaluate a GQA checkpoint under Joint, CLIP-only, and Feature-Swap interventions.

Runs all five GQA evaluation modes reported in the paper:
``normal_joint``, ``clip_swap``, ``region_swap``, ``clip_only_normal``,
and ``clip_only_swap``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from acm_lxmert.data.gqa_dataset import GQADataset, WithGQAClipGrid
from acm_lxmert.models.acm_model import ClipGridVQA
from acm_lxmert.utils import atomic_json, build_derangement, seed_everything

MODES = ("normal_joint", "clip_swap", "region_swap", "clip_only_normal", "clip_only_swap")


class CachedGQAInputs:
    """Cache the 398 GQA testdev images and question tokens on GPU for fast multi-mode diagnostics."""

    def __init__(self, base: GQADataset, clip_root: Path, mapping: dict[str, str], device: torch.device):
        joint = WithGQAClipGrid(base, clip_root)
        self.items = base.items
        self.answers = base.answers
        self.qids = [str(row["question_id"]) for row in base.items]
        self.device = device

        images = sorted(mapping)
        lookup = {image: idx for idx, image in enumerate(images)}
        representative: dict[str, int] = {}
        for idx, row in enumerate(base.items):
            representative.setdefault(str(row["img_id"]), idx)

        bank = [joint[representative[image]] for image in images]
        self.visual = torch.stack([row["visual_features"] for row in bank]).to(device)
        self.boxes = torch.stack([row["spatials"] for row in bank]).to(device)
        self.clip = torch.stack([row["clip_grid"] for row in bank]).to(device)
        self.source = torch.tensor([lookup[str(row["img_id"])] for row in base.items], device=device)
        self.donor = torch.tensor([lookup[mapping[str(row["img_id"])]] for row in base.items], device=device)

        from lxrt.entry import convert_sents_to_features

        tokens = convert_sents_to_features([row["sent"] for row in base.items], 20, base.tokenizer)
        self.tokens = torch.tensor([t.input_ids for t in tokens], device=device)
        self.token_mask = torch.tensor([t.input_mask for t in tokens], device=device)

    def batch(self, positions: slice, mode: str) -> dict:
        src, donor = self.source[positions], self.donor[positions]
        region_idx = donor if mode == "region_swap" else src
        clip_idx = donor if mode in ("clip_swap", "clip_only_swap") else src
        inputs = {
            "visual_features": self.visual[region_idx],
            "spatials": self.boxes[region_idx],
            "clip_grid": self.clip[clip_idx],
            "question_token_ids": self.tokens[positions],
            "question_token_mask": self.token_mask[positions],
        }
        if mode.startswith("clip_only"):
            inputs["use_regions"] = torch.zeros(len(src), device=self.device)
        return inputs


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Trained GQA epoch4.pth checkpoint")
    parser.add_argument("--gqa-root", required=True, type=Path)
    parser.add_argument("--vg-lmdb", required=True, type=Path)
    parser.add_argument("--testdev-lmdb", required=True, type=Path)
    parser.add_argument("--clip-root", required=True, type=Path)
    parser.add_argument("--raw-questions", required=True, type=Path, help="testdev_balanced_questions.json")
    parser.add_argument("--official-checkpoint", required=True, type=Path)
    parser.add_argument("--bert-model", default="bert-base-uncased")
    parser.add_argument("--grid", type=int, default=7)
    parser.add_argument("--swap-seed", type=int, default=9595)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.swap_seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    base = GQADataset(
        args.gqa_root, ["testdev"], args.vg_lmdb, args.testdev_lmdb, max_tokens=20, bert_model=args.bert_model
    )
    mapping, map_meta = build_derangement(base.items, seed=args.swap_seed)
    atomic_json(args.output / "image_derangement.json", {"metadata": map_meta, "mapping": mapping})
    cached = CachedGQAInputs(base, args.clip_root, mapping, device)

    model_config = {
        "num_answers": 1842,
        "official_checkpoint": str(args.official_checkpoint.resolve()),
        "answer_vocab_path": str((args.gqa_root / "trainval_label2ans.json").resolve()),
    }
    model = ClipGridVQA(model_config, grid=args.grid)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt["model"]
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    evaluator = REPO_ROOT / "scripts" / "evaluate_gqa_predictions.py"
    metrics: dict[str, dict] = {}
    for mode in MODES:
        rows = []
        for start in range(0, len(cached.qids), args.batch_size):
            stop = min(start + args.batch_size, len(cached.qids))
            scores = model(**cached.batch(slice(start, stop), mode))["ranking_scores"]
            rows.extend(
                {"questionId": qid, "prediction": cached.answers[idx]}
                for qid, idx in zip(cached.qids[start:stop], scores.argmax(1).cpu().tolist())
            )
        pred_path = args.output / f"{mode}_predictions.json"
        metric_path = args.output / f"{mode}_metrics.json"
        atomic_json(pred_path, rows)
        subprocess.run(
            [
                sys.executable,
                str(evaluator),
                "--questions",
                str(args.raw_questions),
                "--predictions",
                str(pred_path),
                "--output",
                str(metric_path),
            ],
            check=True,
        )
        metrics[mode] = json.loads(metric_path.read_text(encoding="utf-8"))

    summary = {
        "metrics": metrics,
        "drops_pp": {
            "joint_clip_swap": metrics["normal_joint"]["overall"] - metrics["clip_swap"]["overall"],
            "joint_region_swap": metrics["normal_joint"]["overall"] - metrics["region_swap"]["overall"],
            "clip_only_swap": metrics["clip_only_normal"]["overall"] - metrics["clip_only_swap"]["overall"],
        },
    }
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary["drops_pp"], indent=2), flush=True)


if __name__ == "__main__":
    main()

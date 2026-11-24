#!/usr/bin/env python3
"""Evaluate a VQA v2 checkpoint under Joint, CLIP-only, and Joint CLIP Swap protocols.

Also supports exporting predictions on ``test2015`` (test-dev / test-std) in the
official VQA evaluation server JSON format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from acm_lxmert.data.vqa2_dataset import (
    VQA2ClipSwapDataset,
    VQA2RegionDataset,
    WithVQA2ClipGrid,
)
from acm_lxmert.models.acm_model import ClipGridVQA
from acm_lxmert.models.lxmert_backbone import OfficialBaselineVQA
from acm_lxmert.utils import (
    atomic_json,
    batch_to_device,
    build_derangement,
    model_inputs,
    seed_everything,
)


@torch.no_grad()
def run_inference(model, dataset, device, batch_size=256, workers=4, mode="joint"):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    predictions = []
    soft_scores = []
    answer_types = []
    offset = 0

    for raw in loader:
        batch = batch_to_device(raw, device)
        inputs = model_inputs(batch)
        if mode == "plain":
            inputs.pop("clip_grid", None)
        elif mode == "clip_only":
            inputs["use_regions"] = torch.zeros(inputs["clip_grid"].size(0), device=device)
        elif mode == "clip_swap":
            inputs["clip_grid"] = raw["swap_clip_grid"].to(device, non_blocking=True)
        elif mode != "joint":
            raise ValueError(f"Unsupported mode: {mode}")

        logits = model(**inputs)["ranking_scores"]
        chosen = logits.argmax(1).cpu().tolist()
        for local, (qid, ans_id) in enumerate(zip(raw["question_id"].tolist(), chosen)):
            datum = dataset.items[offset + local]
            answer = dataset.answers[ans_id]
            predictions.append({"question_id": int(qid), "answer": answer})
            if "label" in datum:
                soft_scores.append(float(datum["label"].get(answer, 0.0)))
                answer_types.append(datum.get("answer_type", "unknown"))
        offset += len(chosen)

    metrics = {"questions": offset}
    if soft_scores:
        scores_arr = np.asarray(soft_scores, dtype=np.float64)
        types_arr = np.asarray(answer_types)
        metrics["overall"] = 100.0 * float(scores_arr.mean())
        for key in ("yes/no", "number", "other"):
            mask = types_arr == key
            if mask.any():
                metrics[key] = 100.0 * float(scores_arr[mask].mean())
    return metrics, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Trained epoch4.pt checkpoint")
    parser.add_argument("--vqa-data-root", required=True, type=Path)
    parser.add_argument("--train-lmdb", required=True, type=Path)
    parser.add_argument("--val-lmdb", required=True, type=Path)
    parser.add_argument("--test-lmdb", type=Path, default=None, help="Required when --split test")
    parser.add_argument("--clip-root", type=Path, default=None)
    parser.add_argument("--official-checkpoint", required=True, type=Path)
    parser.add_argument("--bert-model", default="bert-base-uncased")
    parser.add_argument("--split", default="minival", help="Evaluation split name (e.g. minival or test)")
    parser.add_argument("--plain", action="store_true", help="Evaluate a Plain LXMERT checkpoint")
    parser.add_argument("--grid", type=int, default=7)
    parser.add_argument("--run-clip-only", action="store_true", help="Also evaluate under CLIP-only input")
    parser.add_argument("--run-clip-swap", action="store_true", help="Also run fixed-derangement Joint CLIP Swap")
    parser.add_argument("--swap-seed", type=int, default=9595)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.swap_seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    base = VQA2RegionDataset(
        args.vqa_data_root,
        [args.split],
        args.train_lmdb,
        args.val_lmdb,
        test_lmdb=args.test_lmdb,
        max_tokens=20,
        bert_model=args.bert_model,
    )
    clip_splits = ("train2014", "val2014", "test2015")
    dataset = base if args.plain else WithVQA2ClipGrid(base, args.clip_root, splits=clip_splits)

    model_config = {
        "num_answers": 3129,
        "official_checkpoint": str(args.official_checkpoint.resolve()),
        "answer_vocab_path": str((args.vqa_data_root / "trainval_label2ans.json").resolve()),
    }
    model = OfficialBaselineVQA(model_config) if args.plain else ClipGridVQA(model_config, grid=args.grid)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt["model_state"]
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    summary: dict = {"split": args.split, "checkpoint": str(args.checkpoint)}
    primary_mode = "plain" if args.plain else "joint"
    primary_metrics, primary_preds = run_inference(
        model, dataset, device, batch_size=args.batch_size, workers=args.workers, mode=primary_mode
    )
    summary[primary_mode] = primary_metrics
    atomic_json(args.output / f"{args.split}_{primary_mode}_predictions.json", primary_preds)

    if not args.plain and args.run_clip_only:
        clip_only_metrics, clip_only_preds = run_inference(
            model, dataset, device, batch_size=args.batch_size, workers=args.workers, mode="clip_only"
        )
        summary["clip_only"] = clip_only_metrics
        atomic_json(args.output / f"{args.split}_clip_only_predictions.json", clip_only_preds)

    if not args.plain and args.run_clip_swap:
        donor_map, map_meta = build_derangement(base.items, seed=args.swap_seed)
        swap_dataset = VQA2ClipSwapDataset(dataset, donor_map)
        swap_metrics, swap_preds = run_inference(
            model, swap_dataset, device, batch_size=args.batch_size, workers=args.workers, mode="clip_swap"
        )
        if "overall" in primary_metrics and "overall" in swap_metrics:
            swap_metrics["drop_pp"] = primary_metrics["overall"] - swap_metrics["overall"]
        summary["clip_swap"] = swap_metrics
        summary["swap_mapping_meta"] = map_meta
        atomic_json(args.output / f"{args.split}_clip_swap_predictions.json", swap_preds)

    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

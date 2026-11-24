#!/usr/bin/env python3
"""Train Plain LXMERT, RG-LXMERT (p=0), or ACM-LXMERT (0 < p < 1) on VQA v2.

Also supports the fine-tuning strategy ablations reported in the paper
(epoch-wise alternation and two-stage projection alignment).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from acm_lxmert.data.vqa2_dataset import VQA2RegionDataset, WithVQA2ClipGrid
from acm_lxmert.models.acm_model import ClipGridVQA
from acm_lxmert.models.lxmert_backbone import OfficialBaselineVQA
from acm_lxmert.utils import (
    atomic_json,
    atomic_save,
    batch_to_device,
    full_vocab_soft_target_bce,
    make_bert_adam,
    make_epoch_loader,
    model_inputs,
    seed_everything,
)


@torch.no_grad()
def evaluate_vqa2(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 256,
    workers: int = 4,
    use_clip: bool = True,
    view: str = "joint",
) -> tuple[dict, list[dict]]:
    """Evaluate on VQA v2 minival under ``joint``, ``clip_only``, or ``region_only`` input."""
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    total = 0.0
    offset = 0
    predictions = []
    by_type: dict[str, list[float]] = {"yes/no": [], "number": [], "other": []}

    for raw in loader:
        inputs = model_inputs(batch_to_device(raw, device))
        if not use_clip:
            inputs.pop("clip_grid", None)
        elif view == "clip_only":
            inputs["use_regions"] = torch.zeros(inputs["clip_grid"].size(0), device=device)
        elif view == "region_only":
            inputs["use_clip"] = torch.zeros(inputs["clip_grid"].size(0), device=device)
        elif view != "joint":
            raise ValueError(f"Unsupported evaluation view: {view}")

        scores = model(**inputs)["ranking_scores"]
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Non-finite evaluation logits")
        chosen = scores.argmax(1).cpu().tolist()
        for local, (qid, answer_id) in enumerate(zip(raw["question_id"].tolist(), chosen)):
            datum = dataset.items[offset + local]
            answer = dataset.answers[answer_id]
            score = float(datum.get("label", {}).get(answer, 0.0))
            total += score
            ans_type = datum.get("answer_type")
            if ans_type in by_type:
                by_type[ans_type].append(score)
            predictions.append({"question_id": int(qid), "answer": answer})
        offset += len(chosen)

    metrics = {"overall": 100.0 * total / max(offset, 1), "questions": offset}
    for key, values in by_type.items():
        if values:
            metrics[key] = 100.0 * float(np.mean(values))
    return metrics, predictions


def align_clip_projector(
    model: ClipGridVQA,
    train_data,
    batch_size: int,
    seed: int,
    workers: int,
    device: torch.device,
    steps: int = 6000,
    lr: float = 1e-3,
    region_mask_prob: float = 0.5,
) -> None:
    """Two-stage ablation helper: freeze LXMERT and train only the CLIP grid interface."""
    new_params = [p for n, p in model.named_parameters() if n.startswith("clip_")]
    frozen_params = [p for n, p in model.named_parameters() if not n.startswith("clip_")]
    for p in frozen_params:
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(new_params, lr=lr)
    model.train()
    loader = make_epoch_loader(train_data, batch_size, seed, epoch=0, num_workers=workers)
    for step, raw in enumerate(loader, 1):
        if step > steps:
            break
        factor = min(step / 100.0, 1.0) * (1.0 - (step - 1) / steps)
        for group in optimizer.param_groups:
            group["lr"] = lr * factor
        batch = batch_to_device(raw, device)
        optimizer.zero_grad()
        keep_regions = (
            (torch.rand(batch["targets"].size(0), device=device) >= region_mask_prob).float()
            if region_mask_prob > 0
            else None
        )
        scores = model(**model_inputs(batch), use_regions=keep_regions)["ranking_scores"]
        loss = full_vocab_soft_target_bce(scores, batch["targets"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(new_params, 5.0)
        optimizer.step()
    for p in frozen_params:
        p.requires_grad_(True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional JSON config file")
    parser.add_argument(
        "--strategy",
        choices=("plain", "rg", "acm", "alternate", "two_stage"),
        default=None,
        help="Training strategy: plain (LXMERT), rg (p=0), acm (sample-wise p), alternate, or two_stage",
    )
    parser.add_argument("--vqa-data-root", required=True, type=Path, help="Official LXMERT VQA JSON directory")
    parser.add_argument("--train-lmdb", required=True, type=Path, help="train2014 obj36 LMDB directory")
    parser.add_argument("--val-lmdb", required=True, type=Path, help="val2014 obj36 LMDB directory")
    parser.add_argument("--clip-root", type=Path, help="Directory with train2014/val2014 CLIP grid memmaps")
    parser.add_argument("--official-checkpoint", required=True, type=Path, help="Pretrained model_LXRT.pth")
    parser.add_argument("--bert-model", default="bert-base-uncased", help="BERT tokenizer name or local directory")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for logs and checkpoints")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--grid", type=int, default=None, help="CLIP grid resolution in {1, 4, 7}")
    parser.add_argument("--region-mask-prob", type=float, default=None, help="ACM masking probability p")
    parser.add_argument(
        "--train-splits",
        nargs="+",
        default=None,
        help="Training splits (default: train nominival; use train nominival minival for official test-dev/std)",
    )
    parser.add_argument("--no-val", action="store_true", help="Disable minival evaluation (for train+val training)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}

    strategy = args.strategy or cfg.get("strategy", "acm")
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 9595))
    epochs = int(args.epochs if args.epochs is not None else cfg.get("epochs", 4))
    batch_size = int(args.batch_size if args.batch_size is not None else cfg.get("batch_size", 32))
    lr = float(args.lr if args.lr is not None else cfg.get("lr", 5e-5))
    grid = int(args.grid if args.grid is not None else cfg.get("grid", 7))
    p_mask = float(
        args.region_mask_prob
        if args.region_mask_prob is not None
        else cfg.get("region_mask_prob", 0.6 if strategy == "acm" else 0.0)
    )
    train_splits = args.train_splits or cfg.get("train_splits", ["train", "nominival"])
    use_clip = strategy != "plain"
    if use_clip and args.clip_root is None:
        raise ValueError("--clip-root is required unless --strategy plain")
    if strategy == "acm" and not (0.0 < p_mask < 1.0):
        raise ValueError("--region-mask-prob p must lie in (0, 1) for --strategy acm")
    if strategy == "rg":
        p_mask = 0.0

    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_base = VQA2RegionDataset(
        args.vqa_data_root,
        train_splits,
        args.train_lmdb,
        args.val_lmdb,
        max_tokens=20,
        bert_model=args.bert_model,
    )
    train_data = WithVQA2ClipGrid(train_base, args.clip_root) if use_clip else train_base

    val_data = None
    if not args.no_val:
        val_base = VQA2RegionDataset(
            args.vqa_data_root,
            ["minival"],
            args.train_lmdb,
            args.val_lmdb,
            max_tokens=20,
            bert_model=args.bert_model,
        )
        val_data = WithVQA2ClipGrid(val_base, args.clip_root) if use_clip else val_base

    model_config = {
        "num_answers": 3129,
        "official_checkpoint": str(args.official_checkpoint.resolve()),
        "answer_vocab_path": str((args.vqa_data_root / "trainval_label2ans.json").resolve()),
    }
    model: torch.nn.Module = (
        OfficialBaselineVQA(model_config)
        if not use_clip
        else ClipGridVQA(model_config, grid=grid)
    )
    model.to(device)

    steps_per_epoch = len(train_data) // batch_size
    optimizer = make_bert_adam(
        model,
        lr=lr,
        total_steps=steps_per_epoch * epochs,
        warmup=0.1,
        weight_decay=0.01,
        epsilon=1e-6,
    )

    run_config = {
        "dataset": "VQA v2",
        "strategy": strategy,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "grid": grid if use_clip else None,
        "region_mask_prob": p_mask,
        "train_splits": train_splits,
        "parameters": model.parameter_counts(),
    }
    atomic_json(args.output / "config.json", run_config)
    print(json.dumps(run_config, indent=2), flush=True)

    history = []
    aligned = False
    for epoch in range(1, epochs + 1):
        if strategy == "two_stage" and epoch == 2 and not aligned:
            align_clip_projector(
                model,
                train_data,
                batch_size=batch_size,
                seed=seed,
                workers=args.workers,
                device=device,
            )
            aligned = True

        cells_on = use_clip and not (strategy == "two_stage" and epoch == 1)
        loader = make_epoch_loader(train_data, batch_size, seed, epoch, num_workers=args.workers)
        model.train()
        loss_sum = 0.0
        score_sum = 0.0
        started = time.monotonic()

        for step, raw in enumerate(loader, 1):
            batch = batch_to_device(raw, device)
            inputs = model_inputs(batch)
            if not cells_on:
                inputs.pop("clip_grid", None)
            elif strategy == "alternate":
                turn = "regions" if epoch % 2 == 1 else "clip"
                off = torch.zeros(batch["targets"].size(0), device=device)
                inputs["use_clip" if turn == "regions" else "use_regions"] = off
            elif strategy in ("acm", "two_stage") and p_mask > 0:
                inputs["use_regions"] = (
                    torch.rand(batch["targets"].size(0), device=device) >= p_mask
                ).float()

            optimizer.zero_grad()
            scores = model(**inputs)["ranking_scores"]
            loss = full_vocab_soft_target_bce(scores, batch["targets"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            loss_sum += float(loss.detach())
            score_sum += float(
                batch["targets"].gather(1, scores.detach().argmax(1, keepdim=True)).mean()
            ) * 100.0
            if step == 1 or step % 1000 == 0 or step == steps_per_epoch:
                print(
                    f"Epoch {epoch}/{epochs} | Step {step}/{steps_per_epoch} | "
                    f"loss={loss_sum/step:.4f} | train_score={score_sum/step:.2f}",
                    flush=True,
                )

        record = {
            "epoch": epoch,
            "train_loss": loss_sum / steps_per_epoch,
            "train_score": score_sum / steps_per_epoch,
            "train_seconds": time.monotonic() - started,
        }
        if val_data is not None:
            joint_metrics, preds = evaluate_vqa2(
                model,
                val_data,
                device,
                workers=args.workers,
                use_clip=cells_on,
                view="joint",
            )
            record["val_joint"] = joint_metrics
            atomic_json(args.output / f"predictions_epoch{epoch}.json", preds)
            if cells_on:
                clip_only_metrics, _ = evaluate_vqa2(
                    model,
                    val_data,
                    device,
                    workers=args.workers,
                    use_clip=True,
                    view="clip_only",
                )
                record["val_clip_only"] = clip_only_metrics

        history.append(record)
        atomic_json(args.output / "history.json", history)
        print(json.dumps(record), flush=True)

        if epoch == epochs:
            atomic_save(
                args.output / f"epoch{epoch}.pt",
                {
                    "config": run_config,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "history": history,
                },
            )


if __name__ == "__main__":
    main()

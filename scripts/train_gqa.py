#!/usr/bin/env python3
"""Train Plain LXMERT, RG-LXMERT (p=0), or ACM-LXMERT (0 < p < 1) on GQA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from acm_lxmert.data.gqa_dataset import GQADataset, WithGQAClipGrid
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
def evaluate_gqa(model, dataset, batch_size, workers, device, joint, output_path: Path) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    correct = 0.0
    count = 0
    predictions = []
    for raw in loader:
        inputs = model_inputs(batch_to_device(raw, device))
        if not joint:
            inputs.pop("clip_grid", None)
        scores = model(**inputs)["ranking_scores"]
        chosen = scores.argmax(1).cpu().tolist()
        for qid, answer_id, target in zip(raw["question_id"], chosen, raw["target"]):
            answer = dataset.answers[answer_id]
            correct += float(target[answer_id])
            count += 1
            predictions.append({"questionId": str(qid), "prediction": answer})
    atomic_json(output_path, predictions)
    return 100.0 * correct / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional JSON config file")
    parser.add_argument("--strategy", choices=("plain", "rg", "acm"), default=None)
    parser.add_argument("--gqa-root", required=True, type=Path, help="Processed GQA JSON splits + vocab")
    parser.add_argument("--vg-lmdb", required=True, type=Path, help="Visual Genome obj36 LMDB")
    parser.add_argument("--testdev-lmdb", required=True, type=Path, help="GQA testdev obj36 LMDB")
    parser.add_argument("--clip-root", type=Path, help="Directory with gqa_grid.f16.npy and gqa_index.json")
    parser.add_argument("--official-checkpoint", required=True, type=Path, help="Pretrained model_LXRT.pth")
    parser.add_argument("--bert-model", default="bert-base-uncased")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--grid", type=int, default=None)
    parser.add_argument("--region-mask-prob", type=float, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    strategy = args.strategy or cfg.get("strategy", "acm")
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 9595))
    epochs = int(args.epochs if args.epochs is not None else cfg.get("epochs", 4))
    batch_size = int(args.batch_size if args.batch_size is not None else cfg.get("batch_size", 32))
    lr = float(args.lr if args.lr is not None else cfg.get("lr", 1e-5))
    grid = int(args.grid if args.grid is not None else cfg.get("grid", 7))
    p_mask = float(
        args.region_mask_prob
        if args.region_mask_prob is not None
        else cfg.get("region_mask_prob", 0.6 if strategy == "acm" else 0.0)
    )
    joint = strategy != "plain"
    if joint and args.clip_root is None:
        raise ValueError("--clip-root is required unless --strategy plain")
    if strategy == "rg":
        p_mask = 0.0

    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_base = GQADataset(
        args.gqa_root, ["train", "valid"], args.vg_lmdb, args.testdev_lmdb, max_tokens=20, bert_model=args.bert_model
    )
    valid_base = GQADataset(
        args.gqa_root, ["testdev"], args.vg_lmdb, args.testdev_lmdb, max_tokens=20, bert_model=args.bert_model
    )
    train_data = WithGQAClipGrid(train_base, args.clip_root) if joint else train_base
    valid_data = WithGQAClipGrid(valid_base, args.clip_root) if joint else valid_base

    model_config = {
        "num_answers": 1842,
        "official_checkpoint": str(args.official_checkpoint.resolve()),
        "answer_vocab_path": str((args.gqa_root / "trainval_label2ans.json").resolve()),
    }
    model = OfficialBaselineVQA(model_config) if not joint else ClipGridVQA(model_config, grid=grid)
    model.to(device)

    steps_per_epoch = len(train_data) // batch_size
    optimizer = make_bert_adam(model, lr=lr, total_steps=steps_per_epoch * epochs, warmup=0.1, weight_decay=0.01)

    run_config = {
        "dataset": "GQA",
        "strategy": strategy,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "grid": grid if joint else None,
        "region_mask_prob": p_mask,
        "parameters": model.parameter_counts(),
    }
    atomic_json(args.output / "config.json", run_config)

    history = []
    for epoch in range(1, epochs + 1):
        loader = make_epoch_loader(train_data, batch_size, seed, epoch, num_workers=args.workers)
        model.train()
        running_loss = 0.0
        correct = 0.0
        seen = 0
        started = time.monotonic()

        for step, raw in enumerate(loader, 1):
            batch = batch_to_device(raw, device)
            inputs = model_inputs(batch)
            if not joint:
                inputs.pop("clip_grid", None)
            elif p_mask > 0:
                inputs["use_regions"] = (
                    torch.rand(batch["targets"].size(0), device=device) >= p_mask
                ).float()

            optimizer.zero_grad()
            scores = model(**inputs)["ranking_scores"]
            loss = full_vocab_soft_target_bce(scores, batch["targets"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            bsz = len(scores)
            running_loss += float(loss.detach()) * bsz
            correct += float(batch["targets"].gather(1, scores.detach().argmax(1, keepdim=True)).sum())
            seen += bsz
            if step == 1 or step % 1000 == 0 or step == steps_per_epoch:
                print(
                    f"Epoch {epoch}/{epochs} | Step {step}/{steps_per_epoch} | "
                    f"loss={running_loss/seen:.4f} | train_acc={100.0*correct/seen:.2f}",
                    flush=True,
                )

        pred_path = args.output / f"testdev_epoch{epoch}_predictions.json"
        acc = evaluate_gqa(model, valid_data, 256, args.workers, device, joint, pred_path)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "train_accuracy": 100.0 * correct / seen,
            "testdev_overall": acc,
            "seconds": time.monotonic() - started,
        }
        history.append(record)
        atomic_json(args.output / "history.json", history)
        print(json.dumps(record), flush=True)

        if epoch == epochs:
            atomic_save(
                args.output / f"epoch{epoch}.pth",
                {"config": run_config, "epoch": epoch, "model_state": model.state_dict(), "history": history},
            )


if __name__ == "__main__":
    main()

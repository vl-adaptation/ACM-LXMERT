#!/usr/bin/env python3
"""Extract CLIP RN50x4 7x7 grid features ([N, 49, 2560], float16) for COCO/VQA v2 images.

Supports reading images from either an extracted directory (e.g., ``train2014/``,
``val2014/``, ``test2015/``) or a COCO ``.zip`` archive. Before extraction, the
script verifies that the convolutional trunk combined with CLIP's pooling head
matches the TorchScript archive's ``encode_image`` output.
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset

SIZE = 448
GRID = 7
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def parse_coco_image_id(name: str) -> int:
    return int(Path(name).stem.rsplit("_", 1)[-1])


class ZipImageDataset(Dataset):
    def __init__(self, archive: Path, names: list[str]):
        self.archive = str(archive)
        self.names = names
        self.handle: zipfile.ZipFile | None = None

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        if self.handle is None:
            self.handle = zipfile.ZipFile(self.archive)
        with Image.open(io.BytesIO(self.handle.read(self.names[index]))) as image:
            width, height = image.size
            pixels = np.asarray(
                image.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC),
                dtype=np.uint8,
            )
        return index, torch.from_numpy(pixels.copy()).permute(2, 0, 1), width, height


class DirImageDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            width, height = image.size
            pixels = np.asarray(
                image.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC),
                dtype=np.uint8,
            )
        return index, torch.from_numpy(pixels.copy()).permute(2, 0, 1), width, height


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Image directory or .zip archive")
    parser.add_argument("--split", required=True, help="Split prefix, e.g. train2014, val2014, or test2015")
    parser.add_argument("--clip-model", required=True, type=Path, help="Path to OpenAI CLIP RN50x4.pt")
    parser.add_argument("--output", required=True, type=Path, help="Output feature directory")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for a quick smoke test")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(args.clip_model), map_location=device).eval()
    visual = model.visual
    dtype = visual.conv1.weight.dtype
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    def trunk(x: torch.Tensor) -> torch.Tensor:
        x = x.type(dtype)
        for conv, bn in ((visual.conv1, visual.bn1), (visual.conv2, visual.bn2), (visual.conv3, visual.bn3)):
            x = visual.relu(bn(conv(x)))
        x = visual.avgpool(x)
        return visual.layer4(visual.layer3(visual.layer2(visual.layer1(x))))

    probe = torch.randn(4, 3, 288, 288, device=device)
    reference = model.encode_image(probe).float()
    rebuilt = visual.attnpool(trunk(probe)).float()
    error = float((reference - rebuilt).abs().max())
    scale = float(reference.abs().max())
    if error > 1e-3 * max(scale, 1.0):
        raise RuntimeError(f"RN50x4 trunk check failed: max diff {error:.4g} at scale {scale:.4g}")

    if args.source.is_file() and args.source.suffix.lower() == ".zip":
        with zipfile.ZipFile(args.source) as handle:
            names = sorted(
                (n for n in handle.namelist() if n.lower().endswith(".jpg")),
                key=parse_coco_image_id,
            )
        if args.limit:
            names = names[: args.limit]
        ids = [parse_coco_image_id(n) for n in names]
        dataset: Dataset = ZipImageDataset(args.source, names)
    elif args.source.is_dir():
        paths = sorted(
            (p for p in args.source.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")),
            key=lambda p: parse_coco_image_id(p.name),
        )
        if args.limit:
            paths = paths[: args.limit]
        ids = [parse_coco_image_id(p.name) for p in paths]
        dataset = DirImageDataset(paths)
    else:
        raise FileNotFoundError(f"Source must be an image directory or .zip archive: {args.source}")

    count = len(ids)
    grid_file = args.output / f"{args.split}_grid.f16.npy"
    sizes_file = args.output / f"{args.split}_sizes.npy"
    progress_file = args.output / f"{args.split}_progress.json"
    done = 0
    if progress_file.is_file() and grid_file.is_file() and sizes_file.is_file():
        prev = json.loads(progress_file.read_text(encoding="utf-8"))
        if prev.get("count") == count and prev.get("first_id") == ids[0]:
            done = int(prev["done"])

    grid = np.lib.format.open_memmap(
        grid_file,
        mode="r+" if done else "w+",
        dtype=np.float16,
        shape=(count, GRID * GRID, 2560),
    )
    sizes = np.load(sizes_file) if done else np.zeros((count, 2), dtype=np.int32)

    if done < count:
        loader = DataLoader(
            Subset(dataset, range(done, count)),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        started, position = time.monotonic(), done
        for step, (indices, pixels, widths, heights) in enumerate(loader, 1):
            if int(indices[0]) != position:
                raise RuntimeError("Images arrived out of order")
            x = (pixels.to(device, non_blocking=True).float().div_(255.0) - mean) / std
            cells = torch.nn.functional.adaptive_avg_pool2d(trunk(x).float(), GRID)
            cells = cells.flatten(2).transpose(1, 2)
            if not torch.isfinite(cells).all():
                raise FloatingPointError(f"Non-finite CLIP grid features at position {position}")
            size = len(indices)
            grid[position : position + size] = cells.half().cpu().numpy()
            sizes[position : position + size, 0] = widths.numpy()
            sizes[position : position + size, 1] = heights.numpy()
            position += size
            if step % 100 == 0 or position == count:
                grid.flush()
                np.save(sizes_file, sizes)
                write_json(progress_file, {"count": count, "first_id": ids[0], "done": position})
                rate = (position - done) / max(time.monotonic() - started, 1e-6)
                print(f"[{args.split}] {position}/{count} ({rate:.1f} img/s)", flush=True)

    write_json(
        args.output / f"{args.split}_index.json",
        {
            "image_ids": ids,
            "width_height": sizes.tolist(),
            "grid": GRID,
            "tokens": GRID * GRID,
            "dim": 2560,
            "dtype": "float16",
            "input_size": SIZE,
            "model": "CLIP RN50x4",
            "rows": count,
            "trunk_check_max_difference": error,
        },
    )
    print(f"Extraction complete: {grid_file} ({count} images)", flush=True)


if __name__ == "__main__":
    main()

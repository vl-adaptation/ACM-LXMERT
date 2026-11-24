#!/usr/bin/env python3
"""Extract the established RN50x4/448/7x7 GQA CLIP representation.

The manifest fixes the exact image order. Output is float16 [N,49,2560] plus
an index JSON, with resumable progress. This script performs no training.
"""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


SIZE, GRID = 448, 7
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


class Images(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from PIL import Image
        with Image.open(self.paths[index]) as image:
            width, height = image.size
            pixels = np.asarray(image.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC),
                                dtype=np.uint8)
        return index, torch.from_numpy(pixels.copy()).permute(2, 0, 1), width, height


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    ids = [str(value) for value in manifest["image_ids"]]
    if args.limit:
        ids = ids[:args.limit]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("manifest is empty or contains duplicate IDs")
    paths = [Path(args.images) / (image_id + ".jpg") for image_id in ids]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing images: %r" % missing[:10])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    model = torch.jit.load(args.model, map_location=device).eval()
    visual, dtype = model.visual, model.visual.conv1.weight.dtype
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    def trunk(x):
        x = x.type(dtype)
        for conv, norm in ((visual.conv1, visual.bn1), (visual.conv2, visual.bn2),
                           (visual.conv3, visual.bn3)):
            x = visual.relu(norm(conv(x)))
        x = visual.avgpool(x)
        return visual.layer4(visual.layer3(visual.layer2(visual.layer1(x))))

    generator = torch.Generator(device="cuda").manual_seed(20261010)
    probe = torch.randn(4, 3, 288, 288, generator=generator, device=device)
    reference = model.encode_image(probe).float()
    rebuilt = visual.attnpool(trunk(probe)).float()
    error, scale = float((reference - rebuilt).abs().max()), float(reference.abs().max())
    if error > 1e-3 * max(scale, 1.0):
        raise RuntimeError("RN50x4 trunk check failed: %.5g at scale %.5g" % (error, scale))

    count = len(ids)
    grid_path, sizes_path = output / "gqa_grid.f16.npy", output / "gqa_sizes.npy"
    progress_path = output / "progress.json"
    done = 0
    if progress_path.is_file() and grid_path.is_file() and sizes_path.is_file():
        old = json.loads(progress_path.read_text())
        if old.get("rows") == count and old.get("id_sha256") == manifest.get("id_sha256"):
            done = int(old["done"])
    grid = np.lib.format.open_memmap(grid_path, mode="r+" if done else "w+",
                                     dtype=np.float16, shape=(count, 49, 2560))
    sizes = np.load(sizes_path) if done else np.zeros((count, 2), dtype=np.int32)
    loader = DataLoader(torch.utils.data.Subset(Images(paths), range(done, count)),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    started, position = time.monotonic(), done
    for step, (indices, pixels, widths, heights) in enumerate(loader, 1):
        if int(indices[0]) != position:
            raise RuntimeError("image order mismatch")
        x = (pixels.to(device, non_blocking=True).float().div_(255) - mean) / std
        cells = torch.nn.functional.adaptive_avg_pool2d(trunk(x).float(), GRID)
        cells = cells.flatten(2).transpose(1, 2)
        if cells.shape[1:] != (49, 2560) or not torch.isfinite(cells).all():
            raise ValueError("bad CLIP cells at row %d" % position)
        size = len(indices)
        grid[position:position + size] = cells.half().cpu().numpy()
        sizes[position:position + size, 0] = widths.numpy()
        sizes[position:position + size, 1] = heights.numpy()
        position += size
        if step % 100 == 0 or position == count:
            grid.flush()
            np.save(sizes_path, sizes)
            atomic_json(progress_path, {"rows": count, "done": position,
                                        "id_sha256": manifest.get("id_sha256")})
            print("done=%d/%d rate=%.2f image/s" %
                  (position, count, (position - done) /
                   max(time.monotonic() - started, 1e-6)), flush=True)
    atomic_json(output / "gqa_index.json", {
        "image_ids": ids, "rows": count, "width_height": sizes.tolist(),
        "grid": 7, "tokens": 49, "dim": 2560, "dtype": "float16",
        "model": "CLIP RN50x4", "input_size": 448,
        "manifest_id_sha256": manifest.get("id_sha256"),
        "trunk_check_max_difference": error})
    print("EXTRACTION_COMPLETE rows=%d" % count)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a memory-safe cache from the official LXMERT obj36 TSV.

The cache contains only the tensors consumed by the official VQA model:
float32 region features and boxes normalized exactly as in vqa_data.py.
Existing complete entries are skipped, so an interrupted conversion can resume.
"""

import argparse
import base64
import csv
import io
import json
import os
import sys
import time

import lmdb
import numpy as np


csv.field_size_limit(sys.maxsize)
FIELDNAMES = [
    "img_id", "img_h", "img_w", "objects_id", "objects_conf",
    "attrs_id", "attrs_conf", "num_boxes", "boxes", "features",
]
META_KEY = b"__official_tsv_cache_meta__"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--map-size-gib", type=int, default=48)
    parser.add_argument("--commit-every", type=int, default=256)
    return parser.parse_args()


def encode_npy(array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def keys(img_id):
    raw = str(int(img_id.rsplit("_", 1)[-1]))
    return (
        ("visual::visual/%s.npy" % raw).encode("utf-8"),
        ("spatial::spatial/%s.npy" % raw).encode("utf-8"),
    )


def main():
    args = parse_args()
    tsv_path = os.path.abspath(args.tsv)
    output_path = os.path.abspath(args.output)
    if not os.path.isfile(tsv_path):
        raise FileNotFoundError(tsv_path)
    os.makedirs(output_path, exist_ok=True)
    env = lmdb.open(
        output_path,
        map_size=args.map_size_gib * 1024 ** 3,
        subdir=True,
        lock=True,
        readahead=False,
        meminit=False,
        max_readers=64,
    )

    started = time.time()
    total = inserted = skipped = 0
    txn = env.begin(write=True)
    try:
        with open(tsv_path, encoding="utf-8") as handle:
            reader = csv.DictReader(handle, FIELDNAMES, delimiter="\t")
            for item in reader:
                total += 1
                visual_key, spatial_key = keys(item["img_id"])
                if txn.get(visual_key) is not None and txn.get(spatial_key) is not None:
                    skipped += 1
                else:
                    num_boxes = int(item["num_boxes"])
                    img_h = int(item["img_h"])
                    img_w = int(item["img_w"])
                    features = np.frombuffer(
                        base64.b64decode(item["features"]), dtype=np.float32
                    ).reshape(num_boxes, -1)
                    boxes = np.frombuffer(
                        base64.b64decode(item["boxes"]), dtype=np.float32
                    ).reshape(num_boxes, 4).copy()
                    if features.shape != (36, 2048) or boxes.shape != (36, 4):
                        raise ValueError(
                            "Unexpected obj36 shape for %s: features=%s boxes=%s" %
                            (item["img_id"], features.shape, boxes.shape)
                        )
                    boxes[:, (0, 2)] /= img_w
                    boxes[:, (1, 3)] /= img_h
                    np.testing.assert_array_less(boxes, 1 + 1e-5)
                    np.testing.assert_array_less(-boxes, 0 + 1e-5)
                    if not np.isfinite(features).all() or not np.isfinite(boxes).all():
                        raise FloatingPointError(
                            "Non-finite TSV tensor for %s" % item["img_id"]
                        )
                    txn.put(visual_key, encode_npy(features))
                    txn.put(spatial_key, encode_npy(boxes))
                    inserted += 1

                if total % args.commit_every == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                if total % 1000 == 0:
                    elapsed = max(time.time() - started, 1e-6)
                    print(
                        "images=%d inserted=%d skipped=%d images_per_sec=%.2f" %
                        (total, inserted, skipped, total / elapsed),
                        flush=True,
                    )

        metadata = {
            "complete": True,
            "source_tsv": tsv_path,
            "source_size": os.path.getsize(tsv_path),
            "images": total,
            "inserted_this_run": inserted,
            "skipped_existing": skipped,
            "visual_shape": [36, 2048],
            "spatial_shape": [36, 4],
            "dtype": "float32",
            "normalization": "official vqa_data.py box normalization",
        }
        txn.put(META_KEY, json.dumps(metadata, sort_keys=True).encode("utf-8"))
        txn.commit()
        txn = None
        env.sync()
    finally:
        if txn is not None:
            txn.abort()
        env.close()

    elapsed = time.time() - started
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print("CACHE_COMPLETE elapsed_sec=%.1f" % elapsed)


if __name__ == "__main__":
    main()

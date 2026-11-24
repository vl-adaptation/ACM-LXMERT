#!/usr/bin/env python3
"""Convert official LXMERT GQA obj36 TSV into a resumable LMDB.

Unlike the older VQA converter, this preserves GQA image IDs verbatim; IDs may
be numeric (``2375429``) or prefixed (``n161313``).
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
FIELDS = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
          "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
META_KEY = b"__gqa_obj36_cache_meta__"


def npy_bytes(array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def keys(image_id):
    image_id = str(image_id)
    if not image_id or "/" in image_id or "\\" in image_id:
        raise ValueError("unsafe image id: %r" % image_id)
    return (("visual::visual/%s.npy" % image_id).encode(),
            ("spatial::spatial/%s.npy" % image_id).encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--map-size-gib", type=int, default=48)
    parser.add_argument("--commit-every", type=int, default=256)
    args = parser.parse_args()
    if not os.path.isfile(args.tsv):
        raise FileNotFoundError(args.tsv)
    os.makedirs(args.output, exist_ok=True)
    env = lmdb.open(args.output, map_size=args.map_size_gib * 1024 ** 3,
                    subdir=True, lock=True, readahead=False, meminit=False)
    started = time.time()
    total = inserted = skipped = 0
    seen = set()
    txn = env.begin(write=True)
    try:
        with open(args.tsv, encoding="utf-8") as handle:
            for item in csv.DictReader(handle, FIELDS, delimiter="\t"):
                total += 1
                image_id = str(item["img_id"])
                if image_id in seen:
                    raise ValueError("duplicate TSV image id: %s" % image_id)
                seen.add(image_id)
                visual_key, spatial_key = keys(image_id)
                if txn.get(visual_key) is not None and txn.get(spatial_key) is not None:
                    skipped += 1
                else:
                    count = int(item["num_boxes"])
                    height, width = int(item["img_h"]), int(item["img_w"])
                    features = np.frombuffer(base64.b64decode(item["features"]),
                                             dtype=np.float32).reshape(count, -1)
                    boxes = np.frombuffer(base64.b64decode(item["boxes"]),
                                          dtype=np.float32).reshape(count, 4).copy()
                    if features.shape != (36, 2048) or boxes.shape != (36, 4):
                        raise ValueError("%s is not fixed obj36: %s %s" %
                                         (image_id, features.shape, boxes.shape))
                    boxes[:, (0, 2)] /= width
                    boxes[:, (1, 3)] /= height
                    if (not np.isfinite(features).all() or not np.isfinite(boxes).all()
                            or boxes.min() < -1e-5 or boxes.max() > 1 + 1e-5):
                        raise ValueError("invalid arrays for %s" % image_id)
                    txn.put(visual_key, npy_bytes(features))
                    txn.put(spatial_key, npy_bytes(boxes))
                    inserted += 1
                if total % args.commit_every == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                if total % 1000 == 0:
                    print("images=%d inserted=%d skipped=%d" %
                          (total, inserted, skipped), flush=True)
        metadata = {"complete": True, "source_tsv": os.path.abspath(args.tsv),
                    "source_size": os.path.getsize(args.tsv), "images": total,
                    "inserted_this_run": inserted, "skipped_existing": skipped,
                    "id_policy": "exact_GQA_string", "visual_shape": [36, 2048],
                    "spatial_shape": [36, 4], "dtype": "float32"}
        txn.put(META_KEY, json.dumps(metadata, sort_keys=True).encode())
        txn.commit()
        txn = None
        env.sync()
    finally:
        if txn is not None:
            txn.abort()
        env.close()
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print("CACHE_COMPLETE elapsed_sec=%.1f" % (time.time() - started))


if __name__ == "__main__":
    main()

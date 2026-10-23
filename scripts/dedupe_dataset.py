"""Remove exact (byte-identical) duplicate images from an existing iconoscope
HDF5 dataset.

Keeps the first path (in stored, sorted order) for each set of identical files
and subsets image paths, sizes, and all stored model features to match.
UMAP and cluster results are deleted, since they were computed on the full set
and must be regenerated.

Usage:
    python scripts/dedupe_dataset.py dataset.h5 [--dry-run]

Back up the dataset first. HDF5 does not reclaim space from deleted data;
run ``h5repack`` afterwards to shrink the file.
"""

import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np


def file_hash(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def subset(group: h5py.Group, name: str, keep: np.ndarray) -> None:
    # datasets can't be resized in place; delete and recreate
    data = group[name][()][keep]
    del group[name]
    group.create_dataset(name, data=data, compression="gzip")


def dedupe(h5_path: Path, dry_run: bool = False) -> None:
    with h5py.File(h5_path, "r" if dry_run else "r+") as f:
        img = f["image"]
        base = Path(img.attrs["image_dir"])
        paths = [p.decode() if isinstance(p, bytes) else p for p in img["paths"][()]]

        seen: dict[str, int] = {}
        keep: list[int] = []
        dupes: list[tuple[str, str]] = []
        for i, rel in enumerate(paths):
            h = file_hash(base / rel)
            if h in seen:
                dupes.append((rel, paths[seen[h]]))
            else:
                seen[h] = i
                keep.append(i)

        print(f"{len(paths)} images, {len(dupes)} exact duplicates")
        for dup, orig in dupes:
            print(f"  {dup}  ==  {orig}")
        if dry_run or not dupes:
            return

        keep_idx = np.array(keep)
        subset(img, "paths", keep_idx)
        subset(img, "sizes", keep_idx)

        models = img.get("models")
        if models is not None:
            for model_grp in models.values():
                for opt_grp in model_grp.values():
                    if "features" in opt_grp:
                        subset(opt_grp, "features", keep_idx)
                    for derived in ("umap", "cluster"):
                        if derived in opt_grp:
                            del opt_grp[derived]
                            print(f"  removed {opt_grp.name}/{derived} (regenerate)")

        print(f"kept {len(keep)} images")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=Path, help="path to iconoscope HDF5 dataset")
    parser.add_argument(
        "--dry-run", action="store_true", help="report duplicates without modifying"
    )
    args = parser.parse_args()
    dedupe(args.dataset, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

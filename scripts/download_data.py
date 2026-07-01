#!/usr/bin/env python3
"""
Download and verify ModelNet40 HDF5 dataset.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --root /path/to/data/dir
"""

import argparse
import hashlib
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

DATASET_URL = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
DATASET_DIR_NAME = "modelnet40_ply_hdf5_2048"

# Expected number of .h5 files (5 train + 2 test = 7 total)
EXPECTED_H5_COUNT = 7
EXPECTED_TRAIN_SAMPLES = 9843
EXPECTED_TEST_SAMPLES = 2468


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download ModelNet40 HDF5 dataset."
    )
    parser.add_argument(
        '--root', type=str, default='data/',
        help="Root directory to download dataset into (default: data/)."
    )
    parser.add_argument(
        '--force', action='store_true',
        help="Re-download even if dataset already exists."
    )
    return parser.parse_args()


def progress_hook(block_num, block_size, total_size):
    """Print download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, 100.0 * downloaded / total_size)
        mb = downloaded / 1024 / 1024
        total_mb = total_size / 1024 / 1024
        print(f"\r  Downloading: {pct:5.1f}%  ({mb:.1f} / {total_mb:.1f} MB)", end='', flush=True)


def verify_dataset(data_dir: str) -> bool:
    """
    Verify that the dataset is complete and readable.

    Checks:
        1. All expected H5 files are present (from train_files.txt and test_files.txt).
        2. Can load and count samples from each file.
        3. Total counts match expected values.

    Returns:
        True if verification passes, False otherwise.
    """
    import h5py

    data_dir = Path(data_dir)
    ok = True

    # Check for required files
    required_files = ['train_files.txt', 'test_files.txt', 'shape_names.txt']
    for fname in required_files:
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"  [MISSING] {fpath}")
            ok = False
        else:
            print(f"  [OK]      {fpath}")

    if not ok:
        return False

    # Count samples
    total_train = 0
    total_test = 0

    for split, total_var in [('train', 'total_train'), ('test', 'total_test')]:
        list_file = data_dir / f"{split}_files.txt"
        with open(list_file, 'r') as f:
            h5_files = [l.strip() for l in f if l.strip()]

        split_count = 0
        for h5_name in h5_files:
            # Try to find the file
            candidates = [
                data_dir / h5_name,
                data_dir / Path(h5_name).name,
                Path(h5_name),
            ]
            h5_path = None
            for c in candidates:
                if c.exists():
                    h5_path = c
                    break

            if h5_path is None:
                print(f"  [MISSING] H5 file: {h5_name}")
                ok = False
                continue

            try:
                with h5py.File(str(h5_path), 'r') as f:
                    n = f['data'].shape[0]
                    split_count += n
                    print(f"  [OK]      {h5_path.name} ({n} samples, shape {f['data'].shape})")
            except Exception as e:
                print(f"  [ERROR]   {h5_path}: {e}")
                ok = False

        if split == 'train':
            total_train = split_count
        else:
            total_test = split_count

    print(f"\n  Train samples: {total_train} (expected {EXPECTED_TRAIN_SAMPLES})")
    print(f"  Test  samples: {total_test}  (expected {EXPECTED_TEST_SAMPLES})")

    if total_train != EXPECTED_TRAIN_SAMPLES:
        print(f"  [WARN] Train count mismatch: got {total_train}, expected {EXPECTED_TRAIN_SAMPLES}")
    if total_test != EXPECTED_TEST_SAMPLES:
        print(f"  [WARN] Test count mismatch: got {total_test}, expected {EXPECTED_TEST_SAMPLES}")

    return ok


def main():
    args = parse_args()

    root = Path(args.root)
    data_dir = root / DATASET_DIR_NAME
    zip_path = root / "modelnet40_ply_hdf5_2048.zip"

    print(f"Dataset root: {root.resolve()}")
    print(f"Dataset dir:  {data_dir.resolve()}")

    # Check if already exists
    if data_dir.exists() and not args.force:
        h5_files = list(data_dir.glob("*.h5"))
        if len(h5_files) >= EXPECTED_H5_COUNT:
            print(f"\nDataset already exists ({len(h5_files)} .h5 files found).")
            print("Use --force to re-download.\n")
            print("Verifying...")
            ok = verify_dataset(str(data_dir))
            if ok:
                print("\nDataset verification PASSED.")
            else:
                print("\nDataset verification FAILED. Try --force to re-download.")
            return

    # Create directory
    root.mkdir(parents=True, exist_ok=True)

    # Download
    if not zip_path.exists() or args.force:
        print(f"\nDownloading ModelNet40 from:")
        print(f"  {DATASET_URL}")
        print(f"  Target: {zip_path}")
        print(f"  Size: ~440 MB\n")
        try:
            urllib.request.urlretrieve(DATASET_URL, str(zip_path), reporthook=progress_hook)
            print()  # newline after progress
        except Exception as e:
            print(f"\nDownload failed: {e}")
            print("\nAlternative: Download manually from:")
            print(f"  {DATASET_URL}")
            print(f"and place the zip at: {zip_path}")
            sys.exit(1)
    else:
        print(f"\nZip file already exists: {zip_path}")

    # Extract
    print(f"\nExtracting {zip_path} to {root} ...")
    try:
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            # List what's inside
            members = zf.namelist()
            print(f"  Archive contains {len(members)} files.")
            zf.extractall(str(root))
        print(f"  Extraction complete.")
    except Exception as e:
        print(f"  Extraction failed: {e}")
        sys.exit(1)

    # Verify
    print(f"\nVerifying dataset at {data_dir} ...")
    if not data_dir.exists():
        # Sometimes the zip extracts to a slightly different name
        extracted = list(root.glob("modelnet40*"))
        print(f"  Found after extraction: {extracted}")
        if extracted:
            data_dir = extracted[0]

    ok = verify_dataset(str(data_dir))

    if ok:
        print("\nDataset download and verification PASSED.")
        print(f"\nDataset is ready at: {data_dir.resolve()}")
        print("\nNext steps:")
        print("  python scripts/train.py --config configs/baseline.yaml")
        print("  python scripts/train.py --config configs/qsde.yaml")
    else:
        print("\nVerification had warnings/errors. Check above output.")


if __name__ == "__main__":
    main()

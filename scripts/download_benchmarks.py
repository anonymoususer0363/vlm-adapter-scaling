"""
Download evaluation benchmarks for G10 downstream correlation.

Benchmarks:
1. VQAv2 (val): questions + annotations + COCO val2014 images
2. TextVQA (val): questions + annotations + images
3. COCO Caption (Karpathy test): karpathy split file + COCO images

Usage:
    python scripts/download_benchmarks.py --data_dir data/benchmarks
    python scripts/download_benchmarks.py --data_dir data/benchmarks --benchmarks vqav2 textvqa
"""

import argparse
import json
import os
import subprocess
import zipfile
from pathlib import Path


def download_file(url: str, dest: str):
    """Download file using wget (more robust for large files)."""
    dest_path = Path(dest)
    if dest_path.exists():
        print(f"  Already exists: {dest}")
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading: {url}")
    subprocess.run(
        ["wget", "-q", "--show-progress", "-O", dest, url],
        check=True,
    )


def extract_zip(zip_path: str, extract_to: str):
    """Extract zip file."""
    print(f"  Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)


def download_vqav2(data_dir: str):
    """Download VQAv2 validation set."""
    vqa_dir = Path(data_dir) / "vqav2"
    vqa_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== VQAv2 Validation Set ===")

    # Questions
    q_url = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip"
    q_zip = str(vqa_dir / "v2_Questions_Val_mscoco.zip")
    q_file = vqa_dir / "v2_OpenEnded_mscoco_val2014_questions.json"
    if not q_file.exists():
        download_file(q_url, q_zip)
        extract_zip(q_zip, str(vqa_dir))

    # Annotations
    a_url = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"
    a_zip = str(vqa_dir / "v2_Annotations_Val_mscoco.zip")
    a_file = vqa_dir / "v2_mscoco_val2014_annotations.json"
    if not a_file.exists():
        download_file(a_url, a_zip)
        extract_zip(a_zip, str(vqa_dir))

    # COCO val2014 images (shared with caption benchmark)
    img_dir = Path(data_dir) / "coco" / "val2014"
    if not img_dir.exists():
        download_coco_val2014(data_dir)

    print(f"  VQAv2 ready: {vqa_dir}")
    if q_file.exists():
        with open(q_file) as f:
            data = json.load(f)
        print(f"  Questions: {len(data['questions']):,}")


def download_textvqa(data_dir: str):
    """Download TextVQA validation set."""
    tvqa_dir = Path(data_dir) / "textvqa"
    tvqa_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== TextVQA Validation Set ===")

    # Annotations
    ann_url = "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json"
    ann_file = str(tvqa_dir / "TextVQA_0.5.1_val.json")
    download_file(ann_url, ann_file)

    # Images
    img_zip_url = "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip"
    img_zip = str(tvqa_dir / "train_val_images.zip")
    img_dir = tvqa_dir / "train_images"
    if not img_dir.exists():
        download_file(img_zip_url, img_zip)
        extract_zip(img_zip, str(tvqa_dir))

    print(f"  TextVQA ready: {tvqa_dir}")
    if Path(ann_file).exists():
        with open(ann_file) as f:
            data = json.load(f)
        print(f"  Val samples: {len(data['data']):,}")


def download_coco_val2014(data_dir: str):
    """Download COCO val2014 images."""
    coco_dir = Path(data_dir) / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== COCO val2014 Images ===")
    img_url = "http://images.cocodataset.org/zips/val2014.zip"
    img_zip = str(coco_dir / "val2014.zip")
    if not (coco_dir / "val2014").exists():
        download_file(img_url, img_zip)
        extract_zip(img_zip, str(coco_dir))
    print(f"  COCO val2014 ready: {coco_dir / 'val2014'}")


def download_coco_karpathy(data_dir: str):
    """Download Karpathy split annotations for COCO Caption."""
    caption_dir = Path(data_dir) / "coco_caption"
    caption_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== COCO Caption (Karpathy Split) ===")

    # Karpathy split file
    karpathy_url = "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
    karpathy_zip = str(caption_dir / "caption_datasets.zip")
    karpathy_file = caption_dir / "dataset_coco.json"
    if not karpathy_file.exists():
        download_file(karpathy_url, karpathy_zip)
        extract_zip(karpathy_zip, str(caption_dir))

    # COCO val2014 images (also used for test split)
    img_dir = Path(data_dir) / "coco" / "val2014"
    if not img_dir.exists():
        download_coco_val2014(data_dir)

    print(f"  Karpathy split ready: {caption_dir}")
    if karpathy_file.exists():
        with open(karpathy_file) as f:
            data = json.load(f)
        test_count = sum(1 for img in data["images"] if img["split"] == "test")
        print(f"  Test images: {test_count:,}")


def main():
    parser = argparse.ArgumentParser(description="Download evaluation benchmarks")
    parser.add_argument("--data_dir", type=str, default="data/benchmarks",
                        help="Root directory for benchmark data")
    parser.add_argument("--benchmarks", nargs="*", default=["vqav2", "textvqa", "coco_caption"],
                        choices=["vqav2", "textvqa", "coco_caption"],
                        help="Which benchmarks to download")
    args = parser.parse_args()

    print(f"Benchmark data directory: {args.data_dir}")

    if "vqav2" in args.benchmarks:
        download_vqav2(args.data_dir)

    if "textvqa" in args.benchmarks:
        download_textvqa(args.data_dir)

    if "coco_caption" in args.benchmarks:
        download_coco_karpathy(args.data_dir)

    print("\n=== Done ===")
    print(f"Total disk usage:")
    subprocess.run(["du", "-sh", args.data_dir])


if __name__ == "__main__":
    main()

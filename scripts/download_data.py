"""
Download datasets for VLM adapter scaling law experiments.

Phase 1: LLaVA-Pretrain-558K (for smoke testing + small experiments)
Phase 2: Convert LLaVA to our unified JSONL format
Phase 3: Download ShareGPT4V-PT + convert to JSONL
Phase 4: Create combined dataset (LLaVA + ShareGPT4V, deduplicated)

ShareGPT4V-PT image sources (~1.2M pairs):
  - sam/images/     : SAM images (~11K archives from ShareGPT4V-SAM-HF)
  - coco/train2017/ : COCO 2017 training images
  - llava/llava_pretrain/images/ : LLaVA-Pretrain CC3M subset
  - share_textvqa/images/ : TextVQA images
  - web-celebrity/images/, web-landmark/images/, wikiart/images/ : web images

Image download strategy:
  - Metadata from HF dataset Lin-Chen/ShareGPT4V
  - SAM images from Lin-Chen/ShareGPT4V-SAM-HF (tar archives)
  - COCO from official source
  - LLaVA images: symlink to existing llava_pretrain/
  - Other web images from Lin-Chen/ShareGPT4V-Data (zip archives)
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path


def download_llava_pretrain(data_dir: str):
    """
    Download LLaVA-Pretrain-558K from HuggingFace.
    This includes CC3M-595K captions + BLIP captions.
    Images come from CC3M (need separate download) or we use the HF dataset version.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    save_dir = Path(data_dir) / "llava_pretrain"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading LLaVA-Pretrain metadata...")
    # Download the caption/conversation file
    hf_hub_download(
        repo_id="liuhaotian/LLaVA-Pretrain",
        filename="blip_laion_cc_sbu_558k.json",
        repo_type="dataset",
        local_dir=str(save_dir),
    )

    # Download images (zip file ~13GB)
    print("Downloading LLaVA-Pretrain images (this may take a while)...")
    hf_hub_download(
        repo_id="liuhaotian/LLaVA-Pretrain",
        filename="images.zip",
        repo_type="dataset",
        local_dir=str(save_dir),
    )

    print(f"Downloaded to {save_dir}")
    print("Unzipping images...")
    os.system(f"cd {save_dir} && unzip -q -o images.zip")
    print("Done!")

    return save_dir


def convert_llava_to_jsonl(data_dir: str, output_dir: str):
    """Convert LLaVA-Pretrain JSON to our unified JSONL format."""
    save_dir = Path(data_dir) / "llava_pretrain"
    json_path = save_dir / "blip_laion_cc_sbu_558k.json"

    if not json_path.exists():
        print(f"LLaVA data not found at {json_path}. Download first.")
        return

    print("Converting LLaVA-Pretrain to JSONL...")
    with open(json_path) as f:
        data = json.load(f)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert to our format
    records = []
    for item in data:
        image_path = item.get("image", "")
        caption = ""
        for conv in item.get("conversations", []):
            if conv.get("from") == "gpt":
                caption = conv["value"]
                break

        if image_path and caption:
            records.append({
                "image": image_path,
                "caption": caption,
            })

    # Shuffle with fixed seed
    random.seed(42)
    random.shuffle(records)

    # Split: 99% train, 1% val
    split_idx = int(len(records) * 0.99)
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    # Write train
    train_path = output_path / "train.jsonl"
    with open(train_path, "w") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    # Write val
    val_path = output_path / "val.jsonl"
    with open(val_path, "w") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")

    print(f"Train: {len(train_records)} samples -> {train_path}")
    print(f"Val:   {len(val_records)} samples -> {val_path}")

    # Create subsets for D control (for smoke testing with small D)
    for subset_size in [1000, 5000, 10000, 50000, 100000, 500000]:
        if subset_size > len(train_records):
            continue
        subset = train_records[:subset_size]
        subset_path = output_path / f"train_{subset_size // 1000}k.jsonl"
        with open(subset_path, "w") as f:
            for r in subset:
                f.write(json.dumps(r) + "\n")
        print(f"Subset D={subset_size}: {subset_path}")


# ──────────────────────────────────────────────────────────────
# ShareGPT4V download + conversion
# ──────────────────────────────────────────────────────────────

# ShareGPT4V-PT metadata filename on HuggingFace (Lin-Chen/ShareGPT4V)
SHAREGPT4V_PT_JSON = "share-captioner_coco_lcs_sam_1246k_1107.json"

# Image source prefixes found in ShareGPT4V-PT metadata
# Each prefix maps to a download source.
SHAREGPT4V_IMAGE_SOURCES = {
    "sam":            "Lin-Chen/ShareGPT4V-SAM-HF",     # tar archives
    "coco":           "coco2017",                         # COCO 2017 official
    "llava":          "symlink",                          # reuse llava_pretrain
    "share_textvqa":  "Lin-Chen/ShareGPT4V-Data",        # zip archive
    "web-celebrity":  "Lin-Chen/ShareGPT4V-Data",        # zip archive
    "web-landmark":   "Lin-Chen/ShareGPT4V-Data",        # zip archive
    "wikiart":        "Lin-Chen/ShareGPT4V-Data",        # zip archive
}


def download_sharegpt4v(data_dir: str, skip_images: bool = False):
    """Download ShareGPT4V-PT metadata + images.

    Directory structure after download:
        data/sharegpt4v/
            share-captioner_coco_lcs_sam_1246k_1107.json   (metadata)
            sam/images/sa_*.jpg
            coco/train2017/*.jpg
            llava/llava_pretrain/images/  -> ../../llava_pretrain/  (symlink)
            share_textvqa/images/*.jpg
            web-celebrity/images/*.jpg
            web-landmark/images/*.jpg
            wikiart/images/*.jpg
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    save_dir = Path(data_dir) / "sharegpt4v"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Download metadata JSON ──
    meta_path = save_dir / SHAREGPT4V_PT_JSON
    if meta_path.exists():
        print(f"Metadata already exists: {meta_path}")
    else:
        print(f"Downloading ShareGPT4V-PT metadata ({SHAREGPT4V_PT_JSON})...")
        hf_hub_download(
            repo_id="Lin-Chen/ShareGPT4V",
            filename=SHAREGPT4V_PT_JSON,
            repo_type="dataset",
            local_dir=str(save_dir),
        )
        print(f"Metadata saved to {meta_path}")

    if skip_images:
        print("Skipping image downloads (--skip-images).")
        return save_dir

    # ── Step 2: Download images from each source ──
    _download_llava_symlink(data_dir, save_dir)
    _download_coco_images(save_dir)
    _download_sam_images(save_dir)
    _download_web_images(save_dir)

    print(f"\nShareGPT4V download complete: {save_dir}")
    return save_dir


def _download_llava_symlink(data_dir: str, save_dir: Path):
    """Create symlink so sharegpt4v/llava/llava_pretrain/images/ -> llava_pretrain/.

    ShareGPT4V metadata references images as "llava/llava_pretrain/images/00001/00001234.jpg"
    which means the actual image is at llava_pretrain/00001/00001234.jpg (same as LLaVA-Pretrain).
    """
    llava_src = Path(data_dir) / "llava_pretrain"
    link_parent = save_dir / "llava" / "llava_pretrain"
    link_target = link_parent / "images"

    if link_target.exists() or link_target.is_symlink():
        print(f"LLaVA symlink already exists: {link_target}")
        return

    if not llava_src.exists():
        print(f"WARNING: LLaVA-Pretrain images not found at {llava_src}")
        print("  Run phase 1 first, or images referencing llava/ prefix will be missing.")
        return

    link_parent.mkdir(parents=True, exist_ok=True)
    # Compute relative path from link location to actual llava_pretrain dir
    rel = os.path.relpath(llava_src, link_parent)
    os.symlink(rel, str(link_target))
    print(f"Symlinked LLaVA images: {link_target} -> {rel}")


def _download_coco_images(save_dir: Path):
    """Download COCO 2017 train images if not already present.

    ShareGPT4V references: coco/train2017/000000000009.jpg
    """
    coco_dir = save_dir / "coco" / "train2017"
    if coco_dir.exists() and any(coco_dir.iterdir()):
        n_imgs = sum(1 for _ in coco_dir.glob("*.jpg"))
        print(f"COCO train2017 already exists ({n_imgs} images): {coco_dir}")
        return

    coco_dir.mkdir(parents=True, exist_ok=True)
    zip_path = save_dir / "coco" / "train2017.zip"

    if not zip_path.exists():
        url = "http://images.cocodataset.org/zips/train2017.zip"
        print(f"Downloading COCO 2017 train images (~18GB)...")
        print(f"  URL: {url}")
        print(f"  Saving to: {zip_path}")
        ret = subprocess.run(
            ["wget", "-c", "--progress=bar:force", "-O", str(zip_path), url],
            check=False,
        )
        if ret.returncode != 0:
            print("ERROR: COCO download failed. You can download manually:")
            print(f"  wget -c {url} -O {zip_path}")
            return

    print(f"Extracting COCO images to {save_dir / 'coco'}...")
    subprocess.run(
        ["unzip", "-q", "-o", str(zip_path), "-d", str(save_dir / "coco")],
        check=True,
    )
    print("COCO extraction complete.")


def _download_sam_images(save_dir: Path):
    """Download SAM images from Lin-Chen/ShareGPT4V-SAM-HF.

    This HF repo contains tar archives (sam_images_0.tar, sam_images_1.tar, ...).
    After extraction, images are at sam/images/sa_*.jpg.
    """
    from huggingface_hub import list_repo_files, hf_hub_download

    sam_dir = save_dir / "sam" / "images"
    if sam_dir.exists() and any(sam_dir.glob("*.jpg")):
        n_imgs = sum(1 for _ in sam_dir.glob("*.jpg"))
        print(f"SAM images already exist ({n_imgs} images): {sam_dir}")
        return

    sam_dir.mkdir(parents=True, exist_ok=True)

    print("Listing SAM image archives from Lin-Chen/ShareGPT4V-SAM-HF...")
    try:
        all_files = list_repo_files("Lin-Chen/ShareGPT4V-SAM-HF", repo_type="dataset")
    except Exception as e:
        print(f"ERROR listing SAM repo: {e}")
        print("  You may need to accept the license at:")
        print("  https://huggingface.co/datasets/Lin-Chen/ShareGPT4V-SAM-HF")
        return

    tar_files = sorted(f for f in all_files if f.endswith(".tar"))
    if not tar_files:
        # Might be zip files or other format
        tar_files = sorted(f for f in all_files if f.endswith((".tar", ".tar.gz", ".zip")))

    print(f"Found {len(tar_files)} SAM archives to download.")

    for i, fname in enumerate(tar_files):
        print(f"  [{i + 1}/{len(tar_files)}] Downloading {fname}...")
        try:
            local_path = hf_hub_download(
                repo_id="Lin-Chen/ShareGPT4V-SAM-HF",
                filename=fname,
                repo_type="dataset",
                local_dir=str(save_dir / "_sam_downloads"),
            )
        except Exception as e:
            print(f"    WARNING: Failed to download {fname}: {e}")
            continue

        # Extract to sam/images/
        print(f"    Extracting {fname}...")
        if fname.endswith(".tar") or fname.endswith(".tar.gz"):
            subprocess.run(
                ["tar", "-xf", local_path, "-C", str(save_dir / "sam" / "images")],
                check=False,
            )
        elif fname.endswith(".zip"):
            subprocess.run(
                ["unzip", "-q", "-o", local_path, "-d", str(save_dir / "sam" / "images")],
                check=False,
            )

    n_imgs = sum(1 for _ in sam_dir.glob("*.jpg"))
    print(f"SAM extraction complete: {n_imgs} images in {sam_dir}")


def _download_web_images(save_dir: Path):
    """Download web images (share_textvqa, web-celebrity, web-landmark, wikiart).

    These are available as zip archives from Lin-Chen/ShareGPT4V-Data.
    """
    from huggingface_hub import list_repo_files, hf_hub_download

    web_prefixes = ["share_textvqa", "web-celebrity", "web-landmark", "wikiart"]

    # Check which ones already exist
    needed = []
    for prefix in web_prefixes:
        img_dir = save_dir / prefix / "images"
        if img_dir.exists() and any(img_dir.iterdir()):
            n = sum(1 for _ in img_dir.glob("*"))
            print(f"{prefix} already exists ({n} files): {img_dir}")
        else:
            needed.append(prefix)

    if not needed:
        print("All web image sources already present.")
        return

    print(f"Downloading web images for: {', '.join(needed)}")
    print("  Source: Lin-Chen/ShareGPT4V-Data")

    try:
        all_files = list_repo_files("Lin-Chen/ShareGPT4V-Data", repo_type="dataset")
    except Exception as e:
        print(f"ERROR listing ShareGPT4V-Data repo: {e}")
        print("  You may need to accept the license at:")
        print("  https://huggingface.co/datasets/Lin-Chen/ShareGPT4V-Data")
        return

    # Download zip archives that match needed prefixes
    for prefix in needed:
        # Look for matching archive files (e.g., share_textvqa.zip, web-celebrity.zip)
        matching = [f for f in all_files if prefix in f and f.endswith((".zip", ".tar", ".tar.gz"))]
        if not matching:
            # Try downloading the entire prefix directory
            matching = [f for f in all_files if f.startswith(prefix + "/")]
            if matching:
                print(f"  Downloading {prefix}/ directory ({len(matching)} files)...")
                for fname in matching:
                    try:
                        hf_hub_download(
                            repo_id="Lin-Chen/ShareGPT4V-Data",
                            filename=fname,
                            repo_type="dataset",
                            local_dir=str(save_dir),
                        )
                    except Exception as e:
                        print(f"    WARNING: Failed to download {fname}: {e}")
            else:
                print(f"  WARNING: No files found for {prefix} in ShareGPT4V-Data.")
            continue

        for fname in matching:
            print(f"  Downloading {fname}...")
            try:
                local_path = hf_hub_download(
                    repo_id="Lin-Chen/ShareGPT4V-Data",
                    filename=fname,
                    repo_type="dataset",
                    local_dir=str(save_dir / "_web_downloads"),
                )
            except Exception as e:
                print(f"    WARNING: Failed to download {fname}: {e}")
                continue

            # Extract
            target_dir = save_dir / prefix
            target_dir.mkdir(parents=True, exist_ok=True)
            print(f"    Extracting to {target_dir}...")
            if fname.endswith(".zip"):
                subprocess.run(
                    ["unzip", "-q", "-o", local_path, "-d", str(target_dir)],
                    check=False,
                )
            elif fname.endswith(".tar") or fname.endswith(".tar.gz"):
                subprocess.run(
                    ["tar", "-xf", local_path, "-C", str(target_dir)],
                    check=False,
                )

    print("Web image download complete.")


# ──────────────────────────────────────────────────────────────
# ShareGPT4V JSONL conversion
# ──────────────────────────────────────────────────────────────

def convert_sharegpt4v_to_jsonl(data_dir: str, output_dir: str):
    """Convert ShareGPT4V-PT JSON to our unified JSONL format.

    Reads share-captioner_coco_lcs_sam_1246k_1107.json and converts each entry
    from the conversation format to our simple {"image": ..., "caption": ...} format.

    Image paths are kept as-is (e.g., "sam/images/sa_545504.jpg") since they are
    relative to data/sharegpt4v/ as image_root.

    Entries with missing conversations or empty captions are skipped.
    """
    save_dir = Path(data_dir) / "sharegpt4v"
    json_path = save_dir / SHAREGPT4V_PT_JSON

    if not json_path.exists():
        print(f"ShareGPT4V metadata not found at {json_path}")
        print("  Run phase 3 first to download.")
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading ShareGPT4V-PT metadata from {json_path}...")
    print("  (This is a ~1.5GB JSON file, may take a moment)")
    with open(json_path) as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} entries.")

    # Convert to our format
    records = []
    skipped_no_image = 0
    skipped_no_caption = 0
    source_counts = {}

    for item in data:
        image_path = item.get("image", "")
        if not image_path:
            skipped_no_image += 1
            continue

        # Extract caption from conversation format
        caption = ""
        for conv in item.get("conversations", []):
            if conv.get("from") == "gpt":
                caption = conv["value"]
                break

        if not caption:
            skipped_no_caption += 1
            continue

        # Track source distribution
        source = image_path.split("/")[0] if "/" in image_path else "unknown"
        source_counts[source] = source_counts.get(source, 0) + 1

        records.append({
            "image": image_path,
            "caption": caption,
        })

    # Report
    print(f"\nConverted {len(records)} valid entries:")
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt:,}")
    if skipped_no_image:
        print(f"  Skipped (no image): {skipped_no_image}")
    if skipped_no_caption:
        print(f"  Skipped (no caption): {skipped_no_caption}")

    # Write JSONL (no split -- this is used for merging into combined dataset)
    out_path = output_path / "sharegpt4v_train.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\nShareGPT4V JSONL: {len(records)} samples -> {out_path}")
    return records


# ──────────────────────────────────────────────────────────────
# Combined dataset creation
# ──────────────────────────────────────────────────────────────

def create_combined_dataset(data_dir: str, output_dir: str):
    """Create combined LLaVA + ShareGPT4V dataset with deduplication.

    Merges:
      - data/processed/train.jsonl (LLaVA-Pretrain, ~552K)
      - data/processed/sharegpt4v_train.jsonl (ShareGPT4V-PT, ~1.2M)

    Image path normalization for combined dataset:
      - LLaVA paths: "00293/002933218.jpg" -> "llava_pretrain/00293/002933218.jpg"
        (relative to data/ as image_root)
      - ShareGPT4V paths: "sam/images/sa_123.jpg" -> "sharegpt4v/sam/images/sa_123.jpg"
        (relative to data/ as image_root)

    This matches generate_phase2_configs.py which uses image_root="data".

    Deduplication: by normalized image path (same image from different sources
    may have the same content -- we keep only the first occurrence).

    Output:
      - data/processed/train_combined.jsonl (shuffled, deduplicated)
      - data/processed/val_combined.jsonl (held from ShareGPT4V only, non-overlapping)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Load LLaVA train data ──
    llava_path = output_path / "train.jsonl"
    if not llava_path.exists():
        print(f"LLaVA JSONL not found: {llava_path}")
        print("  Run phase 2 first.")
        return

    print("Loading LLaVA train data...")
    llava_records = []
    with open(llava_path) as f:
        for line in f:
            r = json.loads(line.strip())
            # Prefix image path for combined dataset
            r["image"] = "llava_pretrain/" + r["image"]
            llava_records.append(r)
    print(f"  LLaVA: {len(llava_records):,} records")

    # ── Load ShareGPT4V train data ──
    sgpt_path = output_path / "sharegpt4v_train.jsonl"
    if not sgpt_path.exists():
        print(f"ShareGPT4V JSONL not found: {sgpt_path}")
        print("  Run phase 3 first (download + convert).")
        return

    print("Loading ShareGPT4V train data...")
    sgpt_records = []
    with open(sgpt_path) as f:
        for line in f:
            r = json.loads(line.strip())
            # Prefix image path for combined dataset
            r["image"] = "sharegpt4v/" + r["image"]
            sgpt_records.append(r)
    print(f"  ShareGPT4V: {len(sgpt_records):,} records")

    # ── Deduplicate by image path ──
    # Some ShareGPT4V entries reference the same COCO/LLaVA images.
    # ShareGPT4V "llava/llava_pretrain/images/00293/002933218.jpg" and
    # LLaVA "llava_pretrain/00293/002933218.jpg" point to the same file.
    # We normalize and deduplicate.
    print("Deduplicating by image path...")

    def _normalize_image_path(path: str) -> str:
        """Normalize image path for dedup comparison."""
        # ShareGPT4V llava refs: "sharegpt4v/llava/llava_pretrain/images/X" -> "llava_pretrain/X"
        # LLaVA refs: "llava_pretrain/X" -> "llava_pretrain/X"
        p = path
        if p.startswith("sharegpt4v/llava/llava_pretrain/images/"):
            p = "llava_pretrain/" + p[len("sharegpt4v/llava/llava_pretrain/images/"):]
        return p

    seen_images = set()
    combined = []

    # Add LLaVA first (higher priority -- already validated)
    for r in llava_records:
        norm = _normalize_image_path(r["image"])
        if norm not in seen_images:
            seen_images.add(norm)
            combined.append(r)
    n_llava = len(combined)

    # Add ShareGPT4V (skip duplicates)
    n_dupes = 0
    for r in sgpt_records:
        norm = _normalize_image_path(r["image"])
        if norm not in seen_images:
            seen_images.add(norm)
            combined.append(r)
        else:
            n_dupes += 1
    n_sgpt = len(combined) - n_llava

    print(f"  LLaVA unique: {n_llava:,}")
    print(f"  ShareGPT4V unique: {n_sgpt:,}")
    print(f"  Duplicates removed: {n_dupes:,}")
    print(f"  Combined total: {len(combined):,}")

    # ── Shuffle with fixed seed ──
    random.seed(42)
    random.shuffle(combined)

    # ── Train / val split ──
    # Use 1% for val (from the combined set), but cap at 10K
    val_size = min(int(len(combined) * 0.01), 10_000)
    val_records = combined[:val_size]
    train_records = combined[val_size:]

    # ── Write outputs ──
    train_path = output_path / "train_combined.jsonl"
    with open(train_path, "w") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    val_path = output_path / "val_combined.jsonl"
    with open(val_path, "w") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")

    print(f"\nCombined train: {len(train_records):,} samples -> {train_path}")
    print(f"Combined val:   {len(val_records):,} samples -> {val_path}")

    # ── Create subsets for D control ──
    for subset_size in [100_000, 500_000, 1_000_000, 1_500_000]:
        if subset_size > len(train_records):
            continue
        subset = train_records[:subset_size]
        if subset_size >= 1_000_000:
            tag = f"{subset_size // 1_000_000}m"
        else:
            tag = f"{subset_size // 1_000}k"
        subset_path = output_path / f"train_combined_{tag}.jsonl"
        with open(subset_path, "w") as f:
            for r in subset:
                f.write(json.dumps(r) + "\n")
        print(f"  Subset D={subset_size:,}: {subset_path}")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"Combined dataset creation complete!")
    print(f"  Total unique pairs: {len(combined):,}")
    print(f"  Train: {len(train_records):,}")
    print(f"  Val:   {len(val_records):,}")
    print(f"  Image root for training: data/  (the parent directory)")
    print(f"{'=' * 60}")


# ──────────────────────────────────────────────────────────────
# Verify image availability
# ──────────────────────────────────────────────────────────────

def verify_images(data_dir: str, jsonl_path: str, image_root: str, max_check: int = 10000):
    """Check how many images in a JSONL file actually exist on disk.

    Useful to verify download completeness before training.
    """
    jsonl = Path(jsonl_path)
    if not jsonl.exists():
        print(f"JSONL not found: {jsonl}")
        return

    print(f"Verifying images for {jsonl}...")
    print(f"  Image root: {image_root}")

    total = 0
    found = 0
    missing_sources = {}

    with open(jsonl) as f:
        for line in f:
            total += 1
            if total > max_check:
                break
            r = json.loads(line.strip())
            img = os.path.join(image_root, r["image"])
            if os.path.exists(img):
                found += 1
            else:
                source = r["image"].split("/")[0]
                missing_sources[source] = missing_sources.get(source, 0) + 1

    checked = min(total, max_check)
    pct = found / checked * 100 if checked > 0 else 0
    print(f"  Checked: {checked:,} / {total:,}")
    print(f"  Found:   {found:,} ({pct:.1f}%)")
    if missing_sources:
        print(f"  Missing by source:")
        for src, cnt in sorted(missing_sources.items(), key=lambda x: -x[1]):
            print(f"    {src}: {cnt:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for VLM adapter scaling law experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
  1  Download LLaVA-Pretrain-558K images + metadata
  2  Convert LLaVA to JSONL (train/val splits + subsets)
  3  Download ShareGPT4V-PT metadata + images, convert to JSONL
  4  Create combined dataset (LLaVA + ShareGPT4V, deduplicated)

Example workflow:
  python scripts/download_data.py --phase 1    # ~13GB download
  python scripts/download_data.py --phase 2    # fast, local conversion
  python scripts/download_data.py --phase 3    # ~30GB download (SAM, COCO, web)
  python scripts/download_data.py --phase 4    # fast, merge + dedup

Verify images:
  python scripts/download_data.py --verify data/processed/train_combined.jsonl --image-root data
""",
    )
    parser.add_argument("--data_dir", type=str,
                        default=os.environ.get("VLM_DATA_DIR", "data"))
    parser.add_argument("--phase", type=int, default=None,
                        help="1: LLaVA download, 2: LLaVA JSONL, 3: ShareGPT4V, 4: Combined")
    parser.add_argument("--skip-images", action="store_true",
                        help="Phase 3: download metadata only, skip large image downloads")
    parser.add_argument("--verify", type=str, default=None,
                        help="Verify image availability for a JSONL file")
    parser.add_argument("--image-root", type=str, default=None,
                        help="Image root directory for --verify")
    parser.add_argument("--max-check", type=int, default=10000,
                        help="Max images to check in --verify mode")
    args = parser.parse_args()

    # Verify mode
    if args.verify:
        image_root = args.image_root or args.data_dir
        verify_images(args.data_dir, args.verify, image_root, args.max_check)
        return

    if args.phase is None:
        parser.print_help()
        return

    if args.phase == 1:
        print("=" * 60)
        print("Phase 1: Downloading LLaVA-Pretrain-558K")
        print("=" * 60)
        download_llava_pretrain(args.data_dir)

    elif args.phase == 2:
        print("=" * 60)
        print("Phase 2: Converting LLaVA to unified JSONL format")
        print("=" * 60)
        convert_llava_to_jsonl(args.data_dir, os.path.join(args.data_dir, "processed"))

    elif args.phase == 3:
        print("=" * 60)
        print("Phase 3: Downloading ShareGPT4V-PT + converting to JSONL")
        print("=" * 60)
        download_sharegpt4v(args.data_dir, skip_images=args.skip_images)
        # Convert metadata to JSONL right after download
        print()
        print("-" * 60)
        print("Converting ShareGPT4V-PT to JSONL...")
        print("-" * 60)
        convert_sharegpt4v_to_jsonl(args.data_dir, os.path.join(args.data_dir, "processed"))

    elif args.phase == 4:
        print("=" * 60)
        print("Phase 4: Creating combined dataset (LLaVA + ShareGPT4V)")
        print("=" * 60)
        create_combined_dataset(args.data_dir, os.path.join(args.data_dir, "processed"))

    else:
        print("Unknown phase. Use --phase 1, 2, 3, or 4")
        print("Run with --help for details.")


if __name__ == "__main__":
    main()

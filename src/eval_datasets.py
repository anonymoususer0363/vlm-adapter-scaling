"""
Evaluation benchmark datasets for G10 downstream correlation.

Supports:
- VQAv2 (val): Visual Question Answering
- TextVQA (val): OCR-dependent VQA
- COCO Caption (Karpathy test): Image captioning
"""

import json
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Prompt templates
# ============================================================

VQA_PROMPT = "Question: {question}\nShort answer:"
CAPTION_PROMPT = "Describe this image briefly."


# ============================================================
# VQAv2 Dataset
# ============================================================

class VQAv2Dataset(Dataset):
    """
    VQAv2 validation set.

    Expected files:
        questions_file: v2_OpenEnded_mscoco_val2014_questions.json
        annotations_file: v2_mscoco_val2014_annotations.json
        image_root: val2014/ images directory
    """

    def __init__(
        self,
        questions_file: str,
        annotations_file: str,
        image_root: str,
        image_processor: Any,
        tokenizer: Any,
        max_samples: int | None = None,
    ):
        self.image_root = image_root
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        with open(questions_file) as f:
            q_data = json.load(f)
        with open(annotations_file) as f:
            a_data = json.load(f)

        # Build question_id → annotation map
        ann_map = {a["question_id"]: a for a in a_data["annotations"]}

        self.samples = []
        for q in q_data["questions"]:
            qid = q["question_id"]
            ann = ann_map.get(qid)
            if ann is None:
                continue
            # Collect all answer strings for VQA accuracy
            answers = [a["answer"] for a in ann["answers"]]
            self.samples.append({
                "question_id": qid,
                "image_id": q["image_id"],
                "question": q["question"],
                "answers": answers,
            })

        if max_samples and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image_file = f"COCO_val2014_{sample['image_id']:012d}.jpg"
        image_path = os.path.join(self.image_root, image_file)
        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
        except Exception:
            pixel_values = torch.zeros(3, 384, 384)

        # Build prompt
        prompt = VQA_PROMPT.format(question=sample["question"])
        tokens = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=128)

        return {
            "pixel_values": pixel_values,
            "prompt_ids": tokens["input_ids"][0],
            "prompt_mask": tokens["attention_mask"][0],
            "question_id": sample["question_id"],
            "answers": sample["answers"],
        }


# ============================================================
# TextVQA Dataset
# ============================================================

class TextVQADataset(Dataset):
    """
    TextVQA validation set.

    Expected files:
        data_file: TextVQA_0.5.1_val.json
        image_root: textvqa/train_images/ directory
    """

    def __init__(
        self,
        data_file: str,
        image_root: str,
        image_processor: Any,
        tokenizer: Any,
        max_samples: int | None = None,
    ):
        self.image_root = image_root
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        with open(data_file) as f:
            data = json.load(f)

        self.samples = []
        for item in data["data"]:
            self.samples.append({
                "question_id": item["question_id"],
                "question": item["question"],
                "image_id": item["image_id"],
                "answers": item.get("answers", []),
            })

        if max_samples and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image_path = os.path.join(self.image_root, f"{sample['image_id']}.jpg")
        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
        except Exception:
            pixel_values = torch.zeros(3, 384, 384)

        prompt = VQA_PROMPT.format(question=sample["question"])
        tokens = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=128)

        return {
            "pixel_values": pixel_values,
            "prompt_ids": tokens["input_ids"][0],
            "prompt_mask": tokens["attention_mask"][0],
            "question_id": sample["question_id"],
            "answers": sample["answers"],
        }


# ============================================================
# COCO Caption Dataset (Karpathy test split)
# ============================================================

class COCOCaptionDataset(Dataset):
    """
    COCO Karpathy test split for captioning.

    Expected files:
        karpathy_file: dataset_coco.json (Karpathy split annotations)
        image_root: directory containing train2014/ and val2014/ (COCO images)
    """

    def __init__(
        self,
        karpathy_file: str,
        image_root: str,
        image_processor: Any,
        tokenizer: Any,
        max_samples: int | None = None,
    ):
        self.image_root = image_root
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        with open(karpathy_file) as f:
            data = json.load(f)

        self.samples = []
        for img in data["images"]:
            if img["split"] != "test":
                continue
            refs = [s["raw"] for s in img["sentences"]]
            self.samples.append({
                "image_id": img["cocoid"],
                "filepath": img["filepath"],  # e.g., "val2014"
                "filename": img["filename"],  # e.g., "COCO_val2014_000000391895.jpg"
                "references": refs,
            })

        if max_samples and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image_path = os.path.join(self.image_root, sample["filepath"], sample["filename"])
        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
        except Exception:
            pixel_values = torch.zeros(3, 384, 384)

        prompt = CAPTION_PROMPT
        tokens = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=64)

        return {
            "pixel_values": pixel_values,
            "prompt_ids": tokens["input_ids"][0],
            "prompt_mask": tokens["attention_mask"][0],
            "image_id": sample["image_id"],
            "references": sample["references"],
        }


# ============================================================
# Collate functions
# ============================================================

def eval_collate_fn(batch: list[dict], pad_token_id: int = 0) -> dict:
    """Collate for eval: pad prompt_ids, stack pixel_values, keep metadata."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])

    # Pad prompts
    max_len = max(b["prompt_ids"].shape[0] for b in batch)
    prompt_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    prompt_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        seq_len = b["prompt_ids"].shape[0]
        prompt_ids[i, :seq_len] = b["prompt_ids"]
        prompt_mask[i, :seq_len] = b["prompt_mask"]

    # Collect metadata (varies by dataset)
    meta = {}
    if "question_id" in batch[0]:
        meta["question_ids"] = [b["question_id"] for b in batch]
    if "answers" in batch[0]:
        meta["answers"] = [b["answers"] for b in batch]
    if "image_id" in batch[0]:
        meta["image_ids"] = [b["image_id"] for b in batch]
    if "references" in batch[0]:
        meta["references"] = [b["references"] for b in batch]

    return {
        "pixel_values": pixel_values,
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "meta": meta,
    }


def build_eval_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 4,
    pad_token_id: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda b: eval_collate_fn(b, pad_token_id=pad_token_id),
    )

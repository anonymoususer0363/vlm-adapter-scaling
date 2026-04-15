"""
Data loading for VLM adapter scaling law experiments.

Supports:
- ShareGPT4V format (json with image paths + conversations)
- COYO-like format (parquet/csv with image URL + caption)
- Generic image-caption pairs

For scaling experiments, we need:
- Controllable dataset size D (subset sampling)
- Consistent tokenization across runs
- Efficient image loading with transforms
"""

import json
import os
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class ImageCaptionDataset(Dataset):
    """
    Generic image-caption dataset.

    Expected data format (JSONL):
    {"image": "path/to/image.jpg", "caption": "A photo of ..."}

    Or ShareGPT4V format (JSON list):
    [{"image": "path.jpg", "conversations": [{"from": "human", ...}, {"from": "gpt", "value": "caption"}]}]
    """

    def __init__(
        self,
        data_path: str,
        image_root: str,
        image_processor: Any,
        tokenizer: Any,
        max_text_len: int = 512,
        num_samples: int | None = None,
        seed: int = 42,
    ):
        self.image_root = image_root
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

        # Load data
        self.samples = self._load_data(data_path)

        # D control: subsample if num_samples < dataset size
        # For D > dataset size, use num_epochs in trainer config instead
        if num_samples is not None and num_samples < len(self.samples):
            rng = random.Random(seed)
            self.samples = rng.sample(self.samples, num_samples)

    def _load_data(self, data_path: str) -> list[dict]:
        path = Path(data_path)

        if path.suffix == ".jsonl":
            samples = []
            with open(path) as f:
                for line in f:
                    samples.append(json.loads(line.strip()))
            return samples

        elif path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            # Handle ShareGPT4V format
            samples = []
            for item in data:
                if "conversations" in item:
                    caption = ""
                    for turn in item["conversations"]:
                        if turn.get("from") == "gpt":
                            caption = turn["value"]
                            break
                    samples.append({
                        "image": item.get("image", ""),
                        "caption": caption,
                    })
                else:
                    samples.append(item)
            return samples

        else:
            raise ValueError(f"Unsupported data format: {path.suffix}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Load and process image
        image_path = os.path.join(self.image_root, sample["image"])
        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
        except Exception:
            # Fallback: return a black image on error
            pixel_values = torch.zeros(3, 384, 384)

        # Tokenize caption
        caption = sample.get("caption", "")
        tokens = self.tokenizer(
            caption,
            max_length=self.max_text_len,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"][0]
        attention_mask = tokens["attention_mask"][0]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def collate_fn(batch: list[dict], pad_token_id: int = 0) -> dict:
    """Collate with dynamic padding for text tokens."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])

    # Pad text to max length in batch
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = b["input_ids"].shape[0]
        input_ids[i, :seq_len] = b["input_ids"]
        attention_mask[i, :seq_len] = b["attention_mask"]

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def build_dataloader(
    data_path: str,
    image_root: str,
    image_processor: Any,
    tokenizer: Any,
    batch_size: int = 32,
    num_samples: int | None = None,
    num_workers: int = 8,
    max_text_len: int = 512,
    seed: int = 42,
) -> DataLoader:
    dataset = ImageCaptionDataset(
        data_path=data_path,
        image_root=image_root,
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_text_len=max_text_len,
        num_samples=num_samples,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id or 0),
        drop_last=True,
    )

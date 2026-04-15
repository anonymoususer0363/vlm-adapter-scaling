"""
Evaluation metrics for G10 downstream benchmarks.

- VQA accuracy (standard VQAv2 protocol)
- CIDEr score (via pycocoevalcap)
"""

import re
import string
from collections import Counter


# ============================================================
# VQA Accuracy (standard protocol from VQAv2)
# ============================================================

def _normalize_answer(answer: str) -> str:
    """Standard VQA answer normalization."""
    answer = answer.lower().strip()
    # Remove punctuation
    answer = answer.translate(str.maketrans("", "", string.punctuation))
    # Remove articles
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    # Collapse whitespace
    answer = " ".join(answer.split())
    return answer


def vqa_accuracy(prediction: str, ground_truths: list[str]) -> float:
    """
    Standard VQA accuracy for a single sample.

    acc = min(1, count(pred in GT) / 3)
    where count = number of GT annotators who gave this exact answer.
    """
    pred = _normalize_answer(prediction)
    gt_counts = Counter([_normalize_answer(gt) for gt in ground_truths])
    return min(1.0, gt_counts.get(pred, 0) / 3.0)


def compute_vqa_metrics(predictions: list[str], answers_list: list[list[str]]) -> dict:
    """Compute VQA accuracy over a full dataset."""
    assert len(predictions) == len(answers_list)
    accs = [vqa_accuracy(pred, gts) for pred, gts in zip(predictions, answers_list)]
    return {
        "vqa_accuracy": sum(accs) / len(accs) * 100,
        "num_samples": len(accs),
    }


# ============================================================
# CIDEr (via pycocoevalcap if available, else lightweight fallback)
# ============================================================

def compute_caption_metrics(
    predictions: list[dict],
    references: list[dict],
) -> dict:
    """
    Compute captioning metrics (CIDEr, BLEU-4).

    Args:
        predictions: [{"image_id": int, "caption": str}, ...]
        references: [{"image_id": int, "references": [str, ...]}, ...]

    Returns:
        dict with CIDEr, BLEU4 scores
    """
    try:
        from pycocoevalcap.eval import COCOEvalCap
        from pycocotools.coco import COCO
        return _compute_with_pycocoevalcap(predictions, references)
    except ImportError:
        return _compute_cider_fallback(predictions, references)


def _compute_with_pycocoevalcap(predictions, references):
    """Use official pycocoevalcap for accurate metrics."""
    import tempfile, json
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    # Build COCO-format reference annotations
    ref_ann = {
        "images": [],
        "annotations": [],
    }
    ann_id = 0
    for ref in references:
        img_id = ref["image_id"]
        ref_ann["images"].append({"id": img_id})
        for caption in ref["references"]:
            ref_ann["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "caption": caption,
            })
            ann_id += 1

    # Write temp files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(ref_ann, f)
        ref_path = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(predictions, f)
        pred_path = f.name

    coco = COCO(ref_path)
    coco_res = coco.loadRes(pred_path)
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.evaluate()

    import os
    os.unlink(ref_path)
    os.unlink(pred_path)

    return {
        "CIDEr": coco_eval.eval.get("CIDEr", 0.0) * 100,
        "BLEU4": coco_eval.eval.get("Bleu_4", 0.0) * 100,
        "num_samples": len(predictions),
    }


def _compute_cider_fallback(predictions, references):
    """Lightweight CIDEr approximation without pycocoevalcap."""
    # Simple n-gram overlap metric as fallback
    from collections import Counter
    import math

    def _get_ngrams(text, n):
        words = text.lower().split()
        return Counter([tuple(words[i:i+n]) for i in range(len(words)-n+1)])

    scores = []
    ref_map = {r["image_id"]: r["references"] for r in references}

    for pred in predictions:
        img_id = pred["image_id"]
        pred_text = pred["caption"]
        ref_texts = ref_map.get(img_id, [])
        if not ref_texts:
            scores.append(0.0)
            continue

        # Simple 4-gram overlap (CIDEr-like)
        pred_ngrams = _get_ngrams(pred_text, 4)
        if not pred_ngrams:
            scores.append(0.0)
            continue

        ref_scores = []
        for ref in ref_texts:
            ref_ngrams = _get_ngrams(ref, 4)
            if not ref_ngrams:
                ref_scores.append(0.0)
                continue
            overlap = sum((pred_ngrams & ref_ngrams).values())
            precision = overlap / max(sum(pred_ngrams.values()), 1)
            recall = overlap / max(sum(ref_ngrams.values()), 1)
            if precision + recall > 0:
                ref_scores.append(2 * precision * recall / (precision + recall))
            else:
                ref_scores.append(0.0)
        scores.append(sum(ref_scores) / len(ref_scores) if ref_scores else 0.0)

    return {
        "CIDEr_approx": sum(scores) / len(scores) * 100 if scores else 0.0,
        "num_samples": len(predictions),
        "_note": "approximate (install pycocoevalcap for official metrics)",
    }

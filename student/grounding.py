"""Offline helpers for the road-damage grounding lab (CP1 and CP2)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.labcommon import CLASSES, read_jsonl


def build_prompt(coord_scale: str = "norm1000") -> str:
    if coord_scale == "norm1000":
        scale_text = "integers on the 0–1000 scale, relative to image width and height"
    elif coord_scale == "pixel":
        scale_text = "integer pixel coordinates [x1, y1, x2, y2] in the original image"
    else:
        raise ValueError("coord_scale must be 'norm1000' or 'pixel'")
    return (
        "Locate every pothole, crack, and manhole visible in the image. "
        "Return only a JSON list. Each item must have a label from pothole, crack, "
        f"manhole and bbox_2d [x1, y1, x2, y2]. Coordinates are {scale_text}. "
        "Use [] if none are visible."
    )


PROMPT = build_prompt()


def load_split(splits_dir: str | Path, split: str) -> list[dict[str, Any]]:
    """Read one supplied split, validating the minimum record shape."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unknown split: {split!r}")
    rows = read_jsonl(Path(splits_dir) / f"{split}.jsonl")
    for row in rows:
        for key in ("image_id", "image_path", "width", "height", "session", "objects"):
            if key not in row:
                raise ValueError(f"{split} record missing {key!r}: {row!r}")
    return rows


def validate_session_disjoint(splits: Mapping[str, Iterable[Mapping[str, Any]]]) -> bool:
    """Raise with offending sessions if a recording session crosses splits."""
    owners: dict[str, str] = {}
    for split, entries in splits.items():
        for entry in entries:
            session = entry.get("session")
            if not session:
                raise ValueError(f"record in {split!r} has no session")
            previous = owners.setdefault(str(session), split)
            if previous != split:
                raise ValueError(f"session {session!r} occurs in both {previous!r} and {split!r}")
    return True


def prepare_training_rows(rows: Sequence[Mapping[str, Any]], coord_scale: str = "norm1000") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Exclude true zero-area source annotations from training targets, retaining every image.

    The supplied split remains untouched for official evaluation. A detailed audit is
    returned so the CLI can print and persist exactly which annotations were excluded.
    """
    cleaned: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    total_objects = 0
    kept_objects = 0
    empty_images = 0
    for row in rows:
        row_copy = dict(row)
        kept = []
        bad = []
        for index, obj in enumerate(row.get("objects", [])):
            total_objects += 1
            if not isinstance(obj, Mapping) or obj.get("label") not in CLASSES:
                raise ValueError(f"invalid label for {row.get('image_id')!r} object {index}: {obj!r}")
            yolo = obj.get("yolo", [])
            if len(yolo) != 4:
                raise ValueError(f"invalid YOLO box for {row.get('image_id')!r} object {index}: expected four values")
            try:
                values = list(map(float, yolo))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid YOLO box for {row.get('image_id')!r} object {index}: {yolo!r}") from exc
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
                raise ValueError(f"invalid YOLO values for {row.get('image_id')!r} object {index}: {yolo!r}")
            if values[2] <= 0 or values[3] <= 0:
                bad.append({"index": index, "label": obj.get("label"), "yolo": list(yolo)})
                continue
            kept.append(obj)
            kept_objects += 1
        row_copy["objects"] = kept
        # Exercise the exact target conversion during preflight so invalid classes,
        # scales, dimensions, or coordinates fail before any model is allocated.
        objects_to_answer(kept, coord_scale, row.get("width"), row.get("height"))
        cleaned.append(row_copy)
        if bad:
            excluded.append({"image_id": row.get("image_id"), "annotations": bad})
            if not kept:
                empty_images += 1
    excluded_count = sum(len(item["annotations"]) for item in excluded)
    return cleaned, {
        "source_images": len(rows),
        "source_annotations": total_objects,
        "retained_annotations": kept_objects,
        "zero_area_annotations_excluded": excluded_count,
        "images_with_zero_area_annotations": len(excluded),
        "images_empty_after_filter": empty_images,
        "excluded": excluded,
    }


def yolo_to_box(label: str, yolo: Sequence[float], coord_scale: str = "norm1000",
                image_width: int | None = None, image_height: int | None = None) -> dict[str, Any]:
    """Convert normalized YOLO center-width-height directly to requested XYXY scale."""
    if label not in CLASSES:
        raise ValueError(f"unknown label {label!r}; expected one of {CLASSES}")
    if len(yolo) != 4:
        raise ValueError("YOLO box must contain xc, yc, width, height")
    xc, yc, width, height = map(float, yolo)
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (xc, yc, width, height)):
        raise ValueError("YOLO coordinates must be finite values in [0, 1]")
    if width <= 0 or height <= 0:
        raise ValueError("box has zero area in source YOLO coordinates")
    if coord_scale == "norm1000":
        sx, sy, max_x, max_y = 1000, 1000, 1000, 1000
    elif coord_scale == "pixel":
        if not image_width or not image_height or image_width <= 0 or image_height <= 0:
            raise ValueError("pixel coordinates require positive image_width and image_height")
        sx, sy, max_x, max_y = image_width, image_height, image_width, image_height
    else:
        raise ValueError("coord_scale must be 'norm1000' or 'pixel'")
    coords = [round((xc - width / 2) * sx), round((yc - height / 2) * sy),
              round((xc + width / 2) * sx), round((yc + height / 2) * sy)]
    coords = [max(0, min(max_x if i % 2 == 0 else max_y, int(v)))
              for i, v in enumerate(coords)]
    # Preserve positive-area source boxes that collapse under integer rounding.
    # Expand to the nearest in-bounds one-unit interval; only source zero-area
    # boxes are rejected above.
    if coords[2] <= coords[0]:
        if coords[0] >= max_x:
            coords[0], coords[2] = max_x - 1, max_x
        else:
            coords[2] = coords[0] + 1
    if coords[3] <= coords[1]:
        if coords[1] >= max_y:
            coords[1], coords[3] = max_y - 1, max_y
        else:
            coords[3] = coords[1] + 1
    return {"label": label, "bbox_2d": coords}


def objects_to_answer(objects: Iterable[Mapping[str, Any]], coord_scale: str = "norm1000",
                      image_width: int | None = None, image_height: int | None = None) -> str:
    boxes = [yolo_to_box(obj["label"], obj["yolo"], coord_scale, image_width, image_height)
             for obj in objects]
    boxes.sort(key=lambda item: (item["bbox_2d"][1], item["bbox_2d"][0],
                                 item["label"], item["bbox_2d"][3], item["bbox_2d"][2]))
    return json.dumps(boxes, ensure_ascii=False, separators=(",", ":"))


def norm1000_to_pixel(box: Sequence[int | float], width: int, height: int) -> tuple[int, int, int, int]:
    """Map XYXY normalized coordinates to pixel bounds using deterministic rounding."""
    if len(box) != 4 or width <= 0 or height <= 0:
        raise ValueError("expected a four-coordinate box and positive image dimensions")
    x1, y1, x2, y2 = map(float, box)
    result = [round(x1 * width / 1000), round(y1 * height / 1000),
              round(x2 * width / 1000), round(y2 * height / 1000)]
    return (max(0, min(width, result[0])), max(0, min(height, result[1])),
            max(0, min(width, result[2])), max(0, min(height, result[3])))


def draw_ground_truth(image: Any, objects: Iterable[Mapping[str, Any]], copy: bool = True) -> Any:
    """Draw supplied YOLO ground truth boxes on a PIL image and return it."""
    from PIL import ImageDraw  # lightweight dependency, imported only when drawing

    canvas = image.copy() if copy else image
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    colors = {"pothole": "red", "crack": "yellow", "manhole": "cyan"}
    for obj in objects:
        x1, y1, bw, bh = map(float, obj["yolo"])
        left, top, right, bottom = norm1000_to_pixel(
            ((x1 - bw / 2) * 1000, (y1 - bh / 2) * 1000,
             (x1 + bw / 2) * 1000, (y1 + bh / 2) * 1000), width, height)
        color = colors.get(obj["label"], "white")
        draw.rectangle((left, top, right, bottom), outline=color, width=max(1, width // 320))
        draw.text((left, max(0, top - 12)), obj["label"], fill=color)
    return canvas


def _apply_template(processor: Any, messages: list[dict[str, Any]], *, generation: bool) -> dict[str, Any]:
    encoded = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=generation,
        return_dict=True, return_tensors="pt")
    return dict(encoded)


def _user_messages(image: Any, prompt: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]


def encode_example(processor: Any, image: Any, prompt: str, answer: str) -> dict[str, Any]:
    """Encode an image/chat pair and mask prompt tokens from the causal loss."""
    user = _user_messages(image, prompt)
    prompt_enc = _apply_template(processor, user, generation=True)
    full_messages = user + [{"role": "assistant", "content": answer}]
    full = _apply_template(processor, full_messages, generation=False)
    prompt_ids, full_ids = prompt_enc["input_ids"], full["input_ids"]
    n = prompt_ids.shape[-1]
    if full_ids.shape[-1] < n or not bool((full_ids[..., :n] == prompt_ids).all()):
        raise ValueError("prompt token sequence is not a prefix of the full conversation")
    labels = full_ids.clone()
    labels[..., :n] = -100
    full["labels"] = labels
    full["prefix_length"] = n
    return full


def vision_token_count(encoded: Mapping[str, Any]) -> int:
    """Count image tokens using Qwen's mm_token_type_ids (1 denotes image)."""
    types = encoded.get("mm_token_type_ids")
    if types is None:
        raise ValueError("processor output lacks mm_token_type_ids; cannot count image tokens")
    return int((types == 1).sum().item())


def collate_examples(batch: Sequence[Mapping[str, Any]], pad_token_id: int) -> dict[str, Any]:
    """Pad token sequences (labels with -100), concatenate variable vision payloads."""
    if not batch:
        raise ValueError("cannot collate an empty batch")
    import torch
    from torch.nn.utils.rnn import pad_sequence

    seq_keys = {"input_ids", "attention_mask", "mm_token_type_ids", "labels"}
    vision_keys = {"pixel_values", "image_grid_thw"}
    all_keys = set.union(*(set(item) for item in batch))
    for key in (seq_keys | vision_keys) & all_keys:
        if any(key not in item for item in batch):
            raise ValueError(f"batch items disagree about presence of {key!r}")
    keys = set.intersection(*(set(item) for item in batch))
    result: dict[str, Any] = {}
    for key in keys:
        if key in seq_keys:
            fill = -100 if key == "labels" else (pad_token_id if key == "input_ids" else 0)
            tensors = [item[key].reshape(-1) for item in batch]
            result[key] = pad_sequence(tensors, batch_first=True, padding_value=fill)
        elif key in vision_keys:
            result[key] = torch.cat([item[key] for item in batch], dim=0)
    if "input_ids" not in result or "labels" not in result:
        raise ValueError("each encoded item must include input_ids and labels")
    return result

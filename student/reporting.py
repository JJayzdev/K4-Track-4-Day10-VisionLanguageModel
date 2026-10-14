"""Create evidence visualizations and summaries from actual scorer outputs."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from scripts.labcommon import read_jsonl
from scripts.score import parse_output
from .grounding import norm1000_to_pixel


def render_predictions(split_entries: Sequence[Mapping[str, Any]], predictions: Mapping[str, str],
                       data_dir: str | Path, out_dir: str | Path, coord: str = "norm1000",
                       limit: int = 6, image_ids: Sequence[str] | None = None) -> list[Path]:
    """Render matched GT (green) and parsed predictions (red); skip missing/bad outputs."""
    if coord not in {"norm1000", "pixel"}:
        raise ValueError("coord must be norm1000 or pixel")
    if limit < 1:
        raise ValueError("limit must be positive")
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    selected_ids = set(image_ids) if image_ids is not None else None
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for entry in split_entries:
        if selected_ids is not None and entry["image_id"] not in selected_ids:
            continue
        raw = predictions.get(entry["image_id"])
        parsed = parse_output(raw) if raw is not None else None
        if parsed is None:
            continue
        image_path = data_dir / entry["image_path"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        for obj in entry["objects"]:
            xc, yc, bw, bh = map(float, obj["yolo"])
            box = norm1000_to_pixel(((xc - bw / 2) * 1000, (yc - bh / 2) * 1000,
                                     (xc + bw / 2) * 1000, (yc + bh / 2) * 1000), width, height)
            draw.rectangle(box, outline="#00d26a", width=2)
            draw.text((box[0], max(0, box[1] - 12)), f"GT {obj['label']}", fill="#00d26a")
        for obj in parsed:
            x1, y1, x2, y2 = obj["bbox"]
            if coord == "pixel":
                box = (max(0, min(width, round(x1))), max(0, min(height, round(y1))),
                       max(0, min(width, round(x2))), max(0, min(height, round(y2))))
            else:
                box = norm1000_to_pixel((x1, y1, x2, y2), width, height)
            draw.rectangle(box, outline="#ff4040", width=2)
            draw.text((box[0], min(height - 12, box[1] + 2)), f"Pred {obj['label']}", fill="#ff4040")
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(entry["image_id"]))
        caption = Image.new("RGB", (width, height + 18), "white")
        caption.paste(image, (0, 18))
        ImageDraw.Draw(caption).text((4, 2), f"{entry['image_id']} | GT: green | prediction: red", fill="black")
        target = out_dir / f"{safe_id}.jpg"
        caption.save(target, quality=92)
        saved.append(target)
        if len(saved) >= limit:
            break
    if len(saved) < min(limit, len(split_entries)):
        raise ValueError(f"only rendered {len(saved)} valid predictions; requested at least {min(limit, len(split_entries))}")
    return saved


def evidence_summary(paths: Sequence[str | Path]) -> str:
    """Build a compact markdown table from score.py JSON files; no values are inferred."""
    lines = ["| Run | F1 | CI95 | FP/image | pothole F1 | crack F1 | manhole F1 | trainable params | VRAM GB | minutes |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        per_class = data.get("per_class", {})
        meta = data.get("meta", {})
        ci = data.get("f1_ci95", [None, None])
        def fmt(value: Any) -> str:
            return "—" if value is None else f"{value:.3f}" if isinstance(value, (int, float)) else str(value)
        lines.append("| {name} | {f1} | {lo}–{hi} | {fp} | {p} | {c} | {m} | {trainable} | {vram} | {mins} |".format(
            name=data.get("name", Path(path).stem), f1=fmt(data.get("micro", {}).get("f1")),
            lo=fmt(ci[0] if len(ci) > 0 else None), hi=fmt(ci[1] if len(ci) > 1 else None),
            fp=fmt(data.get("fp_per_image")), p=fmt(per_class.get("pothole", {}).get("f1")),
            c=fmt(per_class.get("crack", {}).get("f1")), m=fmt(per_class.get("manhole", {}).get("f1")),
            vram=fmt(meta.get("peak_vram_gb")), mins=fmt(meta.get("train_minutes")),
            trainable=fmt(meta.get("trainable_params"))))
    return "\n".join(lines)


def load_predictions(path: str | Path) -> dict[str, str]:
    """Read scorer-compatible JSONL while rejecting duplicate IDs."""
    result = {}
    for row in read_jsonl(path):
        image_id = row["image_id"]
        if image_id in result:
            raise ValueError(f"duplicate prediction for {image_id}")
        result[image_id] = row["raw"]
    return result

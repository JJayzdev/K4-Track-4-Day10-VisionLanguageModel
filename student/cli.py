"""Command-line entry points for student training and prediction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .grounding import load_split, prepare_training_rows
from .modeling import load_qwen, predict, restore_checkpoint, train


def resolve_coord_scale(requested: str | None, run: dict, zero_shot: bool = False) -> str:
    scale = requested or ("norm1000" if zero_shot else run.get("coord_scale", "norm1000"))
    if scale not in {"norm1000", "pixel"}:
        raise ValueError(f"invalid coordinate scale in checkpoint: {scale!r}")
    return scale


def _latest_checkpoint(path: Path) -> Path:
    candidates = sorted(path.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint-* directory under {path}")
    return candidates[-1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    tp = sub.add_parser("train")
    tp.add_argument("--config", required=True)
    tp.add_argument("--train-fraction", type=float, default=1.0)
    tp.add_argument("--rank", type=int)
    tp.add_argument("--coord", choices=["norm1000", "pixel"], default="norm1000")
    tp.add_argument("--out-dir", required=True)
    tp.add_argument("--splits-dir", default="data/splits")
    ap = sub.add_parser("audit-train", help="print train label quality without loading a model")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--coord", choices=["norm1000", "pixel"], default="norm1000")
    ip = sub.add_parser("predict")
    choice = ip.add_mutually_exclusive_group(required=True)
    choice.add_argument("--checkpoint", help="checkpoint-NNNN folder or experiment directory")
    choice.add_argument("--zero-shot", action="store_true", help="predict with the original model, without an adapter")
    ip.add_argument("--run-json", help="run.json to resolve experiment model id/projector setup")
    ip.add_argument("--split", choices=["val", "test"], required=True)
    ip.add_argument("--coord", choices=["norm1000", "pixel"])
    ip.add_argument("--batch-size", type=int, default=4)
    ip.add_argument("--max-new-tokens", type=int, default=256)
    ip.add_argument("--out", required=True)
    ip.add_argument("--splits-dir", default="data/splits")
    args = parser.parse_args(argv)

    if args.command == "audit-train":
        rows = load_split(args.splits_dir, "train")
        _, data_quality = prepare_training_rows(rows, args.coord)
        print("TRAIN_DATA_PREFLIGHT_BEGIN", flush=True)
        print(json.dumps(data_quality, ensure_ascii=False, indent=2), flush=True)
        print("TRAIN_DATA_PREFLIGHT_END", flush=True)
        return

    if args.command == "train":
        if not 0 < args.train_fraction <= 1 or (args.rank is not None and args.rank <= 0):
            parser.error("train-fraction must be in (0, 1] and rank must be positive")
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        if args.rank is not None:
            config["rank"] = args.rank
            config["alpha"] = args.rank * 2
        rows = load_split(args.splits_dir, "train")
        rows, data_quality = prepare_training_rows(rows, args.coord)
        print("TRAIN_DATA_PREFLIGHT_BEGIN", flush=True)
        print(json.dumps(data_quality, ensure_ascii=False, indent=2), flush=True)
        print("TRAIN_DATA_PREFLIGHT_END", flush=True)
        model, processor = load_qwen(config.get("model_id", "Qwen/Qwen3-VL-4B-Instruct"))
        run = train(model, processor, rows, config, args.out_dir,
                    coord_scale=args.coord, train_fraction=args.train_fraction,
                    images_root=Path(args.splits_dir).parent,
                    training_data_quality=data_quality)
        print(json.dumps(run, indent=2))
        return

    checkpoint = None
    run = {}
    if args.checkpoint:
        ckpt_arg = Path(args.checkpoint)
        checkpoint = _latest_checkpoint(ckpt_arg) if not ckpt_arg.name.startswith("checkpoint-") else ckpt_arg
        run_path = Path(args.run_json) if args.run_json else checkpoint / "run.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
    model, processor = load_qwen(run.get("model_id", "Qwen/Qwen3-VL-4B-Instruct"))
    if checkpoint is not None:
        model = restore_checkpoint(model, checkpoint, bool(run.get("train_projector", False)))
    rows = load_split(args.splits_dir, args.split)
    coord = resolve_coord_scale(args.coord, run, args.zero_shot)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        def write_batch(batch: list[dict[str, str]]) -> None:
            for row in batch:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

        predictions = predict(model, processor, rows, args.splits_dir, coord,
                              batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                              on_batch=write_batch)
    print(f"wrote {len(predictions)} predictions to {out}")


if __name__ == "__main__":
    main()

"""Qwen3-VL loading, LoRA setup, training, and inference utilities."""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .grounding import (
    PROMPT, build_prompt, collate_examples, encode_example, load_split,
    objects_to_answer, prepare_training_rows,
)

LLM_TARGET = re.compile(
    r".*language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))"
)
VIT_TARGET = re.compile(
    r".*visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))"
)
PROJECTOR_PARTS = ("visual.merger", "visual.deepstack_merger_list")
QWEN3_VL_BNB_SKIP_MODULES = ("model.visual", "lm_head")


def validate_quantization_layout(model: Any, params4bit_type: Any = None) -> None:
    """Fail fast unless the complete vision tower and lm_head remain fp16."""
    import torch

    if params4bit_type is None:
        from bitsandbytes.nn import Params4bit
        params4bit_type = Params4bit
    named = list(model.named_parameters())
    visual = [(name, param) for name, param in named if "visual." in name]
    head_module = getattr(model, "lm_head", None)
    lm_head = ([(f"lm_head.{name}" if name else "lm_head", param)
                for name, param in head_module.named_parameters()]
               if head_module is not None else [])
    if not visual:
        raise RuntimeError("Qwen3-VL visual parameters were not found after loading")
    if not lm_head:
        raise RuntimeError("Qwen3-VL lm_head parameters were not found after loading")
    quantized = [name for name, param in visual + lm_head
                 if isinstance(param, params4bit_type)]
    if quantized:
        raise RuntimeError(
            "The Qwen3-VL visual tower or lm_head was quantized; refusing training. "
            f"Check llm_int8_skip_modules={list(QWEN3_VL_BNB_SKIP_MODULES)!r}. "
            f"Examples: {quantized[:5]}"
        )
    wrong_dtype = [(name, str(param.dtype)) for name, param in visual + lm_head
                   if param.dtype != torch.float16]
    if wrong_dtype:
        raise RuntimeError(
            "Expected visual tower and lm_head parameters to remain fp16; "
            f"found {wrong_dtype[:5]}"
        )


def load_qwen(model_id: str = "Qwen/Qwen3-VL-4B-Instruct", device_map: Any = None) -> tuple[Any, Any]:
    """Load the specified Qwen3-VL model with 4-bit LLM and fp16 vision/head."""
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=list(QWEN3_VL_BNB_SKIP_MODULES),
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quant, dtype=torch.float16,
        device_map={"": 0} if device_map is None else device_map,
    )
    validate_quantization_layout(model)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def matched_lora_modules(model: Any, setup: str) -> list[str]:
    if setup not in {"E1", "E2", "E3"}:
        raise ValueError("setup must be E1, E2, or E3")
    patterns = [LLM_TARGET] + ([VIT_TARGET] if setup == "E3" else [])
    return [name for name, _ in model.named_modules() if any(p.fullmatch(name) for p in patterns)]


def apply_setup(model: Any, setup: str, rank: int = 16, alpha: int | None = None,
                dropout: float = 0.05) -> tuple[Any, dict[str, int]]:
    """Freeze base weights, inject prescribed LoRA targets, and open projector for E2/E3."""
    from peft import LoraConfig, get_peft_model

    if rank <= 0 or (alpha is not None and alpha <= 0):
        raise ValueError("LoRA rank and alpha must be positive")
    matched = matched_lora_modules(model, setup)
    expected = 348 if setup == "E3" else 252
    if len(matched) != expected:
        raise ValueError(f"{setup} expected {expected} LoRA modules, found {len(matched)}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    target_regex = (LLM_TARGET.pattern if setup != "E3"
                    else f"(?:{LLM_TARGET.pattern})|(?:{VIT_TARGET.pattern})")
    wrapped = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=alpha if alpha is not None else rank * 2,
        lora_dropout=dropout, target_modules=target_regex, bias="none", task_type="CAUSAL_LM",
    ))
    for name, parameter in wrapped.named_parameters():
        if ".lora_" in name and parameter.requires_grad:
            parameter.data = parameter.data.float()
    base = wrapped.get_base_model()
    if setup in {"E2", "E3"}:
        projector_count = 0
        for name, parameter in base.named_parameters():
            if any(part in name for part in PROJECTOR_PARTS):
                parameter.data = parameter.data.float()
                parameter.requires_grad_(True)
                projector_count += parameter.numel()
        if not projector_count:
            raise ValueError("no visual merger parameters found to train")
    counts = {
        "matched_lora_modules": len(matched),
        "trainable_params": sum(p.numel() for p in wrapped.parameters() if p.requires_grad),
        "projector_params": sum(p.numel() for n, p in base.named_parameters()
                                if any(part in n for part in PROJECTOR_PARTS)
                                and p.requires_grad),
    }
    return wrapped, counts


def _projector_state(model: Any) -> dict[str, Any]:
    return {name: parameter.detach().cpu().contiguous()
            for name, parameter in model.get_base_model().named_parameters()
            if any(part in name for part in PROJECTOR_PARTS)}


def save_checkpoint(model: Any, out_dir: str | Path, step: int, run: Mapping[str, Any]) -> None:
    import torch

    out_dir = Path(out_dir)
    target = out_dir / f"checkpoint-{step:04d}"
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target / "adapter")
    if run.get("train_projector"):
        torch.save(_projector_state(model), target / "projector.pt")
    (target / "run.json").write_text(json.dumps({**run, "step": step}, indent=2), encoding="utf-8")


def restore_checkpoint(model: Any, checkpoint: str | Path, train_projector: bool = False) -> Any:
    """Load LoRA and optional trained projector weights into an already loaded base model."""
    import torch
    from peft import PeftModel

    checkpoint = Path(checkpoint)
    wrapped = PeftModel.from_pretrained(model, checkpoint / "adapter", is_trainable=False)
    projector_path = checkpoint / "projector.pt"
    if train_projector:
        if not projector_path.exists():
            raise FileNotFoundError(f"missing trained projector weights: {projector_path}")
        state = torch.load(projector_path, map_location="cpu", weights_only=True)
        load_projector_state(wrapped.get_base_model(), state)
    return wrapped


def load_projector_state(base_model: Any, state: Mapping[str, Any]) -> None:
    """Validate exact projector key/shape coverage and restore every value in fp32."""
    targets = {name: parameter for name, parameter in base_model.named_parameters()
               if any(part in name for part in PROJECTOR_PARTS)}
    if not targets:
        raise ValueError("base model has no projector parameters")
    if set(state) != set(targets):
        missing, extra = sorted(set(targets) - set(state)), sorted(set(state) - set(targets))
        raise ValueError(f"projector keys mismatch; missing={missing[:5]}, extra={extra[:5]}")
    for name, parameter in targets.items():
        if tuple(state[name].shape) != tuple(parameter.shape):
            raise ValueError(f"projector shape mismatch for {name}: {state[name].shape} vs {parameter.shape}")
        parameter.data = parameter.data.float()
        parameter.data.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))


def _schedule(optimizer: Any, total_steps: int):
    from torch.optim.lr_scheduler import LambdaLR

    warmup = max(1, round(total_steps * 0.05))
    def multiplier(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, multiplier)


def train(model: Any, processor: Any, train_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
          out_dir: str | Path, coord_scale: str = "norm1000", train_fraction: float = 1.0,
          images_root: str | Path = "data",
          training_data_quality: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run the GUIDE fp16/GradScaler loop and persist checkpoint plus truthful run metadata."""
    import random
    import torch
    from PIL import Image

    if not 0 < train_fraction <= 1:
        raise ValueError("train_fraction must be in (0, 1]")
    train_rows, detected_quality = prepare_training_rows(train_rows, coord_scale)
    data_quality = dict(training_data_quality or detected_quality)
    if not train_rows:
        raise ValueError("training split is empty")
    seed = int(config.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    setup = str(config["name"])
    rank = int(config.get("rank", config.get("lora", {}).get("r", 16)))
    alpha = int(config.get("alpha", config.get("lora", {}).get("alpha", rank * 2)))
    model, counts = apply_setup(model, setup, rank=rank, alpha=alpha,
                                dropout=float(config.get("lora", {}).get("dropout", 0.05)))
    processor.tokenizer.padding_side = "right"
    frac_n = max(1, int(len(train_rows) * train_fraction))
    rng = random.Random(seed)
    selected = list(train_rows)
    rng.shuffle(selected)
    selected = selected[:frac_n]
    batch_size = int(config.get("batch_size", 2))
    accum = int(config.get("grad_accum", 4))
    max_steps = int(config.get("max_steps", 300))
    if batch_size < 1 or accum < 1 or max_steps < 1:
        raise ValueError("batch_size, grad_accum, and max_steps must be positive")
    images_root = Path(images_root)

    def batches():
        order = list(selected)
        while True:
            rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                group = order[start:start + batch_size]
                examples = []
                for row in group:
                    with Image.open(images_root / row["image_path"]) as im:
                        image = im.convert("RGB")
                    answer = objects_to_answer(row["objects"], coord_scale,
                                               row["width"], row["height"])
                    examples.append(encode_example(processor, image, build_prompt(coord_scale), answer))
                yield collate_examples(examples, processor.tokenizer.pad_token_id)

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(config.get("lr", 2e-4)), weight_decay=0.0)
    scheduler = _schedule(optimizer, max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(batches())
    losses, started = [], time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    out_dir = Path(out_dir)
    run = {"setup": setup, "trainable_params": counts["trainable_params"],
           "model_id": config.get("model_id", "Qwen/Qwen3-VL-4B-Instruct"),
           "matched_lora_modules": counts["matched_lora_modules"],
           "projector_params": counts["projector_params"], "train_projector": setup in {"E2", "E3"},
           "rank": rank, "alpha": alpha, "train_fraction": train_fraction,
           "coord_scale": coord_scale, "lr": float(config.get("lr", 2e-4)),
           "batch_size": batch_size, "grad_accum": accum, "max_steps": max_steps,
           "seed": seed, "selected_train_images": len(selected),
           "grad_scaler_skipped_steps": 0,
           "training_data_quality": data_quality}
    successful_steps = 0
    consecutive_overflows = 0
    while successful_steps < max_steps:
        step = successful_steps + 1
        attempt_losses = []
        for _ in range(accum):
            batch = next(iterator)
            batch = {key: value.to(model.device) if hasattr(value, "to") else value
                     for key, value in batch.items() if key != "prefix_length"}
            with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                loss = model(**batch).loss / accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at optimizer step {step}")
            scaler.scale(loss).backward()
            attempt_losses.append(float(loss.detach()) * accum)
        scaler.unscale_(optimizer)
        old_scale = scaler.get_scale()
        bad_gradients = any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                            for parameter in parameters)
        if bad_gradients and not scaler.is_enabled():
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"non-finite gradient at optimizer step {step} with GradScaler disabled")
        if not bad_gradients:
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.is_enabled() and scaler.get_scale() < old_scale
        if skipped:
            optimizer.zero_grad(set_to_none=True)
            consecutive_overflows += 1
            run["grad_scaler_skipped_steps"] += 1
            print(f"train: GradScaler overflow; optimizer step {step} not counted "
                  f"(scale {old_scale:g}->{scaler.get_scale():g}, "
                  f"consecutive_skips={consecutive_overflows})", flush=True)
            if consecutive_overflows >= 16:
                raise FloatingPointError(
                    f"GradScaler skipped 16 consecutive attempts at optimizer step {step}; "
                    "no skipped attempt was counted toward max_steps"
                )
            continue
        if bad_gradients:
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                f"non-finite gradient at optimizer step {step} without GradScaler scale reduction"
            )
        optimizer.zero_grad(set_to_none=True)
        consecutive_overflows = 0
        successful_steps += 1
        losses.extend(attempt_losses)
        scheduler.step()
        if step == 1 or step % 10 == 0:
            _step_vram = (torch.cuda.max_memory_allocated() / 1024**3
                          if torch.cuda.is_available() else 0.0)
            _step_loss = sum(losses[-accum:]) / len(losses[-accum:])
            print(f"train: step {step}/{max_steps} mean_loss_last_accum={_step_loss:.5f} "
                  f"elapsed_min={(time.perf_counter() - started) / 60:.2f} "
                  f"peak_vram_gb={_step_vram:.2f}", flush=True)
        if step % 100 == 0 or step == max_steps:
            run.update({"step": step, "mean_loss_last_100": sum(losses[-100:]) / len(losses[-100:]),
                        "train_minutes": (time.perf_counter() - started) / 60,
                        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3
                        if torch.cuda.is_available() else 0.0})
            save_checkpoint(model, out_dir, step, run)
    run["train_minutes"] = (time.perf_counter() - started) / 60
    run["peak_vram_gb"] = (torch.cuda.max_memory_allocated() / 1024**3
                           if torch.cuda.is_available() else 0.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    return run


def predict(model: Any, processor: Any, rows: Sequence[Mapping[str, Any]], split_dir: str | Path,
            coord_scale: str = "norm1000", batch_size: int = 4, max_new_tokens: int = 256,
            on_batch: Callable[[list[dict[str, str]]], None] | None = None) -> list[dict[str, str]]:
    """Generate deterministic output in batches; slice after padded prompt width."""
    import torch
    from PIL import Image

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    processor.tokenizer.padding_side = "left"
    root = Path(split_dir).parent
    prompt = build_prompt(coord_scale)
    model.eval()
    outputs = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            group = rows[start:start + batch_size]
            conversations = []
            for row in group:
                with Image.open(root / row["image_path"]) as im:
                    image = im.convert("RGB")
                conversations.append([{"role": "user", "content": [
                    {"type": "image", "image": image}, {"type": "text", "text": prompt}]}])
            inputs = processor.apply_chat_template(
                conversations, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
                processor_kwargs={"padding": True},
            )
            inputs = {key: value.to(model.device) if hasattr(value, "to") else value
                      for key, value in dict(inputs).items()}
            inputs.pop("token_type_ids", None)
            prompt_width = inputs["input_ids"].shape[1]
            with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            decoded = processor.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
            batch_outputs = [{"image_id": row["image_id"], "raw": raw}
                             for row, raw in zip(group, decoded)]
            outputs.extend(batch_outputs)
            if on_batch is not None:
                on_batch(batch_outputs)
            print(f"predict: {len(outputs)}/{len(rows)} images in {time.perf_counter() - started:.1f}s",
                  flush=True)
    return outputs

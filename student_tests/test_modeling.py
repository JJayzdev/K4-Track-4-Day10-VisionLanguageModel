import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from student.cli import resolve_coord_scale
from student.modeling import (
    LLM_TARGET, VIT_TARGET, QWEN3_VL_BNB_SKIP_MODULES, _schedule, load_projector_state,
    validate_quantization_layout,
    matched_lora_modules, train,
)


class ModuleNames:
    def __init__(self, names):
        self.names = names

    def named_modules(self):
        return [(name, object()) for name in self.names]


def test_lora_target_counts_and_coord_resolution():
    names = []
    for layer in range(36):
        names.extend(f"model.language_model.layers.{layer}.self_attn.{name}"
                     for name in ("q_proj", "k_proj", "v_proj", "o_proj"))
        names.extend(f"model.language_model.layers.{layer}.mlp.{name}"
                     for name in ("gate_proj", "up_proj", "down_proj"))
    for layer in range(24):
        names.extend(f"model.visual.blocks.{layer}.attn.{name}" for name in ("qkv", "proj"))
        names.extend(f"model.visual.blocks.{layer}.mlp.{name}" for name in ("linear_fc1", "linear_fc2"))
    model = ModuleNames(names)
    assert len(matched_lora_modules(model, "E1")) == 252
    assert len(matched_lora_modules(model, "E2")) == 252
    assert len(matched_lora_modules(model, "E3")) == 348
    assert resolve_coord_scale(None, {"coord_scale": "pixel"}) == "pixel"
    assert resolve_coord_scale(None, {}, zero_shot=True) == "norm1000"
    assert resolve_coord_scale("pixel", {"coord_scale": "norm1000"}) == "pixel"


def test_quantization_skip_paths_and_layout_validation():
    import torch

    assert QWEN3_VL_BNB_SKIP_MODULES == ("model.visual", "lm_head")

    class Model:
        def __init__(self, visual, head):
            self.params = [
                ("model.visual.blocks.0.attn.qkv.weight", visual),
            ]
            self.lm_head = SimpleNamespace(
                named_parameters=lambda: iter([("weight", head)])
            )

        def named_parameters(self):
            return iter(self.params)

    class FourBit:
        dtype = torch.float16

    fp16_visual = torch.nn.Parameter(torch.ones(2, dtype=torch.float16))
    fp16_head = torch.nn.Parameter(torch.ones(2, dtype=torch.float16))
    validate_quantization_layout(Model(fp16_visual, fp16_head), params4bit_type=FourBit)
    with pytest.raises(RuntimeError, match="visual tower or lm_head was quantized"):
        validate_quantization_layout(Model(FourBit(), fp16_head), params4bit_type=FourBit)
    with pytest.raises(RuntimeError, match="remain fp16"):
        validate_quantization_layout(
            Model(torch.nn.Parameter(torch.ones(2)), fp16_head), params4bit_type=FourBit
        )


def test_quantization_layout_checks_tied_lm_head_even_when_global_params_deduplicate():
    import torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(4, 2).half()
            self.lm_head = torch.nn.Linear(2, 4, bias=False).half()
            self.lm_head.weight = self.embed_tokens.weight
            self.visual = torch.nn.Module()
            self.visual.blocks = torch.nn.Module()
            self.visual.blocks.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float16))

    model = Model()
    assert not any("lm_head" in name for name, _ in model.named_parameters())
    validate_quantization_layout(model, params4bit_type=type("FourBit", (), {}))

    class FourBit:
        dtype = torch.float16

    class Head:
        def named_parameters(self):
            return iter([("weight", FourBit())])

    class BrokenModel:
        lm_head = Head()

        def named_parameters(self):
            return iter([("visual.blocks.weight", model.visual.blocks.weight)])

    with pytest.raises(RuntimeError, match="visual tower or lm_head was quantized"):
        validate_quantization_layout(BrokenModel(), params4bit_type=FourBit)


def test_projector_restore_requires_exact_keys_shapes_and_uses_fp32():
    import torch

    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = torch.nn.Module()
            self.visual.merger = torch.nn.Linear(2, 2, bias=False).half()

    model = Base()
    complete = {name: torch.full_like(param, 0.25).half()
                for name, param in model.named_parameters()}
    load_projector_state(model, complete)
    assert model.visual.merger.weight.dtype == torch.float32
    assert torch.all(model.visual.merger.weight == 0.25)
    with pytest.raises(ValueError, match="keys mismatch"):
        load_projector_state(model, {})
    with pytest.raises(ValueError, match="shape mismatch"):
        load_projector_state(model, {"visual.merger.weight": torch.zeros((3, 2))})


def test_schedule_warmup_then_cosine():
    import torch

    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([param], lr=1.0)
    scheduler = _schedule(optimizer, 100)
    values = [optimizer.param_groups[0]["lr"]]
    for _ in range(100):
        optimizer.step()
        scheduler.step()
        values.append(optimizer.param_groups[0]["lr"])
    assert values[0] < values[1] < values[2]
    assert values[-1] == pytest.approx(0.0)


class TinyProcessor:
    class Tokenizer:
        pad_token_id = 0
        padding_side = "right"

    tokenizer = Tokenizer()

    def apply_chat_template(self, messages, **kwargs):
        import torch
        is_full = len(messages) == 2
        ids = [1, 8] + ([9, 2] if is_full else [])
        return {"input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
                "mm_token_type_ids": torch.tensor([[0] * len(ids)]),
                "pixel_values": torch.ones((1, 2))}


class TinyTrainModel:
    def __init__(self, finite=True):
        import torch
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(use_cache=True)
        self.finite = finite

    @property
    def device(self):
        return self.weight.device

    def parameters(self):
        return [self.weight]

    def named_parameters(self):
        return [("weight", self.weight)]

    def gradient_checkpointing_enable(self, **kwargs):
        pass

    def train(self):
        pass

    def __call__(self, input_ids, labels, **kwargs):
        loss = self.weight * input_ids.float().mean()
        if not self.finite:
            loss = loss * float("nan")
        return SimpleNamespace(loss=loss)

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


def _mock_setup(monkeypatch):
    import student.modeling as modeling
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def setup(model, name, **kwargs):
        model.weight.requires_grad_(True)
        return model, {"trainable_params": 1, "matched_lora_modules": 252, "projector_params": 0}
    monkeypatch.setattr(modeling, "apply_setup", setup)


def test_train_cpu_smoke_uses_custom_images_root_and_saves_truthful_run(tmp_path, monkeypatch):
    from PIL import Image

    _mock_setup(monkeypatch)
    image_root = tmp_path / "custom_images"
    image_root.mkdir()
    Image.new("RGB", (20, 10)).save(image_root / "one.jpg")
    rows = [{"image_path": "one.jpg", "width": 20, "height": 10, "objects": []}]
    model = TinyTrainModel()
    config = {"name": "E1", "seed": 7, "batch_size": 1, "grad_accum": 2,
              "max_steps": 2, "lr": .01, "lora": {"r": 2, "alpha": 4}}
    before = model.weight.detach().clone()
    run = train(model, TinyProcessor(), rows, config, tmp_path / "out", images_root=image_root)
    assert model.weight.detach() != before
    assert run["step"] == 2
    assert run["selected_train_images"] == 1
    assert (tmp_path / "out" / "checkpoint-0002" / "run.json").exists()
    assert json.loads((tmp_path / "out" / "run.json").read_text())["trainable_params"] == 1


def test_train_rejects_empty_rows_and_nonfinite_loss(tmp_path, monkeypatch):
    from PIL import Image

    _mock_setup(monkeypatch)
    model = TinyTrainModel()
    with pytest.raises(ValueError, match="split is empty"):
        train(model, TinyProcessor(), [], {"name": "E1"}, tmp_path / "empty")
    Image.new("RGB", (12, 8)).save(tmp_path / "image.jpg")
    with pytest.raises(FloatingPointError, match="non-finite loss"):
        train(TinyTrainModel(finite=False), TinyProcessor(),
              [{"image_path": "image.jpg", "objects": [], "width": 12, "height": 8}],
              {"name": "E1", "batch_size": 1, "grad_accum": 1, "max_steps": 1},
              tmp_path / "nan", images_root=tmp_path)


def test_train_retries_gradscaler_overflow_without_counting_skipped_update(tmp_path, monkeypatch):
    from PIL import Image
    import torch

    _mock_setup(monkeypatch)
    Image.new("RGB", (12, 8)).save(tmp_path / "image.jpg")

    class OneOverflowScaler:
        def __init__(self):
            self.scale_value = 65536.0
            self.skip = False
            self.injected = False

        def is_enabled(self):
            return True

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            if not self.injected:
                optimizer.param_groups[0]["params"][0].grad.fill_(float("inf"))
                self.injected = True
                self.skip = True

        def get_scale(self):
            return self.scale_value

        def step(self, optimizer):
            if not self.skip:
                optimizer.step()

        def update(self):
            if self.skip:
                self.scale_value /= 2
                self.skip = False

    monkeypatch.setattr(torch.amp, "GradScaler", lambda *args, **kwargs: OneOverflowScaler())
    model = TinyTrainModel()
    before = model.weight.detach().clone()
    run = train(
        model, TinyProcessor(),
        [{"image_path": "image.jpg", "objects": [], "width": 12, "height": 8}],
        {"name": "E1", "batch_size": 1, "grad_accum": 1, "max_steps": 1, "lr": .01},
        tmp_path / "overflow", images_root=tmp_path,
    )
    assert model.weight.detach() != before
    assert run["step"] == 1
    assert run["grad_scaler_skipped_steps"] == 1
    assert json.loads((tmp_path / "overflow" / "run.json").read_text())["step"] == 1


def test_predict_batches_variable_prompt_width_and_slices_after_padding(tmp_path):
    import torch
    from PIL import Image
    from student.modeling import predict

    data_root = tmp_path / "data"
    (data_root / "images").mkdir(parents=True)
    Image.new("RGB", (16, 8)).save(data_root / "images" / "short.jpg")
    Image.new("RGB", (32, 8)).save(data_root / "images" / "long.jpg")

    class Processor:
        class Tokenizer:
            padding_side = "right"
        tokenizer = Tokenizer()

        def __init__(self):
            self.template_kwargs = []

        def apply_chat_template(self, conversations, **kwargs):
            self.template_kwargs.append(kwargs)
            seqs = [torch.ones((conversation[0]["content"][0]["image"].width // 8,), dtype=torch.long)
                    for conversation in conversations]
            width = max(map(len, seqs))
            ids = torch.zeros((len(seqs), width), dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, seq in enumerate(seqs):
                ids[i, width - len(seq):] = seq
                mask[i, width - len(seq):] = 1
            return {"input_ids": ids, "attention_mask": mask,
                    "token_type_ids": torch.zeros_like(ids)}

        def batch_decode(self, rows, **kwargs):
            return [str(int(row[0])) for row in rows]

    class Model:
        device = torch.device("cpu")

        def eval(self):
            return self

        def generate(self, input_ids, **kwargs):
            return torch.cat((input_ids, torch.full((input_ids.shape[0], 1), 99)), dim=1)

    rows = [{"image_id": "s", "image_path": "images/short.jpg"},
            {"image_id": "l", "image_path": "images/long.jpg"}]
    streamed = []
    processor = Processor()
    outputs = predict(Model(), processor, rows, data_root / "splits", batch_size=1,
                      on_batch=lambda batch: streamed.extend(batch))
    assert outputs == [{"image_id": "s", "raw": "99"}, {"image_id": "l", "raw": "99"}]
    assert streamed == outputs
    assert len(processor.template_kwargs) == 2
    assert all(call["processor_kwargs"] == {"padding": True} and "padding" not in call
               for call in processor.template_kwargs)


def test_reporting_uses_real_predictions_and_score_json(tmp_path):
    from PIL import Image
    from student.reporting import evidence_summary, render_predictions

    rows = []
    predictions = {}
    (tmp_path / "data" / "images").mkdir(parents=True)
    for index in range(6):
        image_name = f"{index}.jpg"
        Image.new("RGB", (20, 10), "black").save(tmp_path / "data" / "images" / image_name)
        rows.append({"image_id": str(index), "image_path": f"images/{image_name}",
                     "objects": [{"label": "pothole", "yolo": [.5, .5, .2, .2]}]})
        predictions[str(index)] = '[{"label":"pothole","bbox_2d":[400,400,600,600]}]'
    saved = render_predictions(rows, predictions, tmp_path / "data", tmp_path / "figures", limit=6)
    assert len(saved) == 6 and all(path.exists() for path in saved)
    evidence = {"name": "fixture", "micro": {"f1": 0.5}, "f1_ci95": [0.3, 0.7],
                "fp_per_image": 0.2, "per_class": {"pothole": {"f1": 0.4},
                "crack": {"f1": 0.5}, "manhole": {"f1": 0.6}},
                "meta": {"peak_vram_gb": 7, "train_minutes": 12}}
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence))
    table = evidence_summary([evidence_path])
    assert "0.500" in table and "0.300–0.700" in table and "fixture" in table

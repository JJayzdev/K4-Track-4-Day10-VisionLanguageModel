import json

import pytest

from student.grounding import (
    build_prompt, collate_examples, encode_example, load_split, norm1000_to_pixel,
    objects_to_answer, validate_session_disjoint, vision_token_count, yolo_to_box,
    prepare_training_rows,
)


def test_conversion_clamps_rounds_and_orders_stably():
    objs = [
        {"label": "crack", "yolo": [.5, .6, .2, .2]},
        {"label": "pothole", "yolo": [.4, .2, .1, .1]},
    ]
    raw = objects_to_answer(objs)
    parsed = json.loads(raw)
    assert parsed == [
        {"label": "pothole", "bbox_2d": [350, 150, 450, 250]},
        {"label": "crack", "bbox_2d": [400, 500, 600, 700]},
    ]
    assert objects_to_answer([]) == "[]"
    assert yolo_to_box("pothole", [0, 0, 1, 1])["bbox_2d"] == [0, 0, 500, 500]
    assert norm1000_to_pixel([500, 500, 1000, 1000], 640, 360) == (320, 180, 640, 360)
    assert yolo_to_box("pothole", [.5, .5, .25, .5], "pixel", 640, 360)["bbox_2d"] == [240, 90, 400, 270]
    assert "0–1000 scale" in build_prompt()
    assert "pixel coordinates" in build_prompt("pixel")


def test_invalid_class_or_coordinates_rejected():
    with pytest.raises(ValueError):
        yolo_to_box("debris", [.5, .5, .1, .1])
    with pytest.raises(ValueError):
        yolo_to_box("crack", [.5, .5, float("nan"), .1])
    with pytest.raises(ValueError, match="zero area"):
        yolo_to_box("crack", [.5, .5, 0, .1])


def test_tiny_positive_boxes_keep_minimum_extent_after_rounding_and_clipping():
    # A positive box narrower than one target unit remains a one-unit box.
    assert yolo_to_box("crack", [.5, .5, .0001, .0001]) == {
        "label": "crack", "bbox_2d": [500, 500, 501, 501]
    }
    assert yolo_to_box("crack", [.5, .5, .0001, .0001], "pixel", 10, 10) == {
        "label": "crack", "bbox_2d": [5, 5, 6, 6]
    }
    # At the far edge, the one-unit interval shifts inward and stays in bounds.
    assert yolo_to_box("crack", [1, 1, .0001, .0001]) == {
        "label": "crack", "bbox_2d": [999, 999, 1000, 1000]
    }
    assert yolo_to_box("crack", [1, 1, .0001, .0001], "pixel", 10, 10) == {
        "label": "crack", "bbox_2d": [9, 9, 10, 10]
    }
    with pytest.raises(ValueError, match="zero area"):
        yolo_to_box("crack", [.5, .5, 0, .1])


def test_training_preflight_excludes_only_true_zero_area_annotations_and_keeps_images():
    rows = [
        {"image_id": "mixed", "objects": [
            {"label": "pothole", "yolo": [.5, .5, .2, .2]},
            {"label": "crack", "yolo": [.4, .4, 0, .1]},
        ]},
        {"image_id": "empty_after_filter", "objects": [
            {"label": "manhole", "yolo": [.2, .2, .1, 0]},
        ]},
        {"image_id": "clean_empty", "objects": []},
    ]
    cleaned, report = prepare_training_rows(rows)
    assert [row["image_id"] for row in cleaned] == ["mixed", "empty_after_filter", "clean_empty"]
    assert cleaned[0]["objects"] == rows[0]["objects"][:1]
    assert cleaned[1]["objects"] == [] and cleaned[2]["objects"] == []
    assert rows[0]["objects"][1]["yolo"][2] == 0  # source split is not mutated
    assert report == {
        "source_images": 3, "source_annotations": 3, "retained_annotations": 1,
        "zero_area_annotations_excluded": 2, "images_with_zero_area_annotations": 2,
        "images_empty_after_filter": 1,
        "excluded": [
            {"image_id": "mixed", "annotations": [
                {"index": 1, "label": "crack", "yolo": [.4, .4, 0, .1]},
            ]},
            {"image_id": "empty_after_filter", "annotations": [
                {"index": 0, "label": "manhole", "yolo": [.2, .2, .1, 0]},
            ]},
        ],
    }
    with pytest.raises(ValueError, match="invalid YOLO values"):
        prepare_training_rows([{"image_id": "bad", "objects": [
            {"label": "crack", "yolo": [.5, .5, -0.1, .2]},
        ]}])
    with pytest.raises(ValueError, match="invalid label"):
        prepare_training_rows([{"image_id": "bad-label", "objects": [
            {"label": "unknown", "yolo": [.5, .5, .1, .2]},
        ]}])


def test_session_split_validation():
    assert validate_session_disjoint({"train": [{"session": "a"}], "test": [{"session": "b"}]})
    with pytest.raises(ValueError, match="occurs in both"):
        validate_session_disjoint({"train": [{"session": "a"}], "test": [{"session": "a"}]})


def test_load_split_requires_known_split_and_fields(tmp_path):
    with pytest.raises(ValueError, match="unknown split"):
        load_split(tmp_path, "evaluation")


class TinyProcessor:
    """Small template stub to exercise masking without loading a VLM."""
    def apply_chat_template(self, messages, **kwargs):
        import torch

        is_full = len(messages) == 2
        ids = [1, 20, 21] + ([30, 31, 2] if is_full else [])
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            "mm_token_type_ids": torch.tensor([[0, 1, 1] + ([0, 0, 0] if is_full else [])]),
            "pixel_values": torch.ones((2, 3)),
            "image_grid_thw": torch.tensor([[1, 1, 2]]),
        }


class PrefixMismatchProcessor(TinyProcessor):
    def apply_chat_template(self, messages, **kwargs):
        result = super().apply_chat_template(messages, **kwargs)
        if len(messages) == 2:
            result["input_ids"][0, 1] = 99
        return result


def test_encode_masks_only_prefix_and_collate_pads_labels():
    encoded = encode_example(TinyProcessor(), object(), "locate", "[]")
    assert encoded["prefix_length"] == 3
    assert encoded["labels"].tolist() == [[-100, -100, -100, 30, 31, 2]]
    longer = dict(encoded)
    import torch
    longer["input_ids"] = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    longer["labels"] = torch.tensor([[-100, -100, -100, 4, 5, 6, 7]])
    longer["attention_mask"] = torch.ones((1, 7), dtype=torch.long)
    longer["mm_token_type_ids"] = torch.zeros((1, 7), dtype=torch.long)
    batch = collate_examples([encoded, longer], pad_token_id=0)
    assert batch["input_ids"].shape == (2, 7)
    assert batch["labels"].tolist()[0] == [-100, -100, -100, 30, 31, 2, -100]
    assert batch["pixel_values"].shape[0] == 4
    assert batch["image_grid_thw"].shape[0] == 2
    assert vision_token_count(encoded) == 2


def test_prefix_mismatch_and_optional_multimodal_fields_are_explicit():
    with pytest.raises(ValueError, match="not a prefix"):
        encode_example(PrefixMismatchProcessor(), object(), "locate", "[]")
    import torch
    one = {"input_ids": torch.tensor([[1]]), "labels": torch.tensor([[-100]]),
           "mm_token_type_ids": torch.tensor([[0]])}
    two = {"input_ids": torch.tensor([[1]]), "labels": torch.tensor([[-100]])}
    with pytest.raises(ValueError, match="mm_token_type_ids"):
        collate_examples([one, two], pad_token_id=0)

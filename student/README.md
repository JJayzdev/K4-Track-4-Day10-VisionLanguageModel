# Student implementation

The lab code lives in `student/`; the instructor-owned `scripts/` and `tests/` remain unchanged. The code keeps model imports inside runtime functions so split, coordinate, and JSON helpers can be used without loading a VLM.

## Run on Colab

After installing `requirements-colab.txt` and preparing `data/`, run E0 and the three SFT setups from the notebook. The equivalent commands are:

In notebook cell 3, set `DATA_MODE` to `kaggle`, `zip`, or `source`. Kaggle mode first uses `KAGGLE_USERNAME` and `KAGGLE_KEY` already present in the environment. Otherwise, add both names in Colab **Secrets** (key icon) and enable notebook access for each; keep their values out of notebook cells. For a manually downloaded archive, set `DATA_MODE = 'zip'` and `DATA_PATH` to the uploaded zip file. For an extracted dataset, set `DATA_MODE = 'source'` and `DATA_PATH` to the directory containing `images/` and `labels-YOLO/`. These two modes do not import or access Colab Secrets.

```bash
python -m student.cli predict --zero-shot --split test --out work/e0.jsonl
python -m student.cli train --config configs/e1.yaml --out-dir work/e1
python -m student.cli train --config configs/e2.yaml --out-dir work/e2
python -m student.cli train --config configs/e3.yaml --out-dir work/e3
```

Training saves LoRA adapters every 100 optimizer steps. E2/E3 checkpoints also include the trained visual merger weights in fp32. Each run records the setup, data fraction, coordinate scale, trainable parameter count, peak allocated VRAM, runtime, and mean recent loss in `run.json`. Use `student.cli predict` with a run directory; it selects the latest checkpoint and reads the coordinate scale from the run metadata unless `--coord` is explicitly supplied.

Before allocating the model, inspect source-label quality with `python -m student.cli audit-train --splits-dir data/splits`. Training excludes only annotations whose source YOLO width or height is zero, records each excluded image/object and aggregate counts in `run.json`, and retains every training image (including images whose filtered target becomes `[]`). Preflight validates labels and target-coordinate conversion for every retained annotation; malformed or out-of-range boxes stop before model loading. The source split files are never rewritten. Validation, test labels, and official scoring remain unchanged; source zero-area boxes still count as ground truth during scoring. Use `--coord pixel` for a pixel-coordinate run.

E4 changes one variable at a time. For rank, run `--rank 4` and `--rank 64` against the same selected baseline config. For data size, use `--train-fraction 0.1` and `1.0`. For coordinates, use `--coord pixel` and `norm1000` consistently for training, prediction, and `scripts/score.py`. Predictions and evidence must come from actual runs; this module does not fabricate scores or report values.

`student.reporting.render_predictions` creates labelled ground-truth/prediction images from a real prediction JSONL, and `evidence_summary` formats existing `score.py` JSON files. `student_tests/` contains offline tests, including a tiny CPU training loop; it does not load the 4B model or replace the instructor tests.

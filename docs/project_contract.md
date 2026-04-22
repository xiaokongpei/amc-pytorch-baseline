# Project Contract

- This repository is the main PyTorch execution line for future AMC experiments.
- The formal baseline entrypoints are `scripts/train_fast.py`, `scripts/build_fast_slices.py`, and `configs/fast.yaml`.
- It consumes prepared sliced data and does not own raw data conversion.
- Default semantic data location is `data/processed/`.
- The active processed split format is `pt`.
- Official split semantics are `train.pt`, `validation.pt`, and `test.pt`.
- Run outputs must be written under `runs/<run_name>/`.
- Baseline structure should stay comparable to the Harper TensorFlow baseline.
- Future experiment scripts should extend this baseline rather than duplicating the whole training stack.

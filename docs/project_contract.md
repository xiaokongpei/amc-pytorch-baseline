# Project Contract

- This repository is the main PyTorch execution line for future AMC experiments.
- The formal baseline entrypoints are `scripts/train_fast.py`, `scripts/build_stratified_slices.py`, and `configs/fast.yaml`.
- It consumes prepared sliced data and provides the official stratified slice builder.
- Default semantic data location is `data/processed_v2_stratified_64_16_20/`.
- The active processed split format is `pt`.
- Official split semantics are `train.pt`, `validation.pt`, and `test.pt`.
- The official split rule is `modulation x SNR` stratified `64/16/20`.
- Run outputs must be written under `runs/<run_name>/`.
- Baseline structure should remain stable enough for controlled PyTorch ablations.
- Future experiment scripts should extend this baseline rather than duplicating the whole training stack.

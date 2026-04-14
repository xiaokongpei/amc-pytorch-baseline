# Project Contract

- This repository is the main PyTorch execution line for future AMC experiments.
- It consumes prepared sliced data and does not own raw data conversion.
- Default semantic data location is `data/processed/`.
- Official split formats are `npy`, `npz`, and `pt`.
- Official split semantics are `train/`, `validation/`, and `test/`.
- Recommended shard size is about `2000` samples, following the original Harper slicing rule.
- Run outputs must be written under `runs/<run_name>/`.
- Baseline structure should stay comparable to the Harper TensorFlow baseline.
- Future experiment scripts should extend this baseline rather than duplicating the whole training stack.

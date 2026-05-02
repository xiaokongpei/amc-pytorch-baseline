# CLDNN Architecture Version Tree

```
v1.0  Harper Baseline (CNN + SE)
      │   config: configs/fast.yaml
      │   Conv1d×7 + SEBlock → StatisticalPooling → FC
      │
      └─── v2.0  CLDNN Baseline
               │   config: configs/cldnn.yaml
               │   Conv1d×4 → BiLSTM → TemporalStatsPooling → FC
               │
               ├─── v2.1  ShrinkDenoise (Soft)
               │         config: configs/cldnn_shrink_soft.yaml
               │         +ShrinkDenoise1D(shrinkage=soft) after conv blocks
               │
               ├─── v2.2  ShrinkDenoise (Garrote)
               │         config: configs/cldnn_shrink_garrote.yaml
               │         +ShrinkDenoise1D(shrinkage=garrote) after conv blocks
               │
               ├─── v3.0  [TODO] Dual-Stream
               │         stream_iq + stream_fft
               │
               │         ├─── v3.1  + Attention Fusion
               │         └─── v3.2  + Shared Weights
               │
               └─── v4.0  [TODO] ...
```

## Git Tags

```bash
v1.0    Harper CNN + SE
v2.0    CLDNN baseline
v2.1    CLDNN + ShrinkDenoise soft
v2.2    CLDNN + ShrinkDenoise garrote
```

```bash
git checkout v1.0    # Harper baseline
git checkout v2.0    # CLDNN baseline
git checkout v2.1    # ShrinkDenoise soft
```

## Adding a New Version

1. 新建 config: `configs/<name>.yaml`
2. 新模块写入 `src/models/cldnn.py`，通过 config 控制
3. 更新本文件的树
4. `git tag -a <version> -m "<description>"`
5. `git push origin --tags` 同步到远程

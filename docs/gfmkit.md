# gfmkit

`gfmkit` is an importable Graph Foundation Model toolkit for the ZSLP link-prediction benchmark. It keeps the vendored `baselines/` folder untouched and exposes a stable Python and CLI API over the current models.

## Install

```bash
pip install -e .
```

## Python API

```python
from gfmkit import DatasetSuite, create_model

suite = DatasetSuite.from_preset("link_prediction")
model = create_model("GraphPrompt", seed=0, device="cuda:0", pretrain_epochs=100)

model.pretrain(suite.link1)
model.finetune(suite.link2)
outputs = model.infer(suite.link2)
model.save_outputs(outputs)
```

## CLI

```bash
gfmkit list-models
gfmkit run-lp --model GNN --datasets smoke --seed 0 --device cuda:0 --set pretrain_epochs=1
gfmkit pretrain --model GraphPrompt --datasets link1 --seed 0
gfmkit finetune --model GraphPrompt --datasets link2 --seed 0 --set prompt_epochs=50
gfmkit infer --model GraphPrompt --datasets link2 --seed 0
```

## Adapter Policy

`GNN`, `GraphCL`, `GPF`, `GraphPrompt`, `GFT`, and `GILT` are in-process adapters. `AnyGraph`, `GraphAny`, `Mochi`, and `UniLP` are subprocess-backed adapters because their original repositories remain script/config driven.

Subprocess adapters accept command overrides:

```bash
gfmkit pretrain --model UniLP --set command='python main.py --pretrain_datasets {datasets} --inference_datasets link2'
```

Outputs default to the existing benchmark-compatible structure under `baselines/outputs/{ModelName}/{dataset}_{seed}.pt`.

Mochi's native LP embedding exports are loaded from `Mochi/outputs/lp_{dataset}_{seed}.pt` and returned as `{"z": tensor, ...}` because Mochi does not save LP edge labels in those artifacts.

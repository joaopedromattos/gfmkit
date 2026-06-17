# gfmkit

`gfmkit` is an importable Graph Foundation Model toolkit for the ZSLP link-prediction benchmark. It provides one registry and API for training, adapting, and exporting embeddings from the project’s Graph Foundation Model baselines while keeping the original model repositories untouched.

## Install

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
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

## Models

In-process adapters:

- `GNN`
- `GraphCL`
- `GPF`
- `GraphPrompt`
- `GFT`
- `GILT`

Subprocess-backed adapters:

- `AnyGraph`
- `GraphAny`
- `Mochi`
- `UniLP`

The subprocess adapters expect the corresponding original repositories to exist locally. In the ZSLP workspace, these are resolved relative to the project root. Outside that workspace, pass path overrides through `create_model(...)` or CLI `--set key=value` options.

## Outputs

Link-prediction embedding outputs default to the benchmark-compatible structure:

```python
{"z": z, "edge_label": edge_label, "edge_label_index": edge_label_index}
```

Mochi is the exception: its native LP export contains only embeddings, so `gfmkit` returns `{"z": tensor, ...}` for Mochi outputs.


from __future__ import annotations

from importlib.util import find_spec

import torch

from gfmkit import DatasetSuite, EmbeddingOutput, create_model, list_models


def test_dataset_suite_preserves_link_orders():
    suite = DatasetSuite.from_preset("link_prediction")
    assert suite.link1[0] == "products_tech"
    assert suite.link1[-1] == "email-Enron"
    assert suite.link2[0] == "Photo"
    assert suite.link2[-1] == "roadNet-PA"
    assert suite.resolve("smoke") == ["cora"]
    assert suite.resolve("cora,CS") == ["cora", "CS"]


def test_registry_contains_current_gfm_families():
    names = set(list_models())
    assert {
        "AnyGraph",
        "GFT",
        "GILT",
        "GNN",
        "GPF",
        "GraphAny",
        "GraphCL",
        "GraphPrompt",
        "Mochi",
        "UniLP",
    } <= names


def test_create_lightweight_gnn_adapter_without_training(tmp_path):
    if find_spec("torch_geometric") is None:
        return
    model = create_model(
        "GNNs",
        seed=123,
        device="cpu",
        output_root=tmp_path / "outputs",
        checkpoint_root=tmp_path / "checkpoints",
        pretrain_epochs=0,
        save_outputs=False,
    )
    assert model.name == "GNN"
    assert str(model.device) == "cpu"


def test_subprocess_adapter_dry_run_command():
    model = create_model("AnyGraph", seed=7, device="cuda:3", dry_run=True, pretrain_epochs=2)
    cmd = model.command("pretrain", "link1")
    assert cmd[:2] == ["python", "main.py"]
    assert "--seed" in cmd
    assert "7" in cmd


def test_mochi_adapter_command_points_to_root_repo_data():
    model = create_model("Mochi++", seed=2, device="cuda:4", dry_run=True, train_steps=3)
    cmd = model.command("pretrain", "link1")
    assert cmd[:2] == ["python", "train.py"]
    assert "--model_variant" in cmd
    assert "mochi++" in cmd
    assert "--lp_data_root" in cmd
    assert cmd[cmd.index("--dataset_setting") + 1] == "link1"


def test_embedding_output_legacy_dict_shape():
    out = EmbeddingOutput(
        z=torch.zeros(3, 2),
        edge_label=torch.tensor([1, 0]),
        edge_label_index=torch.tensor([[0, 1], [1, 2]]),
        dataset="toy",
        model_name="GNN",
        seed=0,
    )
    legacy = out.to_dict(include_metadata=False)
    assert set(legacy) == {"z", "edge_label", "edge_label_index"}
    with_meta = out.to_dict()
    assert with_meta["dataset"] == "toy"

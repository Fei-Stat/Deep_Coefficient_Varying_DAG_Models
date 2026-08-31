# run_experiment.py
"""
Example end-to-end entry point.

Expected input file (PyTorch .pt) keys:
    train_z, train_Y, train_mask
    val_z,   val_Y,   val_mask
    test_z,  test_Y,  test_mask

Masks are optional only if the corresponding split is fully observational.
If your data are stored differently, replace load_split_file() only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import DynamicDAG
from train import TrainConfig, train_model
from evaluate import evaluate_test


def load_split_file(path: str):
    data = torch.load(path, map_location="cpu")

    required = [
        "train_z",
        "train_Y",
        "val_z",
        "val_Y",
        "test_z",
        "test_Y",
    ]
    missing = [key for key in required if key not in data]

    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")

    return {
        "train_z": data["train_z"].float(),
        "train_Y": data["train_Y"].float(),
        "train_mask": (
            data["train_mask"].float()
            if "train_mask" in data
            else None
        ),
        "val_z": data["val_z"].float(),
        "val_Y": data["val_Y"].float(),
        "val_mask": (
            data["val_mask"].float()
            if "val_mask" in data
            else None
        ),
        "test_z": data["test_z"].float(),
        "test_Y": data["test_Y"].float(),
        "test_mask": (
            data["test_mask"].float()
            if "test_mask" in data
            else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to split .pt file.")
    parser.add_argument("--output", default="results.pt")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--context-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    data = load_split_file(args.data)

    input_dim = data["train_z"].shape[1]
    n_nodes = data["train_Y"].shape[1]

    model = DynamicDAG(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        context_dim=args.context_dim,
        n_nodes=n_nodes,
        dropout=args.dropout,
        context_transform="tanh",
        power_iteration_steps=15,
    ).to(args.device)

    config = TrainConfig(
        stage1_epochs=300,
        stage2_epochs=1200,
        stage1_lr=2e-3,
        stage2_lr=1e-3,
        lambda_group_stage1=1e-3,
        lambda_group_stage2=1e-3,
        gamma_increment=5e-3,
        gamma_schedule="linear",
        freeze_gamma_at_dag=True,
        freeze_gamma_threshold=0.01,
        edge_threshold=0.1,
        use_screening=False,  # turn on only after validating threshold behavior
        seed=0,
    )

    train_result = train_model(
        model=model,
        train_z=data["train_z"],
        train_Y=data["train_Y"],
        train_mask=data["train_mask"],
        val_z=data["val_z"],
        val_Y=data["val_Y"],
        val_mask=data["val_mask"],
        config=config,
    )

    # Test split is touched only after training / checkpoint selection.
    test_metrics = evaluate_test(
        model=model,
        test_z=data["test_z"],
        test_Y=data["test_Y"],
        test_mask=data["test_mask"],
        edge_threshold=config.edge_threshold,
        batch_size=512,
        verbose=True,
        return_tensors=True,
    )

    payload = {
        "model_state_dict": model.state_dict(),
        "config": vars(config),
        "checkpoint": train_result["checkpoint"],
        "validation": {
            k: v
            for k, v in train_result["validation"].items()
            if not torch.is_tensor(v)
        },
        "test_summary": {
            k: v
            for k, v in test_metrics.items()
            if not torch.is_tensor(v)
        },
        "support": test_metrics["support"],
        "adjacency": test_metrics["adjacency"],
        "raw_adjacency": test_metrics["raw_adjacency"],
        "test_prediction": test_metrics.get("prediction"),
        "test_beta": test_metrics.get("beta"),
        "test_context": test_metrics.get("context"),
        "history": train_result["history"],
    }

    out = Path(args.output)
    torch.save(payload, out)
    print(f"\nSaved experiment output to: {out.resolve()}")


if __name__ == "__main__":
    main()

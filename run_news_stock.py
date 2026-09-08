"""Train/evaluate the independent news-stock application.

All reconstruction metrics use other SAME-DAY stock returns.
They are not forecasts.

Checkpoint/graph selection uses validation data, never test data.

Coefficient modes:
    varying: news-dependent edge coefficients (original model)
    fixed:   constant edge coefficients with news-dependent intercepts
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader, TensorDataset

from models.NewsStockDAG import (
    NewsStockDAG,
    is_dag,
    to_return_units,
)
from news_stock_config import MODEL_DEFAULTS, TRAIN_DEFAULTS
from news_stock_data import file_sha256, load_prepared, read_news


def cpu_state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def device_from_name(name):
    if name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable.")

    return device


def make_loaders(data, batch_size, seed):
    loaders = {}

    for split in ("train", "val", "test"):
        idx = data[f"{split}_indices"]

        dataset = TensorDataset(
            torch.from_numpy(data["news"][idx]),
            torch.from_numpy(data["Y"][idx]),
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            generator=torch.Generator().manual_seed(seed),
        )

    return loaders


@torch.no_grad()
def evaluate(model, loader, device, scale):
    model.eval()

    total_std = 0.0
    total_raw = 0.0
    zero_std = 0.0
    zero_raw = 0.0
    count = 0

    scales = torch.as_tensor(
        scale,
        dtype=torch.float32,
        device=device,
    )

    for news, y in loader:
        news = news.to(device)
        y = y.to(device)

        out = model(news, y)
        residual = out["Y_hat"] - y

        total_std += float(residual.square().sum())
        total_raw += float(
            (residual * scales).square().sum()
        )

        # Zero on the standardized scale is the training-mean baseline.
        zero_std += float(y.square().sum())
        zero_raw += float((y * scales).square().sum())

        count += y.numel()

    return {
        "conditional_reconstruction_mse_standardized":
            total_std / count,
        "conditional_reconstruction_mse_log_return":
            total_raw / count,
        "training_mean_baseline_mse_standardized":
            zero_std / count,
        "training_mean_baseline_mse_log_return":
            zero_raw / count,
    }


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    group,
    gamma,
    grad_clip,
):
    model.train()

    total = 0.0
    samples = 0

    for news, y in loader:
        news = news.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        out = model(
            news,
            y,
            lambda_group=group,
            gamma_acyclicity=gamma,
        )

        if not torch.isfinite(out["loss"]):
            raise FloatingPointError(
                "Non-finite loss; inspect feature scales "
                "and learning rates."
            )

        out["loss"].backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            grad_clip,
            error_if_nonfinite=True,
        )

        optimizer.step()
        model.dag.zero_forbidden_edges()

        total += (
            float(out["reconstruction_loss"].detach())
            * len(y)
        )
        samples += len(y)

    return total / samples


def checkpoint_payload(model, data, config, extra):
    return {
        "format_version": 1,
        "model_config": model.model_config,
        "state_dict": cpu_state(model),
        "stock_ids": data["stock_ids"].tolist(),
        "mean": torch.from_numpy(data["mean"].copy()),
        "scale": torch.from_numpy(data["scale"].copy()),
        "data_metadata": data["metadata"],
        "training_config": config,
        **extra,
    }


@torch.no_grad()
def export_graphs(
    model,
    news,
    dates,
    stock_ids,
    mean,
    scale,
    path,
    device,
    batch_size=64,
    returns=None,
    previous_dates=None,
):
    """Export coefficients of the actual fixed-support fitted model.

    No additional per-day threshold is applied to beta.
    In fixed mode, beta is identical across dates; intercepts may vary.
    """

    model.eval()

    support = (
        model.dag.structural_mask
        .detach()
        .cpu()
        .numpy()
        .astype(np.int8)
    )

    if not is_dag(torch.from_numpy(support)):
        raise ValueError(
            "Export requires a fixed acyclic structural mask."
        )

    coefficients = []
    biases = []
    contexts = []

    for start in range(0, len(news), batch_size):
        tensor = torch.from_numpy(
            news[start:start + batch_size]
        ).to(device)

        out = model(tensor)

        coefficients.append(out["beta"].cpu().numpy())
        biases.append(out["intercept"].cpu().numpy())
        contexts.append(out["context"].cpu().numpy())

    beta = np.concatenate(coefficients)
    intercept = np.concatenate(biases)

    raw_beta, raw_intercept = to_return_units(
        torch.from_numpy(beta).double(),
        torch.from_numpy(intercept).double(),
        mean,
        scale,
    )

    arrays = {
        "dates": np.asarray(dates),
        "stock_ids": np.asarray(stock_ids),
        "global_adjacency": support,
        "beta_standardized": beta,
        "intercept_standardized": intercept,
        "beta_log_return": raw_beta.numpy(),
        "intercept_log_return": raw_intercept.numpy(),
        "context": np.concatenate(contexts),
        "mean": np.asarray(mean),
        "scale": np.asarray(scale),
        "orientation": np.asarray(
            "beta[date,source,target]; source -> target"
        ),
    }

    if returns is not None:
        fitted = (
            np.einsum(
                "bi,bij->bj",
                returns,
                arrays["beta_log_return"],
            )
            + arrays["intercept_log_return"]
        )

        arrays.update(
            log_returns=returns,
            conditional_fitted_log_returns=fitted,
            residual_log_returns=returns - fitted,
        )

    if previous_dates is not None:
        arrays["previous_dates"] = previous_dates

    np.savez_compressed(path, **arrays)


def train(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            "Output directory is not empty. "
            "Choose a new run directory."
        )

    data = load_prepared(args.data)

    config = dict(TRAIN_DEFAULTS)

    for key in config:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value

    if min(
        config["stage1_epochs"],
        config["stage2_epochs"],
        config["refit_epochs"],
    ) < 1:
        raise ValueError(
            "Each stage must have at least one epoch."
        )

    if config["batch_size"] < 1 or config["refit_patience"] < 1:
        raise ValueError(
            "Batch size and patience must be positive."
        )

    for key in (
        "stage1_lr",
        "stage2_lr",
        "refit_lr",
        "grad_clip",
        "gamma_increment",
    ):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(
                f"{key} must be finite and positive."
            )

    for key in (
        "lambda_group_stage1",
        "lambda_group_stage2",
        "lambda_group_refit",
        "screening_threshold",
        "graph_threshold",
        "weight_decay",
    ):
        if not np.isfinite(config[key]) or config[key] < 0:
            raise ValueError(
                f"{key} must be finite and nonnegative."
            )

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    torch.use_deterministic_algorithms(True)

    device = device_from_name(args.device)

    model_config = dict(MODEL_DEFAULTS)
    model_config.update(
        n_stocks=len(data["stock_ids"]),
        context_dim=args.context_dim,
        hidden_dim=args.hidden_dim,
        coefficient_mode=args.coefficient_mode,
    )

    if args.hidden_dim < 1:
        raise ValueError("hidden_dim must be positive.")

    model = NewsStockDAG(**model_config).to(device)

    loaders = make_loaders(
        data,
        config["batch_size"],
        config["seed"],
    )

    args.output.mkdir(parents=True, exist_ok=True)

    run_info = {
        "model": model_config,
        "training": config,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "prepared_sha256": file_sha256(args.data),
        "data": data["metadata"],
    }

    (args.output / "run_config.json").write_text(
        json.dumps(run_info, indent=2),
        encoding="utf-8",
    )

    print(
        f"Training {len(data['stock_ids'])} nodes on {device}; "
        f"coefficient_mode={model.coefficient_mode}; "
        f"splits: {data['metadata']['splits']}",
        flush=True,
    )

    def optimizer(lr):
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=config["weight_decay"],
        )

    fields = (
        "stage",
        "epoch",
        "train_reconstruction_mse",
        "val_reconstruction_mse",
        "gamma",
        "threshold_edges",
        "threshold_is_dag",
    )

    selected = None
    best_rank = None

    with (args.output / "history.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
        )
        writer.writeheader()

        def record(stage, epoch, train_mse, gamma):
            val = evaluate(
                model,
                loaders["val"],
                device,
                data["scale"],
            )

            val_mse = val[
                "conditional_reconstruction_mse_standardized"
            ]

            if not np.isfinite(val_mse):
                raise FloatingPointError(
                    "Non-finite validation reconstruction."
                )

            graph = (
                model.dag.support_matrix().detach()
                > config["graph_threshold"]
            ).to(torch.int64)

            graph.fill_diagonal_(0)
            acyclic = is_dag(graph)

            row = dict(
                stage=stage,
                epoch=epoch,
                train_reconstruction_mse=train_mse,
                val_reconstruction_mse=val_mse,
                gamma=gamma,
                threshold_edges=int(graph.sum()),
                threshold_is_dag=acyclic,
            )

            writer.writerow(row)
            stream.flush()

            if epoch == 1 or epoch % args.log_every == 0:
                print(
                    f"{stage} {epoch}: "
                    f"train={train_mse:.6f} "
                    f"val={val_mse:.6f} "
                    f"gamma={gamma:.4g} "
                    f"edges={row['threshold_edges']} "
                    f"DAG={acyclic}",
                    flush=True,
                )

            return val_mse, acyclic

        # Stage 1: train without acyclicity penalty.
        opt = optimizer(config["stage1_lr"])

        for epoch in range(1, config["stage1_epochs"] + 1):
            loss = train_epoch(
                model,
                loaders["train"],
                opt,
                device,
                config["lambda_group_stage1"],
                0.0,
                config["grad_clip"],
            )

            record("screen", epoch, loss, 0.0)

        # Screen from trained coefficients.
        with torch.no_grad():
            mask = (
                model.dag.support_matrix()
                > config["screening_threshold"]
            ).float()

            mask.fill_diagonal_(0)
            model.dag.set_structural_mask(mask)

        if int(mask.sum()) == 0:
            print(
                "Screening retained no edges; subsequent fit "
                "is an intercept-only model.",
                flush=True,
            )

        # Stage 2: increase the acyclicity penalty.
        opt = optimizer(config["stage2_lr"])

        for epoch in range(1, config["stage2_epochs"] + 1):
            gamma = config["gamma_increment"] * epoch

            loss = train_epoch(
                model,
                loaders["train"],
                opt,
                device,
                config["lambda_group_stage2"],
                gamma,
                config["grad_clip"],
            )

            val_mse, acyclic = record(
                "acyclic",
                epoch,
                loss,
                gamma,
            )

            # Prefer threshold-feasible candidates, then validation MSE.
            # This does not assert acyclicity of the raw continuous graph.
            rank = (0 if acyclic else 1, val_mse)

            if best_rank is None or rank < best_rank:
                best_rank = rank
                selected = {
                    "state": cpu_state(model),
                    "epoch": epoch,
                    "val_mse": val_mse,
                    "threshold_is_dag": acyclic,
                }

        model.load_state_dict(selected["state"])

        selected_info = {
            key: value
            for key, value in selected.items()
            if key != "state"
        }

        torch.save(
            checkpoint_payload(
                model,
                data,
                config,
                {
                    "fixed_dag": False,
                    "selection": selected_info,
                },
            ),
            args.output / "selected_stage2.pt",
        )

        graph_info = model.freeze_graph(
            config["graph_threshold"]
        )

        # Refit using exactly the support that will be exported.
        initial_val = evaluate(
            model,
            loaders["val"],
            device,
            data["scale"],
        )["conditional_reconstruction_mse_standardized"]

        best_val = initial_val
        best_state = cpu_state(model)
        best_epoch = 0
        stale = 0
        epochs_run = 0

        opt = optimizer(config["refit_lr"])

        for epoch in range(1, config["refit_epochs"] + 1):
            loss = train_epoch(
                model,
                loaders["train"],
                opt,
                device,
                config["lambda_group_refit"],
                0.0,
                config["grad_clip"],
            )

            val_mse, _ = record(
                "fixed_graph_refit",
                epoch,
                loss,
                0.0,
            )

            epochs_run = epoch

            if val_mse < best_val:
                best_val = val_mse
                best_state = cpu_state(model)
                best_epoch = epoch
                stale = 0
            else:
                stale += 1

            if stale >= config["refit_patience"]:
                break

        model.load_state_dict(best_state)

    assert is_dag(model.dag.structural_mask)

    graph_info.update(
        refit_selected_epoch=best_epoch,
        refit_epochs_run=epochs_run,
        projected_initial_val_mse=initial_val,
        refit_best_val_mse=best_val,
    )

    metrics = {
        split: evaluate(
            model,
            loader,
            device,
            data["scale"],
        )
        for split, loader in loaders.items()
    }

    report = {
        "coefficient_mode": model.coefficient_mode,
        "graph": graph_info,
        "stage2_selection": selected_info,
        "metrics": metrics,
        "interpretation": (
            "Conditional same-day reconstruction; not forecasting, "
            "causal identification, or confidence intervals."
        ),
    }

    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    torch.save(
        checkpoint_payload(
            model,
            data,
            config,
            {
                "fixed_dag": True,
                "report": report,
                "prepared_sha256": run_info["prepared_sha256"],
            },
        ),
        args.output / "model.pt",
    )

    with (args.output / "global_edges.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.writer(stream)

        strength_name = (
            "absolute_coefficient"
            if model.coefficient_mode == "fixed"
            else "coefficient_basis_norm"
        )

        writer.writerow(
            ("source", "target", strength_name)
        )

        support = (
            model.dag.support_matrix()
            .detach()
            .cpu()
            .numpy()
        )

        structural_mask = (
            model.dag.structural_mask
            .detach()
            .cpu()
            .numpy()
        )

        for i, j in np.argwhere(structural_mask != 0):
            writer.writerow(
                (
                    data["stock_ids"][i],
                    data["stock_ids"][j],
                    float(support[i, j]),
                )
            )

    idx = data["test_indices"]

    export_graphs(
        model,
        data["news"][idx],
        data["dates"][idx],
        data["stock_ids"],
        data["mean"],
        data["scale"],
        args.output / "test_graphs.npz",
        device,
        config["batch_size"],
        data["log_returns"][idx],
        data["previous_dates"][idx],
    )

    print(json.dumps(report, indent=2), flush=True)

    if graph_info["selected_edges"] == 0:
        print(
            "Final graph is empty. Review validation behavior, "
            "regularization, and graph threshold; successful "
            "execution does not establish graph recovery.",
            flush=True,
        )

    print(
        f"Saved final fixed-DAG model and test graphs "
        f"to {args.output}",
        flush=True,
    )


def infer(args):
    if args.output.exists():
        raise FileExistsError(
            "Inference output exists; choose a new path."
        )

    if args.output.suffix != ".npz":
        raise ValueError(
            "Inference output must have the .npz extension."
        )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    if (
        checkpoint.get("format_version") != 1
        or not checkpoint.get("fixed_dag")
    ):
        raise ValueError(
            "Use the final fixed-DAG model.pt, "
            "not selected_stage2.pt."
        )

    device = device_from_name(args.device)

    # The saved model_config determines fixed/varying mode.
    # Older checkpoints without coefficient_mode default to varying.
    model = NewsStockDAG(
        **checkpoint["model_config"]
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])

    dates, news = read_news(
        args.news,
        args.news_sheet,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    export_graphs(
        model,
        news,
        dates,
        checkpoint["stock_ids"],
        checkpoint["mean"].numpy(),
        checkpoint["scale"].numpy(),
        args.output,
        device,
        args.batch_size,
    )

    print(
        f"Saved {len(dates)} daily coefficient matrices "
        f"to {args.output}. "
        "No stock outcomes were used for graph inference.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    train_parser = sub.add_parser("train")

    train_parser.add_argument(
        "--data",
        type=Path,
        required=True,
    )
    train_parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    train_parser.add_argument(
        "--context-dim",
        type=int,
        choices=(16, 32),
        default=16,
    )
    train_parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
    )
    train_parser.add_argument(
        "--log-every",
        type=int,
        default=10,
    )
    train_parser.add_argument(
        "--coefficient-mode",
        choices=("varying", "fixed"),
        default="varying",
        help=(
            "Varying edges (original) or fixed edges; "
            "both retain news-dependent intercepts."
        ),
    )

    for key, default in TRAIN_DEFAULTS.items():
        train_parser.add_argument(
            "--" + key.replace("_", "-"),
            type=type(default),
            default=default,
        )

    inference_parser = sub.add_parser("infer")

    inference_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    inference_parser.add_argument(
        "--news",
        type=Path,
        required=True,
    )
    inference_parser.add_argument("--news-sheet")
    inference_parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    inference_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    for command_parser in (
        train_parser,
        inference_parser,
    ):
        command_parser.add_argument(
            "--device",
            default="auto",
        )
        command_parser.add_argument(
            "--threads",
            type=int,
            default=2,
        )

    args = parser.parse_args()

    if (
        args.threads < 1
        or args.batch_size < 1
        or getattr(args, "log_every", 1) < 1
    ):
        parser.error(
            "threads, batch-size, and log-every must be positive."
        )

    torch.set_num_threads(args.threads)

    if args.command == "train":
        train(args)
    else:
        infer(args)


if __name__ == "__main__":
    main()

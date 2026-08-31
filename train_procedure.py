# train.py
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluate import (
    evaluate_validation,
    extract_raw_adjacency,
    is_dag,
)


@dataclass
class TrainConfig:
    # Stage I: sparse pre-selection / warm-up
    stage1_epochs: int = 300
    stage1_lr: float = 2e-3
    stage1_batch_size: int = 256
    lambda_group_stage1: float = 1e-3

    # Optional hard candidate screening after Stage I
    use_screening: bool = False
    screening_threshold: float = 1e-3

    # Stage II: DAG-constrained refinement
    stage2_epochs: int = 1200
    stage2_lr: float = 1e-3
    stage2_batch_size: int = 256
    lambda_group_stage2: float = 1e-3

    # SDCD-style increasing spectral penalty
    gamma_increment: float = 5e-3
    gamma_schedule: str = "linear"  # "linear", "power_2", "exponential"
    freeze_gamma_at_dag: bool = True

    # A stricter/smaller threshold is used to decide when gamma can be frozen.
    freeze_gamma_threshold: float = 0.01
    edge_threshold: float = 0.1
    gamma_warmup_epochs: int = 2

    # Validation / early stopping
    val_every: int = 20
    early_stopping: bool = True
    patience: int = 10

    # Optimization
    grad_clip: Optional[float] = 5.0

    # Misc
    seed: int = 0
    verbose_every: int = 50


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loader(
    z: torch.Tensor,
    Y: torch.Tensor,
    mask: Optional[torch.Tensor],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    if mask is None:
        mask = torch.ones_like(Y)

    if len(z) != len(Y) or mask.shape != Y.shape:
        raise ValueError("z, Y, and mask shapes are inconsistent.")

    dataset = TensorDataset(z, Y, mask)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def build_optimizer(model, lr: float) -> torch.optim.Optimizer:
    # A fresh optimizer is created for each stage so Adam moments are not
    # carried across a change in objective / structural mask.
    return torch.optim.Adam(model.parameters(), lr=lr)


def _gamma_value(
    index: int,
    increment: float,
    schedule: str,
) -> float:
    x = max(index, 0) * increment

    if schedule == "linear":
        return x
    if schedule.startswith("power_"):
        power = float(schedule.split("_", 1)[1])
        return x ** power
    if schedule == "exponential":
        # Starts at zero and then grows smoothly.
        return float(np.expm1(x))

    raise ValueError(
        f"Unknown gamma_schedule={schedule!r}. "
        "Use 'linear', 'power_2', or 'exponential'."
    )


def _single_batch_step(
    model,
    optimizer,
    z_b,
    Y_b,
    mask_b,
    lambda_group: float,
    gamma_acyc: float,
    grad_clip: Optional[float],
) -> Dict[str, float]:
    optimizer.zero_grad(set_to_none=True)

    loss, details = model.loss(
        z=z_b,
        Y=Y_b,
        observational_mask=mask_b,
        lambda_group=lambda_group,
        gamma_acyc=gamma_acyc,
    )

    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite loss encountered.")

    loss.backward()

    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=grad_clip,
        )

    optimizer.step()

    # Keep masked/self-loop alpha exactly zero.
    with torch.no_grad():
        model.alpha.mul_(model.structural_mask[:, :, None])

    return {
        key: float(value.detach().cpu().item())
        for key, value in details.items()
    }


def _train_epoch(
    model,
    loader,
    optimizer,
    device,
    lambda_group: float,
    gamma_acyc: float,
    grad_clip: Optional[float],
) -> Dict[str, float]:
    model.train()

    sums = {"total": 0.0, "recon": 0.0, "group": 0.0, "dag": 0.0}
    n_batches = 0

    for z_b, Y_b, mask_b in loader:
        z_b = z_b.to(device)
        Y_b = Y_b.to(device)
        mask_b = mask_b.to(device)

        stats = _single_batch_step(
            model=model,
            optimizer=optimizer,
            z_b=z_b,
            Y_b=Y_b,
            mask_b=mask_b,
            lambda_group=lambda_group,
            gamma_acyc=gamma_acyc,
            grad_clip=grad_clip,
        )

        for key in sums:
            sums[key] += stats[key]
        n_batches += 1

    return {key: value / max(n_batches, 1) for key, value in sums.items()}


def build_screening_mask(
    model,
    threshold: float,
) -> torch.Tensor:
    with torch.no_grad():
        S = model.support_matrix()
        mask = (S > threshold).to(dtype=S.dtype)
        mask = mask * model.offdiag_mask
    return mask


def train_stage1(
    model,
    train_z,
    train_Y,
    train_mask,
    config: TrainConfig,
) -> List[Dict[str, Any]]:
    device = next(model.parameters()).device
    loader = _make_loader(
        train_z,
        train_Y,
        train_mask,
        batch_size=config.stage1_batch_size,
        shuffle=True,
    )
    optimizer = build_optimizer(model, lr=config.stage1_lr)

    history: List[Dict[str, Any]] = []

    print(
        "\n"
        "====================================\n"
        "Stage I: sparse pre-selection\n"
        "===================================="
    )

    for epoch in range(config.stage1_epochs):
        stats = _train_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            lambda_group=config.lambda_group_stage1,
            gamma_acyc=0.0,
            grad_clip=config.grad_clip,
        )

        record = {
            "stage": 1,
            "epoch": epoch,
            "gamma": 0.0,
            **stats,
        }
        history.append(record)

        if (
            epoch % config.verbose_every == 0
            or epoch == config.stage1_epochs - 1
        ):
            rho = float(model.spectral_radius_estimate().detach().cpu().item())
            print(
                f"[Stage I] epoch={epoch:4d} | "
                f"recon={stats['recon']:.6f} | "
                f"group={stats['group']:.6f} | "
                f"rho~={rho:.3e}"
            )

    return history


def train_stage2(
    model,
    train_z,
    train_Y,
    train_mask,
    val_z,
    val_Y,
    val_mask,
    config: TrainConfig,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if config.freeze_gamma_threshold > config.edge_threshold:
        raise ValueError(
            "freeze_gamma_threshold should be <= edge_threshold."
        )

    device = next(model.parameters()).device

    loader = _make_loader(
        train_z,
        train_Y,
        train_mask,
        batch_size=config.stage2_batch_size,
        shuffle=True,
    )

    # Fresh Adam state for Stage II.
    optimizer = build_optimizer(model, lr=config.stage2_lr)

    history: List[Dict[str, Any]] = []

    gamma_index = 0
    gamma_cap: Optional[float] = None

    best_feasible_state = None
    best_feasible_val = float("inf")
    best_feasible_epoch = None

    best_any_state = None
    best_any_val = float("inf")
    best_any_epoch = None

    patience_counter = 0

    print(
        "\n"
        "====================================\n"
        "Stage II: spectral DAG refinement\n"
        "===================================="
    )

    for epoch in range(config.stage2_epochs):
        if gamma_cap is None:
            gamma = _gamma_value(
                gamma_index,
                config.gamma_increment,
                config.gamma_schedule,
            )
            gamma_index += 1
        else:
            gamma = gamma_cap

        stats = _train_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            lambda_group=config.lambda_group_stage2,
            gamma_acyc=gamma,
            grad_clip=config.grad_clip,
        )

        record: Dict[str, Any] = {
            "stage": 2,
            "epoch": epoch,
            "gamma": gamma,
            **stats,
        }

        do_val = (
            epoch % config.val_every == 0
            or epoch == config.stage2_epochs - 1
        )

        if do_val:
            val_metrics = evaluate_validation(
                model=model,
                val_z=val_z,
                val_Y=val_Y,
                val_mask=val_mask,
                edge_threshold=config.edge_threshold,
                batch_size=max(config.stage2_batch_size, 512),
                verbose=False,
            )

            S = val_metrics["support"]

            raw_freeze_adj = extract_raw_adjacency(
                S,
                threshold=config.freeze_gamma_threshold,
            )
            raw_main_adj = extract_raw_adjacency(
                S,
                threshold=config.edge_threshold,
            )

            freeze_graph_is_dag = is_dag(raw_freeze_adj)
            main_graph_is_dag = is_dag(raw_main_adj)

            # SDCD-style gamma freezing:
            # once a stricter/lower-threshold raw graph is DAG, stop increasing
            # gamma. If the main-threshold raw graph later becomes cyclic,
            # resume increasing gamma.
            if (
                config.freeze_gamma_at_dag
                and epoch > config.gamma_warmup_epochs
                and gamma_cap is None
                and freeze_graph_is_dag
            ):
                gamma_cap = gamma

            elif (
                config.freeze_gamma_at_dag
                and gamma_cap is not None
                and not main_graph_is_dag
            ):
                gamma_cap = None
                patience_counter = 0

            val_loss = val_metrics["mse"]

            if val_loss < best_any_val:
                best_any_val = val_loss
                best_any_state = copy.deepcopy(model.state_dict())
                best_any_epoch = epoch

            # Feasible checkpoint means the *raw* thresholded graph is already a DAG.
            if main_graph_is_dag and val_loss < best_feasible_val:
                best_feasible_val = val_loss
                best_feasible_state = copy.deepcopy(model.state_dict())
                best_feasible_epoch = epoch
                patience_counter = 0
            elif main_graph_is_dag and (
                not config.freeze_gamma_at_dag or gamma_cap is not None
            ):
                patience_counter += 1

            record.update(
                {
                    "val_mse": val_metrics["mse"],
                    "val_rmse": val_metrics["rmse"],
                    "val_r2_macro": val_metrics["r2_macro"],
                    "rho_est": val_metrics["spectral_radius"],
                    "raw_main_is_dag": main_graph_is_dag,
                    "raw_freeze_is_dag": freeze_graph_is_dag,
                    "gamma_frozen": gamma_cap is not None,
                }
            )

            if (
                config.early_stopping
                and best_feasible_state is not None
                and (
                    not config.freeze_gamma_at_dag
                    or gamma_cap is not None
                )
                and patience_counter >= config.patience
            ):
                history.append(record)
                print(
                    f"Early stopping at Stage-II epoch {epoch}; "
                    f"best feasible epoch={best_feasible_epoch}."
                )
                break

        history.append(record)

        if (
            epoch % config.verbose_every == 0
            or epoch == config.stage2_epochs - 1
        ):
            rho = float(model.spectral_radius_estimate().detach().cpu().item())
            msg = (
                f"[Stage II] epoch={epoch:4d} | "
                f"recon={stats['recon']:.6f} | "
                f"group={stats['group']:.6f} | "
                f"dag={stats['dag']:.6f} | "
                f"gamma={gamma:.3e} | "
                f"rho~={rho:.3e}"
            )
            if gamma_cap is not None:
                msg += " | gamma=frozen"
            print(msg)

    # Restore best validation checkpoint.
    if best_feasible_state is not None:
        model.load_state_dict(best_feasible_state)
        selected = {
            "checkpoint_type": "feasible",
            "best_val_mse": best_feasible_val,
            "best_epoch": best_feasible_epoch,
        }
    elif best_any_state is not None:
        model.load_state_dict(best_any_state)
        selected = {
            "checkpoint_type": "fallback_nonfeasible",
            "best_val_mse": best_any_val,
            "best_epoch": best_any_epoch,
        }
        print(
            "WARNING: no validation checkpoint had a raw thresholded DAG. "
            "The returned discrete graph should therefore use cycle-safe extraction."
        )
    else:
        selected = {
            "checkpoint_type": "final",
            "best_val_mse": float("nan"),
            "best_epoch": config.stage2_epochs - 1,
        }

    return history, selected


def train_model(
    model,
    train_z: torch.Tensor,
    train_Y: torch.Tensor,
    train_mask: Optional[torch.Tensor],
    val_z: torch.Tensor,
    val_Y: torch.Tensor,
    val_mask: Optional[torch.Tensor],
    config: Optional[TrainConfig] = None,
) -> Dict[str, Any]:
    """
    Train on train split, choose checkpoint using validation split.

    Test data is intentionally NOT accepted here to prevent leakage.
    """
    if config is None:
        config = TrainConfig()

    set_seed(config.seed)

    model.reset_edge_mask()

    stage1_history = train_stage1(
        model=model,
        train_z=train_z,
        train_Y=train_Y,
        train_mask=train_mask,
        config=config,
    )

    screening_mask = None

    if config.use_screening:
        screening_mask = build_screening_mask(
            model,
            threshold=config.screening_threshold,
        )

        n_edges = int(screening_mask.sum().item())
        if n_edges == 0:
            raise RuntimeError(
                "Stage-I screening removed every candidate edge. "
                "Lower screening_threshold."
            )

        model.set_edge_mask(screening_mask)

        print(
            f"\nStage-I screening retained {n_edges} candidate directed edges."
        )

    stage2_history, checkpoint = train_stage2(
        model=model,
        train_z=train_z,
        train_Y=train_Y,
        train_mask=train_mask,
        val_z=val_z,
        val_Y=val_Y,
        val_mask=val_mask,
        config=config,
    )

    final_val = evaluate_validation(
        model=model,
        val_z=val_z,
        val_Y=val_Y,
        val_mask=val_mask,
        edge_threshold=config.edge_threshold,
        batch_size=max(config.stage2_batch_size, 512),
        verbose=True,
    )

    return {
        "model": model,
        "history": stage1_history + stage2_history,
        "checkpoint": checkpoint,
        "validation": final_val,
        "screening_mask": (
            None if screening_mask is None else screening_mask.detach().cpu()
        ),
    }

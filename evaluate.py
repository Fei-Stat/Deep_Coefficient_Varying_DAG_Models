# evaluate.py
from __future__ import annotations

from typing import Optional, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def _as_cpu_int_adjacency(adjacency: torch.Tensor) -> np.ndarray:
    A = adjacency.detach().cpu().numpy()
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be a square 2D matrix.")
    return (A != 0).astype(np.int64)


def is_dag(adjacency: torch.Tensor) -> bool:
    """
    Exact DAG check using Kahn's topological-sort algorithm.

    Convention: adjacency[i, j] = 1 means i -> j.
    """
    A = _as_cpu_int_adjacency(adjacency)
    p = A.shape[0]

    indegree = A.sum(axis=0).astype(np.int64)
    stack = [j for j in range(p) if indegree[j] == 0]
    visited = 0

    while stack:
        node = stack.pop()
        visited += 1
        children = np.where(A[node] != 0)[0]

        for child in children:
            indegree[child] -= 1
            if indegree[child] == 0:
                stack.append(int(child))

    return visited == p


def _has_path(A: np.ndarray, source: int, target: int) -> bool:
    """
    Return True if the current directed graph has source -> ... -> target.
    """
    if source == target:
        return True

    p = A.shape[0]
    seen = np.zeros(p, dtype=bool)
    stack = [source]
    seen[source] = True

    while stack:
        node = stack.pop()
        for nxt in np.where(A[node] != 0)[0]:
            nxt = int(nxt)
            if nxt == target:
                return True
            if not seen[nxt]:
                seen[nxt] = True
                stack.append(nxt)

    return False


def extract_raw_adjacency(
    support: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """
    Plain thresholding. This may be cyclic and is useful for diagnostics.
    """
    if threshold < 0:
        raise ValueError("threshold must be nonnegative.")

    adjacency = (support.detach().cpu() > threshold).to(torch.int64)
    adjacency.fill_diagonal_(0)
    return adjacency


def extract_dag_adjacency(
    support: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """
    Cycle-safe graph extraction inspired by SDCD.

    Candidate edges are considered from strongest to weakest. An edge i -> j
    is added only if its weight exceeds threshold and adding it would not
    create a path j -> ... -> i, i.e. a directed cycle.
    """
    if threshold < 0:
        raise ValueError("threshold must be nonnegative.")

    S = support.detach().cpu().numpy()
    p = S.shape[0]
    if S.ndim != 2 or S.shape[1] != p:
        raise ValueError("support must be square.")

    candidates = [
        (float(S[i, j]), i, j)
        for i in range(p)
        for j in range(p)
        if i != j and S[i, j] > threshold
    ]
    candidates.sort(reverse=True, key=lambda t: t[0])

    A = np.zeros((p, p), dtype=np.int64)

    for weight, i, j in candidates:
        # i -> j creates a cycle iff j already reaches i.
        if _has_path(A, j, i):
            continue
        A[i, j] = 1

    return torch.from_numpy(A)


def graph_summary(
    support: torch.Tensor,
    threshold: float,
) -> Dict[str, Any]:
    raw_adj = extract_raw_adjacency(support, threshold)
    dag_adj = extract_dag_adjacency(support, threshold)

    p = support.shape[0]
    max_edges = p * (p - 1)

    selected = support.detach().cpu()[dag_adj.bool()]

    return {
        "n_nodes": p,
        "threshold": float(threshold),
        "raw_n_edges": int(raw_adj.sum().item()),
        "raw_is_dag": is_dag(raw_adj),
        "n_edges": int(dag_adj.sum().item()),
        "density": (
            float(dag_adj.sum().item()) / max_edges if max_edges > 0 else 0.0
        ),
        "is_dag": is_dag(dag_adj),
        "mean_selected_strength": (
            float(selected.mean().item()) if selected.numel() else 0.0
        ),
        "min_selected_strength": (
            float(selected.min().item()) if selected.numel() else 0.0
        ),
        "max_selected_strength": (
            float(selected.max().item()) if selected.numel() else 0.0
        ),
        "raw_adjacency": raw_adj,
        "adjacency": dag_adj,
    }


@torch.no_grad()
def _evaluate_batches(
    model,
    z: torch.Tensor,
    Y: torch.Tensor,
    observational_mask: Optional[torch.Tensor],
    batch_size: int,
    return_tensors: bool,
) -> Dict[str, Any]:
    device = next(model.parameters()).device

    if observational_mask is None:
        mask = torch.ones_like(Y)
    else:
        if observational_mask.shape != Y.shape:
            raise ValueError(
                "observational_mask must have same shape as Y; "
                f"got {observational_mask.shape} vs {Y.shape}."
            )
        mask = observational_mask

    dataset = TensorDataset(z, Y, mask)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    p = Y.shape[1]

    total_n = 0.0
    total_sq = 0.0
    total_abs = 0.0

    node_n = torch.zeros(p, dtype=torch.float64)
    node_sq = torch.zeros(p, dtype=torch.float64)
    node_abs = torch.zeros(p, dtype=torch.float64)
    node_y_sum = torch.zeros(p, dtype=torch.float64)
    node_y2_sum = torch.zeros(p, dtype=torch.float64)

    pred_chunks: List[torch.Tensor] = []
    target_chunks: List[torch.Tensor] = []
    beta_chunks: List[torch.Tensor] = []
    context_chunks: List[torch.Tensor] = []
    mask_chunks: List[torch.Tensor] = []

    model.eval()

    for z_b, Y_b, m_b in loader:
        z_b = z_b.to(device)
        Y_b = Y_b.to(device)
        m_b = m_b.to(device=device, dtype=Y_b.dtype)

        out = model(z_b, Y_b)
        pred = out["Y_hat"]

        residual = Y_b - pred
        sq = residual.pow(2)
        ab = residual.abs()

        total_n += float(m_b.sum().item())
        total_sq += float((sq * m_b).sum().item())
        total_abs += float((ab * m_b).sum().item())

        node_n += m_b.sum(dim=0).detach().cpu().double()
        node_sq += (sq * m_b).sum(dim=0).detach().cpu().double()
        node_abs += (ab * m_b).sum(dim=0).detach().cpu().double()
        node_y_sum += (Y_b * m_b).sum(dim=0).detach().cpu().double()
        node_y2_sum += (Y_b.pow(2) * m_b).sum(dim=0).detach().cpu().double()

        if return_tensors:
            pred_chunks.append(pred.detach().cpu())
            target_chunks.append(Y_b.detach().cpu())
            beta_chunks.append(out["beta"].detach().cpu())
            context_chunks.append(out["x"].detach().cpu())
            mask_chunks.append(m_b.detach().cpu())

    if total_n <= 0:
        raise ValueError("No observational entries available for evaluation.")

    mse = total_sq / total_n
    rmse = mse ** 0.5
    mae = total_abs / total_n

    nodewise: Dict[int, Dict[str, float]] = {}
    valid_r2 = []

    for j in range(p):
        n_j = float(node_n[j].item())

        if n_j <= 0:
            nodewise[j] = {
                "mse": float("nan"),
                "rmse": float("nan"),
                "mae": float("nan"),
                "r2": float("nan"),
                "n_observed": 0,
            }
            continue

        mse_j = float(node_sq[j].item()) / n_j
        mae_j = float(node_abs[j].item()) / n_j
        mean_j = float(node_y_sum[j].item()) / n_j

        ss_tot_j = (
            float(node_y2_sum[j].item())
            - 2.0 * mean_j * float(node_y_sum[j].item())
            + n_j * mean_j * mean_j
        )
        ss_res_j = float(node_sq[j].item())

        if ss_tot_j <= 1e-12:
            r2_j = float("nan")
        else:
            r2_j = 1.0 - ss_res_j / ss_tot_j
            valid_r2.append(r2_j)

        nodewise[j] = {
            "mse": mse_j,
            "rmse": mse_j ** 0.5,
            "mae": mae_j,
            "r2": r2_j,
            "n_observed": int(n_j),
        }

    result = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2_macro": (
            float(np.mean(valid_r2)) if valid_r2 else float("nan")
        ),
        "n_observed": int(total_n),
        "nodewise": nodewise,
    }

    if return_tensors:
        result.update(
            {
                "prediction": torch.cat(pred_chunks, dim=0),
                "target": torch.cat(target_chunks, dim=0),
                "beta": torch.cat(beta_chunks, dim=0),
                "context": torch.cat(context_chunks, dim=0),
                "observational_mask": torch.cat(mask_chunks, dim=0),
            }
        )

    return result


@torch.no_grad()
def evaluate_model(
    model,
    z: torch.Tensor,
    Y: torch.Tensor,
    observational_mask: Optional[torch.Tensor] = None,
    edge_threshold: float = 0.1,
    batch_size: int = 512,
    return_tensors: bool = False,
    exact_eig_max_nodes: int = 256,
) -> Dict[str, Any]:
    """
    Evaluate prediction and learned graph without changing model parameters.
    """
    prediction = _evaluate_batches(
        model=model,
        z=z,
        Y=Y,
        observational_mask=observational_mask,
        batch_size=batch_size,
        return_tensors=return_tensors,
    )

    support = model.support_matrix().detach().cpu()
    graph = graph_summary(support, threshold=edge_threshold)

    rho_est = float(model.spectral_radius_estimate().detach().cpu().item())

    if model.n_nodes <= exact_eig_max_nodes:
        rho_exact = float(model.exact_spectral_radius().detach().cpu().item())
    else:
        rho_exact = None

    result: Dict[str, Any] = {
        **prediction,
        "recon_loss": prediction["mse"],
        "group_penalty": float(support.sum().item()),
        "spectral_radius": rho_est,
        "spectral_radius_exact": rho_exact,
        "support": support,
        "raw_adjacency": graph["raw_adjacency"],
        "adjacency": graph["adjacency"],
        "graph": {
            k: v
            for k, v in graph.items()
            if k not in {"raw_adjacency", "adjacency"}
        },
    }

    return result


def print_evaluation(
    metrics: Dict[str, Any],
    split_name: str = "Validation",
    print_nodewise: bool = False,
) -> None:
    g = metrics["graph"]

    print(
        "\n"
        "========================================\n"
        f"{split_name} results\n"
        "========================================"
    )
    print(f"MSE                 : {metrics['mse']:.6f}")
    print(f"RMSE                : {metrics['rmse']:.6f}")
    print(f"MAE                 : {metrics['mae']:.6f}")
    print(f"Macro R^2           : {metrics['r2_macro']:.6f}")
    print(f"Observed entries    : {metrics['n_observed']}")

    print("\nGraph statistics")
    print("----------------------------------------")
    print(f"Spectral radius (PI): {metrics['spectral_radius']:.3e}")
    if metrics["spectral_radius_exact"] is not None:
        print(f"Spectral radius exact: {metrics['spectral_radius_exact']:.3e}")
    print(f"Raw selected edges  : {g['raw_n_edges']}")
    print(f"Raw graph is DAG    : {g['raw_is_dag']}")
    print(f"Cycle-safe edges    : {g['n_edges']}")
    print(f"Cycle-safe DAG      : {g['is_dag']}")
    print(f"Graph density       : {g['density']:.4f}")

    if print_nodewise:
        print("\nNodewise performance")
        print("----------------------------------------")
        for node, stat in metrics["nodewise"].items():
            print(
                f"Node {node:3d} | "
                f"MSE={stat['mse']:.6f} | "
                f"RMSE={stat['rmse']:.6f} | "
                f"MAE={stat['mae']:.6f} | "
                f"R2={stat['r2']:.4f} | "
                f"n={stat['n_observed']}"
            )


def evaluate_validation(
    model,
    val_z: torch.Tensor,
    val_Y: torch.Tensor,
    val_mask: Optional[torch.Tensor] = None,
    edge_threshold: float = 0.1,
    batch_size: int = 512,
    verbose: bool = False,
) -> Dict[str, Any]:
    metrics = evaluate_model(
        model=model,
        z=val_z,
        Y=val_Y,
        observational_mask=val_mask,
        edge_threshold=edge_threshold,
        batch_size=batch_size,
        return_tensors=False,
    )

    if verbose:
        print_evaluation(metrics, split_name="Validation", print_nodewise=False)

    return metrics


def evaluate_test(
    model,
    test_z: torch.Tensor,
    test_Y: torch.Tensor,
    test_mask: Optional[torch.Tensor] = None,
    edge_threshold: float = 0.1,
    batch_size: int = 512,
    verbose: bool = True,
    return_tensors: bool = True,
) -> Dict[str, Any]:
    metrics = evaluate_model(
        model=model,
        z=test_z,
        Y=test_Y,
        observational_mask=test_mask,
        edge_threshold=edge_threshold,
        batch_size=batch_size,
        return_tensors=return_tensors,
    )

    if verbose:
        print_evaluation(metrics, split_name="Test", print_nodewise=True)

    return metrics

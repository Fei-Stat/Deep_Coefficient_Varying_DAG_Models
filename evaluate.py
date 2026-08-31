# evaluate.py

from typing import Optional, Dict, Any

import numpy as np
import torch


# ============================================================
# Basic utilities
# ============================================================

def get_model_device(model):
    """
    Return the device on which model parameters are stored.
    """
    return next(model.parameters()).device


def move_optional_tensor(
    tensor: Optional[torch.Tensor],
    device: torch.device
):
    """
    Move tensor to device if it is not None.
    """
    if tensor is None:
        return None

    return tensor.to(device)


def prepare_observational_mask(
    Y: torch.Tensor,
    observational_mask: Optional[torch.Tensor]
):
    """
    Construct a float mask with the same shape as Y.

    mask[k, j] = 1:
        sample k is observational for node j.

    mask[k, j] = 0:
        node j is intervened for sample k.

    If observational_mask is None, all entries are treated
    as observational.
    """

    if observational_mask is None:

        mask = torch.ones_like(
            Y,
            dtype=Y.dtype
        )

    else:

        if observational_mask.shape != Y.shape:
            raise ValueError(
                "observational_mask must have the same shape as Y. "
                f"Got mask={observational_mask.shape}, Y={Y.shape}."
            )

        mask = observational_mask.to(
            device=Y.device,
            dtype=Y.dtype
        )

    return mask


# ============================================================
# Overall masked regression metrics
# ============================================================

def masked_regression_metrics(
    Y_true: torch.Tensor,
    Y_pred: torch.Tensor,
    observational_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-12
) -> Dict[str, float]:
    """
    Compute regression metrics using only observational entries.

    Returns
    -------
    mse
    rmse
    mae
    r2
    n_observed
    """

    if Y_true.shape != Y_pred.shape:
        raise ValueError(
            "Y_true and Y_pred must have the same shape. "
            f"Got {Y_true.shape} and {Y_pred.shape}."
        )

    mask = prepare_observational_mask(
        Y_true,
        observational_mask
    )

    n_observed = mask.sum()

    if n_observed.item() <= 0:
        raise ValueError(
            "No observational entries are available for evaluation."
        )

    residual = Y_true - Y_pred

    # --------------------------------------------------------
    # MSE
    # --------------------------------------------------------

    squared_error = residual.pow(2)

    mse = (
        squared_error * mask
    ).sum() / n_observed

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    rmse = torch.sqrt(
        mse.clamp_min(0.0)
    )

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae = (
        residual.abs() * mask
    ).sum() / n_observed

    # --------------------------------------------------------
    # Global masked R^2
    # --------------------------------------------------------

    target_mean = (
        Y_true * mask
    ).sum() / n_observed

    ss_res = (
        squared_error * mask
    ).sum()

    ss_tot = (
        ((Y_true - target_mean) ** 2)
        * mask
    ).sum()

    if ss_tot.item() <= eps:
        r2 = torch.tensor(
            float("nan"),
            device=Y_true.device
        )
    else:
        r2 = 1.0 - ss_res / ss_tot

    return {
        "mse": float(mse.detach().cpu()),
        "rmse": float(rmse.detach().cpu()),
        "mae": float(mae.detach().cpu()),
        "r2": float(r2.detach().cpu()),
        "n_observed": int(
            n_observed.detach().cpu().item()
        )
    }


# ============================================================
# Nodewise evaluation
# ============================================================

def nodewise_regression_metrics(
    Y_true: torch.Tensor,
    Y_pred: torch.Tensor,
    observational_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-12
) -> Dict[int, Dict[str, float]]:
    """
    Compute MSE / RMSE / MAE / R^2 separately for every node.

    Only observational samples for each node are used.

    Returns
    -------
    {
        0: {
            "mse": ...,
            "rmse": ...,
            "mae": ...,
            "r2": ...,
            "n_observed": ...
        },
        1: {...},
        ...
    }
    """

    mask = prepare_observational_mask(
        Y_true,
        observational_mask
    )

    p = Y_true.shape[1]

    results = {}

    for j in range(p):

        node_mask = mask[:, j] > 0

        n_j = int(
            node_mask.sum().item()
        )

        if n_j == 0:

            results[j] = {
                "mse": float("nan"),
                "rmse": float("nan"),
                "mae": float("nan"),
                "r2": float("nan"),
                "n_observed": 0
            }

            continue

        y_j = Y_true[node_mask, j]
        y_hat_j = Y_pred[node_mask, j]

        residual = y_j - y_hat_j

        mse = residual.pow(2).mean()
        rmse = torch.sqrt(
            mse.clamp_min(0.0)
        )
        mae = residual.abs().mean()

        target_mean = y_j.mean()

        ss_res = residual.pow(2).sum()

        ss_tot = (
            (y_j - target_mean) ** 2
        ).sum()

        if ss_tot.item() <= eps:

            r2 = torch.tensor(
                float("nan"),
                device=Y_true.device
            )

        else:

            r2 = (
                1.0
                - ss_res / ss_tot
            )

        results[j] = {
            "mse": float(
                mse.detach().cpu()
            ),
            "rmse": float(
                rmse.detach().cpu()
            ),
            "mae": float(
                mae.detach().cpu()
            ),
            "r2": float(
                r2.detach().cpu()
            ),
            "n_observed": n_j
        }

    return results


# ============================================================
# Graph extraction
# ============================================================

def extract_adjacency(
    model,
    threshold: float = 1e-3
):
    """
    Convert the continuous global support matrix

        S_ij = ||alpha_ij||_2

    into a binary adjacency matrix.

    Convention
    ----------
    adjacency[i, j] = 1
        means i -> j.

    Parameters
    ----------
    threshold:
        Edge i -> j is selected when S_ij > threshold.
    """

    if threshold < 0:
        raise ValueError(
            "threshold must be non-negative."
        )

    model.eval()

    with torch.inference_mode():

        S = model.support_matrix()

        adjacency = (
            S > threshold
        ).to(torch.int64)

        adjacency.fill_diagonal_(0)

    return (
        S.detach().cpu(),
        adjacency.detach().cpu()
    )


# ============================================================
# Exact DAG test
# ============================================================

def is_dag(
    adjacency: torch.Tensor
) -> bool:
    """
    Check whether a binary directed adjacency matrix is a DAG
    using Kahn's topological-sort algorithm.

    adjacency[i, j] = 1 means i -> j.
    """

    A = (
        adjacency
        .detach()
        .cpu()
        .numpy()
    )

    if A.ndim != 2:
        raise ValueError(
            "adjacency must be a 2D matrix."
        )

    p1, p2 = A.shape

    if p1 != p2:
        raise ValueError(
            "adjacency must be square."
        )

    p = p1

    # --------------------------------------------------------
    # indegree[j]
    # --------------------------------------------------------

    indegree = (
        A.sum(axis=0)
        .astype(np.int64)
    )

    # Nodes with no parents
    queue = [
        j
        for j in range(p)
        if indegree[j] == 0
    ]

    visited = 0

    while queue:

        node = queue.pop()

        visited += 1

        # All children of current node
        children = np.where(
            A[node, :] != 0
        )[0]

        for child in children:

            indegree[child] -= 1

            if indegree[child] == 0:
                queue.append(child)

    return visited == p


# ============================================================
# Graph-level summary
# ============================================================

def graph_summary(
    support: torch.Tensor,
    adjacency: torch.Tensor
) -> Dict[str, Any]:
    """
    Summarize the learned global graph.
    """

    p = adjacency.shape[0]

    n_edges = int(
        adjacency.sum().item()
    )

    max_possible_edges = (
        p * (p - 1)
    )

    density = (
        n_edges / max_possible_edges
        if max_possible_edges > 0
        else 0.0
    )

    selected_strengths = support[
        adjacency.bool()
    ]

    if selected_strengths.numel() > 0:

        mean_edge_strength = float(
            selected_strengths
            .mean()
            .item()
        )

        max_edge_strength = float(
            selected_strengths
            .max()
            .item()
        )

        min_edge_strength = float(
            selected_strengths
            .min()
            .item()
        )

    else:

        mean_edge_strength = 0.0
        max_edge_strength = 0.0
        min_edge_strength = 0.0

    return {
        "n_nodes": p,
        "n_edges": n_edges,
        "density": density,

        "mean_selected_strength":
            mean_edge_strength,

        "max_selected_strength":
            max_edge_strength,

        "min_selected_strength":
            min_edge_strength,

        "is_dag": is_dag(
            adjacency
        )
    }


# ============================================================
# Main evaluation function
# ============================================================

def evaluate_model(
    model,
    z: torch.Tensor,
    Y: torch.Tensor,
    observational_mask: Optional[torch.Tensor] = None,
    edge_threshold: float = 1e-3,
    return_tensors: bool = True
) -> Dict[str, Any]:
    """
    Complete evaluation of the DynamicDAG model.

    This function DOES NOT update model parameters.

    It evaluates:

    1. reconstruction loss used by the model;
    2. masked MSE / RMSE / MAE / R^2;
    3. nodewise metrics;
    4. group penalty;
    5. spectral radius;
    6. support matrix;
    7. thresholded adjacency;
    8. exact DAG status;
    9. sample-specific beta matrices;
    10. predictions.

    Parameters
    ----------
    model:
        Trained DynamicDAG model.

    z:
        Context / representation input.
        Shape: (n, input_dim)

    Y:
        Node observations.
        Shape: (n, p)

    observational_mask:
        Shape: (n, p)

        1 = observational
        0 = intervened

    edge_threshold:
        Threshold applied to S_ij = ||alpha_ij||_2.

    return_tensors:
        If False, large tensors such as beta and predictions
        are not stored in the returned dictionary.
    """

    model.eval()

    device = get_model_device(
        model
    )

    z = z.to(device)
    Y = Y.to(device)

    observational_mask = move_optional_tensor(
        observational_mask,
        device
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.inference_mode():

        output = model(
            z,
            Y,
            observational_mask
        )

        Y_hat = output["Y_hat"]

        recon_loss = output[
            "recon_loss"
        ]

        group_penalty = output[
            "group_penalty"
        ]

        spectral_radius = output[
            "acyc_penalty"
        ]

        S = output["S"]

        beta = output["beta"]

        context = output["x"]

    # --------------------------------------------------------
    # Standard prediction metrics
    # --------------------------------------------------------

    overall_metrics = (
        masked_regression_metrics(
            Y_true=Y,
            Y_pred=Y_hat,
            observational_mask=
                observational_mask
        )
    )

    # --------------------------------------------------------
    # Per-node metrics
    # --------------------------------------------------------

    node_metrics = (
        nodewise_regression_metrics(
            Y_true=Y,
            Y_pred=Y_hat,
            observational_mask=
                observational_mask
        )
    )

    # --------------------------------------------------------
    # Global graph
    # --------------------------------------------------------

    S_cpu = (
        S.detach()
        .cpu()
    )

    adjacency = (
        S_cpu > edge_threshold
    ).to(torch.int64)

    adjacency.fill_diagonal_(0)

    graph_metrics = graph_summary(
        support=S_cpu,
        adjacency=adjacency
    )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    result = {
        # Prediction
        "recon_loss": float(
            recon_loss
            .detach()
            .cpu()
        ),

        "mse": overall_metrics["mse"],
        "rmse": overall_metrics["rmse"],
        "mae": overall_metrics["mae"],
        "r2": overall_metrics["r2"],

        "n_observed":
            overall_metrics["n_observed"],

        # Nodewise
        "nodewise":
            node_metrics,

        # Regularization / graph
        "group_penalty": float(
            group_penalty
            .detach()
            .cpu()
        ),

        "spectral_radius": float(
            spectral_radius
            .detach()
            .cpu()
        ),

        "edge_threshold":
            edge_threshold,

        "graph":
            graph_metrics,

        "support":
            S_cpu,

        "adjacency":
            adjacency
    }

    # --------------------------------------------------------
    # Large tensors are optional
    # --------------------------------------------------------

    if return_tensors:

        result["prediction"] = (
            Y_hat.detach().cpu()
        )

        result["target"] = (
            Y.detach().cpu()
        )

        result["beta"] = (
            beta.detach().cpu()
        )

        result["context"] = (
            context.detach().cpu()
        )

        if observational_mask is not None:

            result[
                "observational_mask"
            ] = (
                observational_mask
                .detach()
                .cpu()
            )

    return result


# ============================================================
# Pretty printing
# ============================================================

def print_evaluation(
    metrics: Dict[str, Any],
    split_name: str = "Validation",
    print_nodewise: bool = True
):
    """
    Pretty-print evaluation results.
    """

    graph = metrics["graph"]

    print(
        "\n"
        "========================================\n"
        f"{split_name} results\n"
        "========================================"
    )

    print(
        f"Reconstruction loss : "
        f"{metrics['recon_loss']:.6f}"
    )

    print(
        f"MSE                 : "
        f"{metrics['mse']:.6f}"
    )

    print(
        f"RMSE                : "
        f"{metrics['rmse']:.6f}"
    )

    print(
        f"MAE                 : "
        f"{metrics['mae']:.6f}"
    )

    print(
        f"R^2                 : "
        f"{metrics['r2']:.6f}"
    )

    print(
        f"Observed entries    : "
        f"{metrics['n_observed']}"
    )

    print(
        "\n"
        "----------------------------------------\n"
        "Graph statistics\n"
        "----------------------------------------"
    )

    print(
        f"Spectral radius     : "
        f"{metrics['spectral_radius']:.3e}"
    )

    print(
        f"Group penalty       : "
        f"{metrics['group_penalty']:.6f}"
    )

    print(
        f"Threshold           : "
        f"{metrics['edge_threshold']:.3e}"
    )

    print(
        f"Selected edges      : "
        f"{graph['n_edges']}"
    )

    print(
        f"Graph density       : "
        f"{graph['density']:.4f}"
    )

    print(
        f"Thresholded DAG     : "
        f"{graph['is_dag']}"
    )

    if print_nodewise:

        print(
            "\n"
            "----------------------------------------\n"
            "Nodewise performance\n"
            "----------------------------------------"
        )

        for node, stat in (
            metrics["nodewise"].items()
        ):

            print(
                f"Node {node:3d} | "
                f"MSE={stat['mse']:.6f} | "
                f"RMSE={stat['rmse']:.6f} | "
                f"MAE={stat['mae']:.6f} | "
                f"R2={stat['r2']:.4f} | "
                f"n={stat['n_observed']}"
            )


# ============================================================
# Convenience wrappers
# ============================================================

def evaluate_validation(
    model,
    val_z,
    val_Y,
    val_mask=None,
    edge_threshold=1e-3,
    verbose=True
):
    """
    Convenience wrapper for validation evaluation.
    """

    metrics = evaluate_model(
        model=model,
        z=val_z,
        Y=val_Y,
        observational_mask=val_mask,
        edge_threshold=edge_threshold,
        return_tensors=False
    )

    if verbose:

        print_evaluation(
            metrics,
            split_name="Validation",
            print_nodewise=False
        )

    return metrics


def evaluate_test(
    model,
    test_z,
    test_Y,
    test_mask=None,
    edge_threshold=1e-3,
    verbose=True,
    return_tensors=True
):
    """
    Convenience wrapper for final test evaluation.

    Ideally this should be called only after all
    hyperparameters / checkpoints have been selected
    using the validation set.
    """

    metrics = evaluate_model(
        model=model,
        z=test_z,
        Y=test_Y,
        observational_mask=test_mask,
        edge_threshold=edge_threshold,
        return_tensors=return_tensors
    )

    if verbose:

        print_evaluation(
            metrics,
            split_name="Test",
            print_nodewise=True
        )

    return metrics

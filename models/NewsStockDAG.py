"""News-conditioned stock DAG with varying or fixed edge coefficients.

varying:
    beta_ij(c) = sum_s alpha_ijs * c_s

fixed:
    beta_ij(c) = B_ij

Both modes retain:
    b_j(c) = a_j + U_j^T c

Reuses DynamicDAG without changing any WSI source file.
"""

from __future__ import annotations

import torch
from torch import nn

from .TransMIL_regression import (
    DynamicDAG,
    extract_dag_adjacency,
    is_dag,
)


class FixedCoefficientDAG(DynamicDAG):
    """Fixed edge coefficients with a news-dependent intercept.

    Store each scalar B_ij as alpha[i, j, 0] so that the existing
    support, masking, regularization, and spectral code can be reused.

    In this mode:
        alpha.shape = [n_nodes, n_nodes, 1]
        support[i, j] = abs(B_ij)

    Therefore the inherited group penalty becomes an L1 penalty on B.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Keep encoder/intercept initialization identical to the varying
        # model under the same seed. Retain one scalar per edge.
        self.alpha = nn.Parameter(
            self.alpha.detach()[..., :1].clone()
        )

    def compute_beta(self, context):
        # effective_alpha applies the structural mask.
        coefficients = self.effective_alpha().squeeze(-1)

        # The same B is used for every date in this batch.
        return coefficients.unsqueeze(0).expand(
            context.shape[0], -1, -1
        )


class NewsStockDAG(nn.Module):
    def __init__(
        self,
        n_stocks=99,
        hidden_dim=128,
        context_dim=16,
        dropout=0.0,
        power_iteration_steps=15,
        alpha_init_scale=0.01,
        coefficient_mode="varying",
    ):
        super().__init__()

        if coefficient_mode not in ("varying", "fixed"):
            raise ValueError(
                "coefficient_mode must be 'varying' or 'fixed'."
            )

        self.n_stocks = int(n_stocks)
        self.coefficient_mode = coefficient_mode

        self.model_config = dict(
            n_stocks=self.n_stocks,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            dropout=dropout,
            power_iteration_steps=power_iteration_steps,
            alpha_init_scale=alpha_init_scale,
            coefficient_mode=coefficient_mode,
        )

        dag_class = (
            DynamicDAG
            if coefficient_mode == "varying"
            else FixedCoefficientDAG
        )

        self.dag = dag_class(
            input_dim=25 * 50,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            n_nodes=self.n_stocks,
            dropout=dropout,
            context_transform="tanh",
            power_iteration_steps=power_iteration_steps,
            alpha_init_scale=alpha_init_scale,
            use_context_intercept=True,
        )

    def forward(
        self,
        news,
        Y=None,
        lambda_group=0.0,
        gamma_acyclicity=0.0,
    ):
        if news.ndim != 3 or tuple(news.shape[1:]) != (25, 50):
            raise ValueError("news must have shape [B,25,50].")

        return self.dag(
            embedding=news.flatten(start_dim=1),
            Y=Y,
            lambda_group=lambda_group,
            gamma_acyclicity=gamma_acyclicity,
        )

    @torch.no_grad()
    def freeze_graph(self, threshold):
        support = self.dag.support_matrix()

        raw = (support > threshold).to(torch.int64)
        raw.fill_diagonal_(0)

        selected = extract_dag_adjacency(
            support,
            threshold=threshold,
        ).to(support.device)

        info = {
            "raw_threshold_graph_is_dag": is_dag(raw),
            "raw_threshold_edges": int(raw.sum()),
            "selected_edges": int(selected.sum()),
            "edges_removed_to_break_cycles": int(
                raw.sum() - selected.sum()
            ),
        }

        self.dag.set_structural_mask(selected)
        return info


def to_return_units(beta, intercept, mean, scale):
    """B[i,j] means source i -> target j.

    Convert standardized structural equations to log-return units.
    """

    mean = torch.as_tensor(
        mean,
        device=beta.device,
        dtype=beta.dtype,
    )

    scale = torch.as_tensor(
        scale,
        device=beta.device,
        dtype=beta.dtype,
    )

    raw_beta = (
        beta
        * scale[None, None, :]
        / scale[None, :, None]
    )

    raw_intercept = (
        mean
        + scale * intercept
        - torch.einsum("i,bij->bj", mean, raw_beta)
    )

    return raw_beta, raw_intercept

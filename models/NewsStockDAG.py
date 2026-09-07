"""Frozen [25,50] headlines -> shared MLP -> date-varying stock DAG.

Reuses the original DynamicDAG without changing any WSI source file.
The original module imports TransMIL, so nystrom-attention remains an import
dependency, but no TransMIL model is instantiated or trained here.
"""

from __future__ import annotations

import torch
from torch import nn

from .TransMIL_regression import DynamicDAG, extract_dag_adjacency, is_dag


class NewsStockDAG(nn.Module):
    def __init__(self, n_stocks=99, hidden_dim=128, context_dim=16, dropout=0.0,
                 power_iteration_steps=15, alpha_init_scale=0.01):
        super().__init__()
        self.n_stocks = int(n_stocks)
        self.model_config = dict(n_stocks=self.n_stocks, hidden_dim=hidden_dim,
                                 context_dim=context_dim, dropout=dropout,
                                 power_iteration_steps=power_iteration_steps,
                                 alpha_init_scale=alpha_init_scale)
        self.dag = DynamicDAG(input_dim=25 * 50, hidden_dim=hidden_dim,
                              context_dim=context_dim, n_nodes=self.n_stocks,
                              dropout=dropout, context_transform="tanh",
                              power_iteration_steps=power_iteration_steps,
                              alpha_init_scale=alpha_init_scale,
                              use_context_intercept=True)

    def forward(self, news, Y=None, lambda_group=0.0, gamma_acyclicity=0.0):
        if news.ndim != 3 or tuple(news.shape[1:]) != (25, 50):
            raise ValueError("news must have shape [B,25,50].")
        return self.dag(embedding=news.flatten(start_dim=1), Y=Y,
                        lambda_group=lambda_group, gamma_acyclicity=gamma_acyclicity)

    @torch.no_grad()
    def freeze_graph(self, threshold):
        support = self.dag.support_matrix()
        raw = (support > threshold).to(torch.int64)
        raw.fill_diagonal_(0)
        selected = extract_dag_adjacency(support, threshold=threshold).to(support.device)
        info = {"raw_threshold_graph_is_dag": is_dag(raw),
                "raw_threshold_edges": int(raw.sum()),
                "selected_edges": int(selected.sum()),
                "edges_removed_to_break_cycles": int(raw.sum() - selected.sum())}
        self.dag.set_structural_mask(selected)
        return info


def to_return_units(beta, intercept, mean, scale):
    """B[i,j] means source i -> target j; convert standardized equations."""
    mean = torch.as_tensor(mean, device=beta.device, dtype=beta.dtype)
    scale = torch.as_tensor(scale, device=beta.device, dtype=beta.dtype)
    raw_beta = beta * scale[None, None, :] / scale[None, :, None]
    raw_intercept = mean + scale * intercept - torch.einsum("i,bij->bj", mean, raw_beta)
    return raw_beta, raw_intercept

"""Gene-expression regression baseline and image-conditioned DAG model.

This module deliberately contains two separate model classes so the repository
can keep the original four-script layout:

``TransMILRegression``
    WSI -> gene-expression prediction. This is a baseline only.

``TransMILDAG``
    WSI -> slide embedding -> patient-specific DAG coefficients. The regression
    head is completely bypassed. Paired expression Y is required only while
    fitting/evaluating the structural-equation loss, not for graph inference.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from .TransMIL import TransMILBackbone
except ImportError:  # permits: python models/TransMIL_regression.py
    from TransMIL import TransMILBackbone


def _normalize_vector(vector: torch.Tensor, eps: float) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(vector).clamp_min(eps)


def _has_path(adjacency: np.ndarray, source: int, target: int) -> bool:
    if source == target:
        return True
    seen = np.zeros(adjacency.shape[0], dtype=bool)
    stack = [source]
    seen[source] = True
    while stack:
        node = stack.pop()
        for child in np.flatnonzero(adjacency[node]):
            child = int(child)
            if child == target:
                return True
            if not seen[child]:
                seen[child] = True
                stack.append(child)
    return False


def is_dag(adjacency: torch.Tensor) -> bool:
    array = (adjacency.detach().cpu().numpy() != 0).astype(np.int64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("adjacency must be a square matrix.")
    indegree = array.sum(axis=0)
    stack = list(np.flatnonzero(indegree == 0))
    visited = 0
    while stack:
        node = int(stack.pop())
        visited += 1
        for child in np.flatnonzero(array[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                stack.append(int(child))
    return visited == array.shape[0]


def extract_dag_adjacency(
    strength: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Greedily retain strong edges without introducing a directed cycle."""
    if threshold < 0:
        raise ValueError("threshold must be nonnegative.")
    matrix = strength.detach().cpu().numpy()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("strength must be a square matrix.")

    p = matrix.shape[0]
    candidates = [
        (float(matrix[i, j]), i, j)
        for i in range(p)
        for j in range(p)
        if i != j and matrix[i, j] > threshold
    ]
    candidates.sort(key=lambda item: item[0], reverse=True)
    adjacency = np.zeros((p, p), dtype=np.int64)
    for _, parent, child in candidates:
        if not _has_path(adjacency, child, parent):
            adjacency[parent, child] = 1
    return torch.from_numpy(adjacency)


class PowerIterationSpectralPenalty(nn.Module):
    """SDCD-style spectral-radius gradient surrogate for nonnegative support."""

    def __init__(self, n_nodes: int, n_steps: int = 15, eps: float = 1e-6):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.n_steps = int(n_steps)
        self.eps = float(eps)
        initial = torch.ones(self.n_nodes, dtype=torch.float32)
        initial = _normalize_vector(initial, self.eps)
        self.register_buffer("left", initial.clone())
        self.register_buffer("right", initial.clone())

    @torch.no_grad()
    def reset(self) -> None:
        self.left.fill_(1.0)
        self.right.fill_(1.0)
        self.left.copy_(_normalize_vector(self.left, self.eps))
        self.right.copy_(_normalize_vector(self.right, self.eps))

    @torch.no_grad()
    def iterate(self, support: torch.Tensor) -> None:
        matrix = support.detach() + self.eps
        for _ in range(self.n_steps):
            self.left.copy_(
                _normalize_vector(matrix.T @ self.left, self.eps)
            )
            self.right.copy_(
                _normalize_vector(matrix @ self.right, self.eps)
            )

    def surrogate(self, support: torch.Tensor) -> torch.Tensor:
        self.iterate(support)
        denominator = torch.dot(self.left, self.right).clamp_min(self.eps)
        gradient = torch.outer(self.left, self.right) / denominator
        return (gradient.detach() * support).sum()

    @torch.no_grad()
    def estimate(self, support: torch.Tensor) -> torch.Tensor:
        self.iterate(support)
        denominator = torch.dot(self.left, self.right).clamp_min(self.eps)
        value = torch.dot(self.left, support @ self.right) / denominator
        return value.clamp_min(0.0)


class DynamicDAG(nn.Module):
    """Low-rank varying-coefficient linear structural equation model.

    ``alpha[i, j, s]`` belongs to edge i -> j and basis/context component s.
    ``beta[k, i, j]`` is the corresponding coefficient for patient k.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        context_dim: int,
        n_nodes: int,
        dropout: float = 0.0,
        context_transform: str = "tanh",
        power_iteration_steps: int = 15,
        alpha_init_scale: float = 1e-2,
        use_context_intercept: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.context_dim = int(context_dim)
        self.n_nodes = int(n_nodes)
        self.context_transform = str(context_transform)
        self.use_context_intercept = bool(use_context_intercept)

        self.projector = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.context_dim),
        )
        self.alpha = nn.Parameter(
            alpha_init_scale
            * torch.randn(self.n_nodes, self.n_nodes, self.context_dim)
        )
        # Context-dependent node means prevent WSI-driven expression shifts
        # from being spuriously explained by gene-to-gene edges.
        self.intercept_head = (
            nn.Linear(self.context_dim, self.n_nodes)
            if self.use_context_intercept
            else None
        )
        off_diagonal = 1.0 - torch.eye(self.n_nodes)
        self.register_buffer("off_diagonal", off_diagonal)
        self.register_buffer("structural_mask", off_diagonal.clone())
        self.spectral = PowerIterationSpectralPenalty(
            n_nodes=self.n_nodes,
            n_steps=power_iteration_steps,
        )

    def encode_context(self, embedding: torch.Tensor) -> torch.Tensor:
        context = self.projector(embedding)
        if self.context_transform == "tanh":
            return torch.tanh(context)
        if self.context_transform == "none":
            return context
        raise ValueError("context_transform must be 'tanh' or 'none'.")

    def effective_alpha(self) -> torch.Tensor:
        return self.alpha * self.structural_mask[:, :, None]

    def support_matrix(self) -> torch.Tensor:
        return torch.linalg.vector_norm(self.effective_alpha(), dim=-1)

    @torch.no_grad()
    def exact_spectral_radius(self) -> torch.Tensor:
        eigenvalues = torch.linalg.eigvals(self.support_matrix())
        return eigenvalues.abs().max().real

    def compute_beta(self, context: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ks,ijs->kij", context, self.effective_alpha())

    @staticmethod
    def nodewise_prediction(
        Y: torch.Tensor,
        beta: torch.Tensor,
        intercept: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prediction = torch.einsum("ki,kij->kj", Y, beta)
        if intercept is not None:
            prediction = prediction + intercept
        return prediction

    @staticmethod
    def reconstruction_loss(
        Y: torch.Tensor,
        prediction: torch.Tensor,
        observational_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        squared_error = (Y - prediction).square()
        if observational_mask is None:
            return squared_error.mean()
        if observational_mask.shape != Y.shape:
            raise ValueError(
                "observational_mask must have the same shape as Y."
            )
        mask = observational_mask.to(device=Y.device, dtype=Y.dtype)
        if mask.sum() <= 0:
            raise ValueError("observational_mask contains no observed equations.")
        return (squared_error * mask).sum() / mask.sum()

    @torch.no_grad()
    def zero_forbidden_edges(self) -> None:
        self.alpha.mul_(self.structural_mask[:, :, None])

    @torch.no_grad()
    def set_structural_mask(self, mask: torch.Tensor) -> None:
        if mask.shape != (self.n_nodes, self.n_nodes):
            raise ValueError("structural mask has an invalid shape.")
        mask = mask.to(self.structural_mask) * self.off_diagonal
        self.structural_mask.copy_(mask)
        self.zero_forbidden_edges()
        self.spectral.reset()

    def infer_coefficients(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        context = self.encode_context(embedding)
        beta = self.compute_beta(context)
        if self.intercept_head is None:
            intercept = context.new_zeros((context.shape[0], self.n_nodes))
        else:
            intercept = self.intercept_head(context)
        return {
            "context": context,
            "beta": beta,
            "intercept": intercept,
            "support": self.support_matrix(),
        }

    def forward(
        self,
        embedding: torch.Tensor,
        Y: Optional[torch.Tensor] = None,
        observational_mask: Optional[torch.Tensor] = None,
        lambda_group: float = 0.0,
        gamma_acyclicity: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        output = self.infer_coefficients(embedding)
        if Y is None:
            return output
        if Y.ndim != 2 or Y.shape[1] != self.n_nodes:
            raise ValueError(
                f"Y must have shape [B, {self.n_nodes}], got {tuple(Y.shape)}."
            )
        if Y.shape[0] != embedding.shape[0]:
            raise ValueError("embedding and Y batch sizes differ.")

        prediction = self.nodewise_prediction(
            Y, output["beta"], output["intercept"]
        )
        reconstruction = self.reconstruction_loss(
            Y, prediction, observational_mask
        )
        group = output["support"].sum()
        if gamma_acyclicity > 0:
            acyclicity = self.spectral.surrogate(output["support"])
        else:
            acyclicity = reconstruction.new_zeros(())
        total = (
            reconstruction
            + float(lambda_group) * group
            + float(gamma_acyclicity) * acyclicity
        )
        output.update(
            {
                "Y_hat": prediction,
                "loss": total,
                "reconstruction_loss": reconstruction,
                "group_penalty": group,
                "acyclicity_penalty": acyclicity,
            }
        )
        return output


class TransMILRegression(nn.Module):
    """WSI-to-expression regression baseline. It is not the proposed model."""

    is_dag_model = False

    def __init__(
        self,
        n_genes: int,
        feat_dim: int = 1024,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.n_genes = int(n_genes)
        self.backbone = TransMILBackbone(feat_dim=feat_dim, dropout=dropout)
        self.regression_head = nn.Linear(
            self.backbone.embedding_dim, self.n_genes
        )

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode(data)

    def forward(self, data: torch.Tensor, **_: object) -> Dict[str, torch.Tensor]:
        embedding = self.encode(data)
        prediction = self.regression_head(embedding)
        return {
            "preds": prediction,
            "logits": prediction,
            "embedding": embedding,
        }


class TransMILDAG(nn.Module):
    """Proposed model: WSI directly determines patient-specific DAG weights."""

    is_dag_model = True

    def __init__(
        self,
        n_genes: int,
        feat_dim: int = 1024,
        dropout: float = 0.25,
        dag_hidden_dim: int = 128,
        context_dim: int = 16,
        dag_dropout: float = 0.0,
        context_transform: str = "tanh",
        power_iteration_steps: int = 15,
        alpha_init_scale: float = 1e-2,
        use_context_intercept: bool = True,
        global_edge_threshold: float = 0.1,
        patient_edge_threshold: float = 0.1,
    ):
        super().__init__()
        self.n_genes = int(n_genes)
        self.global_edge_threshold = float(global_edge_threshold)
        self.patient_edge_threshold = float(patient_edge_threshold)
        self.backbone = TransMILBackbone(feat_dim=feat_dim, dropout=dropout)
        self.dynamic_dag = DynamicDAG(
            input_dim=self.backbone.embedding_dim,
            hidden_dim=dag_hidden_dim,
            context_dim=context_dim,
            n_nodes=self.n_genes,
            dropout=dag_dropout,
            context_transform=context_transform,
            power_iteration_steps=power_iteration_steps,
            alpha_init_scale=alpha_init_scale,
            use_context_intercept=use_context_intercept,
        )

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode(data)

    @torch.no_grad()
    def raw_global_adjacency(
        self,
        threshold: Optional[float] = None,
    ) -> torch.Tensor:
        """Plain thresholded support, retained to diagnose DAG convergence."""
        if threshold is None:
            threshold = self.global_edge_threshold
        adjacency = (
            self.dynamic_dag.support_matrix() > float(threshold)
        ).to(torch.int64)
        adjacency.fill_diagonal_(0)
        return adjacency

    @torch.no_grad()
    def raw_global_is_dag(self, threshold: Optional[float] = None) -> bool:
        return is_dag(self.raw_global_adjacency(threshold))

    @torch.no_grad()
    def _discrete_graphs(
        self,
        beta: torch.Tensor,
        support: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        raw_global = (
            support.detach() > self.global_edge_threshold
        ).to(torch.int64)
        raw_global.fill_diagonal_(0)

        # A single cycle-safe union support is shared across patients.
        global_adjacency = extract_dag_adjacency(
            support, threshold=self.global_edge_threshold
        ).to(beta.device)
        raw_patient = (
            beta.abs() > self.patient_edge_threshold
        ).to(torch.int64)
        raw_patient = raw_patient * raw_global.to(beta.device).unsqueeze(0)
        patient_adjacency = raw_patient * global_adjacency.unsqueeze(0)
        return {
            "raw_global_adjacency": raw_global,
            "global_adjacency": global_adjacency,
            "raw_patient_adjacency": raw_patient,
            "patient_adjacency": patient_adjacency,
            "raw_global_is_dag": torch.tensor(
                is_dag(raw_global), dtype=torch.bool
            ),
        }

    def infer_graph(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        embedding = self.encode(data)
        output = self.dynamic_dag(embedding=embedding)
        graphs = self._discrete_graphs(output["beta"], output["support"])
        output.update(
            {
                "embedding": embedding,
                **graphs,
            }
        )
        return output

    def forward(
        self,
        data: torch.Tensor,
        Y: Optional[torch.Tensor] = None,
        observational_mask: Optional[torch.Tensor] = None,
        lambda_group: float = 0.0,
        gamma_acyclicity: float = 0.0,
        return_discrete_graphs: bool = False,
        **_: object,
    ) -> Dict[str, torch.Tensor]:
        embedding = self.encode(data)
        output = self.dynamic_dag(
            embedding=embedding,
            Y=Y,
            observational_mask=observational_mask,
            lambda_group=lambda_group,
            gamma_acyclicity=gamma_acyclicity,
        )
        output["embedding"] = embedding
        if return_discrete_graphs:
            output.update(
                self._discrete_graphs(output["beta"], output["support"])
            )
        return output


if __name__ == "__main__":
    features = torch.randn(2, 121, 1024)
    expression = torch.randn(2, 8)

    baseline = TransMILRegression(n_genes=8)
    print("Regression:", baseline(features)["preds"].shape)

    dag_model = TransMILDAG(n_genes=8)
    train_output = dag_model(features, Y=expression, lambda_group=1e-3)
    print("DAG beta:", train_output["beta"].shape)
    graph_output = dag_model.infer_graph(features)
    print("Patient graphs:", graph_output["patient_adjacency"].shape)

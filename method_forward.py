# model.py
from __future__ import annotations

from typing import Optional, Dict, Any

import torch
import torch.nn as nn


def _normalize_vector(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / torch.linalg.vector_norm(v).clamp_min(eps)


class PowerIterationSpectralPenalty(nn.Module):
    """
    SDCD-style spectral-radius surrogate for a nonnegative adjacency-strength matrix S.

    For a nonnegative matrix S, rho(S) = 0 iff the directed graph represented by S is acyclic.

    Training uses the Perron left/right eigenvector gradient
        d rho / d S = u v^T / (u^T v)
    estimated by power iteration.

    The returned surrogate
        h_tilde(S) = <stop_grad(G), S>
    has gradient G with respect to S. When the eigenvectors are exact,
    h_tilde(S) equals rho(S) by Euler's theorem / the Perron eigenvalue identity.

    This mirrors the core idea used by SDCD's PowerIterationGradient, adapted to
    the global support S_ij = ||alpha_ij||_2 of this varying-coefficient model.
    """

    def __init__(
        self,
        n_nodes: int,
        n_steps: int = 15,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.n_steps = int(n_steps)
        self.eps = float(eps)

        init = torch.ones(self.n_nodes, dtype=torch.float32)
        init = _normalize_vector(init)
        self.register_buffer("u", init.clone())
        self.register_buffer("v", init.clone())

    @torch.no_grad()
    def reset_vectors(self) -> None:
        self.u.fill_(1.0)
        self.v.fill_(1.0)
        self.u.copy_(_normalize_vector(self.u))
        self.v.copy_(_normalize_vector(self.v))

    @torch.no_grad()
    def iterate(
        self,
        S: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> None:
        """
        Update approximate left/right Perron eigenvectors.

        A tiny positive shift is used only inside power iteration to avoid the
        degenerate all-zero case and to keep iterates numerically well-defined.
        """
        steps = self.n_steps if n_steps is None else int(n_steps)
        A = S.detach() + self.eps

        # Keep buffers on the same dtype/device as S.
        if self.u.device != S.device or self.u.dtype != S.dtype:
            self.u = self.u.to(device=S.device, dtype=S.dtype)
            self.v = self.v.to(device=S.device, dtype=S.dtype)

        for _ in range(steps):
            new_u = _normalize_vector(A.T @ self.u, eps=self.eps)
            new_v = _normalize_vector(A @ self.v, eps=self.eps)
            self.u.copy_(new_u)
            self.v.copy_(new_v)

    def gradient_matrix(self, S: torch.Tensor) -> torch.Tensor:
        """
        Return an estimated gradient of rho(S) wrt S.
        """
        self.iterate(S)
        denom = torch.dot(self.u, self.v).clamp_min(self.eps)
        grad = torch.outer(self.u, self.v) / denom
        return grad

    def surrogate(self, S: torch.Tensor) -> torch.Tensor:
        """
        Differentiable scalar used in the training objective.
        """
        grad = self.gradient_matrix(S)
        return (grad.detach() * S).sum()

    @torch.no_grad()
    def estimate(self, S: torch.Tensor) -> torch.Tensor:
        """
        Estimate the Perron root with the same left/right vectors.

        For nonnegative S this is the spectral radius.
        """
        self.iterate(S)
        denom = torch.dot(self.u, self.v).clamp_min(self.eps)
        value = torch.dot(self.u, S @ self.v) / denom
        return value.clamp_min(0.0)


class DynamicDAG(nn.Module):
    """
    Deep varying-coefficient DAG model.

    Convention
    ----------
    alpha[i, j, s] : coefficient for latent/context component s on edge i -> j.
    beta[k, i, j]  : sample-specific edge coefficient i -> j for sample k.

    Model
    -----
        x_k = f_theta(z_k)
        beta_ij^(k) = sum_s alpha_ijs * x_ks
        Yhat_kj = sum_i Y_ki * beta_kij

    Global support
    --------------
        S_ij = ||alpha_ij||_2 >= 0

    Acyclicity is imposed on S, so every sample-specific graph is a subgraph
    of one shared directed acyclic support.
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
        power_iteration_eps: float = 1e-6,
        alpha_init_scale: float = 1e-2,
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.context_dim = int(context_dim)
        self.n_nodes = int(n_nodes)
        self.context_transform = context_transform

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
            * torch.randn(
                self.n_nodes,
                self.n_nodes,
                self.context_dim,
            )
        )

        offdiag = 1.0 - torch.eye(self.n_nodes)
        self.register_buffer("offdiag_mask", offdiag)
        self.register_buffer("structural_mask", offdiag.clone())

        self.spectral_module = PowerIterationSpectralPenalty(
            n_nodes=self.n_nodes,
            n_steps=power_iteration_steps,
            eps=power_iteration_eps,
        )

    def encode_context(self, z: torch.Tensor) -> torch.Tensor:
        """
        Project high-dimensional representations to bounded context features.

        The bounded transform prevents the trivial rescaling
            x -> c x, alpha -> alpha / c
        from driving structural penalties toward zero without changing beta.

        `tanh` preserves sample-dependent magnitude information better than
        unit-norm normalization.
        """
        x = self.projector(z)

        if self.context_transform == "tanh":
            return torch.tanh(x)
        if self.context_transform == "none":
            return x

        raise ValueError(
            f"Unknown context_transform={self.context_transform!r}. "
            "Use 'tanh' or 'none'."
        )

    def effective_alpha(self) -> torch.Tensor:
        return self.alpha * self.structural_mask[:, :, None]

    def compute_beta(self, x: torch.Tensor) -> torch.Tensor:
        """
        beta[k, i, j] = sum_s x[k, s] * alpha[i, j, s]
        """
        return torch.einsum("ks,ijs->kij", x, self.effective_alpha())

    def nodewise_prediction(
        self,
        Y: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Y_hat[k, j] = sum_i Y[k, i] * beta[k, i, j].
        """
        return torch.einsum("ki,kij->kj", Y, beta)

    def reconstruction_loss(
        self,
        Y: torch.Tensor,
        Y_hat: torch.Tensor,
        observational_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Mean squared reconstruction error over observational node/sample pairs.

        observational_mask[k, j] = 1 : equation j is observational in sample k
        observational_mask[k, j] = 0 : node j was intervened on in sample k
        """
        residual_sq = (Y - Y_hat).pow(2)

        if observational_mask is None:
            return residual_sq.mean()

        if observational_mask.shape != Y.shape:
            raise ValueError(
                "observational_mask must have the same shape as Y; "
                f"got {observational_mask.shape} vs {Y.shape}."
            )

        mask = observational_mask.to(device=Y.device, dtype=Y.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (residual_sq * mask).sum() / denom

    def support_matrix(self) -> torch.Tensor:
        """
        S_ij = ||alpha_ij||_2, with masked/self-loop entries fixed at zero.
        """
        return torch.linalg.vector_norm(
            self.effective_alpha(),
            ord=2,
            dim=-1,
        )

    def group_lasso_penalty(self) -> torch.Tensor:
        return self.support_matrix().sum()

    def spectral_acyclicity_penalty(self) -> torch.Tensor:
        """
        SDCD-style power-iteration surrogate used for backpropagation.
        """
        S = self.support_matrix()
        return self.spectral_module.surrogate(S)

    @torch.no_grad()
    def spectral_radius_estimate(self) -> torch.Tensor:
        return self.spectral_module.estimate(self.support_matrix())

    @torch.no_grad()
    def exact_spectral_radius(self) -> torch.Tensor:
        """
        O(p^3) diagnostic only. Do not use this as the training penalty.
        """
        eigvals = torch.linalg.eigvals(self.support_matrix())
        return eigvals.abs().max().real

    @torch.no_grad()
    def set_edge_mask(self, edge_mask: torch.Tensor) -> None:
        """
        Permanently restrict candidate edges for Stage II.

        edge_mask[i, j] = 1 keeps candidate i -> j.
        """
        if edge_mask.shape != (self.n_nodes, self.n_nodes):
            raise ValueError(
                "edge_mask must have shape "
                f"({self.n_nodes}, {self.n_nodes}), got {edge_mask.shape}."
            )
        mask = edge_mask.to(
            device=self.structural_mask.device,
            dtype=self.structural_mask.dtype,
        )
        mask = mask * self.offdiag_mask
        self.structural_mask.copy_(mask)
        self.alpha.mul_(self.structural_mask[:, :, None])
        self.spectral_module.reset_vectors()

    @torch.no_grad()
    def reset_edge_mask(self) -> None:
        self.structural_mask.copy_(self.offdiag_mask)
        self.alpha.mul_(self.structural_mask[:, :, None])
        self.spectral_module.reset_vectors()

    def forward(
        self,
        z: torch.Tensor,
        Y: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        x = self.encode_context(z)
        beta = self.compute_beta(x)
        Y_hat = self.nodewise_prediction(Y, beta)
        S = self.support_matrix()

        return {
            "x": x,
            "beta": beta,
            "Y_hat": Y_hat,
            "S": S,
        }

    def loss(
        self,
        z: torch.Tensor,
        Y: torch.Tensor,
        observational_mask: Optional[torch.Tensor] = None,
        lambda_group: float = 0.0,
        gamma_acyc: float = 0.0,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.forward(z, Y)

        recon = self.reconstruction_loss(
            Y,
            out["Y_hat"],
            observational_mask=observational_mask,
        )
        group = self.group_lasso_penalty()

        if gamma_acyc > 0.0:
            dag = self.spectral_acyclicity_penalty()
        else:
            dag = torch.zeros((), device=Y.device, dtype=Y.dtype)

        total = recon + lambda_group * group + gamma_acyc * dag

        details = {
            "total": total,
            "recon": recon,
            "group": group,
            "dag": dag,
        }
        return total, details

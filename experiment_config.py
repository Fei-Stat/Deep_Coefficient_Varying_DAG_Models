"""Pure-Python experiment configurations.

Edit the dictionaries in this file to change experiment hyperparameters.  The
training entry point takes a defensive deep copy, so it may safely infer and
insert ``n_genes`` and ``feat_dim`` without mutating these defaults.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


DAG_CONFIG: Dict[str, Any] = {
    "seed": 0,
    "model": {
        "name": "TransMIL_regression",
        "class_name": "TransMILDAG",
        # n_genes and feat_dim are inferred from the input data.
        "dropout": 0.25,
        "dag_hidden_dim": 128,
        "context_dim": 16,
        "dag_dropout": 0.0,
        "context_transform": "tanh",
        "power_iteration_steps": 15,
        "alpha_init_scale": 0.01,
        "use_context_intercept": True,
        "global_edge_threshold": 0.1,
        "patient_edge_threshold": 0.1,
    },
    "dag": {
        "stage1_epochs": 300,
        "stage2_epochs": 1200,
        "stage1_lr": 2e-4,
        "stage2_lr": 1e-4,
        "lambda_group_stage1": 1e-3,
        "lambda_group_stage2": 1e-3,
        "use_screening": True,
        "screening_threshold": 1e-3,
        "gamma_increment": 5e-3,
        "gamma_schedule": "linear",
        "gamma_warmup_epochs": 2,
        "freeze_gamma_at_dag": True,
        "freeze_gamma_threshold": 0.01,
        "invalid_graph_penalty": 1e6,
    },
    "optimizer": {
        "name": "AdamW",
        "weight_decay": 1e-4,
        "grad_clip": 5.0,
    },
    "data": {
        "standardize_Y": True,
        "require_ids": True,
        "num_workers": 0,
    },
    "trainer": {
        "accelerator": "auto",
        "devices": 1,
        "precision": "32-true",
        "log_every_n_steps": 10,
        "deterministic": True,
    },
}


REGRESSION_CONFIG: Dict[str, Any] = {
    "seed": 0,
    "model": {
        "name": "TransMIL_regression",
        "class_name": "TransMILRegression",
        # n_genes and feat_dim are inferred from the input data.
        "dropout": 0.25,
    },
    "optimizer": {
        "name": "AdamW",
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "grad_clip": 5.0,
    },
    "data": {
        "standardize_Y": True,
        "require_ids": True,
        "num_workers": 0,
    },
    "trainer": {
        "accelerator": "auto",
        "devices": 1,
        "precision": "32-true",
        "max_epochs": 100,
        "early_stopping_patience": 15,
        "log_every_n_steps": 10,
        "deterministic": True,
    },
}


EXPERIMENT_CONFIGS = {
    "dag": DAG_CONFIG,
    "regression": REGRESSION_CONFIG,
}


def get_experiment_config(task: str) -> Dict[str, Any]:
    """Return an isolated configuration for ``task``."""
    try:
        config = EXPERIMENT_CONFIGS[task]
    except KeyError as error:
        choices = ", ".join(sorted(EXPERIMENT_CONFIGS))
        raise ValueError(f"Unknown task {task!r}; choose one of: {choices}.") from error
    return deepcopy(config)

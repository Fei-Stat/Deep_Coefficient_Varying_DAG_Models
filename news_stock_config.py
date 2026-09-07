"""Defaults for the independent news/stock application; no WSI config changes."""

EXCLUDED_STOCKS = ("AVGO", "FB", "GM", "LYB", "TSLA")
NEWS_SHAPE = (25, 50)
EXPECTED_STOCK_COUNT = 99

MODEL_DEFAULTS = {
    "hidden_dim": 128,
    "context_dim": 16,
    "dropout": 0.0,
    "power_iteration_steps": 15,
    "alpha_init_scale": 0.01,
}

TRAIN_DEFAULTS = {
    "seed": 0,
    "batch_size": 64,
    "stage1_epochs": 300,
    "stage2_epochs": 1200,
    "refit_epochs": 100,
    "stage1_lr": 2e-4,
    "stage2_lr": 1e-4,
    "refit_lr": 1e-4,
    "lambda_group_stage1": 1e-3,
    "lambda_group_stage2": 1e-3,
    "lambda_group_refit": 1e-3,
    "screening_threshold": 1e-3,
    "gamma_increment": 5e-3,
    "graph_threshold": 0.1,
    "weight_decay": 1e-4,
    "grad_clip": 5.0,
    "refit_patience": 25,
}

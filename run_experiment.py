"""End-to-end training and WSI-only graph inference.

Input ``.pt`` dictionary
------------------------
Required keys::

    train_features, train_Y
    val_features,   val_Y
    test_features,  test_Y

Optional keys::

    train_mask, val_mask, test_mask
    train_ids,  val_ids,  test_ids
    gene_ids

Each ``*_features`` value may be a tensor ``[n_slides, n_tiles, feat_dim]``, a
list of ``[n_tiles, feat_dim]`` tensors, or a list of paths to tensors. Variable
tile counts are supported with the required MIL batch size of one.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from models.modelInterfaceRegression import ModelInterfaceRegression


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location="cpu")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("The YAML root must be a mapping.")
    return config


def as_feature_sequence(value: Any) -> Sequence[Any]:
    if torch.is_tensor(value):
        if value.ndim != 3:
            raise ValueError("Feature tensors must have shape [slides, tiles, dim].")
        return [value[index] for index in range(value.shape[0])]
    if isinstance(value, (list, tuple)):
        return value
    raise TypeError("Features must be a 3D tensor, list, or tuple.")


def resolve_feature(item: Any, data_directory: Path) -> torch.Tensor:
    if isinstance(item, (str, Path)):
        path = Path(item)
        if not path.is_absolute():
            path = data_directory / path
        item = torch_load(path)
        if isinstance(item, dict):
            for key in ("features", "feats", "data"):
                if key in item:
                    item = item[key]
                    break
    if not torch.is_tensor(item):
        item = torch.as_tensor(item)
    if item.ndim != 2:
        raise ValueError(
            f"Each slide feature must be [n_tiles, feat_dim], got {tuple(item.shape)}."
        )
    if item.shape[0] == 0:
        raise ValueError("A slide contains no tile features.")
    item = item.float()
    if not torch.isfinite(item).all():
        raise ValueError("A slide feature tensor contains NaN or infinite values.")
    return item


class PairedSlideDataset(Dataset):
    def __init__(
        self,
        features: Sequence[Any],
        expression: torch.Tensor,
        masks: Optional[torch.Tensor],
        slide_ids: Sequence[str],
        data_directory: Path,
    ):
        self.features = features
        self.expression = expression.float()
        self.masks = None if masks is None else masks.float()
        self.slide_ids = [str(value) for value in slide_ids]
        self.data_directory = data_directory

        n = len(self.features)
        if self.expression.ndim != 2 or self.expression.shape[0] != n:
            raise ValueError("Expression must be [n_slides, n_genes].")
        if not torch.isfinite(self.expression).all():
            raise ValueError("Expression contains NaN or infinite values.")
        if self.masks is not None and self.masks.shape != self.expression.shape:
            raise ValueError("An equation mask must have the same shape as expression.")
        if self.masks is not None and not torch.all(
            (self.masks == 0) | (self.masks == 1)
        ):
            raise ValueError("Equation masks must contain only zero and one.")
        if len(self.slide_ids) != n:
            raise ValueError("The number of slide IDs is inconsistent.")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        feature = resolve_feature(self.features[index], self.data_directory)
        mask = (
            torch.ones_like(self.expression[index])
            if self.masks is None
            else self.masks[index]
        )
        return feature, self.expression[index], mask, self.slide_ids[index]


class FeatureOnlyDataset(Dataset):
    def __init__(
        self,
        features: Sequence[Any],
        slide_ids: Sequence[str],
        data_directory: Path,
    ):
        self.features = features
        self.slide_ids = [str(value) for value in slide_ids]
        self.data_directory = data_directory

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return (
            resolve_feature(self.features[index], self.data_directory),
            self.slide_ids[index],
        )


def collate_paired(batch):
    if len(batch) != 1:
        raise ValueError("Variable-length WSI training requires batch_size=1.")
    feature, expression, mask, slide_id = batch[0]
    return (
        feature.unsqueeze(0),
        expression.unsqueeze(0),
        mask.unsqueeze(0),
        [slide_id],
    )


def collate_features(batch):
    if len(batch) != 1:
        raise ValueError("Variable-length WSI inference requires batch_size=1.")
    feature, slide_id = batch[0]
    return feature.unsqueeze(0), [slide_id]


def fit_expression_standardizer(expression: torch.Tensor) -> Dict[str, torch.Tensor]:
    mean = expression.mean(dim=0)
    scale = expression.std(dim=0, unbiased=False).clamp_min(1e-6)
    return {"mean": mean, "scale": scale}


def standardize(expression: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    return (expression.float() - stats["mean"]) / stats["scale"]


def get_split(
    payload: Dict[str, Any],
    split: str,
    data_directory: Path,
    stats: Optional[Dict[str, torch.Tensor]],
    require_ids: bool,
) -> PairedSlideDataset:
    feature_key = f"{split}_features"
    expression_key = f"{split}_Y"
    if feature_key not in payload or expression_key not in payload:
        raise KeyError(f"Missing {feature_key!r} or {expression_key!r}.")

    features = as_feature_sequence(payload[feature_key])
    expression = torch.as_tensor(payload[expression_key]).float()
    if stats is not None:
        expression = standardize(expression, stats)
    masks = payload.get(f"{split}_mask")
    if masks is not None:
        masks = torch.as_tensor(masks)
    ids = payload.get(f"{split}_ids")
    if ids is None:
        if require_ids:
            raise KeyError(
                f"Missing {split}_ids. Stable patient/case IDs are required "
                "to audit split leakage."
            )
        ids = [f"{split}_{index}" for index in range(len(features))]
    return PairedSlideDataset(
        features=features,
        expression=expression,
        masks=masks,
        slide_ids=ids,
        data_directory=data_directory,
    )


def validate_patient_splits(*datasets: PairedSlideDataset) -> None:
    seen = set()
    for dataset in datasets:
        if len(set(dataset.slide_ids)) != len(dataset.slide_ids):
            raise ValueError("Duplicate patient/case IDs exist within a split.")
        ids = set(dataset.slide_ids)
        overlap = seen.intersection(ids)
        if overlap:
            examples = sorted(overlap)[:5]
            raise ValueError(f"Patient/slide leakage across splits: {examples}")
        seen.update(ids)


def make_loader(
    dataset: Dataset,
    shuffle: bool,
    num_workers: int,
    inference: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collate_features if inference else collate_paired,
    )


def detach_predictions(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clean = []
    for item in predictions:
        clean.append(
            {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in item.items()
            }
        )
    return clean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dag"))
    args = parser.parse_args()

    config = load_config(args.config)
    payload = torch_load(args.data)
    if not isinstance(payload, dict):
        raise TypeError("The input .pt file must contain a dictionary.")

    seed = int(config.get("seed", 0))
    pl.seed_everything(seed, workers=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_train_y = torch.as_tensor(payload["train_Y"]).float()
    standardize_y = bool(config.get("data", {}).get("standardize_Y", True))
    require_ids = bool(config.get("data", {}).get("require_ids", True))
    expression_stats = (
        fit_expression_standardizer(raw_train_y) if standardize_y else None
    )
    data_directory = args.data.resolve().parent
    train_data = get_split(
        payload, "train", data_directory, expression_stats, require_ids
    )
    val_data = get_split(
        payload, "val", data_directory, expression_stats, require_ids
    )
    test_data = get_split(
        payload, "test", data_directory, expression_stats, require_ids
    )
    validate_patient_splits(train_data, val_data, test_data)

    first_feature = resolve_feature(train_data.features[0], data_directory)
    n_genes = train_data.expression.shape[1]
    model_cfg = dict(config["model"])
    configured_genes = int(model_cfg.get("n_genes", n_genes))
    configured_dim = int(model_cfg.get("feat_dim", first_feature.shape[1]))
    if configured_genes != n_genes:
        raise ValueError(f"Configured n_genes={configured_genes}, data has {n_genes}.")
    if configured_dim != first_feature.shape[1]:
        raise ValueError(
            f"Configured feat_dim={configured_dim}, data has {first_feature.shape[1]}."
        )
    model_cfg["n_genes"] = n_genes
    model_cfg["feat_dim"] = first_feature.shape[1]

    gene_ids = payload.get("gene_ids")
    if gene_ids is None:
        gene_ids = [f"gene_{index}" for index in range(n_genes)]

    module = ModelInterfaceRegression(
        model_cfg=model_cfg,
        optimizer_cfg=config.get("optimizer", {}),
        data_cfg=config.get("data", {}),
        dag_cfg=config.get("dag", {}),
        gene_ids=gene_ids,
    )

    num_workers = int(config.get("data", {}).get("num_workers", 0))
    train_loader = make_loader(train_data, True, num_workers)
    val_loader = make_loader(val_data, False, num_workers)
    test_loader = make_loader(test_data, False, num_workers)

    is_dag = module.is_dag_model
    monitor = "val_feasible_loss" if is_dag else "val_loss"
    checkpoint = ModelCheckpoint(
        dirpath=args.output_dir / "checkpoints",
        filename="best-{epoch:04d}",
        monitor=monitor,
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    callbacks = [checkpoint]
    patience = config.get("trainer", {}).get("early_stopping_patience")
    # Generic early stopping can terminate during the DAG preselection stage.
    # DAG runs therefore use the configured finite two-stage epoch budget.
    if patience is not None and not is_dag:
        callbacks.append(
            EarlyStopping(monitor=monitor, mode="min", patience=int(patience))
        )

    trainer_cfg = dict(config.get("trainer", {}))
    trainer_cfg.pop("early_stopping_patience", None)
    if "max_epochs" not in trainer_cfg:
        if is_dag:
            trainer_cfg["max_epochs"] = int(config["dag"]["stage1_epochs"]) + int(
                config["dag"]["stage2_epochs"]
            )
        else:
            trainer_cfg["max_epochs"] = 100
    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        callbacks=callbacks,
        **trainer_cfg,
    )
    trainer.fit(module, train_loader, val_loader)

    best_path = checkpoint.best_model_path or checkpoint.last_model_path
    if not best_path:
        raise RuntimeError("Training completed without producing a checkpoint.")
    if is_dag:
        invalid_penalty = float(config["dag"].get("invalid_graph_penalty", 1e6))
        best_score = checkpoint.best_model_score
        if best_score is None or float(best_score) >= invalid_penalty:
            raise RuntimeError(
                "No Stage-II checkpoint had a raw thresholded DAG. "
                "Inspect val_spectral_radius/val_raw_is_dag and adjust training; "
                "cycle-safe post-processing is not accepted as convergence."
            )
    trainer.test(module, test_loader, ckpt_path=best_path)

    inference_data = FeatureOnlyDataset(
        test_data.features, test_data.slide_ids, data_directory
    )
    inference_loader = make_loader(
        inference_data, False, num_workers, inference=True
    )
    predictions = trainer.predict(module, inference_loader, ckpt_path=best_path)

    result = {
        "checkpoint": best_path,
        "model_config": model_cfg,
        "dag_config": config.get("dag", {}),
        "gene_ids": list(gene_ids),
        "test_ids": test_data.slide_ids,
        "expression_standardization": expression_stats,
        "predictions": detach_predictions(predictions),
    }
    output_name = "test_graphs.pt" if is_dag else "test_expression_predictions.pt"
    output_path = args.output_dir / output_name
    torch.save(result, output_path)
    print(f"Saved predictions to {output_path.resolve()}")


if __name__ == "__main__":
    main()

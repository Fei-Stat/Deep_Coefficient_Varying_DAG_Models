"""PyTorch-Lightning interface for the untouched classification baseline."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Iterable, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _model_kwargs(config: Any, excluded: Iterable[str]) -> Dict[str, Any]:
    if isinstance(config, dict):
        items = config.items()
    elif hasattr(config, "items"):
        items = config.items()
    else:
        items = vars(config).items()
    excluded = set(excluded)
    return {key: value for key, value in items if key not in excluded}


class ModelInterface(pl.LightningModule):
    """Classification only; the DAG never passes through this interface."""

    def __init__(
        self,
        model_cfg: Any,
        optimizer_cfg: Any = None,
        data_cfg: Any = None,
        **_: Any,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model_cfg", "optimizer_cfg", "data_cfg"])
        self.model_cfg = model_cfg
        self.optimizer_cfg = optimizer_cfg
        self.data_cfg = data_cfg
        self.model = self._load_model(model_cfg)
        self._validation_outputs = []
        self._test_outputs = []

    @staticmethod
    def _load_model(model_cfg: Any):
        module_name = _cfg_get(model_cfg, "name", "TransMIL")
        class_name = _cfg_get(model_cfg, "class_name", "TransMIL")
        module = importlib.import_module(f"models.{module_name}")
        model_class = getattr(module, class_name)
        kwargs = _model_kwargs(model_cfg, {"name", "class_name"})
        return model_class(**kwargs)

    def _prepare_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError("Classification batch must be (features, labels, ...).")
        features, labels = batch[0], batch[1]

        if isinstance(features, (list, tuple)):
            if len(features) != 1:
                raise ValueError(
                    "Variable-length WSI bags require batch_size=1."
                )
            features = features[0].unsqueeze(0)
        elif features.ndim == 2:
            features = features.unsqueeze(0)

        labels = labels.long().reshape(-1)
        if features.shape[0] != labels.shape[0]:
            raise ValueError("Feature and label batch sizes differ.")
        return features.to(self.device), labels.to(self.device)

    def forward(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(data=data)

    def _shared_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        features, labels = self._prepare_batch(batch)
        output = self(features)
        logits = output["logits"]
        loss = F.cross_entropy(logits, labels)
        prediction = logits.argmax(dim=-1)
        return {"loss": loss, "prediction": prediction, "target": labels}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        result = self._shared_step(batch)
        self.log(
            "train_loss", result["loss"], on_step=False, on_epoch=True,
            prog_bar=True, batch_size=result["target"].shape[0]
        )
        return result["loss"]

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        result = self._shared_step(batch)
        self._validation_outputs.append(
            {key: value.detach().cpu() for key, value in result.items()}
        )

    def on_validation_epoch_end(self) -> None:
        if not self._validation_outputs:
            return
        losses = torch.stack([item["loss"] for item in self._validation_outputs])
        predictions = torch.cat(
            [item["prediction"] for item in self._validation_outputs]
        )
        targets = torch.cat([item["target"] for item in self._validation_outputs])
        accuracy = (predictions == targets).float().mean()
        self.log("val_loss", losses.mean().to(self.device), prog_bar=True)
        self.log("val_accuracy", accuracy.to(self.device), prog_bar=True)
        self._validation_outputs.clear()

    def test_step(self, batch: Any, batch_idx: int) -> None:
        result = self._shared_step(batch)
        self._test_outputs.append(
            {key: value.detach().cpu() for key, value in result.items()}
        )

    def on_test_epoch_end(self) -> None:
        if not self._test_outputs:
            return
        predictions = torch.cat([x["prediction"] for x in self._test_outputs])
        targets = torch.cat([x["target"] for x in self._test_outputs])
        accuracy = (predictions == targets).float().mean()
        self.log("test_accuracy", accuracy.to(self.device))
        self._test_outputs.clear()

    def configure_optimizers(self):
        name = str(_cfg_get(self.optimizer_cfg, "name", "AdamW")).lower()
        learning_rate = float(_cfg_get(self.optimizer_cfg, "lr", 1e-4))
        weight_decay = float(_cfg_get(self.optimizer_cfg, "weight_decay", 1e-4))
        if name == "adam":
            return torch.optim.Adam(
                self.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        if name == "adamw":
            return torch.optim.AdamW(
                self.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        if name == "sgd":
            momentum = float(_cfg_get(self.optimizer_cfg, "momentum", 0.9))
            return torch.optim.SGD(
                self.parameters(), lr=learning_rate,
                weight_decay=weight_decay, momentum=momentum
            )
        raise ValueError(f"Unsupported optimizer: {name}.")

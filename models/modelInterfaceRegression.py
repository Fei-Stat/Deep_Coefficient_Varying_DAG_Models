"""Lightning interface for both the regression baseline and proposed DAG model.

Select the model through configuration:

Regression baseline::

    name: TransMIL_regression
    class_name: TransMILRegression

Proposed model::

    name: TransMIL_regression
    class_name: TransMILDAG

The two objectives never run simultaneously.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Iterable, Optional, Tuple

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


def _pearson_per_gene(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    prediction = prediction.float()
    target = target.float()
    prediction = prediction - prediction.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    numerator = (prediction * target).sum(dim=0)
    denominator = torch.sqrt(
        prediction.square().sum(dim=0) * target.square().sum(dim=0)
    )
    return numerator / denominator.clamp_min(eps)


class ModelInterfaceRegression(pl.LightningModule):
    def __init__(
        self,
        model_cfg: Any,
        optimizer_cfg: Any = None,
        data_cfg: Any = None,
        dag_cfg: Any = None,
        gene_ids: Optional[Iterable[str]] = None,
        **_: Any,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["model_cfg", "optimizer_cfg", "data_cfg", "dag_cfg"]
        )
        self.model_cfg = model_cfg
        self.optimizer_cfg = optimizer_cfg
        self.data_cfg = data_cfg
        self.dag_cfg = dag_cfg
        self.model = self._load_model(model_cfg)
        self.is_dag_model = bool(getattr(self.model, "is_dag_model", False))
        # Two independent optimizers are required for the two SDCD stages.
        # Lightning permits multiple optimizers only with manual optimization.
        self.automatic_optimization = not self.is_dag_model

        n_genes = int(getattr(self.model, "n_genes"))
        self.gene_ids = (
            [str(item) for item in gene_ids]
            if gene_ids is not None
            else [f"gene_{index}" for index in range(n_genes)]
        )
        if len(self.gene_ids) != n_genes:
            raise ValueError(
                "len(gene_ids) must equal the model's n_genes: "
                f"{len(self.gene_ids)} != {n_genes}."
            )

        self._validation_outputs = []
        self._test_outputs = []
        self._screening_applied = False
        self._gamma_cap: Optional[float] = None

    @staticmethod
    def _load_model(model_cfg: Any):
        module_name = _cfg_get(model_cfg, "name", "TransMIL_regression")
        class_name = _cfg_get(
            model_cfg, "class_name", "TransMILRegression"
        )
        module = importlib.import_module(f"models.{module_name}")
        model_class = getattr(module, class_name)
        kwargs = _model_kwargs(model_cfg, {"name", "class_name"})
        return model_class(**kwargs)

    @property
    def stage1_epochs(self) -> int:
        return int(_cfg_get(self.dag_cfg, "stage1_epochs", 300))

    def _dag_hyperparameters(self) -> Tuple[float, float]:
        if not self.is_dag_model:
            return 0.0, 0.0
        if self.current_epoch < self.stage1_epochs:
            group = float(_cfg_get(self.dag_cfg, "lambda_group_stage1", 1e-3))
            return group, 0.0

        group = float(_cfg_get(self.dag_cfg, "lambda_group_stage2", 1e-3))
        increment = float(_cfg_get(self.dag_cfg, "gamma_increment", 5e-3))
        schedule = str(_cfg_get(self.dag_cfg, "gamma_schedule", "linear"))
        stage2_index = self.current_epoch - self.stage1_epochs
        x = increment * stage2_index
        if schedule == "linear":
            gamma = x
        elif schedule == "power_2":
            gamma = x * x
        elif schedule == "exponential":
            gamma = float(torch.expm1(torch.tensor(x)).item())
        else:
            raise ValueError(
                "gamma_schedule must be linear, power_2, or exponential."
            )
        if self._gamma_cap is not None:
            gamma = self._gamma_cap
        return group, gamma

    def _prepare_batch(
        self, batch: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Any]:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError(
                "Batch must be (features, gene_expression, optional_mask, "
                "optional_slide_id)."
            )
        features, target = batch[0], batch[1]
        mask = None
        slide_id = None

        for extra in batch[2:]:
            if torch.is_tensor(extra) and tuple(extra.shape) == tuple(target.shape):
                if mask is not None:
                    raise ValueError("Batch contains more than one equation mask.")
                mask = extra
            elif slide_id is None:
                slide_id = extra

        if isinstance(features, (list, tuple)):
            if len(features) != 1:
                raise ValueError(
                    "Variable-length WSI bags currently require batch_size=1."
                )
            features = features[0].unsqueeze(0)
        elif features.ndim == 2:
            features = features.unsqueeze(0)

        target = target.float()
        if target.ndim == 1:
            target = target.unsqueeze(0)
        if features.ndim != 3 or target.ndim != 2:
            raise ValueError(
                "Expected features [B,N,D] and expression [B,n_genes]."
            )
        if features.shape[0] != target.shape[0]:
            raise ValueError("Feature and expression batch sizes differ.")

        features = features.to(self.device)
        target = target.to(self.device)
        if mask is not None:
            mask = mask.float().to(self.device)
        return features, target, mask, slide_id

    def forward(
        self,
        data: torch.Tensor,
        Y: Optional[torch.Tensor] = None,
        observational_mask: Optional[torch.Tensor] = None,
        return_discrete_graphs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if self.is_dag_model:
            lambda_group, gamma = self._dag_hyperparameters()
            return self.model(
                data=data,
                Y=Y,
                observational_mask=observational_mask,
                lambda_group=lambda_group,
                gamma_acyclicity=gamma,
                return_discrete_graphs=return_discrete_graphs,
            )
        return self.model(data=data)

    def _shared_step(
        self,
        batch: Any,
        return_discrete_graphs: bool = False,
    ) -> Dict[str, Any]:
        features, target, mask, slide_id = self._prepare_batch(batch)

        if not self.is_dag_model:
            output = self.model(data=features)
            prediction = output["preds"]
            if prediction.shape != target.shape:
                raise ValueError(
                    f"preds {tuple(prediction.shape)} != target {tuple(target.shape)}."
                )
            loss = F.mse_loss(prediction, target)
            return {
                "loss": loss,
                "prediction": prediction,
                "target": target,
                "slide_id": slide_id,
            }

        lambda_group, gamma = self._dag_hyperparameters()
        output = self.model(
            data=features,
            Y=target,
            observational_mask=mask,
            lambda_group=lambda_group,
            gamma_acyclicity=gamma,
            return_discrete_graphs=return_discrete_graphs,
        )
        return {
            "loss": output["loss"],
            "prediction": output["Y_hat"],
            "target": target,
            "observational_mask": (
                torch.ones_like(target) if mask is None else mask
            ),
            "reconstruction_loss": output["reconstruction_loss"],
            "group_penalty": output["group_penalty"],
            "acyclicity_penalty": output["acyclicity_penalty"],
            "beta": output["beta"],
            "context": output["context"],
            "support": output["support"],
            "intercept": output["intercept"],
            "raw_global_adjacency": output.get("raw_global_adjacency"),
            "global_adjacency": output.get("global_adjacency"),
            "raw_patient_adjacency": output.get("raw_patient_adjacency"),
            "patient_adjacency": output.get("patient_adjacency"),
            "slide_id": slide_id,
            "lambda_group": lambda_group,
            "gamma_acyclicity": gamma,
        }

    def on_train_epoch_start(self) -> None:
        if not self.is_dag_model:
            return

        # The optional hard screen is applied exactly once between SDCD stages.
        use_screening = bool(_cfg_get(self.dag_cfg, "use_screening", True))
        if (
            use_screening
            and not self._screening_applied
            and self.current_epoch >= self.stage1_epochs
        ):
            threshold = float(
                _cfg_get(self.dag_cfg, "screening_threshold", 1e-3)
            )
            dynamic_dag = self.model.dynamic_dag
            with torch.no_grad():
                mask = (
                    dynamic_dag.support_matrix() > threshold
                ).to(dynamic_dag.structural_mask)
                mask = mask * dynamic_dag.off_diagonal
            if int(mask.sum().item()) == 0:
                raise RuntimeError(
                    "Stage-I screening removed every edge; lower "
                    "screening_threshold."
                )
            dynamic_dag.set_structural_mask(mask)
            self._screening_applied = True

    def on_before_optimizer_step(self, optimizer) -> None:
        if self.is_dag_model:
            return
        clip_value = _cfg_get(self.optimizer_cfg, "grad_clip", None)
        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), float(clip_value)
            )

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        if self.is_dag_model:
            self.model.dynamic_dag.zero_forbidden_edges()

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        result = self._shared_step(batch)
        batch_size = result["target"].shape[0]

        if self.is_dag_model:
            stage1_optimizer, stage2_optimizer = self.optimizers()
            optimizer = (
                stage1_optimizer
                if self.current_epoch < self.stage1_epochs
                else stage2_optimizer
            )
            optimizer.zero_grad()
            self.manual_backward(result["loss"])
            clip_value = _cfg_get(self.optimizer_cfg, "grad_clip", 5.0)
            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), float(clip_value)
                )
            optimizer.step()
            self.model.dynamic_dag.zero_forbidden_edges()

        self.log(
            "train_loss", result["loss"], on_step=False, on_epoch=True,
            prog_bar=True, batch_size=batch_size
        )
        if self.is_dag_model:
            self.log(
                "train_reconstruction", result["reconstruction_loss"],
                on_step=False, on_epoch=True, batch_size=batch_size
            )
            self.log(
                "gamma_acyclicity", float(result["gamma_acyclicity"]),
                on_step=False, on_epoch=True, batch_size=batch_size
            )
        return result["loss"].detach() if self.is_dag_model else result["loss"]

    @staticmethod
    def _cpu_record(result: Dict[str, Any]) -> Dict[str, Any]:
        record = {}
        for key, value in result.items():
            if torch.is_tensor(value):
                record[key] = value.detach().cpu()
            elif key in {"slide_id", "lambda_group", "gamma_acyclicity"}:
                record[key] = value
        return record

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        result = self._shared_step(batch, return_discrete_graphs=False)
        self._validation_outputs.append(self._cpu_record(result))

    def on_validation_epoch_end(self) -> None:
        if not self._validation_outputs:
            return
        prediction = torch.cat([x["prediction"] for x in self._validation_outputs])
        target = torch.cat([x["target"] for x in self._validation_outputs])

        if self.is_dag_model:
            # Checkpoint selection is based on reconstruction among raw DAGs.
            mask = torch.cat(
                [x["observational_mask"] for x in self._validation_outputs]
            )
            reconstruction = (
                (prediction - target).square() * mask
            ).sum() / mask.sum().clamp_min(1.0)
            support = self.model.dynamic_dag.support_matrix().detach().cpu()
            rho = self.model.dynamic_dag.spectral.estimate(
                self.model.dynamic_dag.support_matrix()
            )
            raw_is_dag = self.model.raw_global_is_dag()
            raw_adjacency = self.model.raw_global_adjacency()
            invalid_penalty = float(
                _cfg_get(self.dag_cfg, "invalid_graph_penalty", 1e6)
            )
            checkpoint_eligible = (
                self.current_epoch >= self.stage1_epochs and raw_is_dag
            )
            feasible_loss = reconstruction + (
                0.0 if checkpoint_eligible else invalid_penalty
            )

            freeze_gamma = bool(
                _cfg_get(self.dag_cfg, "freeze_gamma_at_dag", True)
            )
            freeze_threshold = float(
                _cfg_get(self.dag_cfg, "freeze_gamma_threshold", 0.01)
            )
            warmup = int(_cfg_get(self.dag_cfg, "gamma_warmup_epochs", 2))
            if freeze_threshold > self.model.global_edge_threshold:
                raise ValueError(
                    "freeze_gamma_threshold must be <= global_edge_threshold."
                )
            _, current_gamma = self._dag_hyperparameters()
            if (
                freeze_gamma
                and self.current_epoch >= self.stage1_epochs + warmup
                and self._gamma_cap is None
                and self.model.raw_global_is_dag(freeze_threshold)
            ):
                self._gamma_cap = current_gamma
            elif freeze_gamma and self._gamma_cap is not None and not raw_is_dag:
                self._gamma_cap = None

            self.log("val_loss", reconstruction.to(self.device), prog_bar=True)
            self.log(
                "val_feasible_loss", feasible_loss.to(self.device), prog_bar=False
            )
            self.log("val_spectral_radius", rho.to(self.device), prog_bar=True)
            self.log("val_support_l1", support.sum().to(self.device))
            self.log(
                "val_raw_n_edges",
                torch.tensor(float(raw_adjacency.sum()), device=self.device),
            )
            self.log(
                "val_raw_is_dag",
                torch.tensor(float(raw_is_dag), device=self.device),
                prog_bar=True,
            )
            self.log(
                "val_checkpoint_eligible",
                torch.tensor(float(checkpoint_eligible), device=self.device),
            )
            self.log(
                "gamma_frozen",
                torch.tensor(float(self._gamma_cap is not None), device=self.device),
            )
        else:
            mse = F.mse_loss(prediction, target)
            pearson = _pearson_per_gene(prediction, target).mean()
            self.log("val_loss", mse.to(self.device), prog_bar=True)
            self.log("val_pearson", pearson.to(self.device), prog_bar=True)
        self._validation_outputs.clear()

    def test_step(self, batch: Any, batch_idx: int) -> None:
        result = self._shared_step(
            batch, return_discrete_graphs=self.is_dag_model
        )
        self._test_outputs.append(self._cpu_record(result))

    def on_test_epoch_end(self) -> None:
        if not self._test_outputs:
            return
        prediction = torch.cat([x["prediction"] for x in self._test_outputs])
        target = torch.cat([x["target"] for x in self._test_outputs])
        if self.is_dag_model:
            mask = torch.cat(
                [x["observational_mask"] for x in self._test_outputs]
            )
            mse = ((prediction - target).square() * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            mse = F.mse_loss(prediction, target)
            pearson = _pearson_per_gene(prediction, target).mean()
            self.log("test_pearson", pearson.to(self.device))
        self.log("test_reconstruction_mse", mse.to(self.device))
        self._test_outputs.clear()

    @torch.no_grad()
    def infer_graph(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """WSI-only inference; no expression value is used here."""
        if not self.is_dag_model:
            raise RuntimeError("infer_graph is available only for TransMILDAG.")
        if features.ndim == 2:
            features = features.unsqueeze(0)
        return self.model.infer_graph(features.to(self.device))

    def predict_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        # A prediction loader may contain features only or (features, metadata).
        if torch.is_tensor(batch):
            features = batch
            metadata = None
        elif isinstance(batch, (tuple, list)):
            features = batch[0]
            metadata = batch[1:] if len(batch) > 1 else None
        else:
            raise TypeError("Prediction batch must contain WSI feature tensors.")

        if isinstance(features, (tuple, list)):
            if len(features) != 1:
                raise ValueError("Variable-length WSI prediction uses batch_size=1.")
            features = features[0].unsqueeze(0)
        elif features.ndim == 2:
            features = features.unsqueeze(0)

        if self.is_dag_model:
            output = self.model.infer_graph(features.to(self.device))
        else:
            output = self.model(data=features.to(self.device))
        result = {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in output.items()
        }
        result["metadata"] = metadata
        return result

    def configure_optimizers(self):
        name = str(_cfg_get(self.optimizer_cfg, "name", "AdamW")).lower()
        weight_decay = float(_cfg_get(self.optimizer_cfg, "weight_decay", 1e-4))
        if self.is_dag_model:
            stage1_lr = float(_cfg_get(self.dag_cfg, "stage1_lr", 2e-4))
            stage2_lr = float(_cfg_get(self.dag_cfg, "stage2_lr", 1e-4))
            optimizer_class = {
                "adam": torch.optim.Adam,
                "adamw": torch.optim.AdamW,
            }.get(name)
            if optimizer_class is None:
                raise ValueError("DAG training supports Adam or AdamW.")
            # These are intentionally separate instances with separate moments.
            return [
                optimizer_class(
                    self.parameters(), lr=stage1_lr, weight_decay=weight_decay
                ),
                optimizer_class(
                    self.parameters(), lr=stage2_lr, weight_decay=weight_decay
                ),
            ]

        learning_rate = float(_cfg_get(self.optimizer_cfg, "lr", 1e-4))
        if name == "adam":
            return torch.optim.Adam(
                self.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        if name == "adamw":
            return torch.optim.AdamW(
                self.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        raise ValueError("Regression/DAG interface supports Adam or AdamW.")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["dag_training_state"] = {
            "screening_applied": self._screening_applied,
            "gamma_cap": self._gamma_cap,
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        state = checkpoint.get("dag_training_state", {})
        self._screening_applied = bool(state.get("screening_applied", False))
        gamma_cap = state.get("gamma_cap")
        self._gamma_cap = None if gamma_cap is None else float(gamma_cap)

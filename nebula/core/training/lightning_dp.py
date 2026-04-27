import logging
import traceback

import torch

from nebula.config.config import TRAINING_LOGGER
from nebula.core.training.lightning import Lightning
from nebula.core.training.dp import DifferentialPrivacyPlugin, SimpleDPState

logging_training = logging.getLogger(TRAINING_LOGGER)


class LightningDP(Lightning):
    """
    Lightning-based trainer with Differential Privacy support.

    This class inherits the standard Nebula Lightning trainer.
    """

    def __init__(self, model, datamodule, config=None):
        super().__init__(model, datamodule, config)
        self._dp_plugin = self.create_dp_plugin()
        self.dp_epsilon = None
        self.dp_delta = None

    def create_dp_plugin(self):
        dp_config = self.config.participant["training_args"].get("dp")

        if dp_config is None or not dp_config.get("enabled", False):
            raise ValueError("LightningDP was selected, but Differential Privacy is not enabled in the configuration.")

        return DifferentialPrivacyPlugin(
            noise_multiplier=dp_config["noise_multiplier"],
            max_grad_norm=dp_config["max_grad_norm"],
            target_delta=dp_config["target_delta"],
            accountant=dp_config["accountant"],
            secure_mode=dp_config["secure_mode"],
            poisson_sampling=dp_config["poisson_sampling"],
            clipping=dp_config["clipping"],
        )

    def _train_sync(self):
        try:
            self._fit_with_dp()

            validation_metrics = {}
            if hasattr(self.model, "get_latest_validation_metrics"):
                validation_metrics = self.model.get_latest_validation_metrics() or {}

            loss = None
            model_loss = getattr(self.model, "get_loss", None)
            if callable(model_loss):
                raw_loss = model_loss()
                loss = raw_loss.item() if hasattr(raw_loss, "item") else raw_loss

            accuracy = validation_metrics.get("Validation/Accuracy")
            return loss, accuracy

        except Exception as e:
            logging_training.error(f"Error in _train_sync with Differential Privacy: {e}")
            tb = traceback.format_exc()
            logging_training.error(f"Traceback: {tb}")
            raise

    def _get_training_device(self):
        if (
            self.config.participant["device_args"]["accelerator"] == "gpu"
            and torch.cuda.is_available()
            and self.config.participant["device_args"]["gpu_id"]
        ):
            return torch.device(f"cuda:{self.config.participant['device_args']['gpu_id'][0]}")

        return torch.device("cpu")

    def _log_manual_metrics(self, phase, metrics):
        output = metrics.compute()
        output = {
            f"{phase}/{key.replace('Multiclass', '').split('/')[-1]}": value.detach()
            for key, value in output.items()
        }

        if phase == "Validation":
            self.model._latest_validation_metrics = {
                key: float(value.detach().cpu().item())
                for key, value in output.items()
            }

        self._logger.log_data(output, step=self.model.global_number[phase])

    def _fit_with_dp(self):
        state = SimpleDPState()
        original_configure_optimizers = self.model.configure_optimizers

        self.model.train()
        self.datamodule.setup("fit")
        train_dataloader = self.datamodule.train_dataloader()
        val_dataloader = self.datamodule.val_dataloader()

        optimizer = self.model.configure_optimizers()
        state.extras["dataloader"] = train_dataloader

        try:
            self._dp_plugin.on_train_start(self.model, optimizer, state)

            private_model = state.extras["model"]
            private_optimizer = state.extras["optimizer"]
            private_dataloader = state.extras["dataloader"]

            self.model._optimizer = private_optimizer

            def configure_private_optimizers():
                return private_optimizer

            self.model.configure_optimizers = configure_private_optimizers

            self._trainer.fit(
                self.model,
                train_dataloaders=private_dataloader,
                val_dataloaders=val_dataloader,
            )

            if hasattr(private_model, "_module"):
                self.model.load_state_dict(private_model._module.state_dict())
            else:
                self.model.load_state_dict(private_model.state_dict())

            self.model.train()

        finally:
            self.model.configure_optimizers = original_configure_optimizers
            self._dp_plugin.on_train_end(state)
            self.datamodule.teardown("fit")

        dp_epsilon = state.extras.get("dp_epsilon")

        if dp_epsilon is not None:
            dp_delta = state.extras["dp_delta"]

            self.dp_epsilon = float(dp_epsilon)
            self.dp_delta = float(dp_delta)

            if self._logger is not None:
                self._logger.log_data(
                    {
                        "DP/Epsilon": dp_epsilon,
                        "DP/Delta": dp_delta,
                    }
                )

            logging_training.info(
                f"DP privacy budget | epsilon={dp_epsilon:.4f} | delta={dp_delta}"
            )

    def get_privacy_metrics(self):
        return {
            "dp_enabled": True,
            "dp_epsilon": self.dp_epsilon,
            "dp_delta": self.dp_delta,
        }

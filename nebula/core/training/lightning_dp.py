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

    This class inherits the standard Nebula Lightning trainer but overrides
    the synchronous training logic because Opacus needs to privatize the
    model, optimizer and dataloader before the training loop starts.
    """

    def __init__(self, model, datamodule, config=None):
        super().__init__(model, datamodule, config)
        self._dp_plugin = self.create_dp_plugin()

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
            return None, None

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
        train_dataloader = self.datamodule.train_dataloader()
        val_dataloader = self.datamodule.val_dataloader()

        state = SimpleDPState()
        state.extras["dataloader"] = train_dataloader

        self.model.train()
        optimizer = self.model.configure_optimizers()
        device = self._get_training_device()

        try:
            self._dp_plugin.on_train_start(self.model, optimizer, state)

            private_model = state.extras["model"]
            private_optimizer = state.extras["optimizer"]
            private_dataloader = state.extras["dataloader"]

            self.model._optimizer = private_optimizer
            private_model.to(device)

            for epoch in range(self.epochs):
                logging_training.info(f"Starting Epoch {epoch} DP")

                private_model.train()

                for batch_idx, batch in enumerate(private_dataloader):
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    private_optimizer.zero_grad()

                    outputs = private_model(inputs)
                    loss = self.model.criterion(outputs, labels)

                    self.model._current_loss = loss.detach()

                    if self._logger is not None:
                        self._logger.log_data({"Train/Loss": loss.detach()})

                    self.model.train_metrics.update(
                        torch.argmax(outputs.detach(), dim=1),
                        labels.detach(),
                    )

                    loss.backward()
                    private_optimizer.step()

                self._log_manual_metrics("Train", self.model.train_metrics)
                self.model.train_metrics.reset()
                self.model.global_number["Train"] += 1

                logging_training.info(f"Epoch {epoch} finished DP")

                logging_training.info(f"Starting validation for Epoch {epoch} DP")

                private_model.eval()

                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_dataloader):
                        inputs, labels = batch
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outputs = private_model(inputs)
                        loss = self.model.criterion(outputs, labels)

                        self.model._current_loss = loss.detach()

                        self.model.val_metrics.update(
                            torch.argmax(outputs.detach(), dim=1),
                            labels.detach(),
                        )

                self._log_manual_metrics("Validation", self.model.val_metrics)
                self.model.val_metrics.reset()
                self.model.global_number["Validation"] += 1

                logging_training.info(f"Validation for Epoch {epoch} finished DP")

            if hasattr(private_model, "_module"):
                self.model.load_state_dict(private_model._module.state_dict())
            else:
                self.model.load_state_dict(private_model.state_dict())

            self.model.train()

        finally:
            self._dp_plugin.on_train_end(state)

        dp_epsilon = state.extras.get("dp_epsilon")

        if dp_epsilon is not None:
            dp_delta = state.extras["dp_delta"]

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

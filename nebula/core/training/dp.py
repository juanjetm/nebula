class SimpleDPState:
    def __init__(self):
        self.extras = {}


class DifferentialPrivacyPlugin:
    name = "differential_privacy"

    def __init__(
        self,
        *,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        target_delta=1e-5,
        accountant="prv",
        secure_mode=False,
        poisson_sampling=True,
        clipping="flat",
    ):
        self.noise_multiplier = float(noise_multiplier)
        self.max_grad_norm = float(max_grad_norm)
        self.target_delta = target_delta
        self.accountant = accountant
        self.secure_mode = bool(secure_mode)
        self.poisson_sampling = bool(poisson_sampling)
        self.clipping = clipping
        self._privacy_engine = None

    def on_train_start(self, model, optimizer, state):
        from opacus import PrivacyEngine

        dataloader = state.extras["dataloader"]
        model.train()

        if self._privacy_engine is None:
            self._privacy_engine = PrivacyEngine(
                accountant=self.accountant,
                secure_mode=self.secure_mode,
            )
        privacy_engine = self._privacy_engine

        private_model, private_optimizer, private_dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            poisson_sampling=self.poisson_sampling,
            clipping=self.clipping,
        )

        state.extras["privacy_engine"] = privacy_engine
        state.extras["model"] = private_model
        state.extras["optimizer"] = private_optimizer
        state.extras["dataloader"] = private_dataloader

    def on_train_end(self, state):
        privacy_engine = state.extras.get("privacy_engine")
        private_model = state.extras.get("model")

        if privacy_engine is not None and self.target_delta is not None:
            try:
                epsilon = privacy_engine.get_epsilon(delta=self.target_delta)
                state.extras["dp_epsilon"] = float(epsilon)
                state.extras["dp_delta"] = float(self.target_delta)
            except Exception:
                pass

        if private_model is not None:
            try:
                private_model.zero_grad(set_to_none=True)
            except Exception:
                pass

            try:
                private_model.forbid_grad_accumulation()
            except Exception:
                pass

            try:
                private_model.disable_hooks()
            except Exception:
                pass

            try:
                private_model.remove_hooks()
            except Exception:
                pass

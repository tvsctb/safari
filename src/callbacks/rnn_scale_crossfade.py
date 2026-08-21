"""Optimizer-group LR crossfade for learned RNN AUX scales."""

from __future__ import annotations

from pytorch_lightning import Callback


class RNNScaleCrossfade(Callback):
    """Decay target LR while raising rho/tau LR without a hard boundary."""

    def __init__(
        self,
        enabled=False,
        start_step=0,
        transition_steps=12560,
        target_lr_initial=1e-3,
        target_lr_final=0.0,
        gaussian_lr_initial=0.0,
        gaussian_lr_final=1e-4,
        log_interval=157,
    ):
        super().__init__()
        if start_step < 0 or transition_steps <= 0:
            raise ValueError("invalid RNN scale crossfade interval")
        rates = (
            target_lr_initial,
            target_lr_final,
            gaussian_lr_initial,
            gaussian_lr_final,
        )
        if any(rate < 0 for rate in rates):
            raise ValueError("RNN scale crossfade LRs must be non-negative")
        self.enabled = bool(enabled)
        self.start_step = int(start_step)
        self.transition_steps = int(transition_steps)
        self.target_lr_initial = float(target_lr_initial)
        self.target_lr_final = float(target_lr_final)
        self.gaussian_lr_initial = float(gaussian_lr_initial)
        self.gaussian_lr_final = float(gaussian_lr_final)
        self.log_interval = int(log_interval)

    def _rates(self, step):
        x = min(
            1.0,
            max(0.0, (step - self.start_step) / self.transition_steps),
        )
        progress = x * x * (3.0 - 2.0 * x)
        target_lr = self.target_lr_initial + progress * (
            self.target_lr_final - self.target_lr_initial
        )
        gaussian_lr = self.gaussian_lr_initial + progress * (
            self.gaussian_lr_final - self.gaussian_lr_initial
        )
        return progress, target_lr, gaussian_lr

    @staticmethod
    def _set_parameter_lr(optimizer, parameter, learning_rate):
        matches = [
            group
            for group in optimizer.param_groups
            if any(candidate is parameter for candidate in group["params"])
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "expected exactly one optimizer group for crossfade parameter"
            )
        matches[0]["lr"] = learning_rate

    def _apply(self, trainer, pl_module):
        if not self.enabled:
            return
        model = pl_module.model
        required = ("log_memory_scale_target", "log_rho", "log_tau")
        missing = [name for name in required if not hasattr(model, name)]
        if missing:
            raise RuntimeError(
                "RNN scale crossfade requires learned target/rho/tau: "
                + ", ".join(missing)
            )
        if len(trainer.optimizers) != 1:
            raise RuntimeError("RNN scale crossfade requires one optimizer")
        progress, target_lr, gaussian_lr = self._rates(trainer.global_step)
        optimizer = trainer.optimizers[0]
        self._set_parameter_lr(
            optimizer, model.log_memory_scale_target, target_lr
        )
        self._set_parameter_lr(optimizer, model.log_rho, gaussian_lr)
        self._set_parameter_lr(optimizer, model.log_tau, gaussian_lr)
        if (
            trainer.logger
            and self.log_interval > 0
            and trainer.global_step % self.log_interval == 0
        ):
            trainer.logger.log_metrics(
                {
                    "train/aux/crossfade_progress": progress,
                    "train/aux/memory_scale_target_lr": target_lr,
                    "train/aux/gaussian_scale_lr": gaussian_lr,
                },
                step=trainer.global_step,
            )

    def on_fit_start(self, trainer, pl_module):
        self._apply(trainer, pl_module)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._apply(trainer, pl_module)

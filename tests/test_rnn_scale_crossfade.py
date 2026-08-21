import unittest
from types import SimpleNamespace

import torch

from src.callbacks.rnn_scale_crossfade import RNNScaleCrossfade
from src.models.sequence.rnn_aux import RNNAuxLM


class RNNScaleCrossfadeTest(unittest.TestCase):
    def test_smoothly_crossfades_target_and_gaussian_parameter_lrs(self):
        model = RNNAuxLM(
            d_model=8,
            n_layer=1,
            vocab_size=20,
            chunk_size=4,
            rho=8.0,
            tau=12.0,
            gaussian_scale_mode="learned",
            gaussian_scale_learning_rate=1e-4,
            memory_scale_target_mode="learned",
            memory_scale_target_learning_rate=1e-3,
            memory_scale_constraint_weight=0.2,
        )
        optimizer = torch.optim.AdamW(
            [
                {"params": [model.log_memory_scale_target], "lr": 1e-3},
                {"params": [model.log_rho, model.log_tau], "lr": 1e-4},
            ]
        )
        trainer = SimpleNamespace(
            optimizers=[optimizer], global_step=0, logger=None
        )
        module = SimpleNamespace(model=model)
        callback = RNNScaleCrossfade(
            enabled=True,
            transition_steps=100,
            target_lr_initial=1e-3,
            target_lr_final=0.0,
            gaussian_lr_initial=0.0,
            gaussian_lr_final=1e-4,
        )

        expected = {
            0: (1e-3, 0.0),
            50: (5e-4, 5e-5),
            100: (0.0, 1e-4),
        }
        for step, (target_lr, gaussian_lr) in expected.items():
            trainer.global_step = step
            callback._apply(trainer, module)
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], target_lr)
            self.assertAlmostEqual(
                optimizer.param_groups[1]["lr"], gaussian_lr
            )

    def test_requires_exact_learned_parameters(self):
        model = RNNAuxLM(
            d_model=8, n_layer=1, vocab_size=20, chunk_size=4
        )
        trainer = SimpleNamespace(
            optimizers=[torch.optim.AdamW(model.parameters())],
            global_step=0,
            logger=None,
        )
        callback = RNNScaleCrossfade(enabled=True)
        with self.assertRaisesRegex(RuntimeError, "learned target/rho/tau"):
            callback._apply(trainer, SimpleNamespace(model=model))


if __name__ == "__main__":
    unittest.main()

import copy
from collections import defaultdict
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.sequence.auxiliary import (
    AuxCausalLMOutput,
    boundary_inverse_targets,
    chunk_ranges,
    cross_entropy_sum,
    directional_reconstruction_diagnostics,
    gaussian_nll_sum,
    memory_observation,
    memory_reconstruction_diagnostics,
    memory_reconstruction_target,
    terminal_gaussian_nll,
    terminal_reconstruction_mse,
    terminal_vmf_nll,
    validate_memory_observation_gradient_scale,
    vmf_log_normalizer_grid,
    vmf_nll_sum,
)


class StackedRNN(nn.Module):
    """A configurable stacked vanilla RNN exposing every layer trajectory.

    Each layer is evaluated over the complete sequence in one native RNN call.
    This keeps the recurrent computation independent of auxiliary chunking while
    avoiding a Python loop over tokens.
    """

    def __init__(
        self,
        d_model,
        n_layer,
        dropout=0.0,
        activation="tanh",
        recurrent_init="orthogonal",
        recurrent_identity_scale=1.0,
    ):
        super().__init__()
        if activation not in {"tanh", "relu"}:
            raise ValueError("activation must be tanh or relu")
        if recurrent_init not in {"orthogonal", "identity"}:
            raise ValueError("recurrent_init must be orthogonal or identity")
        if recurrent_init == "identity" and activation != "relu":
            raise ValueError("identity recurrent initialization requires relu")
        if recurrent_identity_scale <= 0:
            raise ValueError("recurrent_identity_scale must be positive")
        self.d_model = d_model
        self.n_layer = n_layer
        self.dropout = float(dropout)
        self.activation = activation
        self.recurrent_init = recurrent_init
        self.recurrent_identity_scale = float(recurrent_identity_scale)
        self.layers = nn.ModuleList(
            nn.RNN(
                d_model,
                d_model,
                num_layers=1,
                nonlinearity=activation,
                batch_first=True,
            )
            for _ in range(n_layer)
        )

    def reset_parameters(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight_ih_l0)
            if self.recurrent_init == "orthogonal":
                nn.init.orthogonal_(layer.weight_hh_l0)
            else:
                nn.init.eye_(layer.weight_hh_l0)
                with torch.no_grad():
                    layer.weight_hh_l0.mul_(self.recurrent_identity_scale)
            nn.init.zeros_(layer.bias_ih_l0)
            nn.init.zeros_(layer.bias_hh_l0)

    def forward(self, inputs, state, return_trajectory=False):
        if inputs.ndim != 3:
            raise ValueError("RNN inputs must have shape (batch, length, d_model)")
        expected_state = (self.n_layer, inputs.size(0), self.d_model)
        if tuple(state.shape) != expected_state:
            raise ValueError(f"RNN state must have shape {expected_state}")

        layer_input = inputs
        terminal_states = []
        trajectories = [] if return_trajectory else None
        for index, layer in enumerate(self.layers):
            if index > 0 and self.dropout > 0.0:
                layer_input = F.dropout(
                    layer_input, p=self.dropout, training=self.training
                )
            # Learned initial states are broadcast across the batch with
            # ``expand``. CPU kernels accept that zero-stride view, whereas
            # cuDNN requires a contiguous hidden state. Materialize only the
            # per-layer state at the native-RNN boundary; values and gradients
            # remain identical, including accumulation into the shared
            # learned initial state.
            layer_state = state[index : index + 1].contiguous()
            layer_output, terminal_state = layer(
                layer_input, layer_state
            )
            terminal_states.append(terminal_state)
            if return_trajectory:
                trajectories.append(layer_output)
            layer_input = layer_output

        terminal_state = torch.cat(terminal_states, dim=0)
        if not return_trajectory:
            return layer_input, terminal_state, None
        # Keep native layer outputs separate. AUX gathers only boundary states,
        # avoiding an additional full (layers, batch, time, hidden) allocation.
        return layer_input, terminal_state, tuple(trajectories)


class RNNAuxLM(nn.Module):
    """Vanilla RNN with configurable core initialization and inverse AUX."""

    def __init__(
        self,
        d_model,
        n_layer,
        vocab_size,
        chunk_size=4,
        aux_chunk_sizes=None,
        chunk_offset="random",
        dropout=0.0,
        activation="tanh",
        recurrent_init="orthogonal",
        recurrent_identity_scale=1.0,
        rho=1.0,
        tau=1.0,
        state_aux_distribution="gaussian",
        vmf_kappa_mode="fixed",
        memory_vmf_kappa=1.0,
        terminal_vmf_kappa=1.0,
        auxiliary_probe_only=False,
        stop_gradient_memory_target=False,
        stop_gradient_memory_observation=False,
        memory_observation_gradient_scale=1.0,
        use_chunk_loss=True,
        use_discrete_loss=True,
        use_memory_loss=True,
        exclude_initial_memory_reconstruction=True,
        use_terminal_loss=True,
        condition_memory_reconstruction_on_boundary=False,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            raise TypeError(
                "unexpected RNN AUX options: " + ", ".join(sorted(kwargs))
            )
        if d_model <= 0 or n_layer <= 0:
            raise ValueError("d_model and n_layer must be positive")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if aux_chunk_sizes is None:
            aux_chunk_sizes = (chunk_size,)
        if (
            isinstance(aux_chunk_sizes, (str, bytes))
            or not isinstance(aux_chunk_sizes, Sequence)
            or not aux_chunk_sizes
        ):
            raise ValueError("aux_chunk_sizes must be null or a non-empty list")
        aux_chunk_sizes = tuple(aux_chunk_sizes)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in aux_chunk_sizes
        ):
            raise ValueError("aux_chunk_sizes must contain positive integers")
        if len(set(aux_chunk_sizes)) != len(aux_chunk_sizes):
            raise ValueError("aux_chunk_sizes must not contain duplicates")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if rho <= 0 or tau <= 0:
            raise ValueError("rho and tau must be positive")
        if state_aux_distribution not in {"gaussian", "vmf"}:
            raise ValueError("state_aux_distribution must be gaussian or vmf")
        if vmf_kappa_mode not in {"fixed", "learned"}:
            raise ValueError("vmf_kappa_mode must be fixed or learned")
        if memory_vmf_kappa <= 0 or terminal_vmf_kappa <= 0:
            raise ValueError("vMF kappas must be positive")
        if not isinstance(auxiliary_probe_only, bool):
            raise ValueError("auxiliary_probe_only must be a boolean")
        if not isinstance(exclude_initial_memory_reconstruction, bool):
            raise ValueError(
                "exclude_initial_memory_reconstruction must be a boolean"
            )
        if not isinstance(condition_memory_reconstruction_on_boundary, bool):
            raise ValueError(
                "condition_memory_reconstruction_on_boundary must be a boolean"
            )
        if chunk_offset != "random":
            if isinstance(chunk_offset, bool) or not isinstance(chunk_offset, int):
                raise ValueError(
                    "chunk_offset must be 'random' or an integer in [0, chunk_size)"
                )
            if not all(0 <= chunk_offset < value for value in aux_chunk_sizes):
                raise ValueError(
                    "fixed chunk_offset must be valid for every auxiliary chunk size"
                )

        self.d_model = d_model
        self.d_output = vocab_size
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.aux_chunk_sizes = tuple(aux_chunk_sizes)
        self.chunk_offset = chunk_offset
        self.state_aux_distribution = state_aux_distribution
        self.vmf_kappa_mode = vmf_kappa_mode
        self.auxiliary_probe_only = auxiliary_probe_only
        self.stop_gradient_memory_target = stop_gradient_memory_target
        self.stop_gradient_memory_observation = stop_gradient_memory_observation
        self.memory_observation_gradient_scale = (
            validate_memory_observation_gradient_scale(
                memory_observation_gradient_scale
            )
        )
        self.use_chunk_loss = use_chunk_loss
        self.use_discrete_loss = use_discrete_loss
        self.use_memory_loss = use_memory_loss
        self.exclude_initial_memory_reconstruction = (
            exclude_initial_memory_reconstruction
        )
        self.use_terminal_loss = use_terminal_loss
        self.condition_memory_reconstruction_on_boundary = (
            condition_memory_reconstruction_on_boundary
        )

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.initial_state = nn.Parameter(torch.empty(n_layer, 1, d_model))
        self.rnn = StackedRNN(
            d_model,
            n_layer,
            dropout=dropout,
            activation=activation,
            recurrent_init=recurrent_init,
            recurrent_identity_scale=recurrent_identity_scale,
        )
        self.inverse_rnn = copy.deepcopy(self.rnn)
        self.memory_predictor = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        # The terminal prior mean is always learned.  Keeping it allocated when
        # the terminal term is disabled makes parameterization identical across
        # terminal-loss ablations and detached no-AUX probes.
        self.terminal_target = nn.Parameter(
            torch.zeros(n_layer, 1, d_model)
        )
        self.register_buffer("rho", torch.tensor(float(rho)))
        self.register_buffer("tau", torch.tensor(float(tau)))
        if vmf_kappa_mode == "learned":
            self.log_memory_vmf_kappa = nn.Parameter(
                torch.tensor(float(memory_vmf_kappa)).log()
            )
            self.log_terminal_vmf_kappa = nn.Parameter(
                torch.tensor(float(terminal_vmf_kappa)).log()
            )
            self.log_memory_vmf_kappa._no_weight_decay = True
            self.log_terminal_vmf_kappa._no_weight_decay = True
        else:
            self.register_buffer(
                "memory_vmf_kappa",
                torch.tensor(float(memory_vmf_kappa)),
                persistent=False,
            )
            self.register_buffer(
                "terminal_vmf_kappa",
                torch.tensor(float(terminal_vmf_kappa)),
                persistent=False,
            )
        if state_aux_distribution == "vmf":
            log_kappa, log_normalizer = vmf_log_normalizer_grid(n_layer * d_model)
            self.register_buffer(
                "_vmf_log_kappa_grid", log_kappa, persistent=False
            )
            self.register_buffer(
                "_vmf_log_normalizer_grid", log_normalizer, persistent=False
            )
            self.terminal_target._no_weight_decay = True
        self.reset_parameters()
        self.metrics = {}
        self.loss_components = {}
        self.scale_loss_components = {}

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.initial_state, std=0.02)
        if self.state_aux_distribution == "vmf":
            # Draw a nonzero terminal direction without perturbing the forward
            # model's initialization stream relative to the Gaussian control.
            rng_state = torch.random.get_rng_state()
            nn.init.normal_(self.terminal_target, std=0.02)
            torch.random.set_rng_state(rng_state)
        else:
            nn.init.zeros_(self.terminal_target)
        self.rnn.reset_parameters()
        self.inverse_rnn.load_state_dict(self.rnn.state_dict())

        first = self.memory_predictor[0]
        second = self.memory_predictor[2]
        nn.init.xavier_uniform_(first.weight)
        nn.init.zeros_(first.bias)
        # The residual predictor is exactly the identity at initialization.
        nn.init.zeros_(second.weight)
        nn.init.zeros_(second.bias)

    def _lm_logits(self, hidden):
        return F.linear(hidden, self.embedding.weight)

    def _inverse_embedding(self, token_ids):
        weight = self.embedding.weight
        if self.auxiliary_probe_only:
            weight = weight.detach()
        return F.embedding(token_ids, weight)

    def _inverse_logits(self, hidden):
        weight = self.embedding.weight
        if self.auxiliary_probe_only:
            weight = weight.detach()
        return F.linear(hidden, weight)

    def _predict_memory(self, hidden):
        return hidden + self.memory_predictor(hidden)

    def default_state(self, *batch_shape, device=None):
        if len(batch_shape) != 1:
            raise ValueError("RNNAuxLM expects one batch dimension")
        return self.initial_state.to(device=device).expand(
            -1, batch_shape[0], -1
        )

    def _configured_vmf_kappa(self, name):
        if self.vmf_kappa_mode == "learned":
            return getattr(self, f"log_{name}_vmf_kappa").exp().clamp(
                min=1e-4, max=1e4
            )
        return getattr(self, f"{name}_vmf_kappa")

    def _sample_offsets(self, batch_size, device, chunk_size=None):
        chunk_size = self.chunk_size if chunk_size is None else chunk_size
        if self.chunk_offset == "random" and self.training:
            return torch.randint(
                chunk_size, (batch_size,), device=device
            )
        fixed = 0 if self.chunk_offset == "random" else self.chunk_offset
        return torch.full(
            (batch_size,), fixed, device=device, dtype=torch.long
        )

    def _inverse_batch(self, chunks, boundaries, successor_states):
        data_ids, inverse_targets = boundary_inverse_targets(
            chunks,
            boundaries,
            self.condition_memory_reconstruction_on_boundary,
        )
        if self.auxiliary_probe_only:
            observed_states = successor_states.detach()
        else:
            observed_states = memory_observation(
                successor_states,
                self.stop_gradient_memory_observation,
                self.memory_observation_gradient_scale,
            )
        inverse_outputs, inverse_state, _ = self.inverse_rnn(
            self._inverse_embedding(data_ids), observed_states
        )
        reconstructed_state = self._predict_memory(inverse_state)
        return (
            self._inverse_logits(inverse_outputs),
            inverse_targets,
            reconstructed_state,
        )

    @staticmethod
    def _state_at(initial_state, trajectory, position, indices):
        if position == 0:
            return initial_state.index_select(1, indices)
        return torch.stack(
            [
                layer_states[:, position - 1].index_select(0, indices)
                for layer_states in trajectory
            ],
            dim=0,
        )

    def _collect_inverse_records(
        self,
        input_ids,
        token_stream,
        initial_state,
        trajectory,
        offsets,
        chunk_size,
    ):
        """Select AUX transitions from an already-computed forward trajectory."""
        records = defaultdict(list)
        length = input_ids.size(1)
        for offset in range(chunk_size):
            indices = torch.nonzero(
                offsets == offset, as_tuple=False
            ).squeeze(1)
            if indices.numel() == 0:
                continue
            group_inputs = input_ids.index_select(0, indices)
            group_tokens = token_stream.index_select(0, indices)
            for start, end in chunk_ranges(length, chunk_size, offset):
                records[end - start].append(
                    (
                        group_tokens[:, start:end],
                        group_inputs[:, start],
                        self._state_at(
                            initial_state, trajectory, start, indices
                        ),
                        self._state_at(
                            initial_state, trajectory, end, indices
                        ),
                        start != 0,
                    )
                )
        return records

    def _compute_auxiliary_loss(
        self,
        input_ids,
        token_stream,
        initial_state,
        trajectory,
        offsets,
        chunk_size,
        logits,
        compute_diagnostics,
    ):
        batch_size, length = input_ids.shape
        records_by_length = self._collect_inverse_records(
            input_ids,
            token_stream,
            initial_state,
            trajectory,
            offsets,
            chunk_size,
        )
        chunk_logits = []
        chunk_targets = []
        discrete_logits = []
        discrete_targets = []
        memory_targets = []
        memory_estimates = []

        # All records with equal token length share one native inverse-RNN call,
        # including records selected by different per-sequence offsets.
        for inverse_length in sorted(records_by_length):
            records = records_by_length[inverse_length]
            chunks = torch.cat([record[0] for record in records], dim=0)
            boundaries = torch.cat([record[1] for record in records], dim=0)
            predecessor_states = torch.cat(
                [record[2] for record in records], dim=1
            )
            successor_states = torch.cat(
                [record[3] for record in records], dim=1
            )
            inverse_logits, inverse_targets, reconstructed_states = (
                self._inverse_batch(chunks, boundaries, successor_states)
            )
            token_logits = inverse_logits[:, :inverse_length]
            token_targets = inverse_targets[:, :inverse_length]
            if inverse_length > 1:
                chunk_logits.append(token_logits[:, :-1])
                chunk_targets.append(token_targets[:, :-1])
            discrete_logits.append(token_logits[:, -1:])
            discrete_targets.append(token_targets[:, -1:])
            # By default M_1 -> M_0 remains a token-reconstruction transition,
            # but the learned initial state is excluded from the Gaussian
            # memory term.  The option can restore the legacy M_0 term for a
            # controlled ablation.
            memory_mask = torch.cat(
                [
                    torch.full(
                        (record[0].size(0),),
                        record[4]
                        or not self.exclude_initial_memory_reconstruction,
                        device=chunks.device,
                        dtype=torch.bool,
                    )
                    for record in records
                ]
            )
            if memory_mask.any():
                memory_target = predecessor_states.transpose(0, 1).index_select(
                    0, torch.nonzero(memory_mask, as_tuple=False).squeeze(1)
                )
                memory_estimate = reconstructed_states.transpose(0, 1).index_select(
                    0, torch.nonzero(memory_mask, as_tuple=False).squeeze(1)
                )
                memory_targets.append(
                    memory_reconstruction_target(
                        memory_target,
                        self.stop_gradient_memory_target
                        or self.auxiliary_probe_only,
                    )
                )
                memory_estimates.append(memory_estimate)

        # Record counts differ by inverse length and sampled offset. Pooling the
        # transition-example axis gives the loss helpers one exact rectangular
        # tensor without padding, repeated forward work, or biased group means.
        memory_targets = (
            [torch.cat(memory_targets, dim=0)] if memory_targets else []
        )
        memory_estimates = (
            [torch.cat(memory_estimates, dim=0)] if memory_estimates else []
        )

        sequence_normalizer = float(length)
        loss_chunk = logits.new_zeros(())
        if self.use_chunk_loss:
            loss_chunk = cross_entropy_sum(
                chunk_logits, chunk_targets, logits, batch_size
            ) / sequence_normalizer
        loss_discrete = logits.new_zeros(())
        if self.use_discrete_loss:
            loss_discrete = cross_entropy_sum(
                discrete_logits, discrete_targets, logits, batch_size
            ) / sequence_normalizer
        loss_memory = logits.new_zeros(())
        if self.use_memory_loss:
            if self.state_aux_distribution == "gaussian":
                loss_memory = gaussian_nll_sum(
                    memory_targets,
                    memory_estimates,
                    self.rho,
                    logits,
                    batch_size,
                    scale_axis=1,
                ) / sequence_normalizer
            else:
                loss_memory = vmf_nll_sum(
                    memory_targets,
                    memory_estimates,
                    self._configured_vmf_kappa("memory"),
                    self._vmf_log_kappa_grid,
                    self._vmf_log_normalizer_grid,
                    logits,
                    batch_size,
                ) / sequence_normalizer
        loss_terminal = logits.new_zeros(())
        if self.use_terminal_loss:
            terminal_value = torch.stack(
                [layer_states[:, -1] for layer_states in trajectory], dim=0
            )
            if self.auxiliary_probe_only:
                terminal_value = terminal_value.detach()
            if self.state_aux_distribution == "gaussian":
                loss_terminal = terminal_gaussian_nll(
                    terminal_value,
                    self.tau,
                    batch_size,
                    scale_axis=0,
                    target=self.terminal_target,
                ) / sequence_normalizer
            else:
                loss_terminal = terminal_vmf_nll(
                    terminal_value,
                    self.terminal_target,
                    self._configured_vmf_kappa("terminal"),
                    self._vmf_log_kappa_grid,
                    self._vmf_log_normalizer_grid,
                    batch_size,
                ) / sequence_normalizer
        aux_loss = loss_chunk + loss_discrete + loss_memory + loss_terminal
        components = {
            "chunk_ce": loss_chunk,
            "discrete_ce": loss_discrete,
            "memory_nll": loss_memory,
            "terminal_nll": loss_terminal,
            "total": aux_loss,
        }
        metrics = {
            "aux/chunk_ce": loss_chunk.detach(),
            "aux/discrete_ce": loss_discrete.detach(),
            "aux/memory_nll": loss_memory.detach(),
            "aux/terminal_nll": loss_terminal.detach(),
            "aux/total": aux_loss.detach(),
            "aux/rho": self.rho.detach().reshape(()),
            "aux/rho_mean": self.rho.detach().reshape(()),
            "aux/tau": self.tau.detach().reshape(()),
            "aux/tau_mean": self.tau.detach().reshape(()),
            "aux/probe_only": logits.detach().new_tensor(
                float(self.auxiliary_probe_only)
            ),
            "aux/memory_vmf_kappa": self._configured_vmf_kappa(
                "memory"
            ).detach().reshape(()),
            "aux/terminal_vmf_kappa": self._configured_vmf_kappa(
                "terminal"
            ).detach().reshape(()),
        }
        if compute_diagnostics:
            diagnostics = memory_reconstruction_diagnostics(
                memory_targets, memory_estimates, logits
            )
            metrics.update(
                {
                    "aux/memory_reconstruction_mse": diagnostics[
                        "residual_mse"
                    ],
                    **{
                        f"aux/memory_{name}": value
                        for name, value in diagnostics.items()
                    },
                }
            )
            directional = directional_reconstruction_diagnostics(
                memory_targets, memory_estimates, logits
            )
            metrics.update(
                {
                    f"aux/memory_directional_{name}": value
                    for name, value in directional.items()
                }
            )
            terminal_value = torch.stack(
                [layer_states[:, -1] for layer_states in trajectory], dim=0
            )
            terminal_targets = [
                self.terminal_target.transpose(0, 1).expand(
                    terminal_value.size(1), -1, -1
                )
            ]
            terminal_estimates = [terminal_value.transpose(0, 1)]
            terminal_directional = directional_reconstruction_diagnostics(
                terminal_targets, terminal_estimates, logits
            )
            metrics.update(
                {
                    "aux/terminal_reconstruction_mse": (
                        terminal_reconstruction_mse(
                            terminal_value, self.terminal_target
                        )
                    ),
                    "aux/terminal_state_second_moment": (
                        terminal_value.detach().float().square().mean()
                    ),
                    "aux/terminal_target_second_moment": (
                        self.terminal_target.detach().float().square().mean()
                    ),
                    **{
                        f"aux/terminal_directional_{name}": value
                        for name, value in terminal_directional.items()
                    },
                }
            )
        return aux_loss, components, metrics, memory_targets, memory_estimates

    def forward(
        self,
        input_ids,
        targets=None,
        aux_tokens=None,
        state=None,
        compute_aux=True,
        compute_diagnostics=True,
        **kwargs,
    ):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, length)")
        batch_size, length = input_ids.shape
        if length == 0:
            raise ValueError("input sequence must be non-empty")

        token_stream = None
        if compute_aux:
            token_stream = aux_tokens
            if (
                token_stream is None
                and targets is not None
                and not torch.any(targets < 0)
            ):
                token_stream = targets
            if token_stream is not None and token_stream.shape != input_ids.shape:
                raise ValueError("aux_tokens must have the same shape as input_ids")
            if token_stream is None:
                raise ValueError(
                    "unmasked aux_tokens are required when compute_aux=True"
                )

        initial_state = (
            self.default_state(batch_size, device=input_ids.device)
            if state is None
            else state
        )
        # The complete forward trajectory is computed exactly once. Offsets are
        # sampled only afterwards and can therefore never affect LM computation.
        outputs, terminal_state, trajectory = self.rnn(
            self.embedding(input_ids),
            initial_state,
            return_trajectory=compute_aux,
        )
        logits = self._lm_logits(outputs)
        if not compute_aux:
            self.metrics = {}
            self.loss_components = {}
            self.scale_loss_components = {}
            return (
                AuxCausalLMOutput(logits=logits, aux_loss=logits.new_zeros(())),
                terminal_state,
            )

        aggregate = {
            name: logits.new_zeros(())
            for name in ("chunk_ce", "discrete_ce", "memory_nll", "terminal_nll")
        }
        self.scale_loss_components = {}
        scale_metrics = {}
        all_memory_targets = []
        all_memory_estimates = []
        first_scale_metrics = None
        for chunk_size in self.aux_chunk_sizes:
            offsets = self._sample_offsets(
                batch_size, input_ids.device, chunk_size=chunk_size
            )
            (
                _,
                components,
                metrics,
                memory_targets,
                memory_estimates,
            ) = self._compute_auxiliary_loss(
                input_ids,
                token_stream,
                initial_state,
                trajectory,
                offsets,
                chunk_size,
                logits,
                compute_diagnostics,
            )
            self.scale_loss_components[chunk_size] = components
            if first_scale_metrics is None:
                first_scale_metrics = metrics
            for name in aggregate:
                aggregate[name] = aggregate[name] + components[name]
            scale_metrics.update(
                {
                    f"aux/scale_{chunk_size}/{name.removeprefix('aux/')}": value
                    for name, value in metrics.items()
                }
            )
            all_memory_targets.extend(memory_targets)
            all_memory_estimates.extend(memory_estimates)
        aux_loss = sum(aggregate.values())
        self.loss_components = {**aggregate, "total": aux_loss}
        self.metrics = {
            **{
                f"aux/{name}": value.detach()
                for name, value in self.loss_components.items()
            },
            "aux/rho": self.rho.detach().reshape(()),
            "aux/rho_mean": self.rho.detach().reshape(()),
            "aux/tau": self.tau.detach().reshape(()),
            "aux/tau_mean": self.tau.detach().reshape(()),
            "aux/memory_vmf_kappa": self._configured_vmf_kappa(
                "memory"
            ).detach().reshape(()),
            "aux/terminal_vmf_kappa": self._configured_vmf_kappa(
                "terminal"
            ).detach().reshape(()),
            "aux/num_chunk_scales": logits.detach().new_tensor(
                float(len(self.aux_chunk_sizes))
            ),
            "aux/probe_only": logits.detach().new_tensor(
                float(self.auxiliary_probe_only)
            ),
            **scale_metrics,
        }
        if compute_diagnostics:
            pooled_memory_targets = (
                [torch.cat(all_memory_targets, dim=0)]
                if all_memory_targets else []
            )
            pooled_memory_estimates = (
                [torch.cat(all_memory_estimates, dim=0)]
                if all_memory_estimates else []
            )
            diagnostics = memory_reconstruction_diagnostics(
                pooled_memory_targets, pooled_memory_estimates, logits
            )
            self.metrics.update(
                {
                    "aux/memory_reconstruction_mse": diagnostics["residual_mse"],
                    **{
                        f"aux/memory_{name}": value
                        for name, value in diagnostics.items()
                    },
                }
            )
            directional = directional_reconstruction_diagnostics(
                pooled_memory_targets, pooled_memory_estimates, logits
            )
            self.metrics.update(
                {
                    f"aux/memory_directional_{name}": value
                    for name, value in directional.items()
                }
            )
            if first_scale_metrics is not None:
                self.metrics.update(
                    {
                        name: value
                        for name, value in first_scale_metrics.items()
                        if name.startswith("aux/terminal_")
                    }
                )
        return AuxCausalLMOutput(logits=logits, aux_loss=aux_loss), terminal_state

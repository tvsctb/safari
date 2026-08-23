import copy
import math
from collections import defaultdict
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.sequence.auxiliary import (
    AuxCausalLMOutput,
    boundary_inverse_targets,
    centered_layerwise_directional_diagnostics,
    centered_layerwise_terminal_vmf_nll,
    centered_layerwise_vmf_nll_sum,
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


class NormalizedStateRNNLayer(nn.Module):
    """Elman layer whose recurrent state is the non-affine normalized preactivation.

    The recurrent memory and the value exposed to the next layer are deliberately
    distinct:

        memory_t = LN_nonaffine(W_x input_t + W_m memory_{t-1})
        output_t = tanh(gamma * memory_t + beta)

    This keeps every post-initial recurrent state on the centered unit-RMS
    sphere while retaining a bounded, affine-calibrated inter-layer output.  The
    two matrices plus gamma/beta have exactly the same parameter count as one
    native ``nn.RNN`` layer with its two bias vectors.
    """

    def __init__(self, d_model, epsilon=1e-5):
        super().__init__()
        self.d_model = int(d_model)
        self.epsilon = float(epsilon)
        self.weight_ih_l0 = nn.Parameter(torch.empty(d_model, d_model))
        self.weight_hh_l0 = nn.Parameter(torch.empty(d_model, d_model))
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.gamma._no_weight_decay = True
        self.beta._no_weight_decay = True

    def reset_parameters(self, recurrent_init, recurrent_identity_scale):
        nn.init.xavier_uniform_(self.weight_ih_l0)
        if recurrent_init == "orthogonal":
            nn.init.orthogonal_(self.weight_hh_l0)
        else:
            nn.init.eye_(self.weight_hh_l0)
            with torch.no_grad():
                self.weight_hh_l0.mul_(recurrent_identity_scale)
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, inputs, state, return_trajectory=False):
        if state.shape != (1, inputs.size(0), self.d_model):
            raise ValueError("normalized RNN layer state has an invalid shape")
        memory = state[0]
        outputs = []
        memories = [] if return_trajectory else None
        for token_input in inputs.unbind(dim=1):
            preactivation = F.linear(token_input, self.weight_ih_l0)
            preactivation = preactivation + F.linear(
                memory, self.weight_hh_l0
            )
            memory = F.layer_norm(
                preactivation,
                (self.d_model,),
                weight=None,
                bias=None,
                eps=self.epsilon,
            )
            outputs.append(torch.tanh(memory * self.gamma + self.beta))
            if return_trajectory:
                memories.append(memory)
        return (
            torch.stack(outputs, dim=1),
            memory.unsqueeze(0),
            torch.stack(memories, dim=1) if return_trajectory else None,
        )


class StackedRNN(nn.Module):
    """A configurable stacked vanilla RNN exposing every layer trajectory.

    Native tanh/ReLU layers use fused ``nn.RNN`` calls.  The normalized-state
    variant uses the explicit recurrence required to keep its memory separate
    from its post-tanh output.  In either case AUX chunking is sampled only after
    this complete forward trajectory has been computed.
    """

    def __init__(
        self,
        d_model,
        n_layer,
        dropout=0.0,
        activation="tanh",
        recurrent_init="orthogonal",
        recurrent_identity_scale=1.0,
        normalized_state=False,
        normalization_epsilon=1e-5,
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
        if normalized_state and activation != "tanh":
            raise ValueError("normalized_state currently requires tanh activation")
        if normalization_epsilon <= 0:
            raise ValueError("normalization_epsilon must be positive")
        self.d_model = d_model
        self.n_layer = n_layer
        self.dropout = float(dropout)
        self.activation = activation
        self.recurrent_init = recurrent_init
        self.recurrent_identity_scale = float(recurrent_identity_scale)
        self.normalized_state = bool(normalized_state)
        self.normalization_epsilon = float(normalization_epsilon)
        self.layers = nn.ModuleList(
            (
                NormalizedStateRNNLayer(d_model, normalization_epsilon)
                if normalized_state
                else nn.RNN(
                    d_model,
                    d_model,
                    num_layers=1,
                    nonlinearity=activation,
                    batch_first=True,
                )
            )
            for _ in range(n_layer)
        )

    def reset_parameters(self):
        for layer in self.layers:
            if self.normalized_state:
                layer.reset_parameters(
                    self.recurrent_init, self.recurrent_identity_scale
                )
                continue
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
            if self.normalized_state:
                layer_output, terminal_state, layer_trajectory = layer(
                    layer_input, layer_state, return_trajectory=return_trajectory
                )
            else:
                layer_output, terminal_state = layer(
                    layer_input, layer_state
                )
                layer_trajectory = layer_output
            terminal_states.append(terminal_state)
            if return_trajectory:
                trajectories.append(layer_trajectory)
            layer_input = layer_output

        terminal_state = torch.cat(terminal_states, dim=0)
        if not return_trajectory:
            return layer_input, terminal_state, None
        # Keep native layer outputs separate. AUX gathers only boundary states,
        # avoiding an additional full (layers, batch, time, hidden) allocation.
        return layer_input, terminal_state, tuple(trajectories)


class FactorizedRNNLayer(nn.Module):
    """Low-rank Elman layer with the same hidden-state interface as ``nn.RNN``."""

    def __init__(self, d_model, rank, activation):
        super().__init__()
        self.d_model = int(d_model)
        self.rank = int(rank)
        self.activation = activation
        self.weight_ih_left = nn.Parameter(torch.empty(d_model, rank))
        self.weight_ih_right = nn.Parameter(torch.empty(rank, d_model))
        self.weight_hh_left = nn.Parameter(torch.empty(d_model, rank))
        self.weight_hh_right = nn.Parameter(torch.empty(rank, d_model))
        self.bias_ih = nn.Parameter(torch.empty(d_model))
        self.bias_hh = nn.Parameter(torch.empty(d_model))

    @staticmethod
    def _truncated_svd(weight, rank):
        u, singular, vh = torch.linalg.svd(weight.float(), full_matrices=False)
        root = singular[:rank].sqrt()
        left = u[:, :rank] * root.unsqueeze(0)
        right = root.unsqueeze(1) * vh[:rank]
        return left.to(weight), right.to(weight)

    @torch.no_grad()
    def initialize_from(self, dense):
        ih_left, ih_right = self._truncated_svd(
            dense.weight_ih_l0, self.rank
        )
        hh_left, hh_right = self._truncated_svd(
            dense.weight_hh_l0, self.rank
        )
        self.weight_ih_left.copy_(ih_left)
        self.weight_ih_right.copy_(ih_right)
        self.weight_hh_left.copy_(hh_left)
        self.weight_hh_right.copy_(hh_right)
        self.bias_ih.copy_(dense.bias_ih_l0)
        self.bias_hh.copy_(dense.bias_hh_l0)

    def forward(self, inputs, state, return_trajectory=False):
        if state.shape != (1, inputs.size(0), self.d_model):
            raise ValueError("factorized RNN layer state has an invalid shape")
        hidden = state[0]
        outputs = []
        for token_input in inputs.unbind(dim=1):
            input_term = F.linear(
                F.linear(token_input, self.weight_ih_right),
                self.weight_ih_left,
                self.bias_ih,
            )
            recurrent_term = F.linear(
                F.linear(hidden, self.weight_hh_right),
                self.weight_hh_left,
                self.bias_hh,
            )
            preactivation = input_term + recurrent_term
            hidden = (
                torch.relu(preactivation)
                if self.activation == "relu"
                else torch.tanh(preactivation)
            )
            outputs.append(hidden)
        outputs = torch.stack(outputs, dim=1)
        return outputs, hidden.unsqueeze(0), outputs if return_trajectory else None


class FactorizedStackedRNN(nn.Module):
    """Stacked inverse RNN whose effective hidden dimension remains unchanged."""

    def __init__(self, d_model, n_layer, rank, dropout, activation):
        super().__init__()
        self.d_model = int(d_model)
        self.n_layer = int(n_layer)
        self.rank = int(rank)
        self.dropout = float(dropout)
        self.activation = activation
        self.layers = nn.ModuleList(
            FactorizedRNNLayer(d_model, rank, activation)
            for _ in range(n_layer)
        )

    @torch.no_grad()
    def initialize_from(self, dense):
        if len(dense.layers) != len(self.layers):
            raise ValueError("dense and factorized inverse depths must match")
        for target, source in zip(self.layers, dense.layers):
            if not isinstance(source, nn.RNN):
                raise TypeError("factorized inverse requires native dense RNN layers")
            target.initialize_from(source)

    def forward(self, inputs, state, return_trajectory=False):
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
            layer_output, terminal_state, layer_trajectory = layer(
                layer_input,
                state[index : index + 1],
                return_trajectory=return_trajectory,
            )
            terminal_states.append(terminal_state)
            if return_trajectory:
                trajectories.append(layer_trajectory)
            layer_input = layer_output
        terminal_state = torch.cat(terminal_states, dim=0)
        return (
            layer_input,
            terminal_state,
            tuple(trajectories) if return_trajectory else None,
        )


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
        normalized_state=False,
        normalization_epsilon=1e-5,
        rho=1.0,
        tau=1.0,
        gaussian_scale_mode="fixed",
        state_likelihood_granularity="global",
        gaussian_scale_learning_start_step=0,
        gaussian_scale_learning_rate=1e-5,
        memory_scale_target=None,
        memory_scale_target_mode="fixed",
        memory_scale_target_learning_rate=1e-3,
        memory_scale_constraint_weight=0.0,
        memory_scale_constraint_start_step=None,
        memory_scale_constraint_ramp_steps=0,
        state_aux_distribution="gaussian",
        vmf_kappa_mode="fixed",
        vmf_kappa_learning_rate=1e-4,
        memory_vmf_kappa=1.0,
        terminal_vmf_kappa=1.0,
        inverse_capacity_multiplier=1.0,
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
        if not isinstance(normalized_state, bool):
            raise ValueError("normalized_state must be a boolean")
        if rho <= 0 or tau <= 0:
            raise ValueError("rho and tau must be positive")
        capacity_specs = {
            0.125: (4, 13),
            0.25: (8, 30),
            0.5: (16, 62),
            1.0: (None, 128),
            2.0: (64, 259),
        }
        inverse_capacity_multiplier = float(inverse_capacity_multiplier)
        if inverse_capacity_multiplier not in capacity_specs:
            raise ValueError(
                "inverse_capacity_multiplier must be one of "
                "0.125, 0.25, 0.5, 1.0, or 2.0"
            )
        if inverse_capacity_multiplier != 1.0 and d_model != 64:
            raise ValueError(
                "non-default inverse capacity is calibrated for d_model=64"
            )
        if normalized_state and inverse_capacity_multiplier != 1.0:
            raise ValueError(
                "factorized inverse capacity currently requires a native RNN core"
            )
        if state_likelihood_granularity not in {"global", "layer"}:
            raise ValueError(
                "state_likelihood_granularity must be global or layer"
            )
        if gaussian_scale_mode not in {"fixed", "learned"}:
            raise ValueError("gaussian_scale_mode must be fixed or learned")
        if memory_scale_target_mode not in {"fixed", "learned"}:
            raise ValueError(
                "memory_scale_target_mode must be fixed or learned"
            )
        if (
            isinstance(gaussian_scale_learning_start_step, bool)
            or not isinstance(gaussian_scale_learning_start_step, int)
            or gaussian_scale_learning_start_step < 0
        ):
            raise ValueError(
                "gaussian_scale_learning_start_step must be a non-negative integer"
            )
        if (
            isinstance(gaussian_scale_learning_rate, bool)
            or not math.isfinite(float(gaussian_scale_learning_rate))
            or float(gaussian_scale_learning_rate) <= 0
        ):
            raise ValueError(
                "gaussian_scale_learning_rate must be finite and positive"
            )
        if memory_scale_target is not None and (
            isinstance(memory_scale_target, bool)
            or not math.isfinite(float(memory_scale_target))
            or float(memory_scale_target) <= 0
        ):
            raise ValueError("memory_scale_target must be finite and positive")
        if (
            isinstance(memory_scale_target_learning_rate, bool)
            or not math.isfinite(float(memory_scale_target_learning_rate))
            or float(memory_scale_target_learning_rate) <= 0
        ):
            raise ValueError(
                "memory_scale_target_learning_rate must be finite and positive"
            )
        if (
            isinstance(memory_scale_constraint_weight, bool)
            or not math.isfinite(float(memory_scale_constraint_weight))
            or float(memory_scale_constraint_weight) < 0
        ):
            raise ValueError(
                "memory_scale_constraint_weight must be finite and non-negative"
            )
        if memory_scale_constraint_start_step is None:
            memory_scale_constraint_start_step = gaussian_scale_learning_start_step
        if (
            isinstance(memory_scale_constraint_start_step, bool)
            or not isinstance(memory_scale_constraint_start_step, int)
            or memory_scale_constraint_start_step < 0
        ):
            raise ValueError(
                "memory_scale_constraint_start_step must be a non-negative integer"
            )
        if (
            isinstance(memory_scale_constraint_ramp_steps, bool)
            or not isinstance(memory_scale_constraint_ramp_steps, int)
            or memory_scale_constraint_ramp_steps < 0
        ):
            raise ValueError(
                "memory_scale_constraint_ramp_steps must be a non-negative integer"
            )
        if (
            memory_scale_constraint_weight > 0
            and memory_scale_target is None
            and memory_scale_target_mode == "fixed"
        ):
            raise ValueError(
                "a positive memory_scale_constraint_weight requires memory_scale_target"
            )
        if normalized_state and memory_scale_constraint_weight > 0:
            raise ValueError(
                "normalized_state fixes recurrent scale and cannot use a state-scale constraint"
            )
        if state_aux_distribution not in {"gaussian", "vmf"}:
            raise ValueError("state_aux_distribution must be gaussian or vmf")
        if vmf_kappa_mode not in {"fixed", "learned"}:
            raise ValueError("vmf_kappa_mode must be fixed or learned")
        if (
            isinstance(vmf_kappa_learning_rate, bool)
            or not math.isfinite(float(vmf_kappa_learning_rate))
            or float(vmf_kappa_learning_rate) <= 0
        ):
            raise ValueError("vmf_kappa_learning_rate must be finite and positive")
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
        self.gaussian_scale_mode = gaussian_scale_mode
        self.state_likelihood_granularity = state_likelihood_granularity
        self.normalized_state = bool(normalized_state)
        self.memory_scale_target_mode = memory_scale_target_mode
        self.gaussian_scale_learning_start_step = (
            gaussian_scale_learning_start_step
        )
        self.memory_scale_constraint_weight = float(
            memory_scale_constraint_weight
        )
        self.memory_scale_constraint_start_step = (
            memory_scale_constraint_start_step
        )
        self.memory_scale_constraint_ramp_steps = (
            memory_scale_constraint_ramp_steps
        )
        self._has_memory_scale_constraint = (
            memory_scale_constraint_weight > 0
            and (
                memory_scale_target is not None
                or memory_scale_target_mode == "learned"
            )
        )
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
        self.inverse_capacity_multiplier = inverse_capacity_multiplier
        inverse_rank, predictor_width = capacity_specs[
            inverse_capacity_multiplier
        ]
        self.inverse_rank = inverse_rank
        self.inverse_predictor_width = predictor_width

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.initial_state = nn.Parameter(torch.empty(n_layer, 1, d_model))
        self.rnn = StackedRNN(
            d_model,
            n_layer,
            dropout=dropout,
            activation=activation,
            recurrent_init=recurrent_init,
            recurrent_identity_scale=recurrent_identity_scale,
            normalized_state=normalized_state,
            normalization_epsilon=normalization_epsilon,
        )
        self.inverse_rnn = (
            copy.deepcopy(self.rnn)
            if inverse_rank is None
            else FactorizedStackedRNN(
                d_model,
                n_layer,
                inverse_rank,
                dropout,
                activation,
            )
        )
        baseline_memory_predictor = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        if predictor_width == 2 * d_model:
            self.memory_predictor = baseline_memory_predictor
        else:
            # Match the constructor-time RNG consumption of the historical 1x
            # model exactly. The capacity-specific predictor is created in a
            # fork because its constructor initialization is discarded below.
            with torch.random.fork_rng(devices=[]):
                self.memory_predictor = nn.Sequential(
                    nn.Linear(d_model, predictor_width),
                    nn.GELU(),
                    nn.Linear(predictor_width, d_model),
                )
        # The terminal prior mean is always learned.  Keeping it allocated when
        # the terminal term is disabled makes parameterization identical across
        # terminal-loss ablations and detached no-AUX probes.
        self.terminal_target = nn.Parameter(
            torch.zeros(n_layer, 1, d_model)
        )
        if gaussian_scale_mode == "learned":
            scale_optim = {
                "lr": float(gaussian_scale_learning_rate),
                "weight_decay": 0.0,
            }
            scale_shape = (n_layer,) if state_likelihood_granularity == "layer" else ()
            self.log_rho = nn.Parameter(
                torch.full(scale_shape, float(rho)).log()
            )
            self.log_tau = nn.Parameter(
                torch.full(scale_shape, float(tau)).log()
            )
            self.log_rho._optim = dict(scale_optim)
            self.log_tau._optim = dict(scale_optim)
        else:
            scale_shape = (n_layer,) if state_likelihood_granularity == "layer" else ()
            self.register_buffer("rho", torch.full(scale_shape, float(rho)))
            self.register_buffer("tau", torch.full(scale_shape, float(tau)))
        self.register_buffer(
            "memory_scale_target",
            torch.tensor(
                float("nan")
                if memory_scale_target is None
                else float(memory_scale_target)
            ),
        )
        if memory_scale_target_mode == "learned":
            initial_target = (
                1.0 if memory_scale_target is None else float(memory_scale_target)
            )
            self.log_memory_scale_target = nn.Parameter(
                torch.tensor(initial_target).log()
            )
            self.log_memory_scale_target._optim = {
                "lr": float(memory_scale_target_learning_rate),
                "weight_decay": 0.0,
            }
        self.register_buffer(
            "memory_scale_target_initialized",
            torch.tensor(memory_scale_target is not None, dtype=torch.bool),
        )
        self._memory_scale_target_initialized_value = (
            memory_scale_target is not None
        )
        # This counter is part of checkpoint state. It therefore remains exact
        # when several continuations branch from one fixed-scale warmup.
        self.register_buffer(
            "aux_training_step", torch.zeros((), dtype=torch.long)
        )
        # Keep the hot-path gate as a Python integer. A tensor comparison plus
        # multiplication would produce a zero-valued gradient for log_rho and
        # log_tau during warmup, causing Adam to advance their step counters
        # even though their values were intended to be completely frozen.
        # The persistent tensor remains the checkpoint source of truth.
        self._aux_training_step_value = 0
        if vmf_kappa_mode == "learned":
            kappa_shape = (
                (n_layer,) if state_likelihood_granularity == "layer" else ()
            )
            self.log_memory_vmf_kappa = nn.Parameter(
                torch.full(kappa_shape, float(memory_vmf_kappa)).log()
            )
            self.log_terminal_vmf_kappa = nn.Parameter(
                torch.full(kappa_shape, float(terminal_vmf_kappa)).log()
            )
            kappa_optim = {
                "lr": float(vmf_kappa_learning_rate),
                "weight_decay": 0.0,
            }
            self.log_memory_vmf_kappa._optim = dict(kappa_optim)
            self.log_terminal_vmf_kappa._optim = dict(kappa_optim)
            self.log_memory_vmf_kappa._no_weight_decay = True
            self.log_terminal_vmf_kappa._no_weight_decay = True
        else:
            kappa_shape = (
                (n_layer,) if state_likelihood_granularity == "layer" else ()
            )
            self.register_buffer(
                "memory_vmf_kappa",
                torch.full(kappa_shape, float(memory_vmf_kappa)),
                persistent=False,
            )
            self.register_buffer(
                "terminal_vmf_kappa",
                torch.full(kappa_shape, float(terminal_vmf_kappa)),
                persistent=False,
            )
        if state_aux_distribution == "vmf":
            vmf_dimension = d_model - 1 if normalized_state else n_layer * d_model
            log_kappa, log_normalizer = vmf_log_normalizer_grid(vmf_dimension)
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
        self.layer_loss_components = {}

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
        if self.inverse_rank is None:
            self.inverse_rnn.load_state_dict(self.rnn.state_dict())
        else:
            self.inverse_rnn.initialize_from(self.rnn)

        first = self.memory_predictor[0]
        second = self.memory_predictor[2]
        nn.init.xavier_uniform_(first.weight)
        nn.init.zeros_(first.bias)
        # The residual predictor is exactly the identity at initialization.
        nn.init.zeros_(second.weight)
        nn.init.zeros_(second.bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Checkpoints predating the warmup-scale study have no training-step
        # buffer. Treat them as step zero while retaining strict loading for
        # every learned weight and all other state.
        inserted = []
        replaced_target = None
        step_key = prefix + "aux_training_step"
        target_key = prefix + "memory_scale_target"
        target_initialized_key = prefix + "memory_scale_target_initialized"
        if step_key not in state_dict:
            state_dict[step_key] = self.aux_training_step.detach().clone()
            inserted.append(step_key)
        if target_key not in state_dict:
            state_dict[target_key] = self.memory_scale_target.detach().clone()
            inserted.append(target_key)
        elif (
            self.memory_scale_target_mode == "fixed"
            and torch.isfinite(self.memory_scale_target)
            and not torch.isfinite(state_dict[target_key])
        ):
            # A branch supplies its calibrated target in configuration while
            # its common warmup checkpoint intentionally contains NaN.
            replaced_target = state_dict[target_key]
            state_dict[target_key] = self.memory_scale_target.detach().clone()
        if target_initialized_key not in state_dict:
            state_dict[target_initialized_key] = (
                self.memory_scale_target_initialized.detach().clone()
            )
            inserted.append(target_initialized_key)
        try:
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            self._aux_training_step_value = int(
                self.aux_training_step.detach().cpu().item()
            )
            self._memory_scale_target_initialized_value = bool(
                self.memory_scale_target_initialized.detach().cpu().item()
            )
        finally:
            for key in inserted:
                state_dict.pop(key, None)
            if replaced_target is not None:
                state_dict[target_key] = replaced_target

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

    def inverse_parameter_count(self):
        return sum(
            parameter.numel()
            for module in (self.inverse_rnn, self.memory_predictor)
            for parameter in module.parameters()
        )

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

    def _configured_gaussian_scale(self, name):
        if self.gaussian_scale_mode == "fixed":
            return getattr(self, name)
        value = getattr(self, f"log_{name}").exp().clamp(
            min=1e-4, max=1e4
        )
        if (
            self._aux_training_step_value
            < self.gaussian_scale_learning_start_step
        ):
            return value.detach()
        return value

    @staticmethod
    def _layer_parameter(value, layer):
        return value if value.ndim == 0 else value[layer]

    def _configured_memory_scale_target(self, log_rms=None):
        if self.memory_scale_target_mode == "fixed":
            return self.memory_scale_target
        if not self._memory_scale_target_initialized_value:
            # Lightning runs validation sanity checks before the first
            # optimizer step.  Only a training trajectory may choose the
            # data-derived initial target.
            if log_rms is None or not self.training:
                return self.log_memory_scale_target.exp()
            with torch.no_grad():
                initial_log_rms = log_rms.detach().mean()
                if (
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    torch.distributed.all_reduce(initial_log_rms)
                    initial_log_rms.div_(torch.distributed.get_world_size())
                self.log_memory_scale_target.copy_(initial_log_rms)
                self.memory_scale_target_initialized.fill_(True)
                self._memory_scale_target_initialized_value = True
        return self.log_memory_scale_target.exp().clamp(min=1e-4, max=1e4)

    @staticmethod
    def trajectory_log_rms(trajectory):
        """Return one per-coordinate log RMS for each sequence.

        Every recurrent state M_1..M_K is included exactly once. The learned
        initial state M_0 and any AUX chunk/offset sampling are deliberately
        absent, so this gauge is a property of the forward memory trajectory.
        """
        if not trajectory:
            raise ValueError("trajectory must contain at least one RNN layer")
        per_layer_second_moment = torch.stack(
            [states.float().square().mean(dim=(1, 2)) for states in trajectory],
            dim=0,
        )
        per_sample_second_moment = per_layer_second_moment.mean(dim=0)
        return 0.5 * torch.log(per_sample_second_moment.clamp_min(1e-20))

    def _memory_scale_constraint(self, trajectory, reference):
        log_rms = self.trajectory_log_rms(trajectory)
        current_geometric_rms = log_rms.detach().mean().exp()
        current_arithmetic_rms = log_rms.detach().exp().mean()
        loss = reference.new_zeros(())
        ramp = reference.new_zeros(())
        active = reference.new_zeros(())
        if self._has_memory_scale_constraint:
            is_active = (
                self._aux_training_step_value
                >= self.memory_scale_constraint_start_step
            )
            active = reference.new_tensor(float(is_active))
            if self.memory_scale_constraint_ramp_steps == 0:
                ramp = active
            else:
                ramp = reference.new_tensor(
                    min(
                        1.0,
                        max(
                            0.0,
                            (
                                self._aux_training_step_value
                                - self.memory_scale_constraint_start_step
                                + 1
                            )
                            / self.memory_scale_constraint_ramp_steps,
                        ),
                    )
                )
            target = self._configured_memory_scale_target(log_rms)
            target_log_rms = target.float().log()
            loss = (
                self.memory_scale_constraint_weight
                * ramp
                * (log_rms - target_log_rms).square().mean()
            )
        target = self._configured_memory_scale_target(log_rms).detach().to(
            reference
        )
        ratio = current_geometric_rms.to(reference) / target
        if not self._has_memory_scale_constraint:
            ratio = reference.new_tensor(float("nan"))
        metrics = {
            "aux/state_scale_loss": loss.detach(),
            "aux/state_scale_active": active.detach(),
            "aux/state_scale_ramp": ramp.detach(),
            "aux/state_scale_target_rms": target,
            "aux/state_scale_geometric_rms": current_geometric_rms.to(reference),
            "aux/state_scale_arithmetic_rms": current_arithmetic_rms.to(reference),
            "aux/state_scale_ratio": ratio,
            "aux/gaussian_scale_learning_active": (
                reference.new_tensor(
                    float(
                        self._aux_training_step_value
                        >= self.gaussian_scale_learning_start_step
                    )
                )
                if self.gaussian_scale_mode == "learned"
                else reference.detach().new_zeros(())
            ),
            "aux/training_step": self.aux_training_step.detach().to(reference),
            "aux/inverse_parameter_count": reference.new_tensor(
                float(self.inverse_parameter_count())
            ),
            "aux/inverse_capacity_multiplier": reference.new_tensor(
                self.inverse_capacity_multiplier
            ),
        }
        return loss, metrics

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
                    self._configured_gaussian_scale("rho"),
                    logits,
                    batch_size,
                    scale_axis=1,
                ) / sequence_normalizer
            elif self.normalized_state:
                loss_memory = centered_layerwise_vmf_nll_sum(
                    memory_targets,
                    memory_estimates,
                    self._configured_vmf_kappa("memory"),
                    self._vmf_log_kappa_grid,
                    self._vmf_log_normalizer_grid,
                    logits,
                    batch_size,
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
                    self._configured_gaussian_scale("tau"),
                    batch_size,
                    scale_axis=0,
                    target=self.terminal_target,
                ) / sequence_normalizer
            elif self.normalized_state:
                loss_terminal = centered_layerwise_terminal_vmf_nll(
                    terminal_value,
                    self.terminal_target,
                    self._configured_vmf_kappa("terminal"),
                    self._vmf_log_kappa_grid,
                    self._vmf_log_normalizer_grid,
                    batch_size,
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
        layer_components = {}
        if self.normalized_state:
            terminal_value = torch.stack(
                [layer_states[:, -1] for layer_states in trajectory], dim=0
            )
            if self.auxiliary_probe_only:
                terminal_value = terminal_value.detach()
            for layer in range(self.n_layer):
                layer_memory = logits.new_zeros(())
                layer_terminal = logits.new_zeros(())
                if self.use_memory_loss:
                    layer_targets = [
                        value[:, layer : layer + 1] for value in memory_targets
                    ]
                    layer_estimates = [
                        value[:, layer : layer + 1] for value in memory_estimates
                    ]
                    if self.state_aux_distribution == "gaussian":
                        layer_memory = gaussian_nll_sum(
                            layer_targets,
                            layer_estimates,
                            self._layer_parameter(
                                self._configured_gaussian_scale("rho"), layer
                            ),
                            logits,
                            batch_size,
                            scale_axis=1,
                        ) / sequence_normalizer
                    else:
                        layer_memory = centered_layerwise_vmf_nll_sum(
                            layer_targets,
                            layer_estimates,
                            self._layer_parameter(
                                self._configured_vmf_kappa("memory"), layer
                            ),
                            self._vmf_log_kappa_grid,
                            self._vmf_log_normalizer_grid,
                            logits,
                            batch_size,
                        ) / sequence_normalizer
                if self.use_terminal_loss:
                    if self.state_aux_distribution == "gaussian":
                        layer_terminal = terminal_gaussian_nll(
                            terminal_value[layer : layer + 1],
                            self._layer_parameter(
                                self._configured_gaussian_scale("tau"), layer
                            ),
                            batch_size,
                            scale_axis=0,
                            target=self.terminal_target[layer : layer + 1],
                        ) / sequence_normalizer
                    else:
                        layer_terminal = centered_layerwise_terminal_vmf_nll(
                            terminal_value[layer : layer + 1],
                            self.terminal_target[layer : layer + 1],
                            self._layer_parameter(
                                self._configured_vmf_kappa("terminal"), layer
                            ),
                            self._vmf_log_kappa_grid,
                            self._vmf_log_normalizer_grid,
                            batch_size,
                        ) / sequence_normalizer
                layer_components[layer] = {
                    "memory_nll": layer_memory,
                    "terminal_nll": layer_terminal,
                    "total": layer_memory + layer_terminal,
                }
        aux_loss = loss_chunk + loss_discrete + loss_memory + loss_terminal
        components = {
            "chunk_ce": loss_chunk,
            "discrete_ce": loss_discrete,
            "memory_nll": loss_memory,
            "terminal_nll": loss_terminal,
            "total": aux_loss,
        }
        rho = self._configured_gaussian_scale("rho").detach()
        tau = self._configured_gaussian_scale("tau").detach()
        memory_kappa = self._configured_vmf_kappa("memory").detach()
        terminal_kappa = self._configured_vmf_kappa("terminal").detach()
        metrics = {
            "aux/chunk_ce": loss_chunk.detach(),
            "aux/discrete_ce": loss_discrete.detach(),
            "aux/memory_nll": loss_memory.detach(),
            "aux/terminal_nll": loss_terminal.detach(),
            "aux/total": aux_loss.detach(),
            "aux/rho": rho.float().mean(),
            "aux/rho_mean": rho.float().mean(),
            "aux/tau": tau.float().mean(),
            "aux/tau_mean": tau.float().mean(),
            "aux/probe_only": logits.detach().new_tensor(
                float(self.auxiliary_probe_only)
            ),
            "aux/memory_vmf_kappa": memory_kappa.float().mean(),
            "aux/terminal_vmf_kappa": terminal_kappa.float().mean(),
        }
        if self.state_likelihood_granularity == "layer":
            for layer in range(self.n_layer):
                metrics.update(
                    {
                        f"aux/rho_layer_{layer}": rho[layer],
                        f"aux/tau_layer_{layer}": tau[layer],
                        f"aux/memory_vmf_kappa_layer_{layer}": memory_kappa[layer],
                        f"aux/terminal_vmf_kappa_layer_{layer}": terminal_kappa[layer],
                    }
                )
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
            if self.normalized_state:
                centered_directional = centered_layerwise_directional_diagnostics(
                    memory_targets, memory_estimates, logits
                )
                metrics.update(
                    {
                        f"aux/memory_centered_{name}": value
                        for name, value in centered_directional.items()
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
        return (
            aux_loss,
            components,
            metrics,
            memory_targets,
            memory_estimates,
            layer_components,
        )

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
            self.layer_loss_components = {}
            return (
                AuxCausalLMOutput(logits=logits, aux_loss=logits.new_zeros(())),
                terminal_state,
            )

        aggregate = {
            name: logits.new_zeros(())
            for name in (
                "chunk_ce",
                "discrete_ce",
                "memory_nll",
                "terminal_nll",
                "state_scale",
            )
        }
        state_scale_loss, state_scale_metrics = self._memory_scale_constraint(
            trajectory, logits
        )
        aggregate["state_scale"] = state_scale_loss
        self.scale_loss_components = {}
        self.layer_loss_components = {}
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
                layer_components,
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
            for layer, values in layer_components.items():
                self.layer_loss_components[(chunk_size, layer)] = values
            if first_scale_metrics is None:
                first_scale_metrics = metrics
            for name in (
                "chunk_ce", "discrete_ce", "memory_nll", "terminal_nll"
            ):
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
        rho = self._configured_gaussian_scale("rho").detach()
        tau = self._configured_gaussian_scale("tau").detach()
        memory_kappa = self._configured_vmf_kappa("memory").detach()
        terminal_kappa = self._configured_vmf_kappa("terminal").detach()
        self.metrics = {
            **{
                f"aux/{name}": value.detach()
                for name, value in self.loss_components.items()
            },
            "aux/rho": rho.float().mean(),
            "aux/rho_mean": rho.float().mean(),
            "aux/tau": tau.float().mean(),
            "aux/tau_mean": tau.float().mean(),
            "aux/memory_vmf_kappa": memory_kappa.float().mean(),
            "aux/terminal_vmf_kappa": terminal_kappa.float().mean(),
            "aux/num_chunk_scales": logits.detach().new_tensor(
                float(len(self.aux_chunk_sizes))
            ),
            "aux/probe_only": logits.detach().new_tensor(
                float(self.auxiliary_probe_only)
            ),
            **state_scale_metrics,
            **scale_metrics,
        }
        if self.state_likelihood_granularity == "layer":
            for layer in range(self.n_layer):
                self.metrics.update(
                    {
                        f"aux/rho_layer_{layer}": rho[layer],
                        f"aux/tau_layer_{layer}": tau[layer],
                        f"aux/memory_vmf_kappa_layer_{layer}": memory_kappa[layer],
                        f"aux/terminal_vmf_kappa_layer_{layer}": terminal_kappa[layer],
                    }
                )
        if self.normalized_state:
            for layer, states in enumerate(trajectory):
                detached = states.detach().float()
                self.metrics.update(
                    {
                        f"aux/state_layer_{layer}_coordinate_mean_abs": (
                            detached.mean(dim=-1).abs().mean()
                        ),
                        f"aux/state_layer_{layer}_coordinate_rms": (
                            detached.square().mean().sqrt()
                        ),
                        f"aux/state_layer_{layer}_vector_norm": (
                            detached.norm(dim=-1).mean()
                        ),
                    }
                )
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
            if pooled_memory_targets:
                target_value = pooled_memory_targets[0]
                estimate_value = pooled_memory_estimates[0]
                for layer in range(self.n_layer):
                    layer_diagnostics = memory_reconstruction_diagnostics(
                        [target_value[:, layer : layer + 1]],
                        [estimate_value[:, layer : layer + 1]],
                        logits,
                    )
                    self.metrics.update(
                        {
                            f"aux/memory_layer_{layer}_{name}": value
                            for name, value in layer_diagnostics.items()
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
            if self.normalized_state:
                centered_directional = centered_layerwise_directional_diagnostics(
                    pooled_memory_targets, pooled_memory_estimates, logits
                )
                self.metrics.update(
                    {
                        f"aux/memory_centered_{name}": value
                        for name, value in centered_directional.items()
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
        if self.training:
            self._aux_training_step_value += 1
            self.aux_training_step.fill_(self._aux_training_step_value)
        return AuxCausalLMOutput(logits=logits, aux_loss=aux_loss), terminal_state

import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.sequence.auxiliary import (
    AuxCausalLMOutput,
    boundary_inverse_targets,
    chunk_ranges,
    cross_entropy_sum,
    gaussian_nll_sum,
    memory_observation,
    memory_reconstruction_diagnostics,
    memory_reconstruction_target,
    validate_memory_observation_gradient_scale,
)


class StackedTanhRNN(nn.Module):
    """A stacked vanilla RNN that exposes every layer's state trajectory.

    Each layer is evaluated over the complete sequence in one native RNN call.
    This keeps the recurrent computation independent of auxiliary chunking while
    avoiding a Python loop over tokens.
    """

    def __init__(self, d_model, n_layer, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_layer = n_layer
        self.dropout = float(dropout)
        self.layers = nn.ModuleList(
            nn.RNN(
                d_model,
                d_model,
                num_layers=1,
                nonlinearity="tanh",
                batch_first=True,
            )
            for _ in range(n_layer)
        )

    def reset_parameters(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight_ih_l0)
            nn.init.orthogonal_(layer.weight_hh_l0)
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
            layer_output, terminal_state = layer(
                layer_input, state[index : index + 1]
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
    """Canonical vanilla-tanh RNN with a chunk-inverse auxiliary objective."""

    def __init__(
        self,
        d_model,
        n_layer,
        vocab_size,
        chunk_size=4,
        chunk_offset="random",
        dropout=0.0,
        rho=1.0,
        stop_gradient_memory_target=False,
        stop_gradient_memory_observation=False,
        memory_observation_gradient_scale=1.0,
        use_chunk_loss=True,
        use_discrete_loss=True,
        use_memory_loss=True,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            raise TypeError(
                "unexpected RNN AUX options: " + ", ".join(sorted(kwargs))
            )
        if d_model <= 0 or n_layer <= 0:
            raise ValueError("d_model and n_layer must be positive")
        if chunk_size <= 1:
            raise ValueError("chunk_size must be greater than 1")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if rho <= 0:
            raise ValueError("rho must be positive")
        if chunk_offset != "random":
            if isinstance(chunk_offset, bool) or not isinstance(chunk_offset, int):
                raise ValueError(
                    "chunk_offset must be 'random' or an integer in [0, chunk_size)"
                )
            if not 0 <= chunk_offset < chunk_size:
                raise ValueError("fixed chunk_offset must be in [0, chunk_size)")

        self.d_model = d_model
        self.d_output = vocab_size
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.chunk_offset = chunk_offset
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

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.initial_state = nn.Parameter(torch.empty(n_layer, 1, d_model))
        self.rnn = StackedTanhRNN(d_model, n_layer, dropout=dropout)
        self.inverse_rnn = copy.deepcopy(self.rnn)
        self.memory_predictor = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.register_buffer("rho", torch.tensor(float(rho)))
        self.reset_parameters()
        self.metrics = {}
        self.loss_components = {}

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.initial_state, std=0.02)
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

    def _predict_memory(self, hidden):
        return hidden + self.memory_predictor(hidden)

    def default_state(self, *batch_shape, device=None):
        if len(batch_shape) != 1:
            raise ValueError("RNNAuxLM expects one batch dimension")
        return self.initial_state.to(device=device).expand(
            -1, batch_shape[0], -1
        )

    def _sample_offsets(self, batch_size, device):
        if self.chunk_offset == "random" and self.training:
            return torch.randint(
                self.chunk_size, (batch_size,), device=device
            )
        fixed = 0 if self.chunk_offset == "random" else self.chunk_offset
        return torch.full(
            (batch_size,), fixed, device=device, dtype=torch.long
        )

    def _inverse_batch(self, chunks, boundaries, successor_states):
        data_ids, inverse_targets = boundary_inverse_targets(chunks, boundaries)
        observed_states = memory_observation(
            successor_states,
            self.stop_gradient_memory_observation,
            self.memory_observation_gradient_scale,
        )
        inverse_outputs, inverse_state, _ = self.inverse_rnn(
            self.embedding(data_ids), observed_states
        )
        reconstructed_state = self._predict_memory(inverse_state)
        return self._lm_logits(inverse_outputs), inverse_targets, reconstructed_state

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
        self, input_ids, token_stream, initial_state, trajectory, offsets
    ):
        """Select AUX transitions from an already-computed forward trajectory."""
        records = defaultdict(list)
        length = input_ids.size(1)
        for offset in range(self.chunk_size):
            indices = torch.nonzero(
                offsets == offset, as_tuple=False
            ).squeeze(1)
            if indices.numel() == 0:
                continue
            group_inputs = input_ids.index_select(0, indices)
            group_tokens = token_stream.index_select(0, indices)
            for start, end in chunk_ranges(length, self.chunk_size, offset):
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
        logits,
        compute_diagnostics,
    ):
        batch_size, length = input_ids.shape
        records_by_length = self._collect_inverse_records(
            input_ids, token_stream, initial_state, trajectory, offsets
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
            if inverse_length > 1:
                chunk_logits.append(inverse_logits[:, :-1])
                chunk_targets.append(inverse_targets[:, :-1])
            discrete_logits.append(inverse_logits[:, -1:])
            discrete_targets.append(inverse_targets[:, -1:])
            memory_targets.append(
                memory_reconstruction_target(
                    predecessor_states.transpose(0, 1),
                    self.stop_gradient_memory_target,
                )
            )
            memory_estimates.append(reconstructed_states.transpose(0, 1))

        # Record counts differ by inverse length and sampled offset. Pooling the
        # transition-example axis gives the loss helpers one exact rectangular
        # tensor without padding, repeated forward work, or biased group means.
        memory_targets = [torch.cat(memory_targets, dim=0)]
        memory_estimates = [torch.cat(memory_estimates, dim=0)]

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
            loss_memory = gaussian_nll_sum(
                memory_targets,
                memory_estimates,
                self.rho,
                logits,
                batch_size,
                scale_axis=1,
            ) / sequence_normalizer
        aux_loss = loss_chunk + loss_discrete + loss_memory
        self.loss_components = {
            "chunk_ce": loss_chunk,
            "discrete_ce": loss_discrete,
            "memory_nll": loss_memory,
            "total": aux_loss,
        }
        self.metrics = {
            "aux/chunk_ce": loss_chunk.detach(),
            "aux/discrete_ce": loss_discrete.detach(),
            "aux/memory_nll": loss_memory.detach(),
            "aux/total": aux_loss.detach(),
            "aux/rho": self.rho.detach().reshape(()),
            "aux/rho_mean": self.rho.detach().reshape(()),
        }
        if compute_diagnostics:
            diagnostics = memory_reconstruction_diagnostics(
                memory_targets, memory_estimates, logits
            )
            self.metrics.update(
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
        return aux_loss

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
            return (
                AuxCausalLMOutput(logits=logits, aux_loss=logits.new_zeros(())),
                terminal_state,
            )

        offsets = self._sample_offsets(batch_size, input_ids.device)
        aux_loss = self._compute_auxiliary_loss(
            input_ids,
            token_stream,
            initial_state,
            trajectory,
            offsets,
            logits,
            compute_diagnostics,
        )
        return AuxCausalLMOutput(logits=logits, aux_loss=aux_loss), terminal_state

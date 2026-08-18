import copy

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
        self.rnn = nn.RNN(
            d_model,
            d_model,
            num_layers=n_layer,
            nonlinearity="tanh",
            dropout=dropout if n_layer > 1 else 0.0,
            batch_first=True,
        )
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
        for layer in range(self.n_layer):
            nn.init.xavier_uniform_(getattr(self.rnn, f"weight_ih_l{layer}"))
            nn.init.orthogonal_(getattr(self.rnn, f"weight_hh_l{layer}"))
            nn.init.zeros_(getattr(self.rnn, f"bias_ih_l{layer}"))
            nn.init.zeros_(getattr(self.rnn, f"bias_hh_l{layer}"))
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

    def _inverse_record(self, chunks, boundaries, successor_states):
        data_ids, inverse_targets = boundary_inverse_targets(chunks, boundaries)
        observed_states = memory_observation(
            successor_states,
            self.stop_gradient_memory_observation,
            self.memory_observation_gradient_scale,
        )
        inverse_outputs, inverse_state = self.inverse_rnn(
            self.embedding(data_ids), observed_states
        )
        reconstructed_state = self._predict_memory(inverse_state)
        return self._lm_logits(inverse_outputs), inverse_targets, reconstructed_state

    def _forward_with_offset(
        self,
        input_ids,
        targets,
        aux_tokens,
        state,
        compute_aux,
        compute_diagnostics,
        offset,
    ):
        batch_size, length = input_ids.shape
        if length == 0:
            raise ValueError("input sequence must be non-empty")
        token_stream = aux_tokens
        if token_stream is None and targets is not None and not torch.any(targets < 0):
            token_stream = targets
        if token_stream is not None and token_stream.shape != input_ids.shape:
            raise ValueError("aux_tokens must have the same shape as input_ids")
        if compute_aux and token_stream is None:
            raise ValueError("unmasked aux_tokens are required when compute_aux=True")

        hidden = (
            self.default_state(batch_size, device=input_ids.device)
            if state is None
            else state
        )
        ranges = chunk_ranges(length, self.chunk_size, offset)
        forward_logits = []
        inverse_records = []
        for start, end in ranges:
            state_before = hidden
            outputs, hidden = self.rnn(
                self.embedding(input_ids[:, start:end]), hidden
            )
            forward_logits.append(self._lm_logits(outputs))
            if compute_aux:
                inverse_records.append(
                    (
                        token_stream[:, start:end],
                        input_ids[:, start],
                        state_before,
                        hidden,
                    )
                )

        logits = torch.cat(forward_logits, dim=1)
        aux_loss = logits.new_zeros(())
        if not compute_aux:
            self.metrics = {}
            self.loss_components = {}
            return AuxCausalLMOutput(logits=logits, aux_loss=aux_loss), hidden

        chunk_logits = []
        chunk_targets = []
        discrete_logits = []
        discrete_targets = []
        memory_targets = []
        memory_estimates = []
        for inverse_length in sorted(
            {record[0].size(1) for record in inverse_records}
        ):
            records = [
                record
                for record in inverse_records
                if record[0].size(1) == inverse_length
            ]
            chunks = torch.cat([record[0] for record in records], dim=0)
            boundaries = torch.cat([record[1] for record in records], dim=0)
            successor_states = torch.cat(
                [record[3] for record in records], dim=1
            )
            inverse_logits, inverse_targets, reconstructed_states = (
                self._inverse_record(chunks, boundaries, successor_states)
            )
            if inverse_length > 1:
                chunk_logits.append(inverse_logits[:, :-1])
                chunk_targets.append(inverse_targets[:, :-1])
            discrete_logits.append(inverse_logits[:, -1:])
            discrete_targets.append(inverse_targets[:, -1:])
            memory_targets.extend(
                memory_reconstruction_target(
                    record[2].transpose(0, 1),
                    self.stop_gradient_memory_target,
                )
                for record in records
            )
            memory_estimates.extend(
                value.transpose(0, 1)
                for value in reconstructed_states.split(batch_size, dim=1)
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
        return AuxCausalLMOutput(logits=logits, aux_loss=aux_loss), hidden

    def _forward_per_sequence(
        self,
        input_ids,
        targets,
        aux_tokens,
        state,
        compute_aux,
        compute_diagnostics,
    ):
        batch_size, length = input_ids.shape
        offsets = self._sample_offsets(batch_size, input_ids.device)
        logits = None
        terminal_state = None
        aux_loss = self.embedding.weight.new_zeros(())
        combined_metrics = {}
        combined_loss_components = {}
        for offset in offsets.unique(sorted=True).tolist():
            indices = torch.nonzero(
                offsets == offset, as_tuple=False
            ).squeeze(1)
            group_state = None if state is None else state.index_select(1, indices)
            output, group_terminal = self._forward_with_offset(
                input_ids.index_select(0, indices),
                None if targets is None else targets.index_select(0, indices),
                None if aux_tokens is None else aux_tokens.index_select(0, indices),
                group_state,
                compute_aux,
                compute_diagnostics,
                int(offset),
            )
            weight = indices.numel() / batch_size
            if logits is None:
                logits = output.logits.new_zeros(
                    batch_size, length, self.vocab_size
                )
                terminal_state = group_terminal.new_zeros(
                    self.n_layer, batch_size, self.d_model
                )
            logits = logits.index_copy(0, indices, output.logits)
            terminal_state = terminal_state.index_copy(
                1, indices, group_terminal
            )
            aux_loss = aux_loss + output.aux_loss * weight
            for name, value in self.metrics.items():
                combined_metrics[name] = combined_metrics.get(
                    name, value.new_zeros(())
                ) + value * weight
            for name, value in self.loss_components.items():
                combined_loss_components[name] = combined_loss_components.get(
                    name, value.new_zeros(())
                ) + value * weight
        self.metrics = combined_metrics
        self.loss_components = combined_loss_components
        return AuxCausalLMOutput(logits=logits, aux_loss=aux_loss), terminal_state

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
        if self.chunk_offset == "random" and self.training:
            return self._forward_per_sequence(
                input_ids,
                targets,
                aux_tokens,
                state,
                compute_aux,
                compute_diagnostics,
            )
        offset = 0 if self.chunk_offset == "random" else self.chunk_offset
        return self._forward_with_offset(
            input_ids,
            targets,
            aux_tokens,
            state,
            compute_aux,
            compute_diagnostics,
            offset,
        )

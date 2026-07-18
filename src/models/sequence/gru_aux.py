import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.sequence.auxiliary import (
    AuxCausalLMOutput,
    boundary_inverse_targets,
    chunk_ranges,
    configured_scale,
    cross_entropy_sum,
    gaussian_nll_sum,
    initialize_scale,
    mean_batch_variance,
    memory_reconstruction_target,
    noisy_generation,
    noisy_observation,
    normalize_offset_mode,
    normalize_token_scheme,
    resolve_direction_embedding,
    role_inverse_targets,
    terminal_gaussian_nll,
    validate_generation_noise_std,
    validate_observation_noise_std,
    validate_scale_configuration,
)


class GRUAuxLM(nn.Module):
    """Stacked GRU LM implementing the report's inverse-auxiliary options."""

    def __init__(
        self,
        d_model,
        n_layer,
        vocab_size,
        chunk_size=4,
        dropout=0.0,
        token_scheme="boundary_reverse",
        random_chunk_offset="sequence",
        use_memory_token=True,
        share_inverse=True,
        share_inverse_embedding=True,
        share_inverse_head=True,
        use_direction_embedding=False,
        memory_scale_mode="fixed",
        memory_scale_granularity="global",
        terminal_scale_mode="fixed",
        terminal_scale_granularity="global",
        rho=1.0,
        tau=1.0,
        stop_gradient_memory_target=False,
        learnable_terminal_target=False,
        observation_noise_std=0.0,
        generation_noise_std=0.0,
        use_terminal_chunk=True,
        use_chunk_loss=True,
        use_discrete_loss=True,
        use_memory_loss=True,
        use_terminal_loss=True,
        use_terminal_chunk_loss=True,
        memory_min_scale=1e-4,
        memory_max_scale=1e4,
        terminal_min_scale=1e-4,
        terminal_max_scale=1e4,
        **kwargs,
    ):
        super().__init__()
        if chunk_size <= 1:
            raise ValueError("chunk_size must be greater than 1")
        if "chunk_offset" in kwargs:
            raise ValueError("chunk_offset is not configurable; fixed offset is always 0")
        legacy_scale_options = {
            "scale_mode",
            "scale_granularity",
            "min_scale",
            "max_scale",
        }.intersection(kwargs)
        if legacy_scale_options:
            names = ", ".join(sorted(legacy_scale_options))
            raise ValueError(
                f"{names} must be configured separately for memory and terminal losses"
            )
        self.d_model = d_model
        self.d_output = vocab_size
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.token_scheme = normalize_token_scheme(token_scheme)
        self.chunk_offset_mode = normalize_offset_mode(random_chunk_offset)
        self.random_chunk_offset = self.chunk_offset_mode != "fixed"
        self.use_memory_token = use_memory_token
        self.share_inverse = share_inverse
        self.share_inverse_embedding = share_inverse_embedding
        self.share_inverse_head = share_inverse_head
        self.use_direction_embedding = resolve_direction_embedding(
            use_direction_embedding,
            self.token_scheme,
        )
        self.memory_scale_mode = memory_scale_mode
        self.memory_scale_granularity = memory_scale_granularity
        self.terminal_scale_mode = terminal_scale_mode
        self.terminal_scale_granularity = terminal_scale_granularity
        self.stop_gradient_memory_target = stop_gradient_memory_target
        self.learnable_terminal_target = learnable_terminal_target
        self.observation_noise_std = validate_observation_noise_std(
            observation_noise_std
        )
        self.generation_noise_std = validate_generation_noise_std(
            generation_noise_std
        )
        self.use_terminal_chunk = use_terminal_chunk
        self.use_chunk_loss = use_chunk_loss
        self.use_discrete_loss = use_discrete_loss
        self.use_memory_loss = use_memory_loss
        self.use_terminal_loss = use_terminal_loss
        self.use_terminal_chunk_loss = use_terminal_chunk_loss
        self.memory_min_scale = memory_min_scale
        self.memory_max_scale = memory_max_scale
        self.terminal_min_scale = terminal_min_scale
        self.terminal_max_scale = terminal_max_scale
        self.memory_token_id = vocab_size
        self.inverse_token_id = vocab_size + 1

        validate_scale_configuration(
            memory_scale_mode,
            memory_scale_granularity,
            {"global", "layerwise"},
        )
        validate_scale_configuration(
            terminal_scale_mode,
            terminal_scale_granularity,
            {"global", "layerwise"},
        )
        memory_scale_size = (
            n_layer if memory_scale_granularity == "layerwise" else 1
        )
        terminal_scale_size = (
            n_layer if terminal_scale_granularity == "layerwise" else 1
        )

        # Allocate both special tokens so checkpoints stay shape-compatible across schemes.
        self.embedding = nn.Embedding(vocab_size + 2, d_model)
        self.inverse_embedding = (
            self.embedding if share_inverse_embedding else copy.deepcopy(self.embedding)
        )
        self.gru = nn.GRU(
            d_model,
            d_model,
            num_layers=n_layer,
            dropout=dropout if n_layer > 1 else 0.0,
            batch_first=True,
        )
        self.inverse_gru = self.gru if share_inverse else copy.deepcopy(self.gru)
        if not share_inverse_head:
            self.inverse_head = nn.Linear(d_model, vocab_size, bias=False)

        self.direction_embedding = None
        if self.use_direction_embedding:
            self.direction_embedding = nn.Parameter(torch.zeros(d_model))

        initialize_scale(
            self,
            "rho",
            rho,
            memory_scale_size,
            memory_scale_mode,
            memory_scale_granularity,
        )
        initialize_scale(
            self,
            "tau",
            tau,
            terminal_scale_size,
            terminal_scale_mode,
            terminal_scale_granularity,
        )
        self.terminal_target = (
            nn.Parameter(torch.zeros(n_layer, 1, d_model))
            if learnable_terminal_target
            else None
        )
        nn.init.normal_(self.embedding.weight, std=0.02)
        if self.inverse_embedding is not self.embedding:
            with torch.no_grad():
                self.inverse_embedding.weight.copy_(self.embedding.weight)
        if hasattr(self, "inverse_head"):
            with torch.no_grad():
                self.inverse_head.weight.copy_(self.embedding.weight[:vocab_size])

        self.metrics = {}
        self.loss_components = {}

    def _lm_logits(self, hidden):
        return F.linear(hidden, self.embedding.weight[:self.vocab_size])

    def _inverse_logits(self, hidden):
        if not hasattr(self, "inverse_head"):
            return self._lm_logits(hidden)
        return self.inverse_head(hidden)

    def _scale(self, name):
        if name == "rho":
            mode = self.memory_scale_mode
            minimum = self.memory_min_scale
            maximum = self.memory_max_scale
        elif name == "tau":
            mode = self.terminal_scale_mode
            minimum = self.terminal_min_scale
            maximum = self.terminal_max_scale
        else:
            raise ValueError(f"unknown scale: {name}")
        return configured_scale(
            self, name, mode, minimum, maximum
        )

    def _offset(self, device, compute_aux, forced_offset):
        if forced_offset is not None:
            return forced_offset
        return 0

    def default_state(self, *batch_shape, device=None):
        if len(batch_shape) != 1:
            raise ValueError("GRUAuxLM expects one batch dimension")
        return torch.zeros(self.n_layer, batch_shape[0], self.d_model, device=device)

    def _inverse_record(self, chunks, boundaries, initial_states):
        inverse_length = chunks.size(1)
        if self.token_scheme == "boundary_reverse":
            data_ids, inverse_targets = boundary_inverse_targets(chunks, boundaries)
            inverse_data = self.inverse_embedding(data_ids)
            if self.direction_embedding is not None:
                inverse_data = inverse_data + self.direction_embedding
            inputs = [inverse_data]
        else:
            data_ids, inverse_targets = role_inverse_targets(
                chunks, self.token_scheme
            )
            role_ids = torch.full(
                (chunks.size(0), 1),
                self.inverse_token_id,
                dtype=torch.long,
                device=chunks.device,
            )
            inputs = [self.inverse_embedding(role_ids), self.inverse_embedding(data_ids)]

        if self.use_memory_token:
            memory_ids = torch.full(
                (chunks.size(0), 1),
                self.memory_token_id,
                dtype=torch.long,
                device=chunks.device,
            )
            inputs.append(self.inverse_embedding(memory_ids))

        observed_states = noisy_observation(
            initial_states, self.observation_noise_std, self.training
        )
        inverse_outputs, reconstructed_states = self.inverse_gru(
            torch.cat(inputs, dim=1), observed_states
        )
        if self.token_scheme == "boundary_reverse":
            token_logits = self._inverse_logits(inverse_outputs[:, :inverse_length])
        else:
            # The role token and first B-1 data positions predict all B tokens.
            token_logits = self._inverse_logits(inverse_outputs[:, :inverse_length])
        return token_logits, inverse_targets, reconstructed_states

    def _forward_with_offset(
        self, input_ids, targets, aux_tokens, state, compute_aux, forced_offset
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
        ranges = chunk_ranges(
            length,
            self.chunk_size,
            self._offset(input_ids.device, compute_aux, forced_offset),
        )

        forward_logits = []
        chunk_logits = []
        chunk_targets = []
        discrete_logits = []
        discrete_targets = []
        memory_targets = []
        memory_estimates = []
        inverse_records = []

        transition_ranges = ranges[:-1] if self.use_terminal_chunk else ranges
        terminal_range = ranges[-1] if self.use_terminal_chunk else None
        if self.token_scheme == "boundary_reverse":
            for start, end in transition_ranges:
                state_before = hidden
                outputs, hidden = self.gru(
                    self.embedding(input_ids[:, start:end]), hidden
                )
                forward_logits.append(self._lm_logits(outputs))
                hidden = noisy_generation(
                    hidden, self.generation_noise_std, self.training
                )
                if compute_aux:
                    chunk = token_stream[:, start:end]
                    boundary = input_ids[:, start]
                    inverse_records.append((chunk, boundary, state_before, hidden))

            terminal_memory = hidden
            if terminal_range is not None:
                terminal_start, terminal_end = terminal_range
                terminal_outputs, _ = self.gru(
                    self.embedding(input_ids[:, terminal_start:terminal_end]),
                    terminal_memory,
                )
                final_chunk_logits = self._lm_logits(terminal_outputs)
                forward_logits.append(final_chunk_logits)
            else:
                final_chunk_logits = None
        else:
            # The host LM remains globally shifted. Process the token preceding
            # C_0 first to obtain S_0, then advance exactly through each
            # transition chunk C_j. This makes the stored state boundaries and
            # inverse chunks refer to the same unmasked token stream.
            initial_outputs, hidden = self.gru(
                self.embedding(input_ids[:, :1]), hidden
            )
            forward_logits.append(self._lm_logits(initial_outputs))
            cursor = 1
            for start, end in transition_ranges:
                state_before = hidden
                advance_end = end + 1
                available_end = min(advance_end, length)
                if cursor < available_end:
                    outputs, hidden = self.gru(
                        self.embedding(input_ids[:, cursor:available_end]),
                        hidden,
                    )
                    forward_logits.append(self._lm_logits(outputs))
                if advance_end > length and token_stream is not None:
                    # The final shifted target has no corresponding host-LM input.
                    # It still completes the last auxiliary transition when C_K
                    # is omitted.
                    _, hidden = self.gru(
                        self.embedding(token_stream[:, end - 1:end]), hidden
                    )
                hidden = noisy_generation(
                    hidden, self.generation_noise_std, self.training
                )
                if compute_aux:
                    chunk = token_stream[:, start:end]
                    inverse_records.append((chunk, None, state_before, hidden))
                cursor = advance_end

            terminal_memory = hidden
            if terminal_range is not None and cursor < length:
                terminal_outputs, _ = self.gru(
                    self.embedding(input_ids[:, cursor:]), terminal_memory
                )
                forward_logits.append(self._lm_logits(terminal_outputs))

        logits = torch.cat(forward_logits, dim=1)
        if terminal_range is not None:
            terminal_start, terminal_end = terminal_range
            final_chunk_logits = logits[:, terminal_start:terminal_end]
        aux_loss = logits.new_zeros(())

        if compute_aux:
            final_chunk_targets = (
                None
                if terminal_range is None
                else token_stream[:, terminal_start:terminal_end]
            )

            lengths = sorted({record[0].size(1) for record in inverse_records})
            for inverse_length in lengths:
                records = [
                    record
                    for record in inverse_records
                    if record[0].size(1) == inverse_length
                ]
                chunks = torch.cat([record[0] for record in records], dim=0)
                boundaries = None
                if self.token_scheme == "boundary_reverse":
                    boundaries = torch.cat([record[1] for record in records], dim=0)
                initial_states = torch.cat([record[3] for record in records], dim=1)
                inverse_logits, inverse_targets, reconstructed_states = (
                    self._inverse_record(chunks, boundaries, initial_states)
                )

                if self.token_scheme == "boundary_reverse":
                    if inverse_length > 1:
                        chunk_logits.append(inverse_logits[:, :-1])
                        chunk_targets.append(inverse_targets[:, :-1])
                    discrete_logits.append(inverse_logits[:, -1:])
                    discrete_targets.append(inverse_targets[:, -1:])
                else:
                    chunk_logits.append(inverse_logits)
                    chunk_targets.append(inverse_targets)
                memory_targets.extend([
                    memory_reconstruction_target(
                        record[2], self.stop_gradient_memory_target
                    )
                    for record in records
                ])
                memory_estimates.extend(
                    reconstructed_states.split(batch_size, dim=1)
                )

            rho = self._scale("rho")
            tau = self._scale("tau")
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
                    rho,
                    logits,
                    batch_size,
                    scale_axis=0,
                ) / sequence_normalizer
            loss_terminal = logits.new_zeros(())
            if self.use_terminal_loss:
                loss_terminal = terminal_gaussian_nll(
                    terminal_memory,
                    tau,
                    batch_size,
                    scale_axis=0,
                    target=self.terminal_target,
                ) / sequence_normalizer
            loss_terminal_chunk = logits.new_zeros(())
            if terminal_range is not None and self.use_terminal_chunk_loss:
                loss_terminal_chunk = F.cross_entropy(
                    final_chunk_logits.reshape(-1, self.vocab_size),
                    final_chunk_targets.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                ) / (batch_size * sequence_normalizer)
            aux_loss = (
                loss_chunk
                + loss_discrete
                + loss_memory
                + loss_terminal
                + loss_terminal_chunk
            )
            self.loss_components = {
                "chunk_ce": loss_chunk,
                "discrete_ce": loss_discrete,
                "memory_nll": loss_memory,
                "terminal_nll": loss_terminal,
                "terminal_chunk": loss_terminal_chunk,
                "total": aux_loss,
            }
            self.metrics = {
                "aux/chunk_ce": loss_chunk.detach(),
                "aux/discrete_ce": loss_discrete.detach(),
                "aux/memory_nll": loss_memory.detach(),
                "aux/terminal_nll": loss_terminal.detach(),
                "aux/terminal_chunk": loss_terminal_chunk.detach(),
                "aux/total": aux_loss.detach(),
                "aux/rho_mean": rho.detach().mean(),
                "aux/tau_mean": tau.detach().mean(),
                "aux/memory_batch_variance": mean_batch_variance(
                    [record[3] for record in inverse_records], batch_axis=1
                ),
                "aux/terminal_batch_variance": mean_batch_variance(
                    [terminal_memory], batch_axis=1
                ),
            }
            if rho.numel() == 1:
                self.metrics["aux/rho"] = rho.detach().reshape(())
            if tau.numel() == 1:
                self.metrics["aux/tau"] = tau.detach().reshape(())
            for index, value in enumerate(rho.detach().reshape(-1)):
                self.metrics[f"aux/rho/{index}"] = value
            for index, value in enumerate(tau.detach().reshape(-1)):
                self.metrics[f"aux/tau/{index}"] = value
            for index, record in enumerate(inverse_records):
                self.metrics[f"aux/memory_batch_variance/{index}"] = (
                    mean_batch_variance([record[3]], batch_axis=1)
                )
        else:
            self.metrics = {}
            self.loss_components = {}

        return AuxCausalLMOutput(logits=logits, aux_loss=aux_loss), terminal_memory

    def _forward_per_sequence(
        self, input_ids, targets, aux_tokens, state, compute_aux
    ):
        batch_size, length = input_ids.shape
        offsets = torch.randint(
            self.chunk_size, (batch_size,), device=input_ids.device
        )
        logits = None
        terminal_state = None
        aux_loss = input_ids.new_zeros((), dtype=self.embedding.weight.dtype)
        combined_metrics = {}
        combined_loss_components = {}
        for offset in offsets.unique(sorted=True).tolist():
            indices = torch.nonzero(offsets == offset, as_tuple=False).squeeze(1)
            group_state = None if state is None else state.index_select(1, indices)
            output, group_terminal = self._forward_with_offset(
                input_ids.index_select(0, indices),
                None if targets is None else targets.index_select(0, indices),
                None if aux_tokens is None else aux_tokens.index_select(0, indices),
                group_state,
                compute_aux,
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
            terminal_state = terminal_state.index_copy(1, indices, group_terminal)
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
        **kwargs,
    ):
        if (
            compute_aux
            and self.training
            and self.chunk_offset_mode == "sequence"
        ):
            return self._forward_per_sequence(
                input_ids, targets, aux_tokens, state, compute_aux
            )
        return self._forward_with_offset(
            input_ids,
            targets,
            aux_tokens,
            state,
            compute_aux,
            forced_offset=None,
        )

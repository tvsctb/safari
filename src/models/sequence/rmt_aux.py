import copy
import math

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
    mean_reconstruction_mse,
    memory_reconstruction_target,
    noisy_generation,
    noisy_observation,
    normalize_token_scheme,
    resolve_direction_embedding,
    role_inverse_targets,
    terminal_gaussian_nll,
    terminal_reconstruction_mse,
    validate_generation_noise_std,
    validate_observation_noise_std,
    validate_scale_configuration,
)


class CausalTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_inner, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask):
        normalized = self.norm1(x)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            need_weights=False,
        )
        x = x + self.dropout(attended)
        return x + self.dropout(self.mlp(self.norm2(x)))


def initialize_position_embedding(parameter, mode, std=0.02):
    if mode == "normal":
        nn.init.normal_(parameter, std=std)
        return
    if mode != "sinusoidal":
        raise ValueError("position_initialization must be normal or sinusoidal")
    length, dimension = parameter.shape
    positions = torch.arange(
        length, device=parameter.device, dtype=parameter.dtype
    ).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(
            0, dimension, 2, device=parameter.device, dtype=parameter.dtype
        )
        * (-math.log(10000.0) / dimension)
    )
    values = torch.zeros_like(parameter)
    values[:, 0::2] = torch.sin(positions * frequencies)
    values[:, 1::2] = torch.cos(positions * frequencies[: values[:, 1::2].shape[1]])
    values = values - values.mean()
    values = values * (std / values.std().clamp_min(1e-12))
    with torch.no_grad():
        parameter.copy_(values)


def initialize_transformer_blocks(blocks, mode):
    if mode == "default":
        return
    if mode != "orthogonal_residual":
        raise ValueError(
            "block_initialization must be default or orthogonal_residual"
        )
    residual_gain = 1.0 / math.sqrt(2.0 * len(blocks))
    with torch.no_grad():
        for block in blocks:
            for projection in block.attention.in_proj_weight.chunk(3, dim=0):
                nn.init.orthogonal_(projection)
            if block.attention.in_proj_bias is not None:
                block.attention.in_proj_bias.zero_()
            nn.init.orthogonal_(
                block.attention.out_proj.weight, gain=residual_gain
            )
            if block.attention.out_proj.bias is not None:
                block.attention.out_proj.bias.zero_()
            nn.init.orthogonal_(block.mlp[0].weight)
            block.mlp[0].bias.zero_()
            nn.init.orthogonal_(block.mlp[3].weight, gain=residual_gain)
            block.mlp[3].bias.zero_()


class RMTAuxLM(nn.Module):
    """Decoder-only RMT implementing the report's inverse-auxiliary options."""

    def __init__(
        self,
        d_model,
        n_layer,
        d_inner,
        n_heads,
        vocab_size,
        chunk_size=4,
        num_memory_tokens=4,
        dropout=0.0,
        token_scheme="boundary_reverse",
        share_inverse=True,
        share_inverse_position_embedding=None,
        inverse_position_initialization="copy",
        position_initialization="normal",
        block_initialization="default",
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
        if "random_chunk_offset" in kwargs:
            raise ValueError("RMT chunk offset is fixed at 0 and is not configurable")
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
        if num_memory_tokens <= 0:
            raise ValueError("num_memory_tokens must be positive")
        self.d_model = d_model
        self.d_output = vocab_size
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.num_memory_tokens = num_memory_tokens
        self.token_scheme = normalize_token_scheme(token_scheme)
        self.share_inverse = share_inverse
        self.share_inverse_position_embedding = (
            share_inverse
            if share_inverse_position_embedding is None
            else share_inverse_position_embedding
        )
        if inverse_position_initialization not in {"copy", "independent"}:
            raise ValueError(
                "inverse_position_initialization must be copy or independent"
            )
        self.inverse_position_initialization = inverse_position_initialization
        if position_initialization not in {"normal", "sinusoidal"}:
            raise ValueError(
                "position_initialization must be normal or sinusoidal"
            )
        self.position_initialization = position_initialization
        if block_initialization not in {"default", "orthogonal_residual"}:
            raise ValueError(
                "block_initialization must be default or orthogonal_residual"
            )
        self.block_initialization = block_initialization
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
        self.forward_token_id = vocab_size
        self.inverse_token_id = vocab_size + 1

        validate_scale_configuration(
            memory_scale_mode,
            memory_scale_granularity,
            {"global", "slotwise"},
        )
        validate_scale_configuration(
            terminal_scale_mode,
            terminal_scale_granularity,
            {"global", "slotwise"},
        )
        memory_scale_size = (
            num_memory_tokens
            if memory_scale_granularity == "slotwise"
            else 1
        )
        terminal_scale_size = (
            num_memory_tokens
            if terminal_scale_granularity == "slotwise"
            else 1
        )

        self.embedding = nn.Embedding(vocab_size + 2, d_model)
        self.inverse_embedding = (
            self.embedding if share_inverse_embedding else copy.deepcopy(self.embedding)
        )
        if not share_inverse_head:
            self.inverse_head = nn.Linear(d_model, vocab_size, bias=False)
        self.initial_memory = nn.Parameter(torch.empty(num_memory_tokens, d_model))
        self.forward_queries = nn.Parameter(torch.empty(num_memory_tokens, d_model))
        self.inverse_queries = nn.Parameter(torch.empty(num_memory_tokens, d_model))
        self.direction_embedding = None
        if self.use_direction_embedding:
            self.direction_embedding = nn.Parameter(torch.zeros(d_model))

        max_token_length = chunk_size + 1
        self.position_embedding = nn.Parameter(
            torch.empty(num_memory_tokens + max_token_length, d_model)
        )
        self.inverse_position_embedding = (
            None
            if self.share_inverse_position_embedding
            else nn.Parameter(
                torch.empty(num_memory_tokens + max_token_length, d_model)
            )
        )
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(d_model, n_heads, d_inner, dropout)
            for _ in range(n_layer)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        if share_inverse:
            self.inverse_blocks = self.blocks
            self.inverse_final_norm = self.final_norm
        else:
            self.inverse_blocks = copy.deepcopy(self.blocks)
            self.inverse_final_norm = copy.deepcopy(self.final_norm)

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
            nn.Parameter(torch.zeros(num_memory_tokens, d_model))
            if learnable_terminal_target
            else None
        )
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(
                    2 * num_memory_tokens + max_token_length,
                    2 * num_memory_tokens + max_token_length,
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
            persistent=False,
        )
        self.reset_parameters()
        self.metrics = {}
        self.loss_components = {}
        self._initial_aux_diagnostics_pending = True

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.initial_memory, std=0.02)
        nn.init.normal_(self.forward_queries, std=0.02)
        nn.init.normal_(self.inverse_queries, std=0.02)
        initialize_position_embedding(
            self.position_embedding, self.position_initialization
        )
        initialize_transformer_blocks(self.blocks, self.block_initialization)
        if self.inverse_blocks is not self.blocks:
            initialize_transformer_blocks(
                self.inverse_blocks, self.block_initialization
            )
        with torch.no_grad():
            if self.inverse_embedding is not self.embedding:
                self.inverse_embedding.weight.copy_(self.embedding.weight)
            if hasattr(self, "inverse_head"):
                self.inverse_head.weight.copy_(
                    self.embedding.weight[:self.vocab_size]
                )
            if self.inverse_position_embedding is not None:
                if self.inverse_position_initialization == "copy":
                    self.inverse_position_embedding.copy_(self.position_embedding)
                else:
                    nn.init.normal_(self.inverse_position_embedding, std=0.02)

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

    def default_state(self, *batch_shape, device=None):
        if len(batch_shape) != 1:
            raise ValueError("RMTAuxLM expects one batch dimension")
        memory = self.initial_memory.to(device=device)
        return memory.unsqueeze(0).expand(batch_shape[0], -1, -1)

    def _transform(
        self,
        memory,
        token_embeddings,
        queries,
        blocks,
        final_norm,
        position_embedding=None,
    ):
        batch_size = memory.size(0)
        token_length = token_embeddings.size(1)
        if token_length > self.chunk_size + 1:
            raise ValueError("RMT token input exceeds the configured scheme length")
        position_embedding = (
            self.position_embedding
            if position_embedding is None
            else position_embedding
        )
        positioned_length = self.num_memory_tokens + token_length
        x = torch.cat((memory, token_embeddings), dim=1)
        x = x + position_embedding[:positioned_length]
        if queries is not None:
            query_batch = queries.unsqueeze(0).expand(batch_size, -1, -1)
            x = torch.cat((x, query_batch), dim=1)
        sequence_length = x.size(1)
        mask = self.causal_mask[:sequence_length, :sequence_length]
        for block in blocks:
            x = block(x, mask)
        x = final_norm(x)
        token_start = self.num_memory_tokens
        token_end = token_start + token_length
        next_memory = None if queries is None else x[:, -self.num_memory_tokens:]
        return x[:, token_start:token_end], next_memory

    def _role_embeddings(self, chunks, role_token_id, embedding):
        role_ids = torch.full(
            (chunks.size(0), 1),
            role_token_id,
            dtype=torch.long,
            device=chunks.device,
        )
        return torch.cat((embedding(role_ids), embedding(chunks)), dim=1)

    def _forward_chunk(self, memory, input_chunk, token_chunk, terminal=False):
        if self.token_scheme == "boundary_reverse":
            token_embeddings = self.embedding(input_chunk)
        else:
            if token_chunk is None:
                raise ValueError("role-token RMT requires an unmasked token stream")
            token_embeddings = self._role_embeddings(
                token_chunk, self.forward_token_id, self.embedding
            )
        token_hidden, next_memory = self._transform(
            memory,
            token_embeddings,
            None if terminal else self.forward_queries,
            self.blocks,
            self.final_norm,
        )
        # Role schemes ignore the final data-token output.
        lm_hidden = token_hidden if self.token_scheme == "boundary_reverse" else token_hidden[:, :-1]
        return self._lm_logits(lm_hidden), next_memory

    def _inverse_record(self, chunks, boundaries, successor_memories):
        if self.token_scheme == "boundary_reverse":
            data_ids, inverse_targets = boundary_inverse_targets(chunks, boundaries)
            inverse_embeddings = self.inverse_embedding(data_ids)
            if self.direction_embedding is not None:
                inverse_embeddings = inverse_embeddings + self.direction_embedding
        else:
            data_ids, inverse_targets = role_inverse_targets(
                chunks, self.token_scheme
            )
            inverse_embeddings = self._role_embeddings(
                data_ids, self.inverse_token_id, self.inverse_embedding
            )
        observed_memories = noisy_observation(
            successor_memories, self.observation_noise_std, self.training
        )
        inverse_hidden, reconstructed_memories = self._transform(
            observed_memories,
            inverse_embeddings,
            self.inverse_queries,
            self.inverse_blocks,
            self.inverse_final_norm,
            self.inverse_position_embedding,
        )
        if self.token_scheme != "boundary_reverse":
            inverse_hidden = inverse_hidden[:, :-1]
        return self._inverse_logits(inverse_hidden), inverse_targets, reconstructed_memories

    def _forward_with_offset(
        self, input_ids, targets, aux_tokens, state, compute_aux
    ):
        batch_size, length = input_ids.shape
        if length == 0:
            raise ValueError("input sequence must be non-empty")
        token_stream = aux_tokens
        if token_stream is None and targets is not None and not torch.any(targets < 0):
            token_stream = targets
        if token_stream is not None and token_stream.shape != input_ids.shape:
            raise ValueError("aux_tokens must have the same shape as input_ids")
        if self.token_scheme != "boundary_reverse" and token_stream is None:
            raise ValueError("role-token RMT requires unmasked aux_tokens")
        if compute_aux and token_stream is None:
            raise ValueError("unmasked aux_tokens are required when compute_aux=True")
        memory = (
            self.default_state(batch_size, device=input_ids.device)
            if state is None
            else state
        )
        ranges = chunk_ranges(length, self.chunk_size, offset=0)

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
        for start, end in transition_ranges:
            memory_before = memory
            token_chunk = None if token_stream is None else token_stream[:, start:end]
            logits, memory = self._forward_chunk(
                memory,
                input_ids[:, start:end],
                token_chunk,
                terminal=False,
            )
            forward_logits.append(logits)
            memory = noisy_generation(
                memory, self.generation_noise_std, self.training
            )
            if compute_aux:
                boundary = (
                    input_ids[:, start]
                    if self.token_scheme == "boundary_reverse"
                    else None
                )
                inverse_records.append(
                    (token_chunk, boundary, memory_before, memory)
                )

        terminal_memory = memory
        final_chunk_tokens = None
        final_chunk_logits = None
        if terminal_range is not None:
            terminal_start, terminal_end = terminal_range
            final_chunk_tokens = (
                None
                if token_stream is None
                else token_stream[:, terminal_start:terminal_end]
            )
            final_chunk_logits, no_successor_memory = self._forward_chunk(
                terminal_memory,
                input_ids[:, terminal_start:terminal_end],
                final_chunk_tokens,
                terminal=True,
            )
            if no_successor_memory is not None:
                raise RuntimeError("terminal RMT call must not produce successor memory")
            forward_logits.append(final_chunk_logits)
        logits = torch.cat(forward_logits, dim=1)
        aux_loss = logits.new_zeros(())

        if compute_aux:
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
                successor_memories = torch.cat(
                    [record[3] for record in records], dim=0
                )
                inverse_logits, inverse_targets, reconstructed_memories = (
                    self._inverse_record(chunks, boundaries, successor_memories)
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
                    reconstructed_memories.split(batch_size, dim=0)
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
                    scale_axis=1,
                ) / sequence_normalizer
            loss_terminal = logits.new_zeros(())
            if self.use_terminal_loss:
                loss_terminal = terminal_gaussian_nll(
                    terminal_memory,
                    tau,
                    batch_size,
                    scale_axis=1,
                    target=self.terminal_target,
                ) / sequence_normalizer
            loss_terminal_chunk = logits.new_zeros(())
            if terminal_range is not None and self.use_terminal_chunk_loss:
                loss_terminal_chunk = F.cross_entropy(
                    final_chunk_logits.reshape(-1, self.vocab_size),
                    final_chunk_tokens.reshape(-1),
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
            initial_aux_diagnostics = {}
            if self._initial_aux_diagnostics_pending:
                for name, records in (
                    ("chunk", chunk_logits),
                    ("discrete", discrete_logits),
                ):
                    if records:
                        initial_logits = torch.cat(
                            [record.reshape(-1, self.vocab_size) for record in records],
                            dim=0,
                        ).detach().float()
                        initial_probabilities = initial_logits.softmax(dim=-1)
                        initial_entropy = -(
                            initial_probabilities
                            * initial_probabilities.clamp_min(1e-12).log()
                        ).sum(dim=-1)
                        initial_top_two = initial_logits.topk(2, dim=-1).values
                        initial_aux_diagnostics.update(
                            {
                                f"diagnostic/initial_{name}_logits_std": (
                                    initial_logits.std()
                                ),
                                f"diagnostic/initial_{name}_logits_entropy": (
                                    initial_entropy.mean()
                                ),
                                f"diagnostic/initial_{name}_logits_margin": (
                                    (initial_top_two[:, 0] - initial_top_two[:, 1]).mean()
                                ),
                            }
                        )
                if inverse_records:
                    initial_successor_memories = torch.cat(
                        [record[3] for record in inverse_records], dim=0
                    ).detach().float()
                    initial_aux_diagnostics.update(
                        {
                            "diagnostic/initial_successor_memory_std": (
                                initial_successor_memories.std()
                            ),
                            "diagnostic/initial_successor_memory_norm": (
                                initial_successor_memories.norm(dim=-1).mean()
                            ),
                        }
                    )
                self._initial_aux_diagnostics_pending = False

            self.metrics = {
                "aux/chunk_ce": loss_chunk.detach(),
                "aux/discrete_ce": loss_discrete.detach(),
                "aux/memory_nll": loss_memory.detach(),
                "aux/terminal_nll": loss_terminal.detach(),
                "aux/memory_reconstruction_mse": mean_reconstruction_mse(
                    memory_targets, memory_estimates, logits
                ),
                "aux/terminal_reconstruction_mse": terminal_reconstruction_mse(
                    terminal_memory, self.terminal_target
                ),
                "aux/terminal_chunk": loss_terminal_chunk.detach(),
                "aux/total": aux_loss.detach(),
                "aux/rho_mean": rho.detach().mean(),
                "aux/tau_mean": tau.detach().mean(),
                "aux/memory_batch_variance": (
                    mean_batch_variance(
                        [record[3] for record in inverse_records], batch_axis=0
                    )
                    if inverse_records
                    else logits.detach().new_zeros(())
                ),
                "aux/terminal_batch_variance": mean_batch_variance(
                    [terminal_memory], batch_axis=0
                ),
                **initial_aux_diagnostics,
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
                    mean_batch_variance([record[3]], batch_axis=0)
                )
        else:
            self.metrics = {}
            self.loss_components = {}

        return AuxCausalLMOutput(logits=logits, aux_loss=aux_loss), terminal_memory

    def forward(
        self,
        input_ids,
        targets=None,
        aux_tokens=None,
        state=None,
        compute_aux=True,
        **kwargs,
    ):
        return self._forward_with_offset(
            input_ids,
            targets,
            aux_tokens,
            state,
            compute_aux,
        )

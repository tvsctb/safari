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
    memory_reconstruction_diagnostics,
    mean_batch_variance,
    memory_observation,
    memory_reconstruction_target,
    terminal_gaussian_nll,
    terminal_reconstruction_mse,
    validate_memory_observation_gradient_scale,
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

    def forward(self, x, attention_mask, key_padding_mask=None):
        normalized = self.norm1(x)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(attended)
        return x + self.dropout(self.mlp(self.norm2(x)))


def positioned_memory_and_tokens(
    memory,
    token_embeddings,
    position_embedding,
    num_memory_tokens,
    token_valid=None,
):
    """Add compact per-row token positions while keeping read slots fixed."""

    batch_size, token_length = token_embeddings.shape[:2]
    memory_positions = position_embedding[:num_memory_tokens]
    if token_valid is None:
        token_positions = position_embedding[
            num_memory_tokens : num_memory_tokens + token_length
        ].unsqueeze(0)
    else:
        if token_valid.shape != (batch_size, token_length):
            raise ValueError(
                "token_valid must match the token embedding batch and length"
            )
        compact_token_index = token_valid.long().cumsum(dim=1).sub_(1)
        position_index = compact_token_index.add(num_memory_tokens)
        position_index.clamp_(0, position_embedding.size(0) - 1)
        token_positions = F.embedding(position_index, position_embedding)
    return torch.cat(
        (
            memory + memory_positions.unsqueeze(0),
            token_embeddings + token_positions,
        ),
        dim=1,
    )


class RMTAuxLM(nn.Module):
    """Canonical decoder-only RMT with a fixed inverse auxiliary objective.

    The architecture is deliberately not configurable: boundary reversal,
    unshared copy-initialized inverse blocks/norm/PE, shared token embedding
    and head, and ``memory + fixed write-role query`` are invariant across AR,
    RegBench, and language-model reproduction experiments.  Only objective
    strengths and gradient-flow switches remain experimental options.
    """

    def __init__(
        self,
        d_model,
        n_layer,
        d_inner,
        n_heads,
        vocab_size,
        chunk_size=4,
        num_memory_tokens=2,
        dropout=0.0,
        rho=31.622777,
        tau=11.925695,
        stop_gradient_memory_target=False,
        stop_gradient_memory_observation=False,
        memory_observation_gradient_scale=1.0,
        learnable_terminal_target=True,
        use_chunk_loss=True,
        use_discrete_loss=True,
        use_memory_loss=True,
        use_terminal_loss=True,
        **kwargs,
    ):
        super().__init__()
        if chunk_size <= 1:
            raise ValueError("chunk_size must be greater than 1")
        retired = {
            "chunk_offset",
            "random_chunk_offset",
            "token_scheme",
            "share_inverse",
            "share_inverse_position_embedding",
            "inverse_position_initialization",
            "position_initialization",
            "block_initialization",
            "block_initialization_seed",
            "embedding_initialization",
            "embedding_initialization_std",
            "write_input_mode",
            "share_inverse_embedding",
            "share_inverse_head",
            "use_direction_embedding",
            "memory_scale_mode",
            "memory_scale_granularity",
            "terminal_scale_mode",
            "terminal_scale_granularity",
            "scale_mode",
            "scale_granularity",
            "min_scale",
            "max_scale",
            "observation_noise_std",
            "generation_noise_std",
            "use_terminal_chunk",
            "use_terminal_chunk_loss",
            "memory_min_scale",
            "memory_max_scale",
            "terminal_min_scale",
            "terminal_max_scale",
        }.intersection(kwargs)
        if retired:
            raise TypeError(
                "retired RMT AUX options are fixed by the canonical architecture: "
                + ", ".join(sorted(retired))
            )
        if kwargs:
            raise TypeError(
                "unexpected RMT AUX options: " + ", ".join(sorted(kwargs))
            )
        if num_memory_tokens <= 0:
            raise ValueError("num_memory_tokens must be positive")
        self.d_model = d_model
        self.d_output = vocab_size
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.num_memory_tokens = num_memory_tokens
        if rho <= 0 or tau <= 0:
            raise ValueError("rho and tau must be positive")
        self.stop_gradient_memory_target = stop_gradient_memory_target
        self.stop_gradient_memory_observation = stop_gradient_memory_observation
        self.memory_observation_gradient_scale = validate_memory_observation_gradient_scale(
            memory_observation_gradient_scale
        )
        self.learnable_terminal_target = learnable_terminal_target
        self.use_chunk_loss = use_chunk_loss
        self.use_discrete_loss = use_discrete_loss
        self.use_memory_loss = use_memory_loss
        self.use_terminal_loss = use_terminal_loss
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.inverse_embedding = self.embedding
        self.initial_memory = nn.Parameter(torch.empty(num_memory_tokens, d_model))
        self.forward_queries = nn.Parameter(torch.empty(num_memory_tokens, d_model))
        self.inverse_queries = nn.Parameter(torch.empty(num_memory_tokens, d_model))
        max_token_length = chunk_size
        self.position_embedding = nn.Parameter(
            torch.empty(num_memory_tokens + max_token_length, d_model)
        )
        self.inverse_position_embedding = nn.Parameter(
            torch.empty(num_memory_tokens + max_token_length, d_model)
        )
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(d_model, n_heads, d_inner, dropout)
            for _ in range(n_layer)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.inverse_blocks = copy.deepcopy(self.blocks)
        self.inverse_final_norm = copy.deepcopy(self.final_norm)
        self.register_buffer("rho", torch.tensor(float(rho)))
        self.register_buffer("tau", torch.tensor(float(tau)))
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
        # The fixed write-role query uses the same initialization law as PE.
        nn.init.normal_(self.forward_queries, std=0.02)
        with torch.no_grad():
            self.inverse_queries.copy_(self.forward_queries)
        nn.init.normal_(self.position_embedding, std=0.02)
        # Unshared means independently trainable, not independently seeded.
        self.inverse_blocks.load_state_dict(self.blocks.state_dict())
        self.inverse_final_norm.load_state_dict(self.final_norm.state_dict())
        with torch.no_grad():
            self.inverse_position_embedding.copy_(self.position_embedding)

    def _lm_logits(self, hidden):
        return F.linear(hidden, self.embedding.weight[:self.vocab_size])

    def _inverse_logits(self, hidden):
        return self._lm_logits(hidden)

    def _scale(self, name):
        if name not in {"rho", "tau"}:
            raise ValueError(f"unknown scale: {name}")
        return getattr(self, name)

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
        token_padding_mask=None,
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
        if token_padding_mask is not None and token_padding_mask.shape != (
            batch_size,
            token_length,
        ):
            raise ValueError(
                "token_padding_mask must match the token embedding batch and length"
            )
        x = positioned_memory_and_tokens(
            memory,
            token_embeddings,
            position_embedding,
            self.num_memory_tokens,
            None if token_padding_mask is None else ~token_padding_mask,
        )
        if queries is not None:
            query_batch = memory + queries.unsqueeze(0)
            x = torch.cat((x, query_batch), dim=1)
        key_padding_mask = None
        if token_padding_mask is not None:
            memory_padding = torch.zeros(
                batch_size,
                self.num_memory_tokens,
                dtype=torch.bool,
                device=x.device,
            )
            key_padding_mask = torch.cat(
                (
                    memory_padding,
                    token_padding_mask,
                    memory_padding if queries is not None else memory_padding[:, :0],
                ),
                dim=1,
            )
        sequence_length = x.size(1)
        mask = self.causal_mask[:sequence_length, :sequence_length]
        for block in blocks:
            x = block(x, mask, key_padding_mask)
        x = final_norm(x)
        token_start = self.num_memory_tokens
        token_end = token_start + token_length
        next_memory = None if queries is None else x[:, -self.num_memory_tokens:]
        return x[:, token_start:token_end], next_memory

    def _forward_chunk(self, memory, input_chunk, token_chunk, terminal=False):
        token_embeddings = self.embedding(input_chunk)
        token_hidden, next_memory = self._transform(
            memory,
            token_embeddings,
            None if terminal else self.forward_queries,
            self.blocks,
            self.final_norm,
        )
        return self._lm_logits(token_hidden), next_memory

    def _inverse_record(self, chunks, boundaries, successor_memories):
        data_ids, inverse_targets = boundary_inverse_targets(chunks, boundaries)
        inverse_embeddings = self.inverse_embedding(data_ids)
        observed_memories = memory_observation(
            successor_memories,
            self.stop_gradient_memory_observation,
            self.memory_observation_gradient_scale,
        )
        inverse_hidden, reconstructed_memories = self._transform(
            observed_memories,
            inverse_embeddings,
            self.inverse_queries,
            self.inverse_blocks,
            self.inverse_final_norm,
            self.inverse_position_embedding,
        )
        return self._inverse_logits(inverse_hidden), inverse_targets, reconstructed_memories

    def _padded_inverse_records(self, records):
        """Evaluate variable-length inverse records in one padded transformer call."""
        max_length = max(record[0].size(1) for record in records)
        input_groups = []
        target_groups = []
        padding_groups = []
        has_padding = False
        lengths = []
        successor_memories = []
        positions = torch.arange(max_length, device=records[0][0].device)
        for chunks, boundaries, _, successor in records:
            data_ids, inverse_targets = boundary_inverse_targets(
                chunks, boundaries
            )
            padding = max_length - chunks.size(1)
            if padding:
                has_padding = True
                data_ids = F.pad(data_ids, (0, padding), value=0)
                inverse_targets = F.pad(
                    inverse_targets, (0, padding), value=-100
                )
            padding_groups.append(
                positions.unsqueeze(0).expand(chunks.size(0), -1)
                >= chunks.size(1)
            )
            input_groups.append(data_ids)
            target_groups.append(inverse_targets)
            lengths.extend([chunks.size(1)] * chunks.size(0))
            successor_memories.append(successor)

        data_ids = torch.cat(input_groups, dim=0)
        inverse_targets = torch.cat(target_groups, dim=0)
        token_padding_mask = (
            torch.cat(padding_groups, dim=0) if has_padding else None
        )
        observed_memories = memory_observation(
            torch.cat(successor_memories, dim=0),
            self.stop_gradient_memory_observation,
            self.memory_observation_gradient_scale,
        )
        inverse_embeddings = self.inverse_embedding(data_ids)
        inverse_hidden, reconstructed_memories = self._transform(
            observed_memories,
            inverse_embeddings,
            self.inverse_queries,
            self.inverse_blocks,
            self.inverse_final_norm,
            self.inverse_position_embedding,
            token_padding_mask,
        )
        return (
            self._inverse_logits(inverse_hidden),
            inverse_targets,
            reconstructed_memories,
            torch.tensor(lengths, device=data_ids.device),
        )

    def _forward_with_offset(
        self,
        input_ids,
        targets,
        aux_tokens,
        state,
        compute_aux,
        compute_diagnostics,
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

        for start, end in ranges:
            memory_before = memory
            token_chunk = None if token_stream is None else token_stream[:, start:end]
            logits, memory = self._forward_chunk(
                memory,
                input_ids[:, start:end],
                token_chunk,
                terminal=False,
            )
            forward_logits.append(logits)
            if compute_aux:
                boundary = input_ids[:, start]
                inverse_records.append(
                    (token_chunk, boundary, memory_before, memory)
                )

        terminal_memory = memory
        logits = torch.cat(forward_logits, dim=1)
        aux_loss = logits.new_zeros(())

        if compute_aux:
            if inverse_records:
                (
                    inverse_logits,
                    inverse_targets,
                    reconstructed_memories,
                    lengths,
                ) = self._padded_inverse_records(inverse_records)
                positions = torch.arange(
                    inverse_logits.size(1), device=inverse_logits.device
                ).unsqueeze(0)
                chunk_mask = positions < (lengths - 1).unsqueeze(1)
                if chunk_mask.any():
                    chunk_logits.append(inverse_logits[chunk_mask])
                    chunk_targets.append(inverse_targets[chunk_mask])
                rows = torch.arange(lengths.numel(), device=lengths.device)
                discrete_logits.append(
                    inverse_logits[rows, lengths - 1].unsqueeze(1)
                )
                discrete_targets.append(
                    inverse_targets[rows, lengths - 1].unsqueeze(1)
                )
                memory_targets.extend([
                    memory_reconstruction_target(
                        record[2], self.stop_gradient_memory_target
                    )
                    for record in inverse_records
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
            aux_loss = (
                loss_chunk
                + loss_discrete
                + loss_memory
                + loss_terminal
            )
            self.loss_components = {
                "chunk_ce": loss_chunk,
                "discrete_ce": loss_discrete,
                "memory_nll": loss_memory,
                "terminal_nll": loss_terminal,
                "total": aux_loss,
            }
            initial_aux_diagnostics = {}
            if compute_diagnostics and self._initial_aux_diagnostics_pending:
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
                "aux/total": aux_loss.detach(),
                "aux/rho_mean": rho.detach().mean(),
                "aux/tau_mean": tau.detach().mean(),
                **initial_aux_diagnostics,
            }
            if compute_diagnostics:
                memory_diagnostics = memory_reconstruction_diagnostics(
                    memory_targets, memory_estimates, logits
                )
                self.metrics.update(
                    {
                        "aux/memory_reconstruction_mse": memory_diagnostics[
                            "residual_mse"
                        ],
                        **{
                            f"aux/memory_{name}": value
                            for name, value in memory_diagnostics.items()
                        },
                        "aux/terminal_reconstruction_mse": (
                            terminal_reconstruction_mse(
                                terminal_memory, self.terminal_target
                            )
                        ),
                        "aux/memory_batch_variance": (
                            mean_batch_variance(
                                [record[3] for record in inverse_records],
                                batch_axis=0,
                            )
                            if inverse_records
                            else logits.detach().new_zeros(())
                        ),
                        "aux/terminal_batch_variance": mean_batch_variance(
                            [terminal_memory], batch_axis=0
                        ),
                    }
                )
            if rho.numel() == 1:
                self.metrics["aux/rho"] = rho.detach().reshape(())
            if tau.numel() == 1:
                self.metrics["aux/tau"] = tau.detach().reshape(())
            for index, value in enumerate(rho.detach().reshape(-1)):
                self.metrics[f"aux/rho/{index}"] = value
            for index, value in enumerate(tau.detach().reshape(-1)):
                self.metrics[f"aux/tau/{index}"] = value
            if compute_diagnostics:
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
        compute_diagnostics=True,
        **kwargs,
    ):
        return self._forward_with_offset(
            input_ids,
            targets,
            aux_tokens,
            state,
            compute_aux,
            compute_diagnostics,
        )

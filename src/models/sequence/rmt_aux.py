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
    normalize_token_scheme,
    resolve_direction_embedding,
    role_inverse_targets,
    terminal_gaussian_nll,
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
        share_inverse_embedding=True,
        share_inverse_head=True,
        use_direction_embedding=False,
        scale_mode="fixed",
        scale_granularity="global",
        rho=1.0,
        tau=1.0,
        min_scale=1e-4,
        max_scale=1e4,
        **kwargs,
    ):
        super().__init__()
        if chunk_size <= 1:
            raise ValueError("chunk_size must be greater than 1")
        if "chunk_offset" in kwargs:
            raise ValueError("chunk_offset is not configurable; fixed offset is always 0")
        if "random_chunk_offset" in kwargs:
            raise ValueError("RMT chunk offset is fixed at 0 and is not configurable")
        if num_memory_tokens <= 0:
            raise ValueError("num_memory_tokens must be positive")
        self.d_model = d_model
        self.d_output = vocab_size
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.num_memory_tokens = num_memory_tokens
        self.token_scheme = normalize_token_scheme(token_scheme)
        self.share_inverse = share_inverse
        self.share_inverse_embedding = share_inverse_embedding
        self.share_inverse_head = share_inverse_head
        self.use_direction_embedding = resolve_direction_embedding(
            use_direction_embedding,
            self.token_scheme,
        )
        self.scale_mode = scale_mode
        self.scale_granularity = scale_granularity
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.forward_token_id = vocab_size
        self.inverse_token_id = vocab_size + 1

        validate_scale_configuration(
            scale_mode, scale_granularity, {"global", "slotwise"}
        )
        scale_size = num_memory_tokens if scale_granularity == "slotwise" else 1

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
            torch.empty(2 * num_memory_tokens + max_token_length, d_model)
        )
        self.inverse_position_embedding = (
            None
            if share_inverse
            else nn.Parameter(
                torch.empty(2 * num_memory_tokens + max_token_length, d_model)
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
            self, "rho", rho, scale_size, scale_mode, scale_granularity
        )
        initialize_scale(
            self, "tau", tau, scale_size, scale_mode, scale_granularity
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

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.initial_memory, std=0.02)
        nn.init.normal_(self.forward_queries, std=0.02)
        nn.init.normal_(self.inverse_queries, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)
        with torch.no_grad():
            if self.inverse_embedding is not self.embedding:
                self.inverse_embedding.weight.copy_(self.embedding.weight)
            if hasattr(self, "inverse_head"):
                self.inverse_head.weight.copy_(
                    self.embedding.weight[:self.vocab_size]
                )
            if self.inverse_position_embedding is not None:
                self.inverse_position_embedding.copy_(self.position_embedding)

    def _lm_logits(self, hidden):
        return F.linear(hidden, self.embedding.weight[:self.vocab_size])

    def _inverse_logits(self, hidden):
        if not hasattr(self, "inverse_head"):
            return self._lm_logits(hidden)
        return self.inverse_head(hidden)

    def _scale(self, name):
        return configured_scale(
            self, name, self.scale_mode, self.min_scale, self.max_scale
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
        if queries is None:
            x = torch.cat((memory, token_embeddings), dim=1)
        else:
            query_batch = queries.unsqueeze(0).expand(batch_size, -1, -1)
            x = torch.cat((memory, token_embeddings, query_batch), dim=1)
        sequence_length = x.size(1)
        position_embedding = (
            self.position_embedding
            if position_embedding is None
            else position_embedding
        )
        x = x + position_embedding[:sequence_length]
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
        inverse_hidden, reconstructed_memories = self._transform(
            successor_memories,
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

        for start, end in ranges[:-1]:
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
                boundary = (
                    input_ids[:, start]
                    if self.token_scheme == "boundary_reverse"
                    else None
                )
                inverse_records.append(
                    (token_chunk, boundary, memory_before, memory)
                )

        terminal_memory = memory
        terminal_start, terminal_end = ranges[-1]
        final_chunk_tokens = (
            None if token_stream is None else token_stream[:, terminal_start:terminal_end]
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
                memory_targets.extend([record[2] for record in records])
                memory_estimates.extend(
                    reconstructed_memories.split(batch_size, dim=0)
                )

            rho = self._scale("rho")
            tau = self._scale("tau")
            sequence_normalizer = float(length)
            loss_chunk = cross_entropy_sum(
                chunk_logits, chunk_targets, logits, batch_size
            ) / sequence_normalizer
            loss_discrete = cross_entropy_sum(
                discrete_logits, discrete_targets, logits, batch_size
            ) / sequence_normalizer
            loss_memory = gaussian_nll_sum(
                memory_targets,
                memory_estimates,
                rho,
                logits,
                batch_size,
                scale_axis=1,
            ) / sequence_normalizer
            loss_terminal = terminal_gaussian_nll(
                terminal_memory, tau, batch_size, scale_axis=1
            ) / sequence_normalizer
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
            self.metrics = {
                "aux/chunk_ce": loss_chunk.detach(),
                "aux/discrete_ce": loss_discrete.detach(),
                "aux/memory_nll": loss_memory.detach(),
                "aux/terminal_nll": loss_terminal.detach(),
                "aux/terminal_chunk": loss_terminal_chunk.detach(),
                "aux/total": aux_loss.detach(),
                "aux/rho_mean": rho.detach().mean(),
                "aux/tau_mean": tau.detach().mean(),
            }
            if rho.numel() == 1:
                self.metrics["aux/rho"] = rho.detach().reshape(())
            if tau.numel() == 1:
                self.metrics["aux/tau"] = tau.detach().reshape(())
            for index, value in enumerate(rho.detach().reshape(-1)):
                self.metrics[f"aux/rho/{index}"] = value
            for index, value in enumerate(tau.detach().reshape(-1)):
                self.metrics[f"aux/tau/{index}"] = value
        else:
            self.metrics = {}

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

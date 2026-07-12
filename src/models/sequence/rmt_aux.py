import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.sequence.auxiliary import (
    AuxCausalLMOutput,
    boundary_inverse_targets,
    bounded_exp,
    chunk_ranges,
    cross_entropy_sum,
    gaussian_nll_sum,
    terminal_gaussian_nll,
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
    """Decoder-only recurrent-memory Transformer with inverse reconstruction."""

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
        random_chunk_offset=False,
        share_inverse=True,
        min_scale=1e-4,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_output = vocab_size
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.num_memory_tokens = num_memory_tokens
        self.random_chunk_offset = random_chunk_offset
        self.share_inverse = share_inverse
        self.min_scale = min_scale

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.initial_memory = nn.Parameter(torch.empty(num_memory_tokens, d_model))
        self.forward_queries = nn.Parameter(torch.empty(num_memory_tokens, d_model))
        self.inverse_queries = nn.Parameter(torch.empty(num_memory_tokens, d_model))
        self.direction_embedding = (
            nn.Parameter(torch.zeros(d_model)) if share_inverse else None
        )
        self.position_embedding = nn.Parameter(
            torch.empty(2 * num_memory_tokens + chunk_size, d_model)
        )
        self.inverse_position_embedding = (
            None
            if share_inverse
            else nn.Parameter(
                torch.empty(2 * num_memory_tokens + chunk_size, d_model)
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

        self.log_rho = nn.Parameter(torch.zeros(()))
        self.log_tau = nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(
                    2 * num_memory_tokens + chunk_size,
                    2 * num_memory_tokens + chunk_size,
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
            if self.inverse_position_embedding is not None:
                self.inverse_position_embedding.copy_(self.position_embedding)

    def _lm_logits(self, hidden):
        return F.linear(hidden, self.embedding.weight)

    def _offset(self, device, compute_aux):
        if compute_aux and self.training and self.random_chunk_offset:
            return int(torch.randint(self.chunk_size, (1,), device=device).item())
        return 0

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
        if token_length > self.chunk_size:
            raise ValueError("RMT token input exceeds chunk_size")
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

    def forward(self, input_ids, targets=None, state=None, compute_aux=True, **kwargs):
        batch_size, length = input_ids.shape
        memory = self.default_state(batch_size, device=input_ids.device) if state is None else state
        ranges = chunk_ranges(length, self.chunk_size, self._offset(input_ids.device, compute_aux))

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
            token_hidden, memory = self._transform(
                memory,
                self.embedding(input_ids[:, start:end]),
                self.forward_queries,
                self.blocks,
                self.final_norm,
            )
            logits = self._lm_logits(token_hidden)
            forward_logits.append(logits)

            if compute_aux:
                if targets is None:
                    raise ValueError("targets are required when compute_aux=True")
                chunk = targets[:, start:end]
                if torch.any(chunk < 0):
                    raise ValueError("inverse targets must contain unmasked token IDs during training")
                inverse_records.append(
                    (chunk, input_ids[:, start], memory_before, memory)
                )

        terminal_memory = memory
        terminal_start, terminal_end = ranges[-1]
        terminal_hidden, no_successor_memory = self._transform(
            terminal_memory,
            self.embedding(input_ids[:, terminal_start:terminal_end]),
            None,
            self.blocks,
            self.final_norm,
        )
        if no_successor_memory is not None:
            raise RuntimeError("terminal RMT call must not produce successor memory")
        final_chunk_logits = self._lm_logits(terminal_hidden)
        forward_logits.append(final_chunk_logits)
        final_chunk_targets = None
        if compute_aux:
            if targets is None:
                raise ValueError("targets are required when compute_aux=True")
            final_chunk_targets = targets[:, terminal_start:terminal_end]
            if torch.any(final_chunk_targets < 0):
                raise ValueError("terminal targets must contain unmasked token IDs during training")

        logits = torch.cat(forward_logits, dim=1)
        aux_loss = logits.new_zeros(())

        if compute_aux:
            lengths = sorted({record[0].size(1) for record in inverse_records})
            for inverse_length in lengths:
                records = [
                    record for record in inverse_records
                    if record[0].size(1) == inverse_length
                ]
                chunks = torch.cat([record[0] for record in records], dim=0)
                boundaries = torch.cat([record[1] for record in records], dim=0)
                initial_memories = torch.cat([record[3] for record in records], dim=0)
                data_ids, inverse_targets = boundary_inverse_targets(chunks, boundaries)
                inverse_data_embeddings = self.embedding(data_ids)
                if self.direction_embedding is not None:
                    inverse_data_embeddings = inverse_data_embeddings + self.direction_embedding
                inverse_hidden, reconstructed_memories = self._transform(
                    initial_memories,
                    inverse_data_embeddings,
                    self.inverse_queries,
                    self.inverse_blocks,
                    self.inverse_final_norm,
                    self.inverse_position_embedding,
                )
                inverse_logits = self._lm_logits(inverse_hidden)

                if inverse_length > 1:
                    chunk_logits.append(inverse_logits[:, :-1])
                    chunk_targets.append(inverse_targets[:, :-1])
                discrete_logits.append(inverse_logits[:, -1:])
                discrete_targets.append(inverse_targets[:, -1:])
                memory_targets.extend([record[2] for record in records])
                memory_estimates.extend(
                    reconstructed_memories.split(batch_size, dim=0)
                )

            rho = bounded_exp(self.log_rho, minimum=self.min_scale)
            tau = bounded_exp(self.log_tau, minimum=self.min_scale)
            sequence_normalizer = float(length)
            loss_chunk = cross_entropy_sum(
                chunk_logits, chunk_targets, logits, batch_size
            ) / sequence_normalizer
            loss_discrete = cross_entropy_sum(
                discrete_logits, discrete_targets, logits, batch_size
            ) / sequence_normalizer
            loss_memory = gaussian_nll_sum(
                memory_targets, memory_estimates, rho, logits, batch_size
            ) / sequence_normalizer
            loss_terminal = terminal_gaussian_nll(
                terminal_memory, tau, batch_size
            ) / sequence_normalizer
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
            self.metrics = {
                "aux/chunk_ce": loss_chunk.detach(),
                "aux/discrete_ce": loss_discrete.detach(),
                "aux/memory_nll": loss_memory.detach(),
                "aux/terminal_nll": loss_terminal.detach(),
                "aux/terminal_chunk": loss_terminal_chunk.detach(),
                "aux/total": aux_loss.detach(),
                "aux/rho": rho.detach(),
                "aux/tau": tau.detach(),
            }
        else:
            self.metrics = {}

        return AuxCausalLMOutput(logits=logits, aux_loss=aux_loss), terminal_memory

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.sequence.auxiliary import (
    AuxCausalLMOutput,
    boundary_inverse_batch,
    bounded_exp,
    chunk_ranges,
    cross_entropy_sum,
    gaussian_nll_sum,
    terminal_gaussian_nll,
)


class GRUAuxLM(nn.Module):
    """Stacked GRU language model with the boundary-token inverse auxiliary."""

    def __init__(
        self,
        d_model,
        n_layer,
        vocab_size,
        chunk_size=4,
        dropout=0.0,
        random_chunk_offset=True,
        share_inverse=True,
        min_scale=1e-4,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_output = vocab_size
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.random_chunk_offset = random_chunk_offset
        self.share_inverse = share_inverse
        self.min_scale = min_scale
        self.memory_token_id = vocab_size

        self.embedding = nn.Embedding(vocab_size + 1, d_model)
        self.gru = nn.GRU(
            d_model,
            d_model,
            num_layers=n_layer,
            dropout=dropout if n_layer > 1 else 0.0,
            batch_first=True,
        )
        self.inverse_gru = self.gru if share_inverse else copy.deepcopy(self.gru)
        self.direction_embedding = (
            nn.Parameter(torch.zeros(d_model)) if share_inverse else None
        )
        self.log_rho = nn.Parameter(torch.zeros(()))
        self.log_tau = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.embedding.weight, std=0.02)

        self.metrics = {}

    def _lm_logits(self, hidden):
        return F.linear(hidden, self.embedding.weight[:self.vocab_size])

    def _offset(self, device, compute_aux):
        if compute_aux and self.training and self.random_chunk_offset:
            return int(torch.randint(self.chunk_size, (1,), device=device).item())
        return 0

    def default_state(self, *batch_shape, device=None):
        if len(batch_shape) != 1:
            raise ValueError("GRUAuxLM expects one batch dimension")
        return torch.zeros(self.n_layer, batch_shape[0], self.d_model, device=device)

    def forward(self, input_ids, targets=None, state=None, compute_aux=True, **kwargs):
        batch_size, length = input_ids.shape
        hidden = self.default_state(batch_size, device=input_ids.device) if state is None else state
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
            state_before = hidden
            token_embeddings = self.embedding(input_ids[:, start:end])
            outputs, hidden = self.gru(token_embeddings, hidden)
            logits = self._lm_logits(outputs)
            forward_logits.append(logits)

            if compute_aux:
                if targets is None:
                    raise ValueError("targets are required when compute_aux=True")
                chunk = targets[:, start:end]
                if torch.any(chunk < 0):
                    raise ValueError("inverse targets must contain unmasked token IDs during training")
                inverse_records.append(
                    (chunk, input_ids[:, start], state_before, hidden)
                )

        terminal_memory = hidden
        terminal_start, terminal_end = ranges[-1]
        terminal_outputs, _ = self.gru(
            self.embedding(input_ids[:, terminal_start:terminal_end]),
            terminal_memory,
        )
        final_chunk_logits = self._lm_logits(terminal_outputs)
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
            memory_embedding = self.embedding.weight[self.memory_token_id][None, None, :]
            lengths = sorted({record[0].size(1) for record in inverse_records})
            for inverse_length in lengths:
                records = [
                    record for record in inverse_records
                    if record[0].size(1) == inverse_length
                ]
                chunks = torch.cat([record[0] for record in records], dim=0)
                boundaries = torch.cat([record[1] for record in records], dim=0)
                initial_states = torch.cat([record[3] for record in records], dim=1)
                data_ids, memory_inputs, inverse_targets = boundary_inverse_batch(
                    chunks, boundaries, memory_embedding
                )
                inverse_data_embeddings = self.embedding(data_ids)
                if self.direction_embedding is not None:
                    inverse_data_embeddings = inverse_data_embeddings + self.direction_embedding
                inverse_inputs = torch.cat(
                    (inverse_data_embeddings, memory_inputs), dim=1
                )
                inverse_outputs, reconstructed_states = self.inverse_gru(
                    inverse_inputs, initial_states
                )
                inverse_logits = self._lm_logits(
                    inverse_outputs[:, :inverse_length]
                )

                if inverse_length > 1:
                    chunk_logits.append(inverse_logits[:, :-1])
                    chunk_targets.append(inverse_targets[:, :-1])
                discrete_logits.append(inverse_logits[:, -1:])
                discrete_targets.append(inverse_targets[:, -1:])
                memory_targets.extend([record[2] for record in records])
                memory_estimates.extend(
                    reconstructed_states.split(batch_size, dim=1)
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

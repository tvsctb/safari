from typing import Optional, List, Tuple
from collections.abc import Mapping
import math
import functools
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import ListConfig
from src.models.nn.components import ReversibleInstanceNorm1dInput, ReversibleInstanceNorm1dOutput, \
    TSNormalization, TSInverseNormalization

from src.models.nn.adaptive_softmax import AdaptiveEmbedding, ProjectedAdaptiveLogSoftmax
import src.tasks.metrics as M
from src.tasks.torchmetrics import torchmetric_fns as tm_mine
import src.models.nn.utils as U
import torchmetrics as tm
from src.utils.config import to_list, instantiate
from torchmetrics import MetricCollection

class BaseTask:
    """ Abstract class that takes care of:
    - loss function
    - arbitrary metrics
    - forward pass
    - (optional) encoder module that interfaces with dataset (inputs) and model
    - (optional) decoder module that interfaces with dataset (targets) and model
    """
    encoder = None
    decoder = None

    def __init__(self, dataset=None, model=None, loss=None, loss_val=None, metrics=None, torchmetrics=None):
        """ This class is allowed to grab attributes directly off a constructed dataset and model object """
        self.dataset = dataset
        self.model = model
        if metrics is None: metrics = []
        self.metric_names = to_list(metrics)

        if torchmetrics is None: torchmetrics = []
        self.torchmetric_names = to_list(torchmetrics)
        self._tracked_torchmetrics = {}

        # The decoder might pass through arguments that the loss needs (e.g. sequence lengths)
        # but might also pass through extraneous arguments (e.g. sampling rate)
        # Wrap loss and metrics so that they accept kwargs and

        # Create loss function
        self.loss = instantiate(M.output_metric_fns, loss, partial=True)
        self.loss = U.discard_kwargs(self.loss)
        if loss_val is not None:
            self.loss_val = instantiate(M.output_metric_fns, loss_val, partial=True)
            self.loss_val = U.discard_kwargs(self.loss_val)
        torchmetrics = MetricCollection(self._init_torchmetrics())
        self.train_torchmetrics = torchmetrics.clone(prefix='train/')
        self.val_torchmetrics = torchmetrics.clone(prefix='val/')
        self.test_torchmetrics = torchmetrics.clone(prefix='test/')

    def _init_torchmetrics(self):
        """
        Instantiate torchmetrics.
        """
        tracked_torchmetrics = {}

        for name in self.torchmetric_names:
            if name in tm_mine:
                tracked_torchmetrics[name] = tm_mine[name]().to('cuda')
            elif name in ['AUROC', 'StatScores', 'Precision', 'Recall', 'F1', 'F1Score']:
                tracked_torchmetrics[name] = getattr(tm, name)(average='macro', num_classes=self.dataset.d_output, compute_on_step=False).to('cuda')
            elif '@' in name:
                k = int(name.split('@')[1])
                mname = name.split('@')[0]
                tracked_torchmetrics[name] = getattr(tm, mname)(average='macro', num_classes=self.dataset.d_output, compute_on_step=False, top_k=k).to('cuda')
            else:
                tracked_torchmetrics[name] = getattr(tm, name)(compute_on_step=False).to('cuda')
        
        return tracked_torchmetrics

    def _reset_torchmetrics(self, prefix=None):
        """
        Reset torchmetrics for a prefix
        associated with a particular dataloader (e.g. train, val, test).

        Generally do this at the start of an epoch.
        """
        all_prefixes = [prefix] if prefix is not None else self._tracked_torchmetrics

        for prefix in all_prefixes:
            if prefix in self._tracked_torchmetrics:
                self._tracked_torchmetrics[prefix].reset()

    def get_torchmetrics(self, prefix):
        """
        Compute torchmetrics for a prefix associated with
        a particular dataloader (e.g. train, val, test).

        Generally do this at the end of an epoch.
        """
        return {name: self._tracked_torchmetrics[prefix][name].compute() for name in self.torchmetric_names}

    def torchmetrics(self, x, y, prefix, loss=None):
        """
        Update torchmetrics with new x, y .
        Prefix corresponds to a particular dataloader (e.g. train, val, test).

        Generally call this every batch.
        """
        if prefix not in self._tracked_torchmetrics:
            self._init_torchmetrics(prefix)
        self._tracked_torchmetrics[prefix](x, y, loss=loss)

        # for name in self.torchmetric_names:
        #     if name.startswith('Accuracy'):
        #         if len(x.shape) > 2:
        #             # Multi-dimensional, multi-class
        #             self._tracked_torchmetrics[prefix][name].update(x.transpose(1, 2), y.squeeze())
        #             continue
        #     self._tracked_torchmetrics[prefix][name].update(x, y)

    def get_torchmetrics(self, prefix):
        return self._tracked_torchmetrics[prefix]

    def metrics(self, x, y, **kwargs):
        """
        Metrics are just functions
        output metrics are a function of output and target
        loss metrics are a function of loss (e.g. perplexity)
        """
        output_metrics = {
            name: U.discard_kwargs(M.output_metric_fns[name])(x, y, **kwargs)
            for name in self.metric_names if name in M.output_metric_fns
        }
        loss_metrics = {
            name: U.discard_kwargs(M.loss_metric_fns[name])(x, y, self.loss, **kwargs)
            for name in self.metric_names if name in M.loss_metric_fns
        }
        return {**output_metrics, **loss_metrics}

    def forward(self, batch, encoder, model, decoder, _state):
        """Passes a batch through the encoder, backbone, and decoder"""
        # z holds arguments such as sequence length
        x, y, *z = batch # z holds extra dataloader info such as resolution
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]

        x, w = encoder(x, **z) # w can model-specific constructions such as key_padding_mask for transformers or state for RNNs
        x, state = model(x, **w, state=_state)
        self._state = state
        x, w = decoder(x, state=state, **z)
        return x, y, w


class Scalar(nn.Module):
    def __init__(self, c=1):
        super().__init__()
        self.c = c
    def forward(self, x):
        return x * self.c

class LMTask(BaseTask):
    def forward(self, batch, encoder, model, decoder, _state):
        """Passes a batch through the encoder, backbone, and decoder"""
        # z holds arguments such as sequence length
        x, y, *z = batch # z holds extra dataloader info such as resolution
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]
        x, w = encoder(x, **z) # w can model-specific constructions such as key_padding_mask for transformers or state for RNNs
        x, state = model(x, **w, state=_state)
        self._state = state
        x, w = decoder(x, state=state, **z)

        x = x.logits
        x = rearrange(x, '... C -> (...) C')
        y = rearrange(y, '... -> (...)')

        return x, y, w


def auxiliary_gradient_norm_metrics(
    model,
    lm_loss,
    loss_components,
    aux_weight,
):
    """Measure actual weighted loss gradients without modifying parameter grads."""
    named_parameters = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    parameter_names = tuple(name for name, _ in named_parameters)
    parameters = tuple(parameter for _, parameter in named_parameters)
    if not parameters:
        return {}

    def gradients(loss):
        if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
            return (None,) * len(parameters)
        return torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

    def norm(values, mask=None):
        squares = [
            value.detach().float().pow(2).sum()
            for index, value in enumerate(values)
            if value is not None and (mask is None or mask[index])
        ]
        if not squares:
            return lm_loss.detach().new_zeros(())
        return torch.stack(squares).sum().sqrt()

    def cosine(left, right, mask=None):
        indices = [
            index
            for index in range(len(left))
            if mask is None or mask[index]
        ]
        left_values = [left[index] for index in indices if left[index] is not None]
        right_values = [right[index] for index in indices if right[index] is not None]
        pairs = [
            (left[index], right[index])
            for index in indices
            if left[index] is not None and right[index] is not None
        ]
        if not left_values or not right_values:
            return lm_loss.detach().new_zeros(())
        dot = (
            torch.stack([
            left_value.detach().float().mul(right_value.detach().float()).sum()
            for left_value, right_value in pairs
            ]).sum()
            if pairs
            else lm_loss.detach().new_zeros(())
        )
        left_norm = torch.stack([
            left_value.detach().float().pow(2).sum()
            for left_value in left_values
        ]).sum().sqrt()
        right_norm = torch.stack([
            right_value.detach().float().pow(2).sum()
            for right_value in right_values
        ]).sum().sqrt()
        denominator = left_norm * right_norm
        if denominator.item() == 0.0:
            return dot.new_zeros(())
        return dot / denominator

    lm_gradients = gradients(lm_loss)
    forward_mask = tuple(value is not None for value in lm_gradients)
    block_masks = {}
    for index, name in enumerate(parameter_names):
        leaf_name = name.rsplit(".", 1)[-1]
        if leaf_name in {"rho", "tau", "raw_rho", "raw_tau"}:
            block = "scale"
        elif "terminal_target" in name:
            block = "terminal_target"
        elif any(
            token in name
            for token in ("initial_memory", "forward_queries", "inverse_queries")
        ):
            block = "memory_state"
        elif any(
            token in name
            for token in ("embedding", "position_embedding", "direction_embedding")
        ):
            block = "embedding"
        elif any(token in name for token in ("blocks", "final_norm", "gru")):
            block = "backbone"
        elif "head" in name:
            block = "head"
        else:
            block = "other"
        block_masks.setdefault(block, [False] * len(parameters))[index] = True
    block_masks = {
        name: tuple(mask)
        for name, mask in block_masks.items()
    }
    metrics = {
        "grad_norm/all/lm": norm(lm_gradients),
        "grad_norm/forward/lm": norm(lm_gradients, forward_mask),
    }
    for block, mask in block_masks.items():
        metrics[f"grad_norm/block/{block}/lm"] = norm(lm_gradients, mask)
    component_gradients = {}
    for name, component in loss_components.items():
        values = gradients(aux_weight * component)
        component_gradients[name] = values
        metrics[f"grad_norm/all/aux/{name}"] = norm(values)
        metrics[f"grad_norm/forward/aux/{name}"] = norm(
            values, forward_mask
        )
        metrics[f"grad_cosine/all/lm_aux/{name}"] = cosine(
            lm_gradients, values
        )
        metrics[f"grad_cosine/forward/lm_aux/{name}"] = cosine(
            lm_gradients, values, forward_mask
        )
        for block, mask in block_masks.items():
            metrics[f"grad_norm/block/{block}/aux/{name}"] = norm(values, mask)
            metrics[f"grad_cosine/block/{block}/lm_aux/{name}"] = cosine(
                lm_gradients, values, mask
            )
    component_names = tuple(component_gradients)
    for left_index, left_name in enumerate(component_names):
        for right_name in component_names[left_index + 1:]:
            pair_name = f"{left_name}__{right_name}"
            metrics[f"grad_cosine/all/aux_aux/{pair_name}"] = cosine(
                component_gradients[left_name], component_gradients[right_name]
            )
            metrics[f"grad_cosine/forward/aux_aux/{pair_name}"] = cosine(
                component_gradients[left_name],
                component_gradients[right_name],
                forward_mask,
            )
    return metrics


def scheduled_aux_weight(initial, final, schedule, step, start_step, end_step):
    """Evaluate a fixed, linear, or cosine auxiliary-weight schedule."""
    if schedule not in {"fixed", "linear", "cosine"}:
        raise ValueError("aux_weight_schedule must be fixed, linear, or cosine")
    values = (initial, final)
    if any(isinstance(value, bool) or not math.isfinite(float(value)) or value < 0 for value in values):
        raise ValueError("auxiliary weights must be finite and non-negative")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("auxiliary schedule step must be a non-negative integer")
    if schedule == "fixed":
        return float(initial)
    if (
        isinstance(start_step, bool)
        or isinstance(end_step, bool)
        or not isinstance(start_step, int)
        or not isinstance(end_step, int)
        or start_step < 0
        or end_step <= start_step
    ):
        raise ValueError("auxiliary decay steps must satisfy 0 <= start < end")
    if step <= start_step:
        return float(initial)
    if step >= end_step:
        return float(final)
    progress = (step - start_step) / (end_step - start_step)
    if schedule == "cosine":
        progress = 0.5 * (1.0 - math.cos(math.pi * progress))
    return float(initial) + (float(final) - float(initial)) * progress


_AUX_COMPONENT_NAMES = (
    "chunk_ce",
    "discrete_ce",
    "memory_nll",
    "terminal_nll",
    "terminal_chunk",
)


def normalize_aux_component_weights(weights, label):
    """Validate sparse per-component multipliers, defaulting missing entries to one."""
    if weights is None:
        weights = {}
    if not isinstance(weights, Mapping):
        raise ValueError(f"{label} must be a mapping")
    unknown = set(weights) - set(_AUX_COMPONENT_NAMES)
    if unknown:
        raise ValueError(
            f"{label} contains unknown components: {', '.join(sorted(unknown))}"
        )
    normalized = {}
    for name in _AUX_COMPONENT_NAMES:
        value = weights.get(name, 1.0)
        if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{label}.{name} must be finite and non-negative")
        normalized[name] = float(value)
    return normalized


def scheduled_aux_component_weights(
    initial,
    final,
    schedule,
    step,
    start_step,
    end_step,
):
    """Evaluate component multipliers on the task's auxiliary schedule."""
    return {
        name: scheduled_aux_weight(
            initial[name], final[name], schedule, step, start_step, end_step
        )
        for name in _AUX_COMPONENT_NAMES
    }


def weighted_aux_components(loss_components, component_weights):
    """Apply task-level component multipliers and rebuild the exact total."""
    weighted = {
        name: loss_components[name] * component_weights[name]
        for name in _AUX_COMPONENT_NAMES
        if name in loss_components
    }
    if not weighted:
        return {}, loss_components.get("total")
    total = sum(weighted.values())
    weighted["total"] = total
    return weighted, total


class AuxLMTask(LMTask):
    """Language modeling task that adds a model-provided inverse auxiliary."""

    def __init__(
        self,
        aux_weight=1.0,
        aux_gradient_norm_interval=0,
        aux_weight_schedule="fixed",
        aux_weight_final=0.1,
        aux_weight_decay_start_step=0,
        aux_weight_decay_end_step=1,
        aux_component_weights=None,
        aux_component_weights_final=None,
        **kwargs,
    ):
        if (
            isinstance(aux_gradient_norm_interval, bool)
            or not isinstance(aux_gradient_norm_interval, int)
            or aux_gradient_norm_interval < 0
        ):
            raise ValueError("aux_gradient_norm_interval must be a non-negative integer")
        scheduled_aux_weight(
            aux_weight,
            aux_weight_final,
            aux_weight_schedule,
            0,
            aux_weight_decay_start_step,
            aux_weight_decay_end_step,
        )
        self.aux_weight = float(aux_weight)
        self.aux_weight_schedule = aux_weight_schedule
        self.aux_weight_final = float(aux_weight_final)
        self.aux_weight_decay_start_step = aux_weight_decay_start_step
        self.aux_weight_decay_end_step = aux_weight_decay_end_step
        self.aux_component_weights = normalize_aux_component_weights(
            aux_component_weights, "aux_component_weights"
        )
        self.aux_component_weights_final = normalize_aux_component_weights(
            aux_component_weights_final, "aux_component_weights_final"
        )
        self._aux_schedule_step = 0
        self._current_aux_weight = self.aux_weight
        self._current_aux_component_weights = dict(self.aux_component_weights)
        self.aux_gradient_norm_interval = aux_gradient_norm_interval
        self._aux_gradient_norm_step = 0
        super().__init__(**kwargs)
        self.lm_loss = self.loss
        self.loss = self._training_loss
        self.loss_val = self.lm_loss

    def _training_loss(self, logits, targets, aux_loss=None, **kwargs):
        loss = self.lm_loss(logits, targets)
        if aux_loss is not None:
            loss = loss + self._current_aux_weight * aux_loss
        return loss

    def forward(self, batch, encoder, model, decoder, _state):
        x, y, *z = batch
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict)
            z = z[0]

        aux_tokens = z.pop("aux_tokens", None)
        x, w = encoder(x, **z)
        if model.training:
            self._current_aux_weight = scheduled_aux_weight(
                self.aux_weight,
                self.aux_weight_final,
                self.aux_weight_schedule,
                self._aux_schedule_step,
                self.aux_weight_decay_start_step,
                self.aux_weight_decay_end_step,
            )
            self._current_aux_component_weights = scheduled_aux_component_weights(
                self.aux_component_weights,
                self.aux_component_weights_final,
                self.aux_weight_schedule,
                self._aux_schedule_step,
                self.aux_weight_decay_start_step,
                self.aux_weight_decay_end_step,
            )
        compute_aux = model.training and self._current_aux_weight != 0.0
        output, state = model(
            x,
            **w,
            state=_state,
            targets=y,
            aux_tokens=aux_tokens,
            compute_aux=compute_aux,
        )
        output, w = decoder(output, state=state, **z)
        weighted_components, weighted_aux_loss = weighted_aux_components(
            getattr(model, "loss_components", {}),
            self._current_aux_component_weights,
        )
        w["aux_loss"] = (
            output.aux_loss if weighted_aux_loss is None else weighted_aux_loss
        )
        w["aux_metrics"] = (
            dict(getattr(model, "metrics", {})) if compute_aux else {}
        )
        logits = rearrange(output.logits, '... C -> (...) C')
        targets = rearrange(y, '... -> (...)')
        w["metric_loss"] = self.lm_loss(logits, targets)
        if compute_aux:
            should_log_gradient_norms = (
                self.aux_gradient_norm_interval > 0
                and self._aux_gradient_norm_step
                % self.aux_gradient_norm_interval
                == 0
            )
            self._aux_gradient_norm_step += 1
            if should_log_gradient_norms:
                w["aux_metrics"].update(
                    auxiliary_gradient_norm_metrics(
                        model,
                        w["metric_loss"],
                        weighted_components,
                        self._current_aux_weight,
                    )
                )
            w["aux_metrics"]["aux/weight"] = logits.detach().new_tensor(
                self._current_aux_weight
            )
            for name, weight in self._current_aux_component_weights.items():
                w["aux_metrics"][f"aux/component_weight/{name}"] = (
                    logits.detach().new_tensor(weight)
                )
        if model.training:
            # Advance schedules even when an initial zero overall weight skips
            # the auxiliary forward pass; otherwise a warm-up from zero stalls.
            self._aux_schedule_step += 1
        return logits, targets, w

    def metrics(
        self,
        x,
        y,
        aux_loss=None,
        metric_loss=None,
        aux_metrics=None,
        **kwargs,
    ):
        metrics = super().metrics(x, y, **kwargs)
        metrics["lm_loss"] = (
            self.lm_loss(x, y) if metric_loss is None else metric_loss
        ).detach()
        if aux_loss is not None:
            metrics["aux_loss"] = aux_loss.detach()
        if aux_metrics:
            metrics.update(aux_metrics)
        return metrics

class ForecastingTask(BaseTask):

    class DummyModule(nn.Module):

        def forward(self, *args):
            return args

    def __init__(self, norm='mean', **kwargs):
        super().__init__(**kwargs)

        if norm == 'revnorm':
            self.encoder = ReversibleInstanceNorm1dInput(self.dataset.d_input, transposed=False)
            self.decoder = ReversibleInstanceNorm1dOutput(self.encoder)
        elif norm == 'mean':
            self.encoder = TSNormalization(method='mean', horizon=self.dataset.dataset_train.forecast_horizon)
            self.decoder = TSInverseNormalization(method='mean', normalizer=self.encoder)
        elif norm == 'last':
            self.encoder = TSNormalization(method='last', horizon=self.dataset.dataset_train.forecast_horizon)
            self.decoder = TSInverseNormalization(method='last', normalizer=self.encoder)
        else:
            self.encoder = None
            self.decoder = None

        try:
            if hasattr(self.dataset.dataset_train, 'mean'):
                self.mean = torch.tensor(self.dataset.dataset_train.mean)
                self.std = torch.tensor(self.dataset.dataset_train.std)
            elif hasattr(self.dataset.dataset_train, 'standardization'):
                self.mean = torch.tensor(self.dataset.dataset_train.standardization['means'])
                self.std = torch.tensor(self.dataset.dataset_train.standardization['stds'])
            else:
                self.mean = None
                self.std = None
        except AttributeError:
            raise AttributeError('Dataset does not have mean/std attributes')
            self.mean = torch.tensor(self.dataset.dataset_train.standardization['means'])
            self.std = torch.tensor(self.dataset.dataset_train.standardization['stds'])

        if hasattr(self.dataset.dataset_train, 'log_transform'):
            self.log_transform = self.dataset.dataset_train.log_transform
        else:
            self.log_transform = False
        print("Log Transform", self.log_transform)

    def metrics(self, x, y, state=None, timestamps=None, ids=None): # Explicit about which arguments the decoder might pass through, but can future-proof with **kwargs
        if self.mean is not None:
            means = self.mean[ids].to(x.device)
            stds = self.std[ids].to(x.device)
            x_ = x * stds[:, None, None] + means[:, None, None]
            y_ = y * stds[:, None, None] + means[:, None, None]
        else:
            x_ = x
            y_ = y

        if self.log_transform:
            x_ = torch.exp(x_)
            y_ = torch.exp(y_)

        return super().metrics(x_, y_)

class VideoTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # self._y_to_logits = {}
        self._vid_to_logits = {}
        self._vid_to_label = {}

        # TODO needed to extract the first element of y, which includes the video idea; there should be a cleaner pattern to this
        import copy
        loss_fn = copy.deepcopy(self.loss)
        self.loss = lambda x, y: loss_fn(x, y[0])
        if hasattr(self, 'loss_val'):
            loss_val_fn = copy.deepcopy(self.loss_val)
            self.loss_val = lambda x, y: loss_val_fn(x, y[0])

    def metrics(self, logits, y, **kwargs):
        labels, vids = y
        return super().metrics(logits, labels, **kwargs)

    def torchmetrics(self, logits, y, prefix):
        """
        logits: (batch, n_classes)
        y = tuple of labels and video ids
        labels: (batch)
        vids: (batch)
        """
        for _logits, _label, _vid in zip(logits, y[0], y[1]):
            _vid = _vid.item()
            # Check that labels are consistent per video id
            assert self._vid_to_label[prefix].get(_vid, _label) == _label
            self._vid_to_label[prefix][_vid] = _label

            self._vid_to_logits[prefix][_vid].append(_logits)

    def _reset_torchmetrics(self, prefix):
        self._vid_to_logits[prefix] = collections.defaultdict(list)
        self._vid_to_label[prefix] = {}

    def get_torchmetrics(self, prefix):
        vid_to_average_logits = {vid: torch.mean(torch.stack(logits, dim=0), dim=0) for vid, logits in self._vid_to_logits[prefix].items()}
        # y is (label, vid) pair
        all_labels = torch.stack(list(self._vid_to_label[prefix].values()), dim=0) # (n_videos)
        all_logits = torch.stack(list(vid_to_average_logits.values()), dim=0) # (n_videos, n_classes)
        m = M.accuracy(all_logits, all_labels)
        return {'aggregate_accuracy': m}


class AdaptiveLMTask(BaseTask):
    def __init__(
        self,
        div_val,
        cutoffs : List[int],
        tie_weights : bool,
        tie_projs : List[bool],
        init_scale=1.0,
        bias_scale=0.0,
        dropemb=0.0,
        dropsoft=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        n_tokens = self.dataset.n_tokens
        d_model = self.model.d_model
        d_output = self.model.d_output

        encoder = AdaptiveEmbedding(
            n_tokens,
            d_model,
            d_model,
            cutoffs=cutoffs,
            div_val=div_val,
            init_scale=init_scale,
            dropout=dropemb,
        )

        if tie_weights:
            assert d_model == d_output
            emb_layers = [i.weight for i in encoder.emb_layers]
        else:
            emb_layers = None

        # Construct decoder/loss
        emb_projs = encoder.emb_projs
        loss = ProjectedAdaptiveLogSoftmax(
            n_tokens, d_output, d_output,
            cutoffs, div_val=div_val,
            tie_projs=tie_projs,
            out_projs=emb_projs,
            out_layers_weights=emb_layers,
            bias_scale=bias_scale,
            dropout=dropsoft,
        )

        self.encoder = encoder
        self.loss = loss


class ImageNetTask(BaseTask):
    """
    Imagenet training uses mixup augmentations, which require a separate loss for train and val,
    which we overide the base task here.
    """

    def __init__(self, **kwargs):
        import hydra

        super().__init__(
            dataset=kwargs.get("dataset", None),
            model=kwargs.get("model", None),
            loss=kwargs.get("loss", None),  # we still create the base loss here, but will overide below
            metrics=kwargs.get("metrics", None),
            torchmetrics=kwargs.get("torchmetrics", None)
        )

        # if using mixup, overide loss (train) and loss_val, otherwise
        # we have just one loss from the base task above
        if "loss_val" in kwargs and "loss_train" in kwargs:
            self.loss = hydra.utils.instantiate(kwargs.get("loss_train"))
            self.loss_val = hydra.utils.instantiate(kwargs.get('loss_val'))


registry = {
    'base': BaseTask,
    'lm': LMTask,
    'aux_lm': AuxLMTask,
    'imagenet': ImageNetTask,
    'forecasting': ForecastingTask,
    'video': VideoTask,
}

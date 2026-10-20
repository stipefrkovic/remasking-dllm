# coding=utf-8
# Copyright 2024 The Dream team, HKUNLP Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import (
    GenerationConfig
)
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)
import os

logger = logging.get_logger(__name__)


# def top_p_logits(logits, top_p=None):
#     sorted_logits, sorted_indices = torch.sort(logits, descending=True)
#     cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
#     sorted_indices_to_remove = cumulative_probs > top_p
#     # Shift the indices to the right to keep the first token above the threshold
#     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#     sorted_indices_to_remove[..., 0] = 0

#     mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
#     mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
#     logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
#     return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_k=None, margin_confidence=False, neg_entropy=False):

    # Always compute confidence from ORIGINAL (non-temperature-scaled) logits,
    # matching LLaDA's approach where temperature only affects token selection.
    clean_probs = torch.softmax(logits, dim=-1)

    if top_k is not None:
        logits = top_k_logits(logits, top_k)

    if temperature > 0:
        scaled_logits = logits / temperature
        scaled_probs = torch.softmax(scaled_logits, dim=-1)
        try:
            x0 = dists.Categorical(probs=scaled_probs).sample()
            confidence = torch.gather(clean_probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = clean_probs.max(dim=-1)
    else:
        probs = torch.softmax(logits, dim=-1)
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


def select_topk_positions_by_score_cdf(
    positions: torch.Tensor,
    score: torch.Tensor,) -> torch.Tensor:
    """Adaptive revision rate following paper Sec 3.3 line 257-261:

        |S_t| = ceil(sqrt(n_t * pi_t))

    where pi_t = 1 - F_t(mean_score) is the tail mass above the mean, so
    n_t * pi_t = count of scores above the mean (= m below).

    Previously this function used `k = m` (raw count, capped at 8) which
    diverges from the paper formula.
    """
    n = int(score.numel())
    if n <= 0:
        return positions.new_empty((0,), dtype=positions.dtype)

    score_f = score.to(torch.float32)
    mean_score = score_f.mean()
    m = torch.count_nonzero(score_f > mean_score)
    # Paper formula: k = ceil(sqrt(m))
    k = int(torch.ceil(torch.sqrt(m.to(torch.float32))).to(torch.long).item())

    if k <= 0:
        return positions.new_empty((0,), dtype=positions.dtype)

    num_select = min(k, n)

    topk_idx = torch.topk(score_f, k=num_select, largest=True).indices
    chosen_positions = positions.index_select(0, topk_idx)

    return chosen_positions


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None
    steps_used: Optional[int] = None


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        # self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        self.gen_length: Optional[int] = kwargs.pop("gen_length", None)
        if self.max_new_tokens is None and self.gen_length is not None:
            self.max_new_tokens = int(self.gen_length)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", 'origin')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)

        # threshold-based decoding params
        self.tau_draft: Optional[float] = kwargs.pop("tau_draft", None)
        self.threshold: float = kwargs.pop("threshold", 0.5)  # probability threshold for decoding
        self.log_step_stats: bool = kwargs.pop("log_step_stats", False)
        self.max_unmask_per_step: int = kwargs.pop("max_unmask_per_step", 5)
        self.debug: bool = kwargs.pop("debug", False)

        # Legacy args kept for eval_llada.py compatibility (some may be unused here)
        self.use_cdcr: bool = kwargs.pop("use_cdcr", True)
        self.delta_drop: float = kwargs.pop("delta_drop", 0.10)
        self.beta_dep: float = kwargs.pop("beta_dep", 0.25)
        self.coupling_kappa: float = kwargs.pop("coupling_kappa", 0.02)

        # contextual re-verify (seed KV-cache override) params
        self.use_low_conf_reverify: bool = kwargs.pop("use_low_conf_reverify", True)
        self.use_kv_cache_for_reverify: bool = kwargs.pop("use_kv_cache_for_reverify", True)
        self.use_kv_cache_reverify: Optional[bool] = kwargs.pop("use_kv_cache_reverify", None)
        self.use_attention_score: bool = kwargs.pop("use_attention_score", False)
        self.max_reverify_times: int = kwargs.pop("max_reverify_times", 30)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        self.block_length = kwargs.pop("block_length", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass

class DreamGenerationMixin:
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
    ) -> DreamGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            generation_config = DreamGenerationConfig.from_model_config(self.config)
            using_model_generation_config = True

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
        # exception will be raised in `_validate_model_kwargs`
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            _kwargs = generation_config.update(**kwargs)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id

        # Fallback to model.config.mask_token_id if still not set
        if generation_config.mask_token_id is None:
            model_mask_token_id = getattr(self.config, "mask_token_id", None)
            if model_mask_token_id is not None:
                generation_config.mask_token_id = model_mask_token_id

        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: DreamGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.

        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        gen_length = getattr(generation_config, "gen_length", None)
        if gen_length is not None and generation_config.max_new_tokens is None:
            generation_config.max_new_tokens = int(gen_length)
        generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
        generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)

        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 4. Check input_ids
        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )
        if (
            hasattr(generation_config, "pad_token_id") and
            torch.any(input_ids == generation_config.pad_token_id) and
            attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        result = self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            generation_tokens_hook_func=generation_tokens_hook_func,
            generation_logits_hook_func=generation_logits_hook_func
        )
        return result

    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func,
        track_flip_flop: bool = True
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        if mask_token_id is None:
            raise ValueError(
                "`mask_token_id` must be set in generation_config for diffusion generation. "
                "Please ensure the model's config or generation_config includes a valid mask_token_id."
            )
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        # top_p = generation_config.top_p
        top_k = generation_config.top_k
        block_length = generation_config.block_length
        tau_draft = getattr(generation_config, "tau_draft", None)
        threshold = float(tau_draft) if tau_draft is not None else float(getattr(generation_config, "threshold", 0.5))
        log_step_stats = bool(getattr(generation_config, "log_step_stats", False))
        use_kv_cache_reverify_cfg = getattr(generation_config, "use_kv_cache_reverify", None)
        if use_kv_cache_reverify_cfg is None:
            use_low_conf_reverify = bool(getattr(generation_config, "use_low_conf_reverify", True))
            use_kv_cache_for_reverify = bool(getattr(generation_config, "use_kv_cache_for_reverify", True))
            use_kv_cache_reverify = bool(use_low_conf_reverify and use_kv_cache_for_reverify)
        else:
            use_kv_cache_reverify = bool(use_kv_cache_reverify_cfg)
        use_attention_score = bool(getattr(generation_config, "use_attention_score", False))
        max_reverify_times = int(getattr(generation_config, "max_reverify_times", 30))
        max_unmask_per_step = int(getattr(generation_config, "max_unmask_per_step", 5))
        # EOS token for early stopping
        eos_token_id = getattr(generation_config, "_eos_token_tensor", None)
        if eos_token_id is None:
            eos_token_id = getattr(generation_config, "eos_token_id", None)
        if isinstance(eos_token_id, torch.Tensor):
            eos_token_id = eos_token_id[0].item() if eos_token_id.numel() > 0 else None

        histories = [] if (return_dict_in_generate and output_history) else None

        prompt_len = int(input_ids.shape[1])
        batch_size = int(input_ids.shape[0])
        can_eos_early_stop = eos_token_id is not None and batch_size == 1

        # Flip-flop tracking: position unmasked with token T -> remasked -> unmasked again with SAME token T
        if track_flip_flop:
            position_history = [{} for _ in range(batch_size)]  # {pos: {'last_token': int, 'was_remasked': bool}}
            flip_flop_count = [0] * batch_size
            total_remask_count = [0] * batch_size
            total_unmask_count = [0] * batch_size
            replace_count = [0] * batch_size
            changed_after_remask_count = [0] * batch_size
            keep_count = [0] * batch_size

        # pad input_ids to max_length
        gen_length = int(max_length - prompt_len)
        x = F.pad(input_ids, (0, gen_length), value=mask_token_id)
        env_low_thre = os.environ.get("LOW_THRE")
        if env_low_thre is not None:
            try:
                threshold = float(env_low_thre)
            except ValueError:
                pass

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        if block_length is None or int(block_length) <= 0:
            raise ValueError("`block_length` must be set to a positive int for block-wise generation.")

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks

        # this allows user-defined token control of the intermediate steps
        x = generation_tokens_hook_func(None, x, None)
        # === KV-cache re-verify mode (slow attention path only) ===
        if x.size(0) != 1 or int(generation_config.num_return_sequences) != 1:
            raise ValueError("KV-cache reverify currently supports batch_size=1 and num_return_sequences=1.")

        model_core = getattr(self, "model", None)
        layers = getattr(model_core, "layers", None) if model_core is not None else None
        attn_modules = []
        if layers is not None:
            for layer in layers:
                if hasattr(layer, "self_attn"):
                    attn_modules.append(layer.self_attn)

        # Off-diagonal correction (q=j-1,k=j) is enabled per attention module via env var.
        # Only when enabled do we also freeze predictor-row K/V (j-1) across layers.
        enable_offdiag = bool(attn_modules) and bool(
            getattr(attn_modules[0], "_seed_kv_enable_offdiag", False)
        )

        def _clear_seed_overrides() -> None:
            if not attn_modules:
                return
            for attn in attn_modules:
                attn._seed_kv_positions = None
                attn._seed_kv_key = None
                attn._seed_kv_value = None
                attn._seed_kv_prev_positions = None
                attn._seed_kv_prev_key = None
                attn._seed_kv_prev_value = None

        def _iter_past_key_values(cache_obj):
            if cache_obj is None:
                return []
            if isinstance(cache_obj, (list, tuple)):
                return list(cache_obj)
            if hasattr(cache_obj, "to_legacy_cache"):
                try:
                    legacy = cache_obj.to_legacy_cache()
                    if isinstance(legacy, (list, tuple)):
                        return list(legacy)
                except Exception:
                    pass
            if hasattr(cache_obj, "key_cache") and hasattr(cache_obj, "value_cache"):
                keys = cache_obj.key_cache
                values = cache_obj.value_cache
                if isinstance(keys, (list, tuple)) and isinstance(values, (list, tuple)):
                    return list(zip(keys, values))
            raise TypeError(f"Unsupported past_key_values type: {type(cache_obj)}")

        from transformers.cache_utils import DynamicCache

        reverify_positions: Optional[torch.Tensor] = None
        reverify_seed_kv_k: Optional[torch.Tensor] = None  # (num_layers, B, kv_heads, S, head_dim)
        reverify_seed_kv_v: Optional[torch.Tensor] = None
        reverify_prev_kv_k: Optional[torch.Tensor] = None  # (num_layers, B, kv_heads, S, head_dim)
        reverify_prev_kv_v: Optional[torch.Tensor] = None
        reverify_count = torch.zeros((max_length,), dtype=torch.long, device=x.device)

        def _prune_adjacent_seed_positions(
            positions: torch.Tensor,
            score: torch.Tensor,
        ) -> torch.Tensor:
            if positions.numel() <= 1:
                return positions

            # Greedy: keep highest-score positions, but disallow adjacent indices
            # because Dream predicts token[j] from query[j-1].
            pos_cpu = positions.to("cpu", dtype=torch.long).tolist()
            score_cpu = score.to("cpu", dtype=torch.float32).tolist()
            idx_sorted = sorted(range(len(pos_cpu)), key=lambda i: float(score_cpu[i]), reverse=True)

            chosen = []
            blocked = set()
            for i in idx_sorted:
                p = int(pos_cpu[i])
                if p in blocked:
                    continue
                chosen.append(p)
                blocked.add(p - 1)
                blocked.add(p)
                blocked.add(p + 1)

            chosen.sort()
            return positions.new_tensor(chosen, dtype=positions.dtype, device=positions.device)

        def _schedule_seeds_for_next_step(
            *,
            logits: torch.Tensor,
            last_layer_attention: Optional[torch.Tensor],
            past_cache: object,
            unmasked_before_forward: torch.Tensor,
            seed_positions: Optional[torch.Tensor],
            newly_unmasked_positions: torch.Tensor,
            block_start: int,
            block_end: int,
        ) -> None:
            nonlocal reverify_positions, reverify_seed_kv_k, reverify_seed_kv_v, reverify_prev_kv_k, reverify_prev_kv_v

            if max_reverify_times <= 0:
                return

            block_positions = torch.arange(block_start, block_end, device=x.device)
            eligible_base = (
                unmasked_before_forward[0, block_positions]
                & (reverify_count[block_positions] < max_reverify_times)
            )
            # Dream predicts token[j] from query[j-1], so only schedule seeds whose predictor
            # position (j-1) is also unmasked in this forward.
            block_prev_positions = torch.where(block_positions > 0, block_positions - 1, block_positions)
            eligible_base = eligible_base & unmasked_before_forward[0, block_prev_positions]
            if seed_positions is not None and seed_positions.numel() > 0:
                seed_mask_total = torch.zeros((max_length,), dtype=torch.bool, device=x.device)
                seed_mask_total[seed_positions] = True
                eligible_base = eligible_base & (~seed_mask_total[block_positions])
                eligible_base = eligible_base & (~seed_mask_total[block_prev_positions])

            if torch.count_nonzero(eligible_base) == 0:
                return

            eligible_pos = block_positions[eligible_base]
            token_ids = x[0, eligible_pos]
            log_probs = F.log_softmax(logits[0, eligible_pos, :].to(torch.float32), dim=-1)
            p_self = log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1).exp().clamp(min=1e-9)
            u = (-torch.log(p_self)).to(torch.float32)

            selected_seed_pos = None
            selected_seed_score = None

            if use_attention_score and last_layer_attention is not None:
                attn = last_layer_attention[0]
                eligible_q = torch.where(eligible_pos > 0, eligible_pos - 1, eligible_pos)

                if newly_unmasked_positions is not None and newly_unmasked_positions.numel() > 0:
                    attn_seed_to_new = (
                        attn.index_select(1, eligible_q).index_select(2, newly_unmasked_positions).to(torch.float32)
                    )
                    d_out = attn_seed_to_new.sum(dim=2).mean(dim=0)
                else:
                    d_out = torch.zeros((eligible_pos.numel(),), device=x.device, dtype=torch.float32)

                mask_after = (x[0, block_positions] == mask_token_id)
                masked_pos = block_positions[mask_after]
                if masked_pos.numel() > 0:
                    masked_q = torch.where(masked_pos > 0, masked_pos - 1, masked_pos)
                    attn_masked_to_eligible = (
                        attn.index_select(1, masked_q).index_select(2, eligible_pos).to(torch.float32)
                    )
                    d_in = attn_masked_to_eligible.sum(dim=1).mean(dim=0)
                else:
                    d_in = torch.zeros((eligible_pos.numel(),), device=x.device, dtype=torch.float32)

                # Paper Sec 3.3 line 235-242: Score(j) = u(j) * (1 + d_in(j)) / (1 + d_out(j))
                # (previously had 0.5 weights on d_in/d_out — now matches paper exactly)
                score = u * (1.0 + d_in) / (1.0 + d_out)
                # score = u * d_in * d_out # 2 - 0.58
                # score = u / (d_in*d_out+1e-9) # 3 -
                selected_seed_pos = select_topk_positions_by_score_cdf(
                    eligible_pos, score
                )
                # print(f"xxx {selected_seed_pos}")
                if selected_seed_pos is not None and selected_seed_pos.numel() > 0:
                    sorted_eligible, perm = eligible_pos.sort()
                    sorted_score = score.index_select(0, perm)
                    idx = torch.searchsorted(sorted_eligible, selected_seed_pos)
                    idx = idx.clamp(max=max(sorted_eligible.numel() - 1, 0))
                    match = sorted_eligible.index_select(0, idx).eq(selected_seed_pos)
                    selected_seed_score = torch.where(
                        match,
                        sorted_score.index_select(0, idx),
                        torch.zeros_like(selected_seed_pos, dtype=torch.float32),
                    )
            else:
                eligible_low = p_self < threshold
                if torch.count_nonzero(eligible_low) > 0:
                    seed_pos = eligible_pos[eligible_low]
                    seed_score = u[eligible_low]
                    selected_seed_pos = select_topk_positions_by_score_cdf(
                        seed_pos, seed_score
                    )
                    if selected_seed_pos is not None and selected_seed_pos.numel() > 0:
                        sorted_seed, perm = seed_pos.sort()
                        sorted_score = seed_score.index_select(0, perm)
                        idx = torch.searchsorted(sorted_seed, selected_seed_pos)
                        idx = idx.clamp(max=max(sorted_seed.numel() - 1, 0))
                        match = sorted_seed.index_select(0, idx).eq(selected_seed_pos)
                        selected_seed_score = torch.where(
                            match,
                            sorted_score.index_select(0, idx),
                            torch.zeros_like(selected_seed_pos, dtype=torch.float32),
                        )

            if selected_seed_pos is None or selected_seed_pos.numel() == 0:
                return

            if selected_seed_score is None:
                selected_seed_score = torch.zeros_like(selected_seed_pos, dtype=torch.float32)

            # Dream predicts token[j] from query[j-1], so avoid selecting adjacent seeds
            # (otherwise one seed's predictor row could be masked by another seed).
            selected_seed_pos = _prune_adjacent_seed_positions(selected_seed_pos, selected_seed_score)
            if selected_seed_pos.numel() == 0:
                return

            kv_layers = _iter_past_key_values(past_cache)
            if not kv_layers:
                return

            k0, v0 = kv_layers[0]
            if not (isinstance(k0, torch.Tensor) and isinstance(v0, torch.Tensor)):
                return

            num_layers = len(kv_layers)
            B_kv, n_kv_heads, _, head_dim = k0.shape
            num_selected = int(selected_seed_pos.numel())
            selected_prev_pos = torch.where(selected_seed_pos > 0, selected_seed_pos - 1, selected_seed_pos)

            reverify_positions = selected_seed_pos
            reverify_seed_kv_k = torch.empty(
                (num_layers, B_kv, n_kv_heads, num_selected, head_dim),
                dtype=k0.dtype,
                device=k0.device,
            )
            reverify_seed_kv_v = torch.empty(
                (num_layers, B_kv, n_kv_heads, num_selected, head_dim),
                dtype=v0.dtype,
                device=v0.device,
            )
            if enable_offdiag:
                reverify_prev_kv_k = torch.empty(
                    (num_layers, B_kv, n_kv_heads, num_selected, head_dim),
                    dtype=k0.dtype,
                    device=k0.device,
                )
                reverify_prev_kv_v = torch.empty(
                    (num_layers, B_kv, n_kv_heads, num_selected, head_dim),
                    dtype=v0.dtype,
                    device=v0.device,
                )
            else:
                reverify_prev_kv_k = None
                reverify_prev_kv_v = None
            for layer_idx, (k_all, v_all) in enumerate(kv_layers):
                reverify_seed_kv_k[layer_idx] = k_all.index_select(-2, selected_seed_pos)
                reverify_seed_kv_v[layer_idx] = v_all.index_select(-2, selected_seed_pos)
                if enable_offdiag and reverify_prev_kv_k is not None and reverify_prev_kv_v is not None:
                    reverify_prev_kv_k[layer_idx] = k_all.index_select(-2, selected_prev_pos)
                    reverify_prev_kv_v[layer_idx] = v_all.index_select(-2, selected_prev_pos)

        total_steps = 0  # Track actual inference steps across all blocks

        for num_block in range(num_blocks):
            block_start = prompt_len + num_block * block_length
            block_end = prompt_len + (num_block + 1) * block_length

            # clear any pending seeds at block boundary
            reverify_positions = None
            reverify_seed_kv_k = None
            reverify_seed_kv_v = None
            reverify_prev_kv_k = None
            reverify_prev_kv_v = None

            step_count = 0
            while (x[:, block_start:block_end] == mask_token_id).any():
                unmasked_before_forward = (x != mask_token_id)

                seed_positions_tensor = None
                seed_old_ids = None
                if (
                    reverify_positions is not None
                    and reverify_seed_kv_k is not None
                    and reverify_seed_kv_v is not None
                    and attn_modules
                ):
                    in_block_mask = (reverify_positions >= block_start) & (reverify_positions < block_end)
                    seed_positions_tensor = reverify_positions[in_block_mask]
                    if seed_positions_tensor.numel() > 0:
                        seed_old_ids = x[0, seed_positions_tensor].clone()
                        x[0, seed_positions_tensor] = mask_token_id


                        prev_positions_tensor = torch.where(
                            seed_positions_tensor > 0, seed_positions_tensor - 1, seed_positions_tensor
                        )

                        for layer_idx, attn in enumerate(attn_modules):
                            if layer_idx >= reverify_seed_kv_k.size(0):
                                break
                            attn._seed_kv_positions = seed_positions_tensor
                            attn._seed_kv_key = reverify_seed_kv_k[layer_idx, :, :, in_block_mask, :]
                            attn._seed_kv_value = reverify_seed_kv_v[layer_idx, :, :, in_block_mask, :]
                            if (
                                enable_offdiag
                                and reverify_prev_kv_k is not None
                                and reverify_prev_kv_v is not None
                            ):
                                attn._seed_kv_prev_positions = prev_positions_tensor
                                attn._seed_kv_prev_key = reverify_prev_kv_k[layer_idx, :, :, in_block_mask, :]
                                attn._seed_kv_prev_value = reverify_prev_kv_v[layer_idx, :, :, in_block_mask, :]
                            else:
                                attn._seed_kv_prev_positions = None
                                attn._seed_kv_prev_key = None
                                attn._seed_kv_prev_value = None

                    reverify_positions = None
                    reverify_seed_kv_k = None
                    reverify_seed_kv_v = None
                    reverify_prev_kv_k = None
                    reverify_prev_kv_v = None

                need_attn_base = bool(use_attention_score) or (
                    seed_positions_tensor is not None and seed_positions_tensor.numel() > 0
                )
                need_attn = need_attn_base

                past_cache = DynamicCache()
                outputs = self(
                    input_ids=x,
                    attention_mask=attention_mask,
                    position_ids=tok_idx,
                    past_key_values=past_cache,
                    use_cache=True,
                    output_attentions=need_attn,
                    output_attentions_last_only=need_attn,
                )
                logits = torch.cat([outputs.logits[:, :1], outputs.logits[:, :-1]], dim=1)
                logits = generation_logits_hook_func(step_count, x, logits)

                last_layer_attention = None
                if use_attention_score and getattr(outputs, "attentions", None) is not None and len(outputs.attentions) > 0:
                    last_layer_attention = outputs.attentions[-1]

                _clear_seed_overrides()

                mask_index = (x == mask_token_id)
                mask_index[:, :block_start] = False
                mask_index[:, block_end:] = False
                if seed_positions_tensor is not None and seed_positions_tensor.numel() > 0:
                    mask_index[0, seed_positions_tensor] = False
                    # Dream's shifted logits: masking seed position `j` corrupts the predictor row `j`,
                    # which is used to predict token at position `j+1`. Avoid drafting those positions in
                    # the same forward pass.
                    seed_next = seed_positions_tensor + 1
                    in_block_next = (seed_next >= block_start) & (seed_next < block_end)
                    seed_next = seed_next[in_block_next]
                    if seed_next.numel() > 0:
                        mask_index[0, seed_next] = False

                if torch.count_nonzero(mask_index) == 0:
                    newly_unmasked_positions = x.new_empty((0,), dtype=torch.long)
                    will_unmask_cnt = 0
                else:
                    mask_logits = logits[mask_index]
                    confidence, x0 = sample_tokens(
                        mask_logits, temperature=temperature, top_k=top_k
                    )

                    full_confidence = torch.full_like(x, -torch.inf, device=x.device, dtype=logits.dtype)
                    full_confidence[mask_index] = confidence
                    # print(threshold)
                    above_threshold = full_confidence > threshold
                    candidate_cnt = int(above_threshold.sum().item())
                    # Apply max_unmask_per_step limit
                    if max_unmask_per_step > 0 and candidate_cnt > max_unmask_per_step:
                        # Select top max_unmask_per_step positions by confidence
                        _, top_indices = torch.topk(full_confidence[0], k=max_unmask_per_step)
                        above_threshold = torch.zeros_like(above_threshold)
                        above_threshold[0, top_indices] = True
                    # print(candidate_cnt)
                    if not above_threshold.any():
                        batch_conf = full_confidence[0]
                        if (batch_conf > -torch.inf).any():
                            max_idx = batch_conf.argmax()
                            above_threshold[0, max_idx] = True

                    x_ = torch.full_like(x, mask_token_id)
                    x_[mask_index] = x0.clone()
                    x = torch.where(above_threshold, x_, x)

                    newly_unmasked_positions = above_threshold[0].nonzero(as_tuple=False).squeeze(-1)
                    will_unmask_cnt = int(newly_unmasked_positions.numel())

                    # UNMASK tracking: record which positions are being unmasked and with which tokens
                    if track_flip_flop and newly_unmasked_positions.numel() > 0:
                        total_unmask_count[0] += int(newly_unmasked_positions.numel())
                        for pos in newly_unmasked_positions.tolist():
                            current_token = x[0, pos].item()
                            if pos in position_history[0]:
                                hist = position_history[0][pos]
                                if hist['was_remasked']:
                                    if hist['last_token'] == current_token:
                                        flip_flop_count[0] += 1
                                    else:
                                        changed_after_remask_count[0] += 1
                            position_history[0][pos] = {'last_token': current_token, 'was_remasked': False}

                    # unmask_confidences = full_confidence[0, newly_unmasked_positions].tolist()
                    # print(
                    #     f"[dream][unmask] block={num_block} step={step_count},above_threshold = {candidate_cnt} will_unmask={will_unmask_cnt} confidences={unmask_confidences}"
                    # )

                if (
                    seed_positions_tensor is not None
                    and seed_positions_tensor.numel() > 0
                    and seed_old_ids is not None
                ):
                    seed_logits = logits[0, seed_positions_tensor, :]
                    seed_new_ids = seed_logits.argmax(dim=-1)
                    seed_log_probs = F.log_softmax(seed_logits.to(torch.float32), dim=-1)
                    p_new = seed_log_probs.gather(-1, seed_new_ids.unsqueeze(-1)).squeeze(-1).exp()

                    same_mask = seed_new_ids.eq(seed_old_ids)
                    low_conf_mask = p_new <= threshold
                    high_conf_mask = ~low_conf_mask

                    remask_mask = low_conf_mask & (~same_mask)
                    keep_mask = same_mask
                    replace_mask = high_conf_mask & (~same_mask)

                    # for idx in range(seed_positions_tensor.numel()):
                    #     pos = seed_positions_tensor[idx].item()
                    #     prob = p_new[idx].item()
                    #     is_same = same_mask[idx].item()
                    #     if remask_mask[idx].item():
                    #         action = "remask"
                    #     elif keep_mask[idx].item():
                    #         action = "keep"
                    #     else:
                    #         action = "replace"
                    #     print(
                    #         f"[dream][seed_detail] block={num_block} step={step_count} "
                    #         f"pos={pos} p_new={prob:.4f} same={is_same} action={action}"
                    #     )

                    keep_pos = seed_positions_tensor[keep_mask]
                    replace_pos = seed_positions_tensor[replace_mask]
                    remask_pos = seed_positions_tensor[remask_mask]

                    reverify_count.index_add_(0, seed_positions_tensor, torch.ones_like(seed_positions_tensor))

                    if keep_pos.numel() > 0:
                        x[0, keep_pos] = seed_old_ids[keep_mask]
                        if track_flip_flop:
                            keep_count[0] += int(keep_pos.numel())
                    if replace_pos.numel() > 0:
                        x[0, replace_pos] = seed_new_ids[replace_mask]
                        # Update position_history for replaced positions (token changed)
                        if track_flip_flop:
                            replace_count[0] += int(replace_pos.numel())
                            for idx, pos in enumerate(replace_pos.tolist()):
                                new_token = seed_new_ids[replace_mask][idx].item()
                                position_history[0][pos] = {'last_token': new_token, 'was_remasked': False}
                    if remask_pos.numel() > 0:
                        # Apply max_remask_allowed limit
                        max_remask_allowed = max(0, will_unmask_cnt - 1)
                        remask_p = p_new[remask_mask]
                        sort_idx = torch.argsort(remask_p)
                        remask_pos = remask_pos.index_select(0, sort_idx)
                        if max_remask_allowed > 0:
                            x[0, remask_pos[:max_remask_allowed]] = mask_token_id

                            # REMASK tracking: mark remasked positions so future unmask can detect flip-flop
                            if track_flip_flop:
                                remasked_positions_list = remask_pos[:max_remask_allowed].tolist()
                                total_remask_count[0] += len(remasked_positions_list)
                                for pos in remasked_positions_list:
                                    if pos in position_history[0]:
                                        position_history[0][pos]['was_remasked'] = True

                        keep_back_pos = remask_pos[max_remask_allowed:]
                        if keep_back_pos.numel() > 0:
                            keep_back_ids = seed_old_ids[remask_mask].index_select(0, sort_idx)[
                                max_remask_allowed:
                            ]
                            x[0, keep_back_pos] = keep_back_ids

                    # print(
                    #     f"[dream][reverify] block={num_block} step={step_count} "
                    #     f"keep={keep_pos.numel()} replace={replace_pos.numel()} "
                    #     f"remask={actual_remask_cnt} (candidates={remask_pos.numel() if remask_pos.numel() > 0 else 0})"
                    # )

                _schedule_seeds_for_next_step(
                    logits=logits,
                    last_layer_attention=last_layer_attention,
                    past_cache=past_cache,
                    unmasked_before_forward=unmasked_before_forward,
                    seed_positions=seed_positions_tensor,
                    newly_unmasked_positions=newly_unmasked_positions,
                    block_start=block_start,
                    block_end=block_end,
                )

                x = generation_tokens_hook_func(step_count, x, logits)
                if histories is not None:
                    histories.append(x.clone())

                # === EOS early stopping (kv-cache reverify threshold mode) ===
                if can_eos_early_stop:
                    response_tokens = x[0, prompt_len:block_end]
                    eos_mask = response_tokens == eos_token_id
                    if torch.count_nonzero(eos_mask) > 0:
                        first_eos_rel = torch.argmax(eos_mask.to(torch.int32)).item()
                        first_eos_abs = prompt_len + first_eos_rel
                        all_unmasked_before_eos = (x[0, prompt_len:first_eos_abs] != mask_token_id).all().item()
                        if all_unmasked_before_eos:
                            # Prepare flip-flop stats for early return
                            if track_flip_flop:
                                flip_flop_stats = {
                                    'flip_flop_count': flip_flop_count,
                                    'total_remask_count': total_remask_count,
                                    'total_unmask_count': total_unmask_count,
                                    'replace_count': replace_count,
                                    'changed_after_remask_count': changed_after_remask_count,
                                    'keep_count': keep_count,
                                    'per_sample_flip_flops': flip_flop_count.copy(),
                                    'per_sample_remasked': total_remask_count.copy(),
                                }
                            else:
                                flip_flop_stats = None
                            if return_dict_in_generate:
                                return DreamModelOutput(sequences=x[:, :first_eos_abs + 1], history=histories, steps_used=total_steps), total_steps, flip_flop_stats
                            else:
                                return x[:, :first_eos_abs + 1], total_steps, flip_flop_stats

                step_count += 1
                total_steps += 1

        # Prepare flip-flop stats for final return
        if track_flip_flop:
            flip_flop_stats = {
                'flip_flop_count': flip_flop_count,
                'total_remask_count': total_remask_count,
                'total_unmask_count': total_unmask_count,
                'replace_count': replace_count,
                'changed_after_remask_count': changed_after_remask_count,
                'keep_count': keep_count,
                'per_sample_flip_flops': flip_flop_count.copy(),
                'per_sample_remasked': total_remask_count.copy(),
            }
        else:
            flip_flop_stats = None

        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
                steps_used=total_steps,
            ), total_steps, flip_flop_stats
        else:
            return x, total_steps, flip_flop_stats



def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def add_gumbel_noise(logits, temperature):
    if temperature == 0.0:
        return logits
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    # x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, float("-inf")))

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens_eff = mask_index.sum(dim=1)
    elif isinstance(num_transfer_tokens, torch.Tensor):
        num_transfer_tokens_eff = num_transfer_tokens.view(-1)
    else:
        num_transfer_tokens_eff = torch.full(
            (confidence.shape[0],),
            int(num_transfer_tokens),
            device=confidence.device,
            dtype=torch.long,
        )
    for j in range(confidence.shape[0]):
        k = int(num_transfer_tokens_eff[j].item())
        if k <= 0:
            continue
        _, select_index = torch.topk(confidence[j], k=k)
        transfer_index[j, select_index] = True
        if threshold is not None:
            for kk in range(1, k):
                if confidence[j, select_index[kk]] < threshold:
                    transfer_index[j, select_index[kk]] = False
    return x0, transfer_index

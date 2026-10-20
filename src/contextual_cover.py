import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import torch
import torch.nn.functional as F

try:
    from .modeling_llada_kv_cover import LLaDAModelLM
except ImportError:  # pragma: no cover
    # Fallback for running this file as a standalone script.
    from modeling_llada_kv_cover import LLaDAModelLM


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits64 = logits.to(torch.float64)
    noise = torch.rand_like(logits64, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits64.exp() / gumbel_noise


def select_topk_positions_by_score_cdf(positions: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    n = int(score.numel())
    if n <= 0:
        return positions.new_empty((0,), dtype=positions.dtype)

    score_f = score.to(torch.float32)
    mean_score = score_f.mean()
    m = torch.count_nonzero(score_f > mean_score)
    k = torch.ceil(torch.sqrt(m.to(torch.float32))).to(torch.long)

    k_max = int(math.ceil(math.sqrt(n)))
    if k_max <= 0:
        return positions.new_empty((0,), dtype=positions.dtype)

    topk_idx = torch.topk(score_f, k=k_max, largest=True).indices
    ranks = torch.arange(k_max, device=score.device)
    chosen_idx = topk_idx[ranks < k]
    return positions.index_select(0, chosen_idx)


class UGAGenerator:
    def __init__(self, model, tokenizer, mask_id: int = 126336):
        self.model = model
        self.tokenizer = tokenizer
        self.mask_id = mask_id
        self.device = model.device

        # Pre-allocated KV cache buffer (lazy init)
        self._kv_buffer_k: torch.Tensor | None = None
        self._kv_buffer_v: torch.Tensor | None = None
        self._num_layers: int | None = None

    @torch.no_grad()
    def generate_causal(
        self,
        prompt: torch.Tensor,
        gen_length: int = 128,
        block_length: int = 128,
        tau_draft: float = 0.5,
        max_unmask_per_step: int = 5,
        temperature: float = 0.0,
        max_reverify_times: int = 30,
        use_kv_cache_for_reverify: bool = True,
        use_attention_score: bool = False,
        batch_size: int = 1,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int, dict]:
        batch_size = int(prompt.shape[0])
        prompt_len = int(prompt.shape[1])
        total_length = prompt_len + int(gen_length)

        x = torch.full(
            (batch_size, total_length),
            self.mask_id,
            dtype=torch.long,
            device=self.device,
        )
        x[:, :prompt_len] = prompt.clone()

        # === Locate transformer blocks (used for per-layer seed KV override) ===
        all_blocks = []
        model_core = getattr(self.model, "model", self.model)
        transformer = getattr(model_core, "transformer", None)
        if transformer is not None:
            try:
                if "blocks" in transformer:
                    all_blocks = list(transformer["blocks"])
                elif "block_groups" in transformer:
                    for group in transformer["block_groups"]:
                        if hasattr(group, "blocks"):
                            all_blocks.extend(group.blocks)
                        else:
                            all_blocks.extend(list(group))
            except TypeError:
                if hasattr(transformer, "blocks"):
                    all_blocks = list(transformer.blocks)
                elif hasattr(transformer, "block_groups"):
                    for group in transformer.block_groups:
                        if hasattr(group, "blocks"):
                            all_blocks.extend(group.blocks)
                        else:
                            all_blocks.extend(list(group))

        # Tracking
        current_confidences = torch.zeros((batch_size, total_length), device=self.device)
        current_confidences[:, :prompt_len] = 1.0
        actual_steps = 0

        # Flip-flop tracking is per sample and persists across steps.
        position_history = [{} for _ in range(batch_size)]  # pos -> {last_token, was_remasked}
        flip_flop_count = [0 for _ in range(batch_size)]
        total_remask_count = [0 for _ in range(batch_size)]
        total_unmask_count = [0 for _ in range(batch_size)]
        keep_count = [0 for _ in range(batch_size)]
        replace_count = [0 for _ in range(batch_size)]
        seed_total_count = [0 for _ in range(batch_size)]
        deferred_remask_count = [0 for _ in range(batch_size)]
        changed_after_remask_count = [0 for _ in range(batch_size)]

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        # Per-sample seed re-verify state (cross-step)
        reverify_positions_list: list[torch.Tensor | None] = [None] * batch_size
        reverify_kv_list: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * batch_size
        reverify_count = torch.zeros((batch_size, total_length), dtype=torch.long, device=self.device)
        reverify_capped_reported: list[set[int]] = [set() for _ in range(batch_size)]  # positions already reported as hitting the M_remask (max_reverify_times) cap, so each is printed only once per sample

        # Per-sample completion tracking
        done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # Build attention_bias (zero) to prevent causal masking when attention_mask is provided
        attn_bias_for_fwd = None
        attn_mask_for_fwd = None
        position_ids_for_fwd = None
        if attention_mask is not None:
            # attention_mask from tokenizer has shape (B, prompt_len) -- only covers prompt.
            # Extend to (B, total_length) by appending 1s for the generation region,
            # so the model correctly ignores padding positions but attends to generated tokens.
            gen_ones = torch.ones(
                (batch_size, gen_length), dtype=attention_mask.dtype, device=self.device
            )
            attn_mask_for_fwd = torch.cat([attention_mask, gen_ones], dim=1)  # (B, total_length)

            # Build per-sample position_ids from the mask.
            # For left-padded inputs, this ensures real tokens start at position 0
            # regardless of padding length, giving correct RoPE positions.
            position_ids_for_fwd = attn_mask_for_fwd.long().cumsum(dim=1) - 1
            position_ids_for_fwd = position_ids_for_fwd.clamp(min=0)  # (B, total_length)

            # If the extended mask is all 1s (no padding), skip masking entirely
            # so the model uses the same code path as batch_size=1 (pure bidirectional attn).
            if (attn_mask_for_fwd == 1).all():
                attn_mask_for_fwd = None
                position_ids_for_fwd = None
            else:
                # Zero attention_bias prevents causal masking; the attention_mask handles padding.
                attn_bias_for_fwd = torch.zeros(
                    (1, 1, total_length, total_length),
                    dtype=torch.float,
                    device=self.device,
                )

        num_blocks = (gen_length + block_length - 1) // block_length
        if num_blocks <= 0:
            stats = {
                "flip_flop_count": flip_flop_count,
                "total_remask_count": total_remask_count,
                "total_unmask_count": total_unmask_count,
                "keep_count": keep_count,
                "replace_count": replace_count,
                "seed_total_count": seed_total_count,
                "deferred_remask_count": deferred_remask_count,
                "changed_after_remask_count": changed_after_remask_count,
            }
            return x, actual_steps, stats

        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * block_length
            block_end = min(prompt_len + (block_idx + 1) * block_length, total_length)
            if block_end <= block_start:
                break

            block_slice = slice(block_start, block_end)
            block_len = block_end - block_start
            block_mask_buf = torch.empty(
                (batch_size, block_len),
                device=self.device,
                dtype=torch.bool,
            )
            masked_after_draft_blk = torch.empty(
                (batch_size, block_len),
                device=self.device,
                dtype=torch.bool,
            )

            t = 0
            max_steps_per_block = block_len * 2  # Safety limit
            # Loop while there are still MASK positions in this block for any non-done sample
            active_mask = ~done
            block_mask_count = torch.zeros(batch_size, device=self.device, dtype=torch.long)
            for b in range(batch_size):
                if not done[b]:
                    block_mask_count[b] = torch.count_nonzero(x[b, block_start:block_end] == self.mask_id)
            while (block_mask_count[active_mask].sum() > 0 if active_mask.any() else False) and t < max_steps_per_block:
                # Positions that are already unmasked BEFORE this forward (prompt + previously unmasked tokens).
                unmasked_before_forward = (x != self.mask_id)
                new_positions_topk_list = [None] * batch_size
                new_positions_mask_list = [None] * batch_size
                masked_after_draft_blk.zero_()

                # === Step t+1: apply cached seeds from step t (per-sample) ===
                seed_positions_tensor_list: list[torch.Tensor | None] = [None] * batch_size
                seed_old_ids_list: list[torch.Tensor | None] = [None] * batch_size

                # Build per-layer KV override lists (length B) for the attention blocks
                need_kv_override = False
                seed_pos_per_layer: list[list[torch.Tensor | None]] | None = None
                seed_k_per_layer: list[list[torch.Tensor | None]] | None = None
                seed_v_per_layer: list[list[torch.Tensor | None]] | None = None

                for b in range(batch_size):
                    if done[b]:
                        continue
                    rev_pos = reverify_positions_list[b]
                    if rev_pos is None:
                        continue

                    in_block_mask = (rev_pos >= block_start) & (rev_pos < block_end)
                    seed_pos_b = rev_pos[in_block_mask]

                    if seed_pos_b.numel() > 0:
                        seed_positions_tensor_list[b] = seed_pos_b

                        if use_kv_cache_for_reverify:
                            seed_old_ids_list[b] = x[b, seed_pos_b].clone()
                        x[b, seed_pos_b] = self.mask_id

                        if (
                            use_kv_cache_for_reverify
                            and reverify_kv_list[b] is not None
                            and all_blocks
                        ):
                            kv_k_b, kv_v_b = reverify_kv_list[b]
                            total_seeds = kv_k_b.size(2)
                            all_kv_indices = torch.arange(total_seeds, device=self.device)
                            kv_indices = all_kv_indices[in_block_mask]

                            if seed_pos_per_layer is None:
                                num_layers_kv = kv_k_b.size(0)
                                seed_pos_per_layer = [[None] * batch_size for _ in range(num_layers_kv)]
                                seed_k_per_layer = [[None] * batch_size for _ in range(num_layers_kv)]
                                seed_v_per_layer = [[None] * batch_size for _ in range(num_layers_kv)]

                            for layer_idx in range(min(len(all_blocks), kv_k_b.size(0))):
                                seed_pos_per_layer[layer_idx][b] = seed_pos_b
                                seed_k_per_layer[layer_idx][b] = kv_k_b[layer_idx, :, kv_indices, :]
                                seed_v_per_layer[layer_idx][b] = kv_v_b[layer_idx, :, kv_indices, :]
                            need_kv_override = True

                        elif not use_kv_cache_for_reverify:
                            reverify_count[b].index_add_(
                                0,
                                seed_pos_b,
                                torch.ones_like(seed_pos_b),
                            )

                    reverify_positions_list[b] = None
                    reverify_kv_list[b] = None

                # Set per-layer KV overrides on blocks (list of length B per layer)
                if need_kv_override and seed_pos_per_layer is not None:
                    for layer_idx, block in enumerate(all_blocks):
                        if layer_idx >= len(seed_pos_per_layer):
                            break
                        block._seed_kv_positions = seed_pos_per_layer[layer_idx]
                        block._seed_kv_key = seed_k_per_layer[layer_idx]
                        block._seed_kv_value = seed_v_per_layer[layer_idx]

                actual_steps += 1

                need_attn = bool(use_attention_score)

                if use_kv_cache_for_reverify and all_blocks:
                    for block in all_blocks:
                        block._save_kv_for_seed = True

                # === Full forward ===
                fwd_kwargs = dict(
                    input_ids=x,
                    output_attentions=False,
                    output_attentions_last_only=need_attn,
                    use_cache=False,
                )
                if attn_mask_for_fwd is not None:
                    fwd_kwargs["attention_mask"] = attn_mask_for_fwd
                    fwd_kwargs["attention_bias"] = attn_bias_for_fwd
                if position_ids_for_fwd is not None:
                    fwd_kwargs["position_ids"] = position_ids_for_fwd
                outputs = self.model(**fwd_kwargs)
                logits = outputs.logits
                logits_blk = logits[:, block_slice, :]

                probs_blk = F.softmax(logits_blk, dim=-1)
                conf_blk, pred_blk = torch.max(probs_blk, dim=-1)

                last_layer_attention = None
                if need_attn and getattr(outputs, "attentions", None) is not None and len(outputs.attentions) > 0:
                    last_layer_attention = outputs.attentions[-1]  # (B, heads, L, L)

                # Clear per-layer KV overrides (avoid leaking into future steps)
                if use_kv_cache_for_reverify and all_blocks:
                    any_had_seeds = any(s is not None for s in seed_positions_tensor_list)
                    if any_had_seeds:
                        for block in all_blocks:
                            block._seed_kv_positions = None
                            block._seed_kv_key = None
                            block._seed_kv_value = None

                # Per-sample remasked positions for excluding from drafting
                remasked_pos_list: list[torch.Tensor | None] = [None] * batch_size

                # === Per-sample verification: keep/replace/remask (three-way) ===
                for b in range(batch_size):
                    if done[b]:
                        continue

                    seed_positions_b = seed_positions_tensor_list[b]
                    seed_old_ids_b = seed_old_ids_list[b]

                    if not (
                        use_kv_cache_for_reverify
                        and seed_positions_b is not None
                        and seed_positions_b.numel() > 0
                        and seed_old_ids_b is not None
                    ):
                        continue

                    seed_total_count[b] += int(seed_positions_b.numel())

                    seed_positions_local = seed_positions_b - block_start
                    seed_new_ids = pred_blk[b, seed_positions_local]

                    p_new = probs_blk[b, seed_positions_local, seed_new_ids].to(torch.float32)

                    # Three-way revision: Keep / Replace / ReMask
                    same_mask = seed_new_ids.eq(seed_old_ids_b)
                    low_conf_mask = p_new <= tau_draft
                    remask_mask = (~same_mask) & low_conf_mask
                    replace_mask_b = (~same_mask) & (~low_conf_mask)

                    keep_pos = seed_positions_b[same_mask]
                    keep_ids = seed_old_ids_b[same_mask]
                    keep_p = p_new[same_mask].to(current_confidences.dtype)

                    replace_pos = seed_positions_b[replace_mask_b]
                    replace_ids = seed_new_ids[replace_mask_b]
                    replace_p = p_new[replace_mask_b].to(current_confidences.dtype)

                    remask_pos = seed_positions_b[remask_mask]
                    remask_old_ids_b = seed_old_ids_b[remask_mask]
                    remask_p = p_new[remask_mask].to(current_confidences.dtype)

                    if remask_pos.numel() > 0:
                        sort_idx = torch.argsort(remask_p)
                        remask_pos = remask_pos.index_select(0, sort_idx)
                        remask_old_ids_b = remask_old_ids_b.index_select(0, sort_idx)
                        remask_p = remask_p.index_select(0, sort_idx)

                    # Preview drafting for this sample
                    torch.eq(x[b:b+1, block_start:block_end], self.mask_id, out=block_mask_buf[b:b+1])
                    temp_block_mask_b = block_mask_buf[b]
                    high_conf_preview = (conf_blk[b] > tau_draft) & temp_block_mask_b
                    will_unmask_cnt = int(high_conf_preview.sum().item())
                    if will_unmask_cnt == 0 and temp_block_mask_b.sum() > 0:
                        will_unmask_cnt = 1
                    if max_unmask_per_step > 0:
                        will_unmask_cnt = min(will_unmask_cnt, max_unmask_per_step)

                    max_remask_allowed = max(0, will_unmask_cnt - 1)
                    remask_cnt = int(remask_pos.numel())
                    actual_remask_cnt = min(remask_cnt, max_remask_allowed)

                    if keep_pos.numel() > 0:
                        reverify_count[b].index_add_(0, keep_pos, torch.ones_like(keep_pos))
                        x[b, keep_pos] = keep_ids
                        current_confidences[b, keep_pos] = keep_p
                        keep_count[b] += int(keep_pos.numel())

                    if replace_pos.numel() > 0:
                        reverify_count[b].index_add_(0, replace_pos, torch.ones_like(replace_pos))
                        x[b, replace_pos] = replace_ids
                        current_confidences[b, replace_pos] = replace_p
                        replace_count[b] += int(replace_pos.numel())

                    if remask_pos.numel() > 0:
                        reverify_count[b].index_add_(0, remask_pos, torch.ones_like(remask_pos))

                        if actual_remask_cnt > 0:
                            remasked_pos_b = remask_pos[:actual_remask_cnt]
                            remasked_old_ids_step = remask_old_ids_b[:actual_remask_cnt]
                            x[b, remasked_pos_b] = self.mask_id
                            current_confidences[b, remasked_pos_b] = 0.0
                            remasked_pos_list[b] = remasked_pos_b
                            total_remask_count[b] += actual_remask_cnt
                            for pos, old_id in zip(remasked_pos_b.tolist(), remasked_old_ids_step.tolist()):
                                hist = position_history[b].get(pos)
                                if hist is None:
                                    position_history[b][pos] = {
                                        "last_token": int(old_id),
                                        "was_remasked": True,
                                    }
                                else:
                                    hist["last_token"] = int(old_id)
                                    hist["was_remasked"] = True

                        if remask_cnt > actual_remask_cnt:
                            keep_pos2 = remask_pos[actual_remask_cnt:]
                            keep_ids2 = remask_old_ids_b[actual_remask_cnt:]
                            keep_p2 = remask_p[actual_remask_cnt:]
                            x[b, keep_pos2] = keep_ids2
                            current_confidences[b, keep_pos2] = keep_p2
                            deferred_remask_count[b] += int(remask_cnt - actual_remask_cnt)

                # Sampling (optional)
                if temperature > 0:
                    noisy_probs_blk = add_gumbel_noise(logits_blk, temperature)
                    predictions_blk = torch.argmax(noisy_probs_blk, dim=-1)
                else:
                    predictions_blk = pred_blk

                # Recompute mask after verification may have ReMasked some positions
                torch.eq(x[:, block_start:block_end], self.mask_id, out=block_mask_buf)
                block_mask_mask_blk = block_mask_buf.clone()

                # Exclude per-sample remasked positions from drafting
                for b in range(batch_size):
                    rp_b = remasked_pos_list[b]
                    if rp_b is not None and rp_b.numel() > 0:
                        remasked_local = rp_b - block_start
                        block_mask_mask_blk[b, remasked_local] = False

                # === Drafting: Unmask positions based on confidence (batched topk) ===
                if torch.count_nonzero(block_mask_mask_blk) > 0:
                    scores = conf_blk.clone()
                    scores[~block_mask_mask_blk] = -1.0

                    k_draft = max_unmask_per_step if max_unmask_per_step > 0 else scores.size(1)
                    k_draft = min(k_draft, scores.size(1))
                    topv, topi = torch.topk(scores, k=k_draft, dim=1)
                    sel = topv > tau_draft
                    sel[:, 0] = True

                    to_unmask_blk = torch.zeros_like(block_mask_mask_blk)
                    to_unmask_blk.scatter_(1, topi, sel)

                    # Skip done samples
                    for b in range(batch_size):
                        if done[b]:
                            to_unmask_blk[b] = False

                    block_view = x[:, block_slice]
                    block_view[to_unmask_blk] = predictions_blk[to_unmask_blk]
                    conf_view = current_confidences[:, block_slice]
                    conf_view[to_unmask_blk] = conf_blk[to_unmask_blk].float()

                    # Count unmask operations and detect cross-step flip-flops.
                    for b in range(batch_size):
                        if not done[b]:
                            newly_unmasked_positions = torch.where(to_unmask_blk[b])[0] + block_start
                            total_unmask_count[b] += int(newly_unmasked_positions.numel())
                            for pos in newly_unmasked_positions.tolist():
                                local_pos = pos - block_start
                                current_token = int(predictions_blk[b, local_pos].item())
                                hist = position_history[b].get(pos)
                                if hist is not None and hist.get("was_remasked"):
                                    if hist.get("last_token") == current_token:
                                        flip_flop_count[b] += 1
                                    else:
                                        changed_after_remask_count[b] += 1
                                position_history[b][pos] = {
                                    "last_token": current_token,
                                    "was_remasked": False,
                                }

                    for b in range(batch_size):
                        if not done[b]:
                            new_positions_topk_list[b] = topi[b] + block_start
                            new_positions_mask_list[b] = sel[b]

                    masked_after_draft_blk.copy_(block_mask_mask_blk)
                    masked_after_draft_blk.logical_and_(~to_unmask_blk)

                # === Per-sample EOS handling ===
                if eos_token_id is not None:
                    for b in range(batch_size):
                        if done[b]:
                            continue
                        response_tokens = x[b, prompt_len:block_end]
                        eos_mask = response_tokens == eos_token_id
                        if torch.count_nonzero(eos_mask) > 0:
                            first_eos_rel = torch.argmax(eos_mask.to(torch.int32)).item()
                            first_eos_abs = prompt_len + first_eos_rel
                            all_unmasked_before_eos = (x[b, prompt_len:first_eos_abs] != self.mask_id).all().item()
                            if all_unmasked_before_eos:
                                done[b] = True

                    # If all done, exit loop early
                    if done.all():
                        if use_kv_cache_for_reverify and all_blocks:
                            for block in all_blocks:
                                block._save_kv_for_seed = False
                                block._last_k = None
                                block._last_v = None
                        stats = {
                            "flip_flop_count": flip_flop_count,
                            "total_remask_count": total_remask_count,
                            "total_unmask_count": total_unmask_count,
                            "keep_count": keep_count,
                            "replace_count": replace_count,
                            "seed_total_count": seed_total_count,
                            "deferred_remask_count": deferred_remask_count,
                            "changed_after_remask_count": changed_after_remask_count,
                        }
                        return x, actual_steps, stats

                # === Step t: schedule low-confidence positions as seeds for step t+1 ===
                if max_reverify_times > 0:
                    for b in range(batch_size):
                        if done[b]:
                            continue

                        block_positions = torch.arange(block_start, block_end, device=self.device)
                        block_tokens = x[b, block_start:block_end]
                        block_positions_local = torch.arange(block_end - block_start, device=self.device)
                        p_self_block = probs_blk[b, block_positions_local, block_tokens]

                        eligible_base = (
                            unmasked_before_forward[b, block_positions]
                            & (reverify_count[b, block_positions] < max_reverify_times)
                        )

                        # A position hits the M_remask cap when it is unmasked but reverify_count has reached max_reverify_times, so it can no longer be re-verified; print each such position once per sample.
                        # capped = block_positions[unmasked_before_forward[b, block_positions] & (reverify_count[b, block_positions] >= max_reverify_times)]
                        # newly_capped = [int(p) for p in capped.tolist() if int(p) not in reverify_capped_reported[b]]
                        # if newly_capped:
                        #     reverify_capped_reported[b].update(newly_capped)
                        #     print(f"[cover] M_remask cap ({max_reverify_times}) hit at positions {newly_capped} (sample {b}, step {actual_steps})", flush=True)

                        seed_positions_b = seed_positions_tensor_list[b]
                        if seed_positions_b is not None and seed_positions_b.numel() > 0:
                            seed_mask_total = torch.zeros((total_length,), dtype=torch.bool, device=self.device)
                            seed_mask_total[seed_positions_b] = True
                            eligible_base = eligible_base & (~seed_mask_total[block_positions])

                        selected_seed_pos = None

                        if torch.count_nonzero(eligible_base) > 0:
                            # Stability-aware seed selection with attention-based scoring
                            if need_attn and last_layer_attention is not None:
                                eps = 1e-6
                                attn = last_layer_attention[b]  # (heads, Q, K)
                                eligible_pos = block_positions[eligible_base]

                                p_self = p_self_block[eligible_base].clamp(min=1e-9)
                                u = (-torch.log(p_self)).to(torch.float32)

                                eligible_pos_q = eligible_pos

                                new_positions_topk = new_positions_topk_list[b]
                                new_positions_mask = new_positions_mask_list[b]

                                if new_positions_topk is not None:
                                    attn_seed_to_new = (
                                        attn.index_select(1, eligible_pos_q).index_select(2, new_positions_topk)
                                    )
                                    new_sel = new_positions_mask.to(attn_seed_to_new.dtype).view(1, 1, -1)
                                    d_out = (attn_seed_to_new * new_sel).to(torch.float32).sum(dim=2).mean(dim=0)
                                else:
                                    d_out = torch.zeros((eligible_pos.numel(),), device=self.device, dtype=torch.float32)

                                block_positions_q = block_positions
                                attn_block_queries = attn.index_select(1, block_positions_q)
                                attn_masked_to_eligible = attn_block_queries.index_select(2, eligible_pos).to(torch.float32)
                                mask_block = masked_after_draft_blk[b].to(attn_masked_to_eligible.dtype)
                                d_in = (
                                    attn_masked_to_eligible * mask_block.view(1, -1, 1)
                                ).sum(dim=1).mean(dim=0)

                                score = u * (1+d_in)/(1+d_out)

                                # Adaptive revision rate via CDF
                                selected_seed_pos = select_topk_positions_by_score_cdf(eligible_pos, score)
                            else:
                                eligible = eligible_base & (p_self_block < tau_draft)
                                if torch.count_nonzero(eligible) > 0:
                                    seed_pos = block_positions[eligible]
                                    seed_p = p_self_block[eligible].clamp(min=1e-9)
                                    score = (-torch.log(seed_p)).to(torch.float32)

                                    # Adaptive revision rate via CDF
                                    selected_seed_pos = select_topk_positions_by_score_cdf(seed_pos, score)

                        if selected_seed_pos is not None and selected_seed_pos.numel() > 0:
                            reverify_positions_list[b] = selected_seed_pos

                            if use_kv_cache_for_reverify and all_blocks:
                                first_block_with_kv = next(
                                    (block for block in all_blocks if getattr(block, "_last_k", None) is not None),
                                    None,
                                )
                                if first_block_with_kv is not None:
                                    k0 = first_block_with_kv._last_k
                                    v0 = first_block_with_kv._last_v
                                    _, n_heads, seq_len_kv, head_dim = k0.shape
                                    num_layers = len(all_blocks)
                                    num_selected = selected_seed_pos.numel()

                                    kv_k_b = torch.empty(
                                        (num_layers, n_heads, num_selected, head_dim),
                                        dtype=k0.dtype, device=k0.device
                                    )
                                    kv_v_b = torch.empty(
                                        (num_layers, n_heads, num_selected, head_dim),
                                        dtype=v0.dtype, device=v0.device
                                    )
                                    for layer_idx, block in enumerate(all_blocks):
                                        k_all = getattr(block, "_last_k", None)
                                        v_all = getattr(block, "_last_v", None)
                                        if k_all is None or v_all is None:
                                            continue
                                        kv_k_b[layer_idx] = k_all[b].index_select(-2, selected_seed_pos)
                                        kv_v_b[layer_idx] = v_all[b].index_select(-2, selected_seed_pos)
                                    reverify_kv_list[b] = (kv_k_b, kv_v_b)

                if use_kv_cache_for_reverify and all_blocks:
                    for block in all_blocks:
                        block._save_kv_for_seed = False
                        block._last_k = None
                        block._last_v = None

                # Update mask count for while loop condition
                active_mask = ~done
                block_mask_count.zero_()
                for b in range(batch_size):
                    if not done[b]:
                        block_mask_count[b] = torch.count_nonzero(x[b, block_start:block_end] == self.mask_id)

                t += 1

        stats = {
            "flip_flop_count": flip_flop_count,
            "total_remask_count": total_remask_count,
            "total_unmask_count": total_unmask_count,
            "keep_count": keep_count,
            "replace_count": replace_count,
            "seed_total_count": seed_total_count,
            "deferred_remask_count": deferred_remask_count,
            "changed_after_remask_count": changed_after_remask_count,
        }
        return x, actual_steps, stats

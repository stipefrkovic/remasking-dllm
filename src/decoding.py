import torch
import torch.nn.functional as F
import numpy as np

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


@torch.no_grad()
def decoding_wino(model, prompt, gen_length=256, block_length=32, temperature=0., mask_id=126336, threshold=0.6, threshold_back=0.9, max_remask=None, logit_shift=False, equal_shadow_remask=False, disable_max_accept=True, debug_prob=False, use_shadow=None, shadow_diagonal=False, extra_ff_stats=False, s=0, t=0, remask_sample=False, remask_temperature=1.0):
    # remask_sample: positions currently masked because they were remasked are re-predicted by sampling at
    # remask_temperature (default 1.0) instead of greedily, until they are accepted again. Applies to expansion
    # (s/t) neighbors as well as seeds; untouched positions keep the global `temperature`.

    if s > 0 or t > 0:
        assert max_remask is not None, "block remasking (s>0 or t>0) requires max_remask to bound per-position seed remasks and guard the unmasking loop"
    assert not (s > 0 and t > 0), "spatial (s) and temporal (t) block remasking are mutually exclusive"
    device = model.device
    use_shadow = (equal_shadow_remask or threshold_back > 0) if use_shadow is None else use_shadow
    print(f"[decoding_wino] use_shadow={use_shadow}")
    x_block = torch.full((1, prompt.shape[1] + gen_length + (block_length if use_shadow else 0)), mask_id, dtype=torch.long).to(model.device)
    x_block[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    step = 0
    total_remasked_positions = 0
    total_flipflop = 0
    total_diff_token = 0
    debug_records = [] if debug_prob else None
    ff_records = [] if extra_ff_stats else None

    for num_block in range(num_blocks):
        block_step = 0
        mask_index_block = (x_block == mask_id) # b, l
        mask_index_block[:, prompt.shape[1] + (num_block + 1) * block_length:] = False

        if use_shadow:
            unmask_index_block = torch.full_like(mask_index_block, False)
            unmask_index_block[:, -block_length:] = ~mask_index_block[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length]
            shadow_start = prompt.shape[1] + num_block * block_length - 1 if logit_shift else prompt.shape[1] + num_block * block_length
            position_ids = torch.cat([torch.arange(prompt.shape[1] + gen_length, device=device), torch.arange(shadow_start, shadow_start + block_length, device=device)])
            attention_mask = torch.ones(1, 1, x_block.shape[1], x_block.shape[1], dtype=torch.bool).to(device)
            attention_mask[:, :, :, -block_length:] = False
            shadow_self = torch.eye(block_length, dtype=torch.bool, device=device) if shadow_diagonal else torch.ones(block_length, block_length, dtype=torch.bool, device=device)
            attention_mask[:, :, -block_length:, -block_length:] = shadow_self
            if logit_shift:
                block_mask = torch.eye(block_length, dtype=torch.bool, device=device)  # block main_j (independence)
                block_mask |= torch.diag(torch.ones(block_length - 1, dtype=torch.bool, device=device), diagonal=-1)  # block main_{j-1} (same pos_id as shadow_j)
                attention_mask[:, :, -block_length:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] = ~block_mask
                attention_mask[:, :, -block_length, prompt.shape[1] + num_block * block_length - 1] = False  # shadow_0 shares pos_id with last token of prev block
            else:
                attention_mask[:, :, -block_length:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] = ~torch.eye(block_length, dtype=torch.bool, device=device)
        else:
            position_ids = torch.arange(prompt.shape[1] + gen_length, device=device)
            attention_mask = torch.ones(1, 1, x_block.shape[1], x_block.shape[1], dtype=torch.bool).to(device)
        last_accept = block_length - 1
        been_remasked = torch.zeros(1, block_length, dtype=torch.long, device=device)
        pre_remask_token = torch.full((1, block_length), -1, dtype=torch.long, device=device)
        unmask_step = torch.full((1, block_length), -1, dtype=torch.long, device=device)
        remask_pending = torch.zeros(1, block_length, dtype=torch.bool, device=device)  # masked *because* it was remasked (seeds + s/t neighbors)
        flipflop_count = 0
        diff_token_count = 0
        _prev_accept_block = torch.zeros(block_length, dtype=torch.bool, device=device)
        while mask_index_block.any():
            max_accept = block_length if disable_max_accept else min(max(int(mask_index_block.sum() * 0.7), 5), 20)
            logits = model(x_block, attention_mask=attention_mask, position_ids=position_ids).logits # b, l, vocab_size
            if logit_shift:
                if use_shadow:
                    main_len = prompt.shape[1] + gen_length
                    logits = torch.cat([torch.cat([logits[:, :1], logits[:, :main_len - 1]], dim=1), logits[:, main_len:]], dim=1)
                else:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            if remask_sample and remask_pending.any():
                # re-predict remasked positions by sampling instead of greedily; confidence below still uses p(x0)
                x0_resample = torch.argmax(add_gumbel_noise(logits, temperature=remask_temperature), dim=-1)
                resample_index = torch.zeros_like(x0, dtype=torch.bool)
                resample_index[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] = remask_pending
                x0 = torch.where(resample_index, x0_resample, x0)
            if use_shadow:
                unmask_index_block_shift_left = torch.zeros_like(unmask_index_block)
                unmask_index_block_shift_left[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] = unmask_index_block[:, -block_length:]
                x0[unmask_index_block] = x_block[unmask_index_block_shift_left]

            # The float64 softmax is only ever read at two places: the current block (via
            # mask_index_block -> confidence) and the shadow tail (via unmask_index_block ->
            # confidence_back), plus the equal_shadow_remask / debug_prob paths, which read those
            # same two slices. Over the full sequence each float64 temporary is ~700 MB at
            # gen_length 256; restricting it to 2*block_length positions is ~10x smaller and
            # arithmetically identical. p_sel is [block | shadow] concatenated along dim 1.
            _b0 = prompt.shape[1] + num_block * block_length
            _b1 = prompt.shape[1] + (num_block + 1) * block_length
            _sel = torch.cat([logits[:, _b0:_b1], logits[:, -block_length:]], dim=1) if use_shadow \
                else logits[:, _b0:_b1]
            p_sel = F.softmax(_sel.to(torch.float64), dim=-1)  # b, block(+shadow), vocab_size
            del _sel

            # 0 outside the two slices, which is never read: confidence is -inf there and
            # confidence_back is +inf there.
            x0_p = torch.zeros_like(x0, dtype=torch.float64)  # b, l
            x0_p[:, _b0:_b1] = p_sel[:, :block_length].gather(
                -1, x0[:, _b0:_b1].unsqueeze(-1)).squeeze(-1)
            if use_shadow:
                x0_p[:, -block_length:] = p_sel[:, block_length:].gather(
                    -1, x0[:, -block_length:].unsqueeze(-1)).squeeze(-1)
            x0 = torch.where(mask_index_block, x0, x_block) # replace the masked tokens with the predicted tokens
            confidence = torch.where(mask_index_block, x0_p, -np.inf) # keep the confidence of the masked tokens
            if use_shadow:
                confidence_back = torch.where(unmask_index_block, x0_p, np.inf)

            transfer_index = confidence > threshold
            if max_remask is None and transfer_index.sum() > max_accept:
                _, indices = torch.topk(confidence, k=max_accept, largest=True)
                transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
                transfer_index.view(-1)[indices] = True
            elif not transfer_index.any():
                if temperature > 0:
                    noisy_scores = logits_with_noise.max(dim=-1).values
                    noisy_scores = torch.where(mask_index_block, noisy_scores, torch.tensor(-np.inf, device=device))
                    max_confidence_index = torch.argmax(noisy_scores)
                else:
                    max_confidence_index = torch.argmax(confidence)
                transfer_index.view(-1)[max_confidence_index] = True

            # Check re-accepted positions: did the token change from pre-remask?
            transfer_block = transfer_index[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length]

            was_remasked = pre_remask_token >= 0
            reaccepted = transfer_block & was_remasked
            if reaccepted.any():
                new_tokens = x0[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length]
                same = (new_tokens == pre_remask_token) & reaccepted
                flipflop_count += same.sum().item()
                diff_token_count += (reaccepted & ~same).sum().item()
                if extra_ff_stats:
                    for _pos in reaccepted[0].nonzero(as_tuple=True)[0].tolist():
                        ff_records.append({
                            'block_idx': num_block,
                            'block_pos': _pos,
                            'num_remasks': been_remasked[0, _pos].item(),
                            'is_flipflop': same[0, _pos].item(),
                        })
                pre_remask_token[reaccepted] = -1

            x_block[transfer_index] = x0[transfer_index]
            unmask_step[transfer_block] = block_step
            remask_pending[transfer_block] = False  # accepted again -> back to greedy unless remasked once more

            num_accept = transfer_index.sum()

            if debug_prob and use_shadow:
                _bstart = prompt.shape[1] + num_block * block_length
                _shad_p = p_sel[0, block_length:].float()
                _transfer_block = transfer_index[0, _bstart:_bstart + block_length]

                _accepted_positions = _transfer_block.nonzero(as_tuple=True)[0].tolist()
                _, _, _pos_probs = _forward(model, x_block.clone(), attention_mask, position_ids,
                                            _bstart, _bstart + block_length, mask_id, positions=_accepted_positions)

                _main_accept = []
                for _j in _accepted_positions:
                    _top5 = torch.topk(_pos_probs[_j], k=5)
                    _main_accept.append({
                        "block_pos": _j,
                        "main_top5": list(zip(_top5.indices.tolist(), [round(v, 4) for v in _top5.values.tolist()])),
                        "accepted_token": x_block[0, _bstart + _j].item(),
                    })

                _shadow_check = []
                _native_check = []
                _main_p = p_sel[0, :block_length].float()
                for _j in _prev_accept_block.nonzero(as_tuple=True)[0].tolist():
                    _sp = _shad_p[_j]
                    _top5s = torch.topk(_sp, k=5)
                    _shadow_check.append({
                        "block_pos": _j,
                        "shadow_top5": list(zip(_top5s.indices.tolist(), [round(v, 4) for v in _top5s.values.tolist()])),
                    })
                    _np = _main_p[_j]
                    _top5n = torch.topk(_np, k=5)
                    _native_check.append({
                        "block_pos": _j,
                        "native_top5": list(zip(_top5n.indices.tolist(), [round(v, 4) for v in _top5n.values.tolist()])),
                    })

                debug_records.append({
                    "block": num_block,
                    "step": block_step,
                    "prompt_length": prompt.shape[1],
                    "main_accept": _main_accept,
                    "shadow_check": _shadow_check,
                    "native_check": _native_check,
                })
                _prev_accept_block = _transfer_block.clone()

            if num_accept > 1 and use_shadow:
                if equal_shadow_remask:
                    shadow_preds = p_sel[:, block_length:].argmax(dim=-1)
                    main_tokens = x_block[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length].clone()
                    shadow_active = unmask_index_block[:, -block_length:]
                    disagree = (shadow_preds != main_tokens) & shadow_active
                    if max_remask is not None:
                        disagree &= (been_remasked < max_remask)
                    remask_index = torch.zeros_like(x_block, dtype=torch.bool)
                    remask_index[:, -block_length:] = disagree
                else:
                    remask_index = confidence_back < threshold_back
                    if max_remask is not None:
                        remask_index[:, -block_length:] &= (been_remasked < max_remask)

                if max_remask is None and remask_index.sum() >= last_accept:
                    num_remask = last_accept - 1
                    confidence_flat = confidence_back.view(-1)
                    temp_mask = torch.zeros_like(confidence_flat, dtype=torch.bool)
                    _, indices = torch.topk(confidence_flat, k=num_remask, largest=False)
                    temp_mask[indices] = True
                    remask_index = temp_mask.view(confidence_back.shape)

                seed_remask_block = remask_index[:, -block_length:].clone()
                if (s > 0 or t > 0) and seed_remask_block.any():
                    expanded = seed_remask_block.clone()
                    if s > 0:
                        block_rm = seed_remask_block.float().unsqueeze(1)
                        block_rm = F.max_pool1d(block_rm, kernel_size=2 * s + 1, stride=1, padding=s)
                        expanded |= block_rm.squeeze(1).bool()
                    if t > 0:
                        seed_steps = unmask_step[seed_remask_block]
                        for u in torch.unique(seed_steps).tolist():
                            expanded |= (unmask_step >= u) & (unmask_step < u + t)
                    remask_index[:, -block_length:] = expanded & unmask_index_block[:, -block_length:]
            else:
                remask_index = torch.zeros_like(transfer_index)
                seed_remask_block = remask_index[:, -block_length:]

            # Only seed remasks count toward flipflop/diff_token stats; expansion (s>0 / t>0) neighbors are excluded
            pre_remask_token[seed_remask_block] = x_block[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length][seed_remask_block]
            been_remasked += seed_remask_block.long()
            unmask_step[remask_index[:, -block_length:]] = -1
            remask_pending |= remask_index[:, -block_length:]

            remask_index_shift = torch.zeros_like(remask_index)
            remask_index_shift[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] = remask_index[:, -block_length:]
            x_block[remask_index_shift] = mask_id
            mask_index_block[transfer_index] = False
            mask_index_block[remask_index_shift] = True
            block_step += 1
            if use_shadow:
                transfer_index_shift = torch.zeros_like(transfer_index)
                transfer_index_shift[:, -block_length:] = transfer_index[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length]
                unmask_index_block[transfer_index_shift] = True
                unmask_index_block[remask_index] = False
            last_accept = num_accept

        total_remasked_positions += (been_remasked > 0).sum().item()
        total_flipflop += flipflop_count
        total_diff_token += diff_token_count
        step += block_step

    if debug_prob and debug_records is not None:
        import json
        append_json = "_diag" if shadow_diagonal else "_full"
        append_json += f"_{block_length}"
        append_json += f"_t{f'{threshold:g}'.replace('.', '')}"
        with open(f"debug_prob{append_json}.json", "a") as _f:
            for rec in debug_records:
                _f.write(json.dumps(rec) + "\n")

    return x_block[:, :prompt.shape[1] + gen_length], step, {
        'remasked_positions': total_remasked_positions,
        'reaccept_flipflop': total_flipflop, 'reaccept_diff_token': total_diff_token,
        'ff_records': ff_records,
    }


@torch.no_grad()
def _forward(model, x, attention_mask, position_ids, block_start, block_end, mask_id, native=False, logits=None, positions=None, return_argmax=False):  # logits required when native=True; positions is a list/tensor of block-relative indices for selective mode
    """Compute oracle confidence for accepted tokens in [block_start, block_end).

    native=False (default): mask each accepted token individually and run the model once per token.
    native=True: read confidence directly from pre-computed logits (0 extra NFEs; logits must be provided).
    positions (list/tensor of block-relative indices): only evaluate the specified positions, skip all others.
        Returns (token_conf, nfe, pos_probs) where pos_probs is (block_len, vocab_size) with full distributions
        at queried positions and zeros elsewhere.
    return_argmax: also return token_argmax, the oracle's top-1 token id per position (-1 where the position
        was still masked and hence not evaluated). Free -- the softmax is already computed. Ignored in
        positions mode, whose pos_probs already carries the full distributions.

    Returns (token_conf, nfe) for native/default modes, plus token_argmax when return_argmax;
    (token_conf, nfe, pos_probs) for positions mode.
    """
    block_len = block_end - block_start
    token_conf = torch.full((block_len,), 2.0, dtype=torch.float64, device=x.device)
    token_argmax = torch.full((block_len,), -1, dtype=torch.long, device=x.device)

    if native:
        p = F.softmax(logits[0, block_start:block_end].to(torch.float64), dim=-1)
        for j in range(block_len):
            tok = x[0, block_start + j].item()
            if tok != mask_id:
                token_conf[j] = p[j, tok]
                token_argmax[j] = torch.argmax(p[j])
        return (token_conf, 0, token_argmax) if return_argmax else (token_conf, 0)

    if positions is not None:
        vocab_size = model.config.vocab_size
        pos_probs = torch.zeros(block_len, vocab_size, dtype=torch.float64, device=x.device)
        nfe = 0
        for j in positions:
            j = int(j)
            if x[0, block_start + j] != mask_id:
                original = x[0, block_start + j].item()
                x[0, block_start + j] = mask_id
                out_logits = model(x, attention_mask=attention_mask, position_ids=position_ids).logits
                nfe += 1
                p_j = F.softmax(out_logits[0, block_start + j].to(torch.float64), dim=-1)
                token_conf[j] = p_j[original]
                pos_probs[j] = p_j
                x[0, block_start + j] = original
            # else: still masked — token_conf stays 2.0, pos_probs row stays zero
        return token_conf, nfe, pos_probs

    nfe = 0
    for j in range(block_len):
        if x[0, block_start + j] != mask_id:  # unmasked: mask it out and run model
            original = x[0, block_start + j].item()
            x[0, block_start + j] = mask_id
            logits = model(x, attention_mask=attention_mask, position_ids=position_ids).logits
            nfe += 1
            p_j = F.softmax(logits[0, block_start + j].to(torch.float64), dim=-1)
            token_conf[j] = p_j[original]
            token_argmax[j] = torch.argmax(p_j)
            x[0, block_start + j] = original
        # else: still masked — token_conf stays 2.0, token_argmax stays -1

    return (token_conf, nfe, token_argmax) if return_argmax else (token_conf, nfe)


@torch.no_grad()
def decoding_faithful(model, prompt, gen_length=256, block_length=32, temperature=0., mask_id=126336, threshold=0.6, threshold_back=0.9, max_remask=None, disable_max_accept=True, use_shadow=None, shadow_diagonal=False, native=False, equal_oracle_remask=False):
    # equal_oracle_remask: remask a position when the oracle's top-1 token disagrees with the token actually
    # sitting there, instead of when its oracle confidence falls below threshold_back. Oracle counterpart of
    # decoding_wino's equal_shadow_remask; threshold_back is then unused except by the top-k overflow fallback.

    device = model.device
    use_shadow = (equal_oracle_remask or threshold_back > 0) if use_shadow is None else use_shadow
    print(f"[decoding_faithful] use_shadow={use_shadow} equal_oracle_remask={equal_oracle_remask}")
    x_block = torch.full((1, prompt.shape[1] + gen_length + (block_length if use_shadow else 0)), mask_id, dtype=torch.long).to(model.device)
    x_block[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    step = 0
    total_oracle_nfe = 0
    total_remasked_positions = 0
    total_flipflop = 0
    total_diff_token = 0

    for num_block in range(num_blocks):
        block_step = 0
        mask_index_block = (x_block == mask_id) # b, l
        mask_index_block[:, prompt.shape[1] + (num_block + 1) * block_length:] = False

        if use_shadow:
            unmask_index_block = torch.full_like(mask_index_block, False)
            unmask_index_block[:, -block_length:] = ~mask_index_block[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length]
            shadow_start = prompt.shape[1] + num_block * block_length
            position_ids = torch.cat([torch.arange(prompt.shape[1] + gen_length, device=device), torch.arange(shadow_start, shadow_start + block_length, device=device)])
            attention_mask = torch.ones(1, 1, x_block.shape[1], x_block.shape[1], dtype=torch.bool).to(device)
            attention_mask[:, :, :, -block_length:] = False
            shadow_self = torch.eye(block_length, dtype=torch.bool, device=device) if shadow_diagonal else torch.ones(block_length, block_length, dtype=torch.bool, device=device)
            attention_mask[:, :, -block_length:, -block_length:] = shadow_self
            attention_mask[:, :, -block_length:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] = ~torch.eye(block_length, dtype=torch.bool, device=device)
        else:
            position_ids = torch.arange(prompt.shape[1] + gen_length, device=device)
            attention_mask = torch.ones(1, 1, x_block.shape[1], x_block.shape[1], dtype=torch.bool).to(device)
        last_accept = block_length - 1
        been_remasked = torch.zeros(1, block_length, dtype=torch.long, device=device)
        pre_remask_token = torch.full((1, block_length), -1, dtype=torch.long, device=device)
        flipflop_count = 0
        diff_token_count = 0
        while mask_index_block.any():
            max_accept = block_length if disable_max_accept else min(max(int(mask_index_block.sum() * 0.7), 5), 20)
            logits = model(x_block, attention_mask=attention_mask, position_ids=position_ids).logits # b, l, vocab_size
            if use_shadow:
                _oracle_out = _forward(model, x_block.clone(), attention_mask, position_ids,
                                      prompt.shape[1] + num_block * block_length, prompt.shape[1] + (num_block + 1) * block_length, mask_id, native=native, logits=logits if native else None,
                                      return_argmax=equal_oracle_remask)
                if equal_oracle_remask:
                    token_conf, oracle_nfe, oracle_argmax = _oracle_out
                    # Block tokens as they stood when the oracle ran. Positions accepted later in this step were
                    # still masked then (argmax -1), so the equality test below skips them -- same as the
                    # threshold path, where their sentinel confidence of 2.0 never falls under threshold_back.
                    oracle_tokens = x_block[0, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length].clone()
                else:
                    token_conf, oracle_nfe = _oracle_out
                total_oracle_nfe += oracle_nfe
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            if use_shadow:
                unmask_index_block_shift_left = torch.zeros_like(unmask_index_block)
                unmask_index_block_shift_left[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] = unmask_index_block[:, -block_length:]
                x0[unmask_index_block] = x_block[unmask_index_block_shift_left]

            p = F.softmax(logits.to(torch.float64), dim=-1) # b, l, vocab_size

            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            x0 = torch.where(mask_index_block, x0, x_block) # replace the masked tokens with the predicted tokens
            confidence = torch.where(mask_index_block, x0_p, -np.inf) # keep the confidence of the masked tokens
            if use_shadow:
                confidence_back = torch.full((1, x_block.shape[1]), np.inf, dtype=torch.float64, device=device)
                print(num_blocks, block_step)
                print('>>>')
                print(confidence_back[0, -block_length:])
                print('<<<')
                print(token_conf)
                confidence_back[0, -block_length:] = token_conf

            transfer_index = confidence > threshold
            if max_remask is None and transfer_index.sum() > max_accept:
                _, indices = torch.topk(confidence, k=max_accept, largest=True)
                transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
                transfer_index.view(-1)[indices] = True
            elif not transfer_index.any():
                if temperature > 0:
                    noisy_scores = logits_with_noise.max(dim=-1).values
                    noisy_scores = torch.where(mask_index_block, noisy_scores, torch.tensor(-np.inf, device=device))
                    max_confidence_index = torch.argmax(noisy_scores)
                else:
                    max_confidence_index = torch.argmax(confidence)
                transfer_index.view(-1)[max_confidence_index] = True

            # Check re-accepted positions: did the token change from pre-remask?
            transfer_block = transfer_index[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length]
            was_remasked = pre_remask_token >= 0
            reaccepted = transfer_block & was_remasked
            if reaccepted.any():
                new_tokens = x0[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length]
                same = (new_tokens == pre_remask_token) & reaccepted
                flipflop_count += same.sum().item()
                diff_token_count += (reaccepted & ~same).sum().item()
                pre_remask_token[reaccepted] = -1

            x_block[transfer_index] = x0[transfer_index]

            num_accept = transfer_index.sum()

            if num_accept > 1 and use_shadow:
                if equal_oracle_remask:
                    disagree = (oracle_argmax >= 0) & (oracle_argmax != oracle_tokens)
                    remask_index = torch.zeros_like(x_block, dtype=torch.bool)
                    remask_index[:, -block_length:] = disagree
                else:
                    remask_index = confidence_back < threshold_back
                if max_remask is not None:
                    remask_index[:, -block_length:] &= (been_remasked < max_remask)

                if max_remask is None and remask_index.sum() >= last_accept:
                    num_remask = last_accept - 1
                    confidence_flat = confidence_back.view(-1)
                    temp_mask = torch.zeros_like(confidence_flat, dtype=torch.bool)
                    _, indices = torch.topk(confidence_flat, k=num_remask, largest=False)
                    temp_mask[indices] = True
                    remask_index = temp_mask.view(confidence_back.shape)
            else:
                remask_index = torch.zeros_like(transfer_index)

            pre_remask_token[remask_index[:, -block_length:]] = x_block[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length][remask_index[:, -block_length:]]
            been_remasked += remask_index[:, -block_length:].long()

            remask_index_shift = torch.zeros_like(remask_index)
            remask_index_shift[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] = remask_index[:, -block_length:]
            x_block[remask_index_shift] = mask_id
            mask_index_block[transfer_index] = False
            mask_index_block[remask_index_shift] = True
            block_step += 1
            if use_shadow:
                transfer_index_shift = torch.zeros_like(transfer_index)
                transfer_index_shift[:, -block_length:] = transfer_index[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length]
                unmask_index_block[transfer_index_shift] = True
                unmask_index_block[remask_index] = False
            last_accept = num_accept

        total_remasked_positions += (been_remasked > 0).sum().item()
        total_flipflop += flipflop_count
        total_diff_token += diff_token_count
        step += block_step

    return x_block[:, :prompt.shape[1] + gen_length], step, {
        'remasked_positions': total_remasked_positions,
        'reaccept_flipflop': total_flipflop, 'reaccept_diff_token': total_diff_token,
        'oracle_nfe': total_oracle_nfe,
    }


@torch.no_grad()
def decoding_original_saber(model, prompt, n=2, mu=2, gen_length=256, block_length=256, temperature=0., mask_id=126336, logit_shift=False, remask_scope='all_prior', drop_conf='argmax', remask='on'):
    # Reference implementation from Saber/decoding.py::generate_with_saber (logic unmodified;
    # remask stats tracking added in parallel; logit_shift added for DREAM compatibility).
    # Ablations (defaults preserve original behavior):
    #   remask_scope: 'all_prior' (orig) | 'block' (restrict remask to current block).
    #   drop_conf:   'argmax' (orig) | 'same_token' (delta tracks p[x_committed]).
    #   remask:      'on' (orig) | 'off' (no remasking; mu becomes inert).
    assert remask_scope in ('all_prior', 'block'), remask_scope
    assert drop_conf in ('argmax', 'same_token'), drop_conf
    assert remask in ('on', 'off'), remask
    step = 0
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)
    mask_index = (x == mask_id)
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    global_transfer_index = torch.zeros_like(x, dtype=torch.bool, device=x.device)
    initial_confidence = torch.zeros_like(x, dtype=torch.float32, device=x.device)
    final_confidence = torch.full_like(x, fill_value=-np.inf, dtype=torch.float32)
    last_confidence = torch.zeros_like(x, dtype=torch.float32, device=x.device)

    # Remask stats (parallel to decoding_saber's tracking; not used by reference logic).
    flipflop_count = 0
    diff_token_count = 0
    been_remasked = torch.zeros_like(x, dtype=torch.long, device=x.device)
    pre_remask_token = torch.full_like(x, fill_value=-1, dtype=torch.long, device=x.device)

    for num_block in range(num_blocks):
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_start = prompt.shape[1] + num_block * block_length
        for i in range(1024):
            step += 1
            logits = model(x).logits
            if logit_shift:  # DREAM predicts position i from logits at i-1
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            if drop_conf == 'same_token':
                # Ablation: use prob of currently-committed token for delta basis.
                # At still-masked positions x==mask_id, so p[x] is irrelevant (those positions
                # are excluded from remasking via positions_mask_now mask below).
                x_p_committed = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x, -1)), -1)
                delta_basis = torch.where(mask_index, x_p_committed, -np.inf)
            else:
                delta_basis = confidence
            confidence_delta = delta_basis - last_confidence
            mask_finite = torch.isfinite(confidence_delta)
            confidence_delta = torch.where(mask_finite, confidence_delta, torch.full_like(confidence_delta, float('inf')))
            last_confidence = delta_basis
            confidence = torch.where(global_transfer_index, -np.inf, confidence)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            fix_list = []
            for j in range(confidence.shape[0]):
                block_final_confidence = final_confidence[j, :block_end]
                generated_probs = block_final_confidence[block_final_confidence > -np.inf]
                if generated_probs.numel() > 0:
                    threshold = generated_probs.mean()
                else:
                    threshold = 1
                select_index = torch.where(confidence[j] > threshold)[0]
                if select_index.numel() < n:
                    _, select_index = torch.topk(confidence[j], k=n)

                fix_list.append(max(n, select_index.numel()))

                transfer_index[j, select_index] = True
                global_transfer_index[j, select_index] = True

            # Re-acceptance stats: positions being unmasked that had a pending pre_remask_token.
            reacc = transfer_index & (pre_remask_token >= 0)
            if reacc.any():
                same = reacc & (x0 == pre_remask_token)
                flipflop_count += int(same.sum().item())
                diff_token_count += int((reacc & ~same).sum().item())
                pre_remask_token[reacc] = -1

            x[transfer_index] = x0[transfer_index]

            new_generated_mask = transfer_index & (initial_confidence == 0.0)
            if new_generated_mask.any():
                initial_confidence = torch.where(new_generated_mask, x0_p, initial_confidence)

            final_confidence = torch.where(transfer_index, x0_p, final_confidence)

            all_masked_transferred = torch.all(global_transfer_index[mask_index]).item()

            if all_masked_transferred:
                return x, step, {
                    'remasked_positions': int((been_remasked > 0).sum().item()),
                    'reaccept_flipflop': flipflop_count, 'reaccept_diff_token': diff_token_count,
                }

            if torch.all(x[:, :block_end] != mask_id):
                break
            if remask == 'off':
                continue
            delta_for_removal = confidence_delta.clone()

            delta_for_removal = torch.where(mask_index, delta_for_removal, torch.full_like(delta_for_removal, float('inf')))
            if block_end < delta_for_removal.shape[1]:
                delta_for_removal[:, block_end:] = float('inf')
            if remask_scope == 'block' and block_start > 0:
                # Ablation: only allow remasking within current block (exclude prior blocks).
                delta_for_removal[:, :block_start] = float('inf')
            positions_mask_now = (x == mask_id)
            delta_for_removal[positions_mask_now] = float('inf')
            delta_for_removal[prompt_index] = float('inf')

            for j in range(delta_for_removal.shape[0]):
                num_to_remask = max(int(n / 2), (fix_list[j] + mu - 1) // mu)

                if num_to_remask > fix_list[j] - 1:
                    num_to_remask = fix_list[j] - 1

                if remask_scope == 'block':
                    elig = torch.nonzero(torch.isfinite(delta_for_removal[j]), as_tuple=False).squeeze(-1)
                    k = min(num_to_remask, elig.numel())
                    if k == 0:
                        continue
                    _, loc = torch.topk(delta_for_removal[j, elig], k=k, largest=False)
                    remove_index = elig[loc]
                else:
                    _, remove_index = torch.topk(delta_for_removal[j], k=num_to_remask, largest=False)
                # Stash pre-remask token + bump counter before blanking (stats only).
                pre_remask_token[j, remove_index] = x[j, remove_index]
                been_remasked[j, remove_index] += 1
                x[j, remove_index] = mask_id
                initial_confidence[j, remove_index] = 0.0
                global_transfer_index[j, remove_index] = False
    return x, step, {
        'remasked_positions': int((been_remasked > 0).sum().item()),
        'reaccept_flipflop': flipflop_count, 'reaccept_diff_token': diff_token_count,
    }


def _cover_remask_result(stats):
    """Map a COVER stats dict (LLaDA or Dream; per-sample lists, batch_size 1) onto WINO's
    remask_result shape so eval.py prints the aggregate flip-flop line for COVER through the same
    code path. A remasked position later re-accepted with the same token is a flip-flop; with a
    different token it's a diff-token re-accept; total_remask_count is COVER's remask magnitude
    (events). No per-position records are produced."""
    if not stats:
        return {}
    return {
        'remasked_positions': stats['total_remask_count'][0],
        'reaccept_flipflop': stats['flip_flop_count'][0],
        'reaccept_diff_token': stats['changed_after_remask_count'][0],
    }


@torch.no_grad()
def _decoding_cover_dream(model, prompt, tokenizer=None, gen_length=256, block_length=32,
                          temperature=0., tau_draft=0.5, max_unmask_per_step=15,
                          max_reverify_times=30, use_attention_score=True, steps=None, **kwargs):
    """COVER decoding for Dream (ported from the COVER repo's DreamGenerationMixin.diffusion_generate).

    Unlike the LLaDA path there is no standalone generator: the COVER algorithm lives in the custom
    Dream model itself (modeling_dream_cover.DreamModel, whose DreamAttention applies the seed-KV
    cache-override plus the shifted-logit off-diagonal corrections). We just drive diffusion_generate.

    The off-diagonal corrections are toggled by the DREAM_VERIFY_ENABLE_OFFDIAG env var, read at model
    init (default on). The reference's use_low_conf_reverify / use_kv_cache_for_reverify /
    max_reverify_per_step knobs are inert in the Dream generator, so they are not exposed here.
    """
    # steps must divide num_blocks (gen_length // block_length); steps == gen_length always satisfies
    # this and matches the paper's Dream script (length 256, steps 256).
    if steps is None:
        steps = gen_length
    result = model.diffusion_generate(
        inputs=prompt,
        attention_mask=None,
        max_new_tokens=gen_length,
        output_history=False,
        return_dict_in_generate=True,
        steps=steps,
        temperature=temperature,
        alg="threshold",  # COVER's threshold-based drafting/verification
        block_length=block_length,
        tau_draft=tau_draft,
        threshold=tau_draft,  # tau is used for both drafting and verification
        max_unmask_per_step=max_unmask_per_step,
        max_reverify_times=max_reverify_times,
        use_attention_score=use_attention_score,
        log_step_stats=bool(kwargs.get('debug', False)),
    )
    # _sample returns (output, steps_used, flip_flop_stats); output is a DreamModelOutput with
    # .sequences (full prompt+generation, possibly EOS-truncated) or a raw tensor.
    if isinstance(result, tuple) and len(result) == 3:
        output, steps_used, stats = result
    elif isinstance(result, tuple) and len(result) == 2:
        output, steps_used, stats = result[0], result[1], None
    else:
        output, steps_used, stats = result, steps, None
    x = output.sequences if hasattr(output, 'sequences') else output
    return x, steps_used, _cover_remask_result(stats)


@torch.no_grad()
def decoding_cover(model, prompt, tokenizer=None, gen_length=256, block_length=32,
                   temperature=0., mask_id=126336, tau_draft=None, threshold=None,
                   max_unmask_per_step=5, max_reverify_times=30,
                   use_kv_cache_for_reverify=True, use_attention_score=False, **kwargs):
    """
    COVER decoding (ported from the COVER repo's UGAGenerator.generate_causal).

    Wraps the COVER generator to match this repo's decoding interface: takes
    (model, prompt, ...) and returns (x, steps, remask_result), where remask_result
    is shaped like WINO's third element so eval.py prints the aggregate flip-flop line
    for COVER through the same code path (per-position ff_records are not produced).

    LLaDA uses the KV-cache model (modeling_llada_kv_cover.LLaDAModelLM); Dream uses the
    KV-cache Dream model and a different generation entry point, so it is dispatched to
    _decoding_cover_dream. eval.py loads the right model and sets model._cover_backend.

    Args:
        tokenizer: used only for eos_token_id early-stopping; may be None.
        tau_draft: draft/verify confidence threshold (a.k.a. `threshold` in WINO-style configs).
    """
    # Backwards-compat with WINO-style configs: `threshold` aliases `tau_draft`
    # (mirrors COVER's own inference._normalize_cover_gen_kwargs). Explicit tau_draft wins.
    if tau_draft is None:
        tau_draft = threshold if threshold is not None else 0.5

    if getattr(model, "_cover_backend", "llada") == "dream":
        return _decoding_cover_dream(
            model, prompt, tokenizer=tokenizer, gen_length=gen_length, block_length=block_length,
            temperature=temperature, tau_draft=tau_draft, max_unmask_per_step=max_unmask_per_step,
            max_reverify_times=max_reverify_times, use_attention_score=use_attention_score, **kwargs,
        )

    # Lazy import: keeps `import decoding` (for the other methods) from importing the
    # COVER model, which registers the "llada" AutoConfig/AutoModel at import time.
    from contextual_cover import UGAGenerator

    if tokenizer is None:
        tokenizer = getattr(model, "_cover_tokenizer", None)
    generator = UGAGenerator(model, tokenizer, mask_id=mask_id)
    x, steps, stats = generator.generate_causal(
        prompt,
        gen_length=gen_length,
        block_length=block_length,
        tau_draft=tau_draft,
        max_unmask_per_step=max_unmask_per_step,
        temperature=temperature,
        max_reverify_times=max_reverify_times,
        use_kv_cache_for_reverify=use_kv_cache_for_reverify,
        use_attention_score=use_attention_score,
    )
    return x, steps, _cover_remask_result(stats)

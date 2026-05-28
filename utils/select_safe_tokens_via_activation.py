import torch

from typing import List, Tuple
from jaxtyping import Float, Int
from torch import Tensor

from model_utils.model_base import ModelBase
from utils.hook_utils import add_hooks, get_activation_addition_input_pre_hook


def _logits_over_horizon(
    model,
    tokenizer,
    instructions,
    tokenize_instructions_fn,
    horizon,
    fwd_pre_hooks,
    fwd_hooks,
    batch_size,
    precomputed_continuations=None,
    do_sample=False,
    temperature=1.0,
    top_p=1.0,
    num_samples_per_prompt=1,
):
    """Variant of select_safe_tokens._logits_over_horizon where the generation
    pass for the `horizon-1` extra tokens is *itself* wrapped in `add_hooks`,
    so the supplied hooks are active during sampling as well as during the
    final teacher-forced forward.

    Used to produce a steered (refusal-aligned) reference trajectory in
    `get_safe_tokens`. Pass empty hook lists + `precomputed_continuations`
    on the second call to score those same tokens under the un-hooked model.
    """
    all_logits = None
    continuations = []

    if num_samples_per_prompt > 1:
        instructions = [p for p in instructions for _ in range(num_samples_per_prompt)]

    for bi, i in enumerate(range(0, len(instructions), batch_size)):
        batch = instructions[i:i + batch_size]

        if precomputed_continuations is not None:
            input_ids, attention_mask = precomputed_continuations[bi]
        else:
            tokenized = tokenize_instructions_fn(instructions=batch)
            input_ids = tokenized.input_ids.to(model.device)
            attention_mask = tokenized.attention_mask.to(model.device)

            if horizon > 1:
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                gen_kwargs = dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=horizon - 1,
                    min_new_tokens=horizon - 1,
                    pad_token_id=pad_id,
                    do_sample=do_sample,
                )
                if do_sample:
                    gen_kwargs["temperature"] = temperature
                    gen_kwargs["top_p"] = top_p
                with torch.no_grad():
                    with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                        input_ids = model.generate(**gen_kwargs)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(
                        (attention_mask.size(0), horizon - 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ], dim=1)

            continuations.append((input_ids, attention_mask))

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits  # (B, L_ext, V)

        horizon_logits = logits[:, -horizon:, :]  # (B, horizon, V)
        all_logits = horizon_logits if all_logits is None else torch.cat((all_logits, horizon_logits), dim=0)

    return all_logits, continuations


def _collect_banned_token_ids(tokenizer):
    banned = set(int(t) for t in (tokenizer.all_special_ids or []))
    try:
        all_tokens = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    except Exception:
        all_tokens = []
    for tid, s in enumerate(all_tokens):
        if isinstance(s, str) and "<|" in s:
            banned.add(tid)
    return banned


def _collect_eot_ids(tokenizer):
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    for tok in ("<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "<|endoftext|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid >= 0 and tid != tokenizer.unk_token_id:
                ids.add(int(tid))
        except Exception:
            pass
    return ids


def get_safe_tokens(
    model_base: ModelBase,
    instructions: List[str],
    direction: Float[Tensor, "d_model"],
    layer: int,
    coeff: float = 1.0,
    top_k: int = 100,
    batch_size: int = 32,
    horizon: int = 1,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    num_samples_per_prompt: int = 1,
    selection_method: str = "mean",
    vote_top_k_inner: int = 200,
    exclude_special_tokens: bool = True,
    min_steered_prob: float = 1e-6,
) -> Tuple[Int[Tensor, "top_k"], Float[Tensor, "top_k"], Float[Tensor, "top_k"], Float[Tensor, "top_k"]]:
    """
    Reverse-trajectory variant of safe-token selection. Same name and
    signature as `utils.select_safe_tokens.get_safe_tokens` so callers can
    swap the import without touching anything else.

    Original (`utils.select_safe_tokens.get_safe_tokens`):
        1. Generate horizon-1 continuations from the un-hooked model.
        2. Teacher-force them through (a) un-hooked and (b) hooked passes.
        3. Pick top-k by Δ = log p_steered − log p_baseline.
        Effectively asks: "on a *normal* trajectory, which tokens does the
        refusal direction push up?"  → tends to surface tokens that simply
        co-occur with the hook's perturbation on benign continuations
        (formatting, ellipses, etc., once horizon grows).

    This reverse version:
        1. Generate horizon-1 continuations from the *hooked* model — so the
           teacher-forced sequence sits on a safety-aligned trajectory ("I
           cannot help with...", "It is illegal to...", etc.).
        2. Teacher-force them through (a) hooked and (b) un-hooked passes.
        3. Pick top-k by Δ = log p_steered − log p_baseline (same sign).
        Now asks: "on the safety-aligned trajectory, which tokens does the
        refusal direction *uniquely* prefer over the un-hooked model?"  →
        the answer is the tokens that drive the safety alignment itself,
        which is what we actually want to upweight in the OPD step.

    All filtering (eot masking on the un-hooked argmax, special-token
    blacklist) and selection backends (`mean` / `vote`) are unchanged.

    Parameters mirror the original. Note that with `horizon=1` there is no
    continuation to generate, so the result is *identical* to the original
    at `horizon=1`; the reverse logic only diverges for `horizon > 1`.
    """
    fwd_pre_hooks = [(
        model_base.model_block_modules[layer],
        get_activation_addition_input_pre_hook(
            vector=direction,
            coeff=torch.tensor(coeff, dtype=direction.dtype),
        ),
    )]

    # 1 + 2(a): hooked pass — refusal direction is active during BOTH the
    # `generate` call (so continuations are sampled from the steered model)
    # and the teacher-forced forward that scores them.
    steered_logits, continuations = _logits_over_horizon(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        instructions=instructions,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        horizon=horizon,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=[],
        batch_size=batch_size,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        num_samples_per_prompt=num_samples_per_prompt,
    )  # (n*M, horizon, V)

    # 2(b): un-hooked pass on the *same* teacher-forced (steered) sequence.
    baseline_logits, _ = _logits_over_horizon(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        instructions=instructions,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        horizon=horizon,
        fwd_pre_hooks=[],
        fwd_hooks=[],
        batch_size=batch_size,
        precomputed_continuations=continuations,
        num_samples_per_prompt=num_samples_per_prompt,
    )  # (n*M, horizon, V)

    base_f = baseline_logits.to(torch.float32)
    steer_f = steered_logits.to(torch.float32)
    del baseline_logits, steered_logits

    lp_b = torch.log_softmax(base_f, dim=-1)
    lp_s = torch.log_softmax(steer_f, dim=-1)
    del base_f, steer_f

    # past_eot detection runs on the un-hooked argmax: if the un-hooked model
    # would already have ended the turn by step t, its distribution beyond t
    # is degenerate and the delta there is noise. Reuses the existing rule.
    top1_b = lp_b.argmax(-1)  # (n*M, H)
    eos_ids = _collect_eot_ids(model_base.tokenizer)
    is_eot = torch.zeros_like(top1_b, dtype=torch.bool)
    if eos_ids:
        hit = torch.zeros(top1_b.size(0), dtype=torch.bool, device=top1_b.device)
        for t in range(top1_b.size(1)):
            tok_t = top1_b[:, t]
            is_this = torch.zeros_like(tok_t, dtype=torch.bool)
            for eid in eos_ids:
                is_this |= (tok_t == eid)
            hit = hit | is_this
            is_eot[:, t] = hit
    del top1_b

    # Absolute mean probabilities over valid (sample, position) records.
    # n_valid_pos is the count of (sample, position) pairs not flagged is_eot.
    mask_f = (~is_eot).unsqueeze(-1).to(lp_b.dtype)  # (n*M, H, 1)
    n_valid_pos = max(int((~is_eot).sum().item()), 1)
    prob_b_avg = (lp_b.exp() * mask_f).sum(dim=(0, 1)) / n_valid_pos  # (V,)
    prob_s_avg = (lp_s.exp() * mask_f).sum(dim=(0, 1)) / n_valid_pos  # (V,)
    del mask_f

    delta = lp_s - lp_b  # (n*M, horizon, V)
    del lp_b, lp_s

    banned_ids: set = set()
    if exclude_special_tokens:
        banned_ids |= set(_collect_banned_token_ids(model_base.tokenizer))
    if min_steered_prob > 0.0:
        low_prob_ids = (prob_s_avg < min_steered_prob).nonzero(as_tuple=True)[0].tolist()
        banned_ids |= set(low_prob_ids)
    if banned_ids:
        banned_tensor = torch.tensor(sorted(banned_ids), device=delta.device, dtype=torch.long)
    else:
        banned_tensor = None

    if selection_method == "mean":
        delta.masked_fill_(is_eot.unsqueeze(-1), 0.0)
        n_valid = int((~is_eot).sum().item())
        delta_avg = delta.sum(dim=(0, 1)) / max(n_valid, 1)  # (V,)
        if banned_tensor is not None:
            delta_avg[banned_tensor] = float("-inf")
        scores, token_ids = torch.topk(delta_avg, k=top_k)
        # Drop banned fillers (-inf) — they get pulled into the topk when
        # fewer than top_k tokens have a real score. Result may be < top_k.
        keep = torch.isfinite(scores)
        token_ids, scores = token_ids[keep], scores[keep]
        return token_ids, scores, prob_b_avg[token_ids], prob_s_avg[token_ids]

    if selection_method == "vote":
        H, V = delta.size(1), delta.size(2)
        vote_count = torch.zeros(V, dtype=torch.long, device=delta.device)
        for t in range(H):
            valid_rows = ~is_eot[:, t]
            if not valid_rows.any():
                continue
            delta_t = delta[valid_rows, t, :]
            if banned_tensor is not None:
                delta_t = delta_t.clone()
                delta_t[:, banned_tensor] = float("-inf")
            _, top_idx = delta_t.topk(vote_top_k_inner, dim=-1)  # (n_valid, K')
            flat = top_idx.reshape(-1)
            vote_count.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.long))
        if banned_tensor is not None:
            vote_count[banned_tensor] = -1
        scores_long, token_ids = torch.topk(vote_count, k=top_k)
        # Drop tokens with no actual support: banned (-1) or zero votes.
        # Result may be < top_k when vote support is concentrated.
        keep = scores_long > 0
        token_ids, scores_long = token_ids[keep], scores_long[keep]
        return token_ids, scores_long.to(torch.float32), prob_b_avg[token_ids], prob_s_avg[token_ids]

    raise ValueError(f"unknown selection_method: {selection_method!r}")


def print_safe_tokens(
    tokenizer,
    token_ids: Int[Tensor, "top_k"],
    scores: Float[Tensor, "top_k"],
) -> None:
    print(f"{'Rank':<6} {'Token ID':<10} {'Token':<20} {'Δ log-prob':<12}")
    print("-" * 50)
    for rank, (tid, score) in enumerate(zip(token_ids.tolist(), scores.tolist()), start=1):
        token_str = repr(tokenizer.decode([tid]))
        print(f"{rank:<6} {tid:<10} {token_str:<20} {score:+.4f}")

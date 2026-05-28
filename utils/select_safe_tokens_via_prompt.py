"""Safe-token selection by toggling system prompt (instead of refusal-vector hook).

Mirrors `utils.select_safe_tokens_reverse.get_safe_tokens` but the "steered"
teacher is built by prepending a safety system prompt rather than injecting an
activation-addition hook. Same vote/mean selection backends, same special-token
and is_eot masking, same return signature: (token_ids, scores, prob_baseline,
prob_steered).

The two passes use DIFFERENT prefix tokenizations (with vs without the safety
system prompt) but share the SAME answer tokens (sampled from the with-prompt
teacher's safety-aligned trajectory). Logits are taken from the last `horizon`
positions of each forward pass — these correspond to identical answer-position
predictions despite the prefix length differing.
"""
import torch

from typing import List, Tuple
from jaxtyping import Float, Int
from torch import Tensor

from model_utils.model_base import ModelBase

def _logits_pair_over_horizon(
    model,
    tokenizer,
    instructions,
    safe_system_prompt,
    tokenize_instructions_fn,
    horizon,
    batch_size,
    do_sample,
    temperature,
    top_p,
    num_samples_per_prompt,
):
    """Returns (steered_logits, baseline_logits) of shape (n*M, horizon, V).

    1. Encode each prompt with and without the safety system prompt.
    2. From the with-prompt teacher, sample horizon-1 continuation tokens (steered trajectory).
    3. Append the same continuation to BOTH sequences.
    4. Forward both → take last horizon logits.
    """
    M = num_samples_per_prompt
    if M > 1:
        instructions = [p for p in instructions for _ in range(M)]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    all_steered = None
    all_baseline = None

    for i in range(0, len(instructions), batch_size):
        batch = instructions[i:i + batch_size]
        with_enc = tokenize_instructions_fn(instructions=batch, system=safe_system_prompt)
        wo_enc = tokenize_instructions_fn(instructions=batch, system="")

        with_ids = with_enc.input_ids.to(model.device)
        with_mask = with_enc.attention_mask.to(model.device)
        wo_ids = wo_enc.input_ids.to(model.device)
        wo_mask = wo_enc.attention_mask.to(model.device)

        if horizon > 1:
            gen_kwargs = dict(
                input_ids=with_ids, attention_mask=with_mask,
                max_new_tokens=horizon - 1, min_new_tokens=horizon - 1,
                pad_token_id=pad_id, do_sample=do_sample,
            )
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            with torch.no_grad():
                gen_out = model.generate(**gen_kwargs)
            cont_tokens = gen_out[:, with_ids.size(1):]  # (B, horizon-1)
            with_ids = gen_out
            with_mask = torch.cat(
                [with_mask, torch.ones_like(cont_tokens, dtype=with_mask.dtype)], dim=1,
            )
            wo_ids = torch.cat([wo_ids, cont_tokens], dim=1)
            wo_mask = torch.cat(
                [wo_mask, torch.ones_like(cont_tokens, dtype=wo_mask.dtype)], dim=1,
            )

        with torch.no_grad():
            logits_s = model(input_ids=with_ids, attention_mask=with_mask).logits[:, -horizon:, :]
            logits_b = model(input_ids=wo_ids, attention_mask=wo_mask).logits[:, -horizon:, :]

        all_steered = logits_s if all_steered is None else torch.cat([all_steered, logits_s], dim=0)
        all_baseline = logits_b if all_baseline is None else torch.cat([all_baseline, logits_b], dim=0)

    return all_steered, all_baseline


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


def get_safe_tokens_via_prompt(
    model_base: ModelBase,
    instructions: List[str],
    safe_system_prompt: str,
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
    """Prompt-based analog of `select_safe_tokens_reverse.get_safe_tokens`.

    Same return signature: (token_ids, scores, prob_baseline, prob_steered).
    """
    if not safe_system_prompt:
        raise ValueError("safe_system_prompt must be a non-empty string.")

    steered_logits, baseline_logits = _logits_pair_over_horizon(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        instructions=instructions,
        safe_system_prompt=safe_system_prompt,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        horizon=horizon,
        batch_size=batch_size,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        num_samples_per_prompt=num_samples_per_prompt,
    )

    base_f = baseline_logits.to(torch.float32)
    steer_f = steered_logits.to(torch.float32)
    del baseline_logits, steered_logits

    lp_b = torch.log_softmax(base_f, dim=-1)
    lp_s = torch.log_softmax(steer_f, dim=-1)
    del base_f, steer_f

    # past_eot detection on baseline argmax (consistent with _reverse).
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
    mask_f = (~is_eot).unsqueeze(-1).to(lp_b.dtype)  # (n*M, H, 1)
    n_valid_pos = max(int((~is_eot).sum().item()), 1)
    prob_b_avg = (lp_b.exp() * mask_f).sum(dim=(0, 1)) / n_valid_pos  # (V,)
    prob_s_avg = (lp_s.exp() * mask_f).sum(dim=(0, 1)) / n_valid_pos
    del mask_f

    delta = lp_s - lp_b
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
        delta_avg = delta.sum(dim=(0, 1)) / max(n_valid, 1)
        if banned_tensor is not None:
            delta_avg[banned_tensor] = float("-inf")
        scores, token_ids = torch.topk(delta_avg, k=top_k)
        # Drop banned fillers (-inf). Result may be < top_k.
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
            _, top_idx = delta_t.topk(vote_top_k_inner, dim=-1)
            flat = top_idx.reshape(-1)
            vote_count.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.long))
        if banned_tensor is not None:
            vote_count[banned_tensor] = -1
        scores_long, token_ids = torch.topk(vote_count, k=top_k)
        # Drop tokens with no actual support: banned (-1) or zero votes.
        keep = scores_long > 0
        token_ids, scores_long = token_ids[keep], scores_long[keep]
        return token_ids, scores_long.to(torch.float32), prob_b_avg[token_ids], prob_s_avg[token_ids]

    raise ValueError(f"unknown selection_method: {selection_method!r}")

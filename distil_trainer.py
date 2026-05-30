# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

import os
import json
import torch
import torch.utils.data
import random
import inspect
import datasets
import transformers

from pathlib import Path
from functools import partial
from typing import Any, Callable, Optional, Union
from collections import defaultdict, deque

from accelerate import logging
from accelerate.utils import gather, gather_object, is_peft_model, set_seed

from datasets import Dataset, IterableDataset

from torch import nn
from torch.utils.data import DataLoader, Sampler
from torch.nn.functional import log_softmax, kl_div

from transformers import (
    AutoConfig,
    AutoProcessor,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    is_wandb_available,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_flash_attn_2_available, is_peft_available, is_rich_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template, prepare_multimodal_messages
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.models import prepare_peft_model, unwrap_model_for_generation
from trl.trainer.base_trainer import BaseTrainer
from trl.trainer.utils import (
    RepeatSampler,
    disable_dropout_in_model,
    entropy_from_logits,
    identity,
    nanmax,
    nanmin,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
    shuffle_sequence_dict,
    split_pixel_values_by_grid,
    split_tensor_dict,
    unsplit_pixel_values_by_grid,
)

from model_utils.model_factory import construct_model_base
from utils.refusal_direction_utils import load_dataset_split, filter_data, select_and_save_direction, generate_and_save_candidate_directions
from utils.select_safe_tokens_via_activation import get_safe_tokens
from utils.select_safe_tokens_via_prompt import get_safe_tokens_via_prompt
from distil_config import DistilConfig

if is_peft_available():
    from peft import PeftConfig, PeftModel

if is_wandb_available():
    import wandb

logger = logging.get_logger(__name__)

def _dump_safe_tokens_trace(output_dir, step, horizon, token_ids, scores, tokenizer,
                            prob_baseline=None, prob_steered=None, is_main_process=True):
    """Append one safe-tokens snapshot to {output_dir}/safe_tokens_trace.jsonl (main process only)."""
    if not is_main_process:
        return
    try:
        ids = token_ids.detach().cpu().tolist()
        sc = scores.detach().cpu().float().tolist() if scores is not None else None
        pb = prob_baseline.detach().cpu().float().tolist() if prob_baseline is not None else None
        ps = prob_steered.detach().cpu().float().tolist() if prob_steered is not None else None
        decoded = [tokenizer.decode([t]) for t in ids]
        record = {
            "step": int(step),
            "horizon": int(horizon),
            "token_ids": ids,
            "tokens": decoded,
            "scores": sc,
            "prob_baseline": pb,
            "prob_steered": ps,
        }
        print(f"[safe_tokens_trace] Step {step}, horizon {horizon}: \nsafe_token_ids={ids} \ntokens={decoded} \nscores={sc} \nprob_baseline={pb} \nprob_steered={ps}")    
        path = os.path.join(output_dir, "safe_tokens_trace.jsonl")
        os.makedirs(output_dir, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[safe_tokens_trace] dump failed: {e}")


class DynamicRefusalVectorCallback(TrainerCallback):
    """
    A HuggingFace TrainerCallback that periodically recomputes the "refusal
    direction" used for activation-level steering during training.

    Use when:
      - use_refusal_vector == True       (Teacher steering is via the hook)
      - update_refusal_vector == True    (direction is allowed to refresh)
      - freeze_teacher == False          (Teacher is being synced)

    Notes:
      - The sync-cadence check `state.global_step % args.ref_model_sync_steps == 0`
        must match the Teacher-sync callback's cadence exactly; otherwise the
        direction is extracted from a Teacher that has not yet been re-synced.
      - Co-registration order: when both this callback and
        DynamicSafeTokenViaRefusalVectorCallback are registered, this one must
        be added FIRST so the safe-token callback consumes the freshly updated
        direction. DistilTrainer.__init__ guarantees this via add_callback
        ordering.
      - The hidden ModelBase wrapper built in __init__ holds a reference to the
        unwrapped ref_model. The sync callback mutates ref_model weights
        in-place, so the wrapper automatically reflects the latest Teacher
        without needing to be rebuilt.
    """
    def __init__(self, trainer, harmful_train, harmless_train, harmful_val, harmless_val):
        self.trainer = trainer
        self.harmful_train = harmful_train
        self.harmless_train = harmless_train
        self.harmful_val = harmful_val
        self.harmless_val = harmless_val

        # Unwrap the reference model from the accelerator (handles DDP wrapping)
        unwrapped_ref_model = self.trainer.accelerator.unwrap_model(self.trainer.ref_model)

        # Build a ModelBase wrapper around the reference model for direction extraction
        self.model_base = construct_model_base(unwrapped_ref_model, self.trainer.processing_class, trainer.args.model_name)

    def on_step_end(self, args, state, control, **kwargs):
        # The extraction frequency must exactly match the Teacher synchronization frequency.
        # Otherwise, the refusal vector will be extracted from stale reference model weights.
        if self.trainer.ref_model is not None and state.global_step > 0 and state.global_step % args.ref_model_sync_steps == 0:
            with torch.no_grad():
            
                # Only the main process prints logs to avoid log spam in multi-GPU runs
                is_main_process = self.trainer.accelerator.is_main_process

                # Directory to persist per-step artifacts (candidate directions, selected direction, etc.)
                step_artifact_dir = os.path.join(args.output_dir, "refusal_direction", f"step_{state.global_step}")
                if is_main_process:
                    print(f"\n[Step {state.global_step}] 🔄 Teacher model synchronized. Recomputing refusal vector...")
                
                # Step 0: Disable refusal injection during the extraction process.
                # If left active, the stale injected direction would contaminate the hidden states 
                # we are trying to use to compute the new direction.
                self.trainer.refusal_state["is_active"] = False
                
                # Filter datasets to isolate samples exhibiting the desired refusal/compliance behaviors
                harmful_train_filted, harmless_train_filted, harmful_val_filted, harmless_val_filted = filter_data(self.model_base, self.harmful_train, self.harmless_train, self.harmful_val, self.harmless_val, is_main_process)
                
                # Step 1: Generate candidate refusal directions. 
                # This computes the mean activation difference between harmful and harmless prompts 
                # across various token positions and network layers.
                candidate_directions = generate_and_save_candidate_directions(self.model_base, harmful_train_filted, harmless_train_filted)
                
                if is_main_process:
                    # Persist raw candidate directions for later inspection and debugging
                    if not os.path.exists(os.path.join(step_artifact_dir, 'generate_directions')):
                        os.makedirs(os.path.join(step_artifact_dir, 'generate_directions'))
                    torch.save(candidate_directions, os.path.join(step_artifact_dir, 'generate_directions/mean_diffs.pt'))
                
                # Step 2: Select the optimal (position, layer) pair.
                # Evaluates which direction is most effective at suppressing refusal on the validation split.
                pos, layer, direction = select_and_save_direction(self.model_base, harmful_val_filted, harmless_val_filted, candidate_directions, step_artifact_dir, is_main_process=is_main_process)
                
                # Step 3: Update the injection layer index in the trainer's shared state.
                # `candidate_directions` has shape [num_positions, num_layers, d_model].
                # `layer` is the optimal layer index and `pos` is the chosen token position.
                self.trainer.refusal_state["layer_idx"] = layer
                
                # Step 4: Copy the new direction into the trainer's state buffer.
                # Match the original dtype and device to ensure the forward hook can apply it without casting overhead.
                self.trainer.refusal_state["direction"] = direction.to(
                    self.trainer.refusal_state["direction"].dtype
                ).to(self.trainer.refusal_state["direction"].device)
                
                # Re-enable refusal injection now that the updated direction is in place
                self.trainer.refusal_state["is_active"] = True
                torch.cuda.empty_cache()
                
                if is_main_process:
                    print(f"SafeSteer✅ Refusal direction recomputed from the latest Teacher and updated.")


class DynamicSafeTokenViaRefusalVectorCallback(TrainerCallback):
    """
    A HuggingFace TrainerCallback that periodically recomputes the safe_tokens
    using the refusal-vector-steered Teacher.

    Use when:
      - voca_selection_mode == 2     (safe-token selection is active)
      - freeze_teacher == False      (Teacher is being synced, so its distribution drifts)
      - freeze_safe_token == False   (safe_tokens are allowed to refresh)
      - use_refusal_vector == True   (Teacher steering is via injected refusal direction)

    Notes:
      - Decoupled from refusal-direction recomputation. Always reads the *current*
        direction/layer from trainer.refusal_state, regardless of whether the
        direction itself was refreshed this step.
      - When co-registered with DynamicRefusalVectorCallback, this callback must
        be added AFTER it so the safe-token computation sees the updated direction.
    """
    def __init__(self, trainer, harmful_train, harmless_train, harmful_val, harmless_val):
        self.trainer = trainer
        self.harmful_train = harmful_train
        self.harmless_train = harmless_train
        self.harmful_val = harmful_val
        self.harmless_val = harmless_val

        # Unwrap the reference model (handles DDP wrapping)
        unwrapped_ref_model = self.trainer.accelerator.unwrap_model(self.trainer.ref_model)
        # Build the ModelBase adapter once; the underlying ref_model is mutated in place
        # by the sync callback, so this adapter automatically sees fresh weights.
        self.model_base = construct_model_base(
            unwrapped_ref_model, self.trainer.processing_class, trainer.args.model_name,
        )

    def on_step_end(self, args, state, control, **kwargs):
        # Match Teacher-sync cadence exactly so safe_tokens reflect the latest Teacher.
        if self.trainer.ref_model is not None and state.global_step > 0 and state.global_step % args.ref_model_sync_steps == 0:

            with torch.no_grad():
                is_main_process = self.trainer.accelerator.is_main_process
                if is_main_process:
                    print(f"\n[Step {state.global_step}] 🔄 Recomputing safe_tokens via refusal-vector-steered Teacher...")

                # Disable the global injection hook during this pass: get_safe_tokens()
                # applies the direction internally via model_base; leaving the hook
                # active would double-inject and bias the steered distribution.
                self.trainer.refusal_state["is_active"] = False

                # Refilter under the current Teacher (its refusal behavior drifts over training)
                _, _, _, harmless_val_filted = filter_data(
                    self.model_base, self.harmful_train, self.harmless_train,
                    self.harmful_val, self.harmless_val, is_main_process,
                )

                # Consume whatever direction/layer is currently in refusal_state.
                direction = self.trainer.refusal_state["direction"]
                layer = self.trainer.refusal_state["layer_idx"]

                _M = self.trainer.args.num_samples_per_prompt
                safe_token_ids, safe_token_scores, safe_prob_baseline, safe_prob_steered = get_safe_tokens(
                    model_base=self.model_base,
                    instructions=harmless_val_filted,
                    direction=direction,
                    layer=layer,
                    top_k=self.trainer.args.voca_selection_num,
                    horizon=self.trainer.args.safe_token_horizon,
                    do_sample=(_M > 1),
                    temperature=self.trainer.args.safe_token_temperature,
                    top_p=self.trainer.args.safe_token_top_p,
                    num_samples_per_prompt=_M,
                    selection_method=self.trainer.args.selection_method,
                    vote_top_k_inner=self.trainer.args.vote_top_k_inner,
                    exclude_special_tokens=self.trainer.args.exclude_special_tokens,
                    min_steered_prob=self.trainer.args.min_steered_prob,
                )

                self.trainer.refusal_state["safe_tokens"] = safe_token_ids.cpu()
                _dump_safe_tokens_trace(
                    output_dir=self.trainer.args.output_dir,
                    step=state.global_step,
                    horizon=self.trainer.args.safe_token_horizon,
                    token_ids=safe_token_ids,
                    scores=safe_token_scores,
                    tokenizer=self.trainer.processing_class,
                    prob_baseline=safe_prob_baseline,
                    prob_steered=safe_prob_steered,
                    is_main_process=is_main_process,
                )

                # Re-enable injection for normal training forward passes.
                self.trainer.refusal_state["is_active"] = True
                torch.cuda.empty_cache()

                if is_main_process:
                    print(f"SafeSteer✅ safe_tokens recomputed via refusal vector "
                        f"(K={self.trainer.args.voca_selection_num}, horizon={self.trainer.args.safe_token_horizon}).")


class DynamicSafeTokenViaSystemPromptCallback(TrainerCallback):
    """
    A HuggingFace TrainerCallback that periodically recomputes the safe_tokens
    using the system-prompt-steered Teacher.

    Use when:
      - voca_selection_mode == 2     (safe-token selection is active)
      - freeze_teacher == False      (Teacher is being synced, so its distribution drifts)
      - freeze_safe_token == False   (safe_tokens are allowed to refresh)
      - use_refusal_vector == False  (Teacher steering is via safe_system_prompt)
    """
    def __init__(self, trainer, harmful_train, harmless_train, harmful_val, harmless_val):
        self.trainer = trainer
        self.harmful_train = harmful_train
        self.harmless_train = harmless_train
        self.harmful_val = harmful_val
        self.harmless_val = harmless_val

        unwrapped_ref_model = self.trainer.accelerator.unwrap_model(self.trainer.ref_model)
        self.model_base = construct_model_base(
            unwrapped_ref_model, self.trainer.processing_class, trainer.args.model_name,
        )

        # The prompt-based selector requires a non-empty steering prompt.
        safe_system_prompt = getattr(self.trainer.args, "safe_system_prompt", None)
        if not safe_system_prompt:
            raise ValueError(
                "DynamicSafeTokenViaSystemPromptCallback requires args.safe_system_prompt."
            )
        self.safe_system_prompt = safe_system_prompt

    def on_step_end(self, args, state, control, **kwargs):

        if self.trainer.ref_model is not None and state.global_step > 0 and state.global_step % args.ref_model_sync_steps == 0:

            with torch.no_grad():
                is_main_process = self.trainer.accelerator.is_main_process
                if is_main_process:
                    print(f"\n[Step {state.global_step}] 🔄 Recomputing safe_tokens via system-prompt-steered Teacher...")

                _, _, harmful_val_filted, harmless_val_filted = filter_data(
                    self.model_base, self.harmful_train, self.harmless_train,
                    self.harmful_val, self.harmless_val, is_main_process,
                )

                _M = self.trainer.args.num_samples_per_prompt
                safe_token_ids, safe_token_scores, safe_prob_baseline, safe_prob_steered = get_safe_tokens_via_prompt(
                    model_base=self.model_base,
                    instructions=harmful_val_filted,
                    safe_system_prompt=self.safe_system_prompt,
                    top_k=self.trainer.args.voca_selection_num,
                    horizon=self.trainer.args.safe_token_horizon,
                    do_sample=(_M > 1),
                    temperature=self.trainer.args.safe_token_temperature,
                    top_p=self.trainer.args.safe_token_top_p,
                    num_samples_per_prompt=_M,
                    selection_method=self.trainer.args.selection_method,
                    vote_top_k_inner=self.trainer.args.vote_top_k_inner,
                    exclude_special_tokens=self.trainer.args.exclude_special_tokens,
                    min_steered_prob=self.trainer.args.min_steered_prob,
                )

                self.trainer.refusal_state["safe_tokens"] = safe_token_ids.cpu()
                _dump_safe_tokens_trace(
                    output_dir=self.trainer.args.output_dir,
                    step=state.global_step,
                    horizon=self.trainer.args.safe_token_horizon,
                    token_ids=safe_token_ids,
                    scores=safe_token_scores,
                    tokenizer=self.trainer.processing_class,
                    prob_baseline=safe_prob_baseline,
                    prob_steered=safe_prob_steered,
                    is_main_process=is_main_process,
                )
                torch.cuda.empty_cache()

                if is_main_process:
                    print(f"SafeSteer✅ safe_tokens recomputed via system prompt "
                        f"(K={self.trainer.args.voca_selection_num}, horizon={self.trainer.args.safe_token_horizon}).")


class MemoryEfficientSyncRefModelCallback(TrainerCallback):
    """
    Memory-efficient callback to synchronize the model with a reference model.
    
    Unlike the default SyncRefModelCallback, this version iterates through parameters
    one at a time instead of gathering all parameters at once. This reduces peak memory
    usage from O(full_model_size) to O(single_param_size).
    """

    def __init__(
        self,
        ref_model: Union[PreTrainedModel, nn.Module],
        accelerator: Optional[Any],
    ):
        self.accelerator = accelerator
        self.ref_model = ref_model

    @staticmethod
    def _sync_param(model_param, ref_param, alpha):
        """Sync a single parameter: ref = alpha * model + (1 - alpha) * ref"""
        # Move the Student's weights to the Teacher's device
        model_param_data = model_param.data.to(ref_param.device)
        ref_param.data.mul_(1.0 - alpha).add_(model_param_data, alpha=alpha)

    @staticmethod
    def sync_target_model_memory_efficient(model, target_model, alpha):
        """
        Sync target_model to track model, gathering one parameter at a time.
        
        This is O(1) in memory overhead instead of O(N) where N is model size.
        """

        # Non-ZeRO-3: just iterate normally
        for model_param, ref_param in zip(model.parameters(), target_model.parameters()):
            MemoryEfficientSyncRefModelCallback._sync_param(model_param, ref_param, alpha)

    def on_step_end(self, args, state, control, **kwargs):
        model: PreTrainedModel = kwargs["model"]

        if self.ref_model is not None and state.global_step % args.ref_model_sync_steps == 0:
            if self.accelerator:
                model = self.accelerator.unwrap_model(model)
            self.sync_target_model_memory_efficient(model, self.ref_model, args.ref_model_mixup_alpha)

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class DistilTrainer(BaseTrainer):
    """
    Trainer for the Self-Distillation method. 

    Example:

    ```python
    from datasets import load_dataset
    from trl import DistilTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")


    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]


    trainer = DistilTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keyword arguments in
              `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. Custom reward
                  functions can also return `None` when the reward is not applicable to those samples. This is useful
                  for multi-task training where different reward functions apply to different types of samples. When a
                  reward function returns `None` for a sample, that reward function is excluded from the reward
                  calculation for that sample. For more details, see [Using a custom reward
                  function](#using-a-custom-reward-function).

                  The trainer's state is also passed to the reward function. The trainer's state is an instance of
                  [`~transformers.TrainerState`] and can be accessed by accessing the `trainer_state` argument to the
                  reward function's signature.
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`DistilConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoProcessor.from_pretrained`]. A
            padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
            `tokenizer.eos_token` will be used as the default.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "distil"]
    _name = "Distil"

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        ref_model: Union[str, PreTrainedModel],
        args: Optional[DistilConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # =====================================================================
        # 1. Resolve args + load Student model
        # =====================================================================

        # 1.1 Default config if not provided.
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = DistilConfig(f"{model_name}-Distil")

        # 1.2 Materialize the Student model.
        # ✅ SafeSteer setup always passes a pre-loaded PreTrainedModel, so the
        # `isinstance(model, str)` branch never fires and model_init_kwargs is empty.
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            dtype = model_init_kwargs.get("dtype")
            if isinstance(dtype, torch.dtype) or dtype == "auto" or dtype is None:
                pass  # dtype is already a torch.dtype or "auto" or None
            elif isinstance(dtype, str):  # it's a str, but not "auto"
                dtype = getattr(torch, dtype)
                model_init_kwargs["dtype"] = dtype
            else:
                raise ValueError(
                    "Invalid `dtype` passed to `DistilConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            config = AutoConfig.from_pretrained(model_id)
            architecture = getattr(transformers, config.architectures[0])
            model = architecture.from_pretrained(model_id, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `DistilConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )

        # 1.3 Cache the kwarg names that model.forward() accepts.
        # Used downstream to decide whether we can pass `logits_to_keep`
        # (some VLMs like SmolVLM/Idefics3 don't support it).
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        # 1.4 PEFT/LoRA wrapping (no-op in ✅ SafeSteer setup: peft_config is None and
        # model is not a PeftModel).
        if peft_config is not None or (is_peft_available() and isinstance(model, PeftModel)):
            model = prepare_peft_model(model, peft_config, args)

        # =====================================================================
        # 2. Tokenizer + pad/eos tokens
        # =====================================================================

        # 2.1 Auto-load a processor if none was passed.
        # ✅ SafeSteer setup always passes a tokenizer, so this branch never fires.
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(model.config._name_or_path, truncation_side="left")

        # 2.2 Extract the underlying tokenizer from either a Processor or a Tokenizer.
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        # 2.3 Fall back pad_token to eos_token (common for Qwen / Llama families).
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        # =====================================================================
        # 3. Copy training hyperparameters from args
        # =====================================================================
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_transformers_paged = args.use_transformers_paged
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.importance_sampling_level = args.importance_sampling_level
        self.mask_truncated_completions = args.mask_truncated_completions
        self.top_entropy_quantile = args.top_entropy_quantile
        self.num_loss_tokens_to_skip = args.num_loss_tokens_to_skip
        self.num_loss_tokens_to_keep = args.num_loss_tokens_to_keep

        # =====================================================================
        # 4. Dataset validation
        # =====================================================================
        self.shuffle_dataset = args.shuffle_dataset

        # IterableDataset is not supported (see https://github.com/huggingface/trl/issues/3213).
        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            raise NotImplementedError(
                "Iterable datasets are not yet supported in DistilTrainer. Please use a standard dataset instead."
            )

        # =====================================================================
        # 5. Multi-iteration sampling state
        # =====================================================================
        self.num_iterations = args.num_iterations
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # Counts forward+backward passes, including those inside one grad-accum cycle.
        self._step = 0
        # Cache the generation outputs so multiple updates can reuse them.
        # See `_get_train_sampler` and `_prepare_inputs` for the read sites.
        self._buffered_inputs = None

        # =====================================================================
        # 6. Call parent Trainer
        # =====================================================================

        # Suppress the "could not estimate FLOPs" warning. The parent Trainer                                                                                                                                                                                                                                                                                                                                    
        # counts elements of "input_ids" to compute throughput, but ✅ SafeSteer batches
        # carry text fields ("prompt") instead. Marking the                                                                                                                                                                                                                                                                                                                                       
        # warning as already-issued silences it; FLOPs metric is unaffected
        # either way.
        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=identity,  # No data collation is needed in Distil
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            # `compute_loss_func` is set to a non-None placeholder so the parent's
            # `training_step` skips its built-in gradient-accumulation scaling.
            # We scale ourselves based on total completion tokens in the global
            # accumulated batch (DAPO-style). Any non-None value works.
            compute_loss_func="non-None value to disable scaling",
        )

        # =====================================================================
        # 7. Reference (Teacher) model
        # =====================================================================
        self.beta = args.beta
        self.alpha = args.alpha

        # ✅ SafeSteer setup always passes a non-None ref_model, so the first branch fires.
        if ref_model is not None:
            self.ref_model = ref_model
        else:
            raise ValueError("DistilTrainer requires a reference model for self-distillation. Please provide one via the `ref_model` argument.")

        # =====================================================================
        # 8. Disable dropout in models (no-op in ✅ SafeSteer setup)
        # =====================================================================
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # =====================================================================
        # 9. Metrics + log buffers
        # =====================================================================
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print

        # Bounded deques so we only retain entries from the latest generation batch.
        self._logs = {
            "images": deque(maxlen=args.generation_batch_size),
            "prompt": deque(maxlen=args.generation_batch_size),
            "teacher_prompt": deque(maxlen=args.generation_batch_size),
            "completion": deque(maxlen=args.generation_batch_size),
            "completion_teacher": deque(maxlen=args.generation_batch_size),
            "rewards": defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
            "advantages": deque(maxlen=args.generation_batch_size),
        }

        # =====================================================================
        # 10. Seeds + generation config
        # =====================================================================

        # Per-process seed so different ranks generate different completions
        # when num_generations > per_device_train_batch_size.
        set_seed(args.seed, device_specific=True)

        generation_kwargs = {
            "max_new_tokens": self.max_completion_length,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "cache_implementation": args.cache_implementation,
        }
        if args.generation_kwargs is not None:
            generation_kwargs.update(args.generation_kwargs)
        self.generation_config = GenerationConfig(**generation_kwargs)

        # =====================================================================
        # 11. Loss-scaling flag + model tags
        # =====================================================================

        # We compute loss ourselves, so disable the parent's automatic scaling
        # behavior that depends on whether the model accepts loss kwargs.
        self.model_accepts_loss_kwargs = False

        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            # SafeSteer✅ is using this one, put the student on cuda:0 and the teacher on cuda:1
            print("🚀 Placing Teacher model on cuda:1...")
            self.ref_model = self.ref_model.to("cuda:1")
            self.ref_model.eval()
        else:
            raise ValueError("DistilTrainer requires a reference model for self-distillation. Please provide one via the `ref_model` argument.")

        # =====================================================================
        # 12. ✅ SafeSteer: Teacher steering setup
        #
        # Decision tree:
        #   L1: use_refusal_vector       -> Teacher steering style (vector vs prompt)
        #   L2: voca_selection_mode == 2 -> whether to compute & use safe_tokens
        #   L3: sync_ref_model           -> (= NOT freeze_teacher) whether to
        #                                   register dynamic refresh callbacks
        #
        # Four leaf cases:
        #   (1) RV=True,  mode==2 : direction + safe_tokens.
        #       If synced: optional direction refresh, optional safe-token refresh.
        #   (2) RV=True,  mode!=2 : direction only.
        #       If synced: optional direction refresh.
        #   (3) RV=False, mode==2 : safe_tokens via system prompt (no direction).
        #       If synced: optional safe-token refresh.
        #   (4) RV=False, mode!=2 : nothing to steer.
        #       If synced: only the weight-sync callback.
        # =====================================================================

        self.use_refusal_vector = getattr(args, "use_refusal_vector", False)

        if self.ref_model is None:
            # No Teacher -> nothing to steer. Install a stub so downstream code
            # can read self.refusal_state["safe_tokens"] safely.
            
            self.refusal_state = {"safe_tokens": None}

            raise ValueError("DistilTrainer requires a reference model for self-distillation. Please provide one via the `ref_model` argument.")

        elif self.use_refusal_vector:
            # ================================================================
            # L1 = True: refusal-vector steering.
            # Shared setup for cases (1) and (2): datasets + model_base + direction.
            # ================================================================
            with torch.no_grad():
                if self.accelerator.is_main_process:
                    print("🔄 Generating initial refusal vector using the reference model as Teacher...")
                harmful_train, harmless_train, harmful_val, harmless_val = \
                    self.load_and_sample_datasets_for_refusal_direction(args.seed)
                unwrapped_ref_model = self.accelerator.unwrap_model(self.ref_model)
                model_base = construct_model_base(
                    unwrapped_ref_model, self.processing_class, args.model_name,
                )
                harmful_train_filted, harmless_train_filted, harmful_val_filted, harmless_val_filted = filter_data(
                    model_base, harmful_train, harmless_train, harmful_val, harmless_val,
                    self.accelerator.is_main_process,
                )
                candidate_directions = generate_and_save_candidate_directions(
                    model_base, harmful_train_filted, harmless_train_filted,
                )
                step_artifact_dir = os.path.join(args.output_dir, "refusal_direction", "step_0")
                if self.accelerator.is_main_process:
                    os.makedirs(os.path.join(step_artifact_dir, "generate_directions"), exist_ok=True)
                    torch.save(
                        candidate_directions,
                        os.path.join(step_artifact_dir, "generate_directions/mean_diffs.pt"),
                    )
                pos, layer, direction = select_and_save_direction(
                    model_base, harmful_val_filted, harmless_val_filted,
                    candidate_directions, step_artifact_dir,
                    is_main_process=self.accelerator.is_main_process,
                )
                if self.accelerator.is_main_process:
                    print(f"SafeSteer✅ Initial refusal vector generated. Saved to {step_artifact_dir}")
                torch.cuda.empty_cache()

            # refusal_state always has "safe_tokens"; filled below if mode==2.
            self.refusal_state = {
                "direction": direction.to(self.ref_model.dtype).to(self.accelerator.device),
                "coeff": 1.0,
                "layer_idx": layer,
                "is_active": True,
                "safe_tokens": None,
            }

            # Attach the ActAdd hook once, before the mode split. Case (1)
            # below toggles refusal_state["is_active"] off/on around its
            # get_safe_tokens call so the hook does not double-inject
            # (get_safe_tokens applies the direction itself via model_base).
            self._attach_teacher_actadd_hook()

            if self.args.voca_selection_mode == 2:
                # ============================================================
                # Case (1): use_refusal_vector=True, mode==2
                # ============================================================
                # Init safe_tokens via refusal-vector-steered Teacher.
                # Disable the hook around the call so get_safe_tokens's internal
                # injection is not stacked on top of the hook's injection.
                self.refusal_state["is_active"] = False
                with torch.no_grad():
                    if self.accelerator.is_main_process:
                        print("🔄 Generating initial safe_tokens via refusal-vector-steered Teacher...")
                    _M = self.args.num_samples_per_prompt
                    safe_token_ids, safe_token_scores, safe_prob_baseline, safe_prob_steered = get_safe_tokens(
                        model_base=model_base,
                        instructions=harmless_val_filted,
                        direction=direction,
                        layer=layer,
                        top_k=self.args.voca_selection_num,
                        horizon=self.args.safe_token_horizon,
                        do_sample=(_M > 1),
                        temperature=self.args.safe_token_temperature,
                        top_p=self.args.safe_token_top_p,
                        num_samples_per_prompt=_M,
                        selection_method=self.args.selection_method,
                        vote_top_k_inner=self.args.vote_top_k_inner,
                        exclude_special_tokens=self.args.exclude_special_tokens,
                        min_steered_prob=self.args.min_steered_prob,
                    )
                    torch.cuda.empty_cache()
                self.refusal_state["is_active"] = True
                self.refusal_state["safe_tokens"] = safe_token_ids.cpu()
                _dump_safe_tokens_trace(
                    output_dir=self.args.output_dir, step=0,
                    horizon=self.args.safe_token_horizon,
                    token_ids=safe_token_ids, scores=safe_token_scores,
                    tokenizer=self.processing_class,
                    prob_baseline=safe_prob_baseline, prob_steered=safe_prob_steered,
                    is_main_process=self.accelerator.is_main_process,
                )
                if self.accelerator.is_main_process:
                    print(f"SafeSteer✅ Initial safe_tokens computed via refusal-vector "
                          f"(K={self.args.voca_selection_num}, horizon={self.args.safe_token_horizon})")

                # L3: callbacks (only when Teacher is not frozen)
                if args.sync_ref_model:
                    self.add_callback(MemoryEfficientSyncRefModelCallback(
                        ref_model=self.ref_model, accelerator=self.accelerator,
                    ))
                    if self.accelerator.is_main_process:
                            print("SafeSteer✅  Registered reference model synchronization callback.")
                    if self.args.update_refusal_vector:
                        self.add_callback(DynamicRefusalVectorCallback(
                            trainer=self,
                            harmful_train=harmful_train, harmless_train=harmless_train,
                            harmful_val=harmful_val, harmless_val=harmless_val,
                        ))
                        if self.accelerator.is_main_process:
                            print("SafeSteer✅  Registered dynamic update-refusal-vector callback.")
                    elif self.accelerator.is_main_process:
                        print("⏸️  update_refusal_vector=False → direction frozen at init.")
                    if not self.args.freeze_safe_token:
                        self.add_callback(DynamicSafeTokenViaRefusalVectorCallback(
                            trainer=self,
                            harmful_train=harmful_train, harmless_train=harmless_train,
                            harmful_val=harmful_val, harmless_val=harmless_val,
                        ))
                        if self.accelerator.is_main_process:
                            print("SafeSteer✅ Registered dynamic safe-token callback (refusal-vector).")
                    elif self.accelerator.is_main_process:
                        print("⏸️  freeze_safe_token=True → safe_tokens frozen at init.")
                elif self.accelerator.is_main_process:
                    print("⏸️  sync_ref_model=False → no dynamic refresh callbacks registered.")
            else:
                # ============================================================
                # Case (2): use_refusal_vector=True, mode!=2
                # ============================================================
                # No safe_tokens; hook is already attached above the mode split.
                if args.sync_ref_model:
                    self.add_callback(MemoryEfficientSyncRefModelCallback(
                        ref_model=self.ref_model, accelerator=self.accelerator,
                    ))
                    if self.args.update_refusal_vector:
                        self.add_callback(DynamicRefusalVectorCallback(
                            trainer=self,
                            harmful_train=harmful_train, harmless_train=harmless_train,
                            harmful_val=harmful_val, harmless_val=harmless_val,
                        ))
                        if self.accelerator.is_main_process:
                            print("SafeSteer✅  Registered dynamic update-refusal-vector callback.")
                    elif self.accelerator.is_main_process:
                        print("⏸️  update_refusal_vector=False → direction frozen at init.")
                elif self.accelerator.is_main_process:
                    print("⏸️  sync_ref_model=False → no dynamic refresh callbacks registered.")

        else:
            # ================================================================
            # L1 = False: system-prompt steering (no direction, no hook).
            # ================================================================
            if self.args.voca_selection_mode == 2:
                # ============================================================
                # Case (3): use_refusal_vector=False, mode==2
                # ============================================================
                safe_system_prompt = getattr(self.args, "safe_system_prompt", None)
                if not safe_system_prompt:
                    raise ValueError(
                        "voca_selection_mode=2 with use_refusal_vector=False requires safe_system_prompt."
                    )

                with torch.no_grad():
                    harmful_train, harmless_train, harmful_val, harmless_val = \
                        self.load_and_sample_datasets_for_refusal_direction(args.seed)
                    unwrapped_ref_model = self.accelerator.unwrap_model(self.ref_model)
                    model_base = construct_model_base(
                        unwrapped_ref_model, self.processing_class, args.model_name,
                    )
                    _, _, harmful_val_filted, harmless_val_filted = filter_data(
                        model_base, harmful_train, harmless_train, harmful_val, harmless_val,
                        self.accelerator.is_main_process,
                    )
                    _M = self.args.num_samples_per_prompt
                    safe_token_ids, safe_token_scores, safe_prob_baseline, safe_prob_steered = get_safe_tokens_via_prompt(
                        model_base=model_base,
                        instructions=harmful_val_filted,
                        safe_system_prompt=safe_system_prompt,
                        top_k=self.args.voca_selection_num,
                        horizon=self.args.safe_token_horizon,
                        do_sample=(_M > 1),
                        temperature=self.args.safe_token_temperature,
                        top_p=self.args.safe_token_top_p,
                        num_samples_per_prompt=_M,
                        selection_method=self.args.selection_method,
                        vote_top_k_inner=self.args.vote_top_k_inner,
                        exclude_special_tokens=self.args.exclude_special_tokens,
                        min_steered_prob=self.args.min_steered_prob,
                    )
                    torch.cuda.empty_cache()

                self.refusal_state = {"safe_tokens": safe_token_ids.cpu()}
                _dump_safe_tokens_trace(
                    output_dir=self.args.output_dir, step=0,
                    horizon=self.args.safe_token_horizon,
                    token_ids=safe_token_ids, scores=safe_token_scores,
                    tokenizer=self.processing_class,
                    prob_baseline=safe_prob_baseline, prob_steered=safe_prob_steered,
                    is_main_process=self.accelerator.is_main_process,
                )
                if self.accelerator.is_main_process:
                    print(f"SafeSteer✅ Initial safe_tokens computed via system-prompt "
                          f"(K={self.args.voca_selection_num}, horizon={self.args.safe_token_horizon})")

                if args.sync_ref_model:
                    self.add_callback(MemoryEfficientSyncRefModelCallback(
                        ref_model=self.ref_model, accelerator=self.accelerator,
                    ))
                    if self.accelerator.is_main_process:
                            print("SafeSteer✅  Registered reference model synchronization callback.")
                    if not self.args.freeze_safe_token:
                        self.add_callback(DynamicSafeTokenViaSystemPromptCallback(
                            trainer=self,
                            harmful_train=harmful_train, harmless_train=harmless_train,
                            harmful_val=harmful_val, harmless_val=harmless_val,
                        ))
                        if self.accelerator.is_main_process:
                            print("SafeSteer✅ Registered dynamic safe-token callback (system-prompt).")
                    elif self.accelerator.is_main_process:
                        print("⏸️  update_refusal_vector=False → direction frozen at init.")
                elif self.accelerator.is_main_process:
                    print("⏸️  sync_ref_model=False → no dynamic refresh callbacks registered.")
            else:
                # ============================================================
                # Case (4): use_refusal_vector=False, mode!=2
                # ============================================================
                # No refusal vector, no safe_tokens. Nothing to compute or refresh.
                self.refusal_state = {"safe_tokens": None}

                if args.sync_ref_model:
                    self.add_callback(MemoryEfficientSyncRefModelCallback(
                        ref_model=self.ref_model, accelerator=self.accelerator,
                    ))
                    if self.accelerator.is_main_process:
                        print("SafeSteer✅  Registered reference model synchronization callback.")
                elif self.accelerator.is_main_process:
                    print("⏸️  sync_ref_model=False → no dynamic refresh callbacks registered.")

    def _attach_teacher_actadd_hook(self):
        """Register a forward-pre-hook on the Teacher that adds
        coeff * direction to the hidden states at refusal_state["layer_idx"]
        whenever refusal_state["is_active"] is True."""
        def teacher_act_add_hook(module, input_args):
            if not self.refusal_state["is_active"]:
                return input_args
            if isinstance(input_args, tuple):
                hidden_states = input_args[0]
            else:
                hidden_states = input_args
            _dir = self.refusal_state["direction"].to(
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
            hidden_states = hidden_states + self.refusal_state["coeff"] * _dir
            if isinstance(input_args, tuple):
                return (hidden_states, *input_args[1:])
            return hidden_states

        target_layer = self.ref_model.model.layers[self.refusal_state["layer_idx"]]
        target_layer.register_forward_pre_hook(teacher_act_add_hook)
        if self.accelerator.is_main_process:
            print(f"SafeSteer✅ Attached ActAdd hook to Teacher "
                  f"(layer={self.refusal_state['layer_idx']}, coeff={self.refusal_state['coeff']})")

    def load_and_sample_datasets_for_refusal_direction(self, seed):
        """
        Load datasets and sample them based on the configuration.

        Returns:
            Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
        """
        random.seed(seed)
        harmful_train = random.sample(load_dataset_split(harmtype='harmful', split='train', instructions_only=True), 128)
        harmless_train = random.sample(load_dataset_split(harmtype='harmless', split='train', instructions_only=True), 128)
        harmful_val = random.sample(load_dataset_split(harmtype='harmful', split='val', instructions_only=True), 32)
        harmless_val = random.sample(load_dataset_split(harmtype='harmless', split='val', instructions_only=True), 32)
        
        return harmful_train, harmless_train, harmful_val, harmless_val

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In DistilTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "teacher_prompt", "image", "images"]

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch (i.e., `per_device_batch_size), our dataloader loads an
    # *generation* batch (i.e., `per_device_batch_size × steps_per_generation`). This allows us to generate completions
    # once every steps_per_generation step—rather than once per accumulation step—which is significantly more
    # efficient. The only change from the original implementation is multiplying the batch size by
    # `steps_per_generation`. Thus, `_prepare_inputs` is called with this *generation* batch, and it handles the
    # splitting internally.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification. As a result, some parts of the method aren't relevant to Distil, but we keep them to stay one line
    # apart from the super method, ensuring easier maintenance in the future.
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.steps_per_generation,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )

            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self, dataset: Optional[Dataset] = None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    @profiling_decorator
    def _get_last_hidden_state(
        self,
        unwrapped_model,
        input_ids,
        attention_mask,
        logits_to_keep,
        pixel_values=None,
        image_grid_thw=None,
        pixel_attention_mask=None,
        image_sizes=None,
    ):
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model

        # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # For Qwen models:
        if image_grid_thw is not None and pixel_values is not None:
            model_inputs["image_grid_thw"] = image_grid_thw
        # For Gemma, SmolVLM2, LLaVa-Next etc.:
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        # For SmolVLM2
        if pixel_attention_mask is not None:
            model_inputs["pixel_attention_mask"] = pixel_attention_mask
        # For LLaVa-Next
        if image_sizes is not None:
            model_inputs["image_sizes"] = image_sizes

        # Only add logits_to_keep if the model supports it
        if "logits_to_keep" in self.model_kwarg_keys:
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

        last_hidden_state = unwrapped_model.model(**model_inputs).last_hidden_state
        # Exclude the last value: it corresponds to the next token pred
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
        last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    def get_high_entropy_mask(self, entropies: torch.Tensor, mask: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Returns a binary mask identifying tokens whose entropy exceeds a given quantile threshold.

        Args:
            entropies (`torch.Tensor`):
                Tensor of shape (batch_size, seq_len) with per-token entropy values.
            mask (`torch.Tensor`):
                Binary mask of the same shape as `entropies`, where `1` indicates valid tokens and `0` padding.
            threshold (`float`):
                Quantile threshold between `0.0` and `1.0` to select high-entropy tokens.

        Returns:
            `torch.Tensor`:
                Boolean mask of shape (batch_size, seq_len), where `True` indicates tokens with entropy >= threshold
                and `False` otherwise.
        """
        local = entropies[mask.bool()].float()

        # Use a negative pad_value as a sentinel because entropy values are always >= 0.
        # This guarantees that the sentinel cannot collide with any real entropy value.
        pad_value = -1e9

        # Pad across processes so that every rank has the same tensor length
        padded = self.accelerator.pad_across_processes(local, dim=0, pad_index=pad_value)
        gathered = self.accelerator.gather(padded)

        # Drop sentinel values (safe because no entropy can be negative)
        gathered = gathered[gathered != pad_value]

        if gathered.numel() == 0:
            return torch.zeros_like(entropies, dtype=torch.bool)

        entropy_threshold = torch.quantile(gathered, threshold)
        masked_entropies = entropies * mask.float()
        entropy_mask = masked_entropies >= entropy_threshold
        return entropy_mask & mask.bool()  # ensure padding tokens are always masked out

    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
        compute_all_logps=True,
    ) -> dict[str, Optional[torch.Tensor]]:
        """Compute per-token log-probs (and optionally entropies) over the last
        `logits_to_keep` positions of each sequence.

        Works for both the Student (cuda:0) and the Teacher (cuda:1): inputs
        are moved to the model's own device for the forward pass, then logits
        are moved back to the caller's device. The input batch is chunked to
        keep peak memory bounded.

        Returns
        -------
        selected_logps : (B, K)    log-prob of the actual completion tokens
        logps          : (B, K, V) full per-token log-prob distribution,
                         or None if compute_all_logps=False
        entropies      : (B, K)    per-token entropy,
                         or None if compute_entropy=False
        """
        # ----- 1. Device bookkeeping ---------------------------------------
        # Student is on cuda:0 and Teacher on cuda:1. Forward on the model's
        # own device, then bring logits back to where the caller's input lived.
        target_device = next(model.parameters()).device
        original_device = input_ids.device

        # Chunk the batch to bound peak activation memory.
        batch_size = batch_size or input_ids.size(0)
        all_selected_logps = []
        all_logps = []
        all_entropies = []

        # ----- 2. Batched forward loop -------------------------------------
        for start in range(0, input_ids.size(0), batch_size):
            # 2.1 Slice the current chunk.
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            # 2.2 Build model_inputs.
            # The multimodal branches below are no-ops for our text-only safety
            # datasets (all image / multimodal kwargs are None on entry).
            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
            if image_grid_thw is not None and pixel_values is not None:
                rows_per_image = image_grid_thw.prod(dim=-1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
                row_start, row_end = cum_rows[start].item(), cum_rows[start + batch_size].item()
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
                model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size]
            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[start : start + batch_size]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start : start + batch_size]
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[start : start + batch_size]

            # 2.3 Optional model kwargs.
            # `logits_to_keep`: ask the model to only compute the last K+1
            # logits if it supports it (older / VLM models may not). The +1 is
            # the next-token-pred position that gets dropped in 2.5 below.
            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            # use_cache is only meaningful during generation; turn it off here
            # to silence the framework warning during training forward passes.
            model_inputs["use_cache"] = False

            # 2.4 Move all input tensors to the model's device, run forward,
            # and bring the logits back to the caller's device.
            for k, v in model_inputs.items():
                if isinstance(v, torch.Tensor):
                    model_inputs[k] = v.to(target_device)

            logits = model(**model_inputs).logits
            logits = logits.to(original_device)

            # 2.5 Trim + scale logits:
            #   - drop the last position (it's the next-token pred beyond the seq)
            #   - keep only the last `logits_to_keep` positions (the completion)
            #   - divide by sampling temperature (see HF RLHF blog, section
            #     "policy training implementation details")
            logits = logits[:, :-1, :]                       # (B, L-1, H)
            logits = logits[:, -logits_to_keep:, :]          # (B, logits_to_keep, H)
            logits = logits / self.temperature

            # 2.6 Per-token logprobs of the actual completion tokens, plus the
            # full log-prob distribution and entropy if requested.
            completion_ids = input_ids_batch[:, -logits_to_keep:]
            selected_logps = selective_log_softmax(logits, completion_ids)
            if compute_all_logps:
                logps = log_softmax(logits, dim=-1)
            else:
                logps = None
            all_selected_logps.append(selected_logps)
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        # ----- 3. Concatenate across chunks --------------------------------
        selected_logps = torch.cat(all_selected_logps, dim=0)
        if compute_all_logps:
            logps = torch.cat(all_logps, dim=0)
        else:
            logps = None
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return selected_logps, logps, entropies

    @profiling_decorator
    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(generation_batch)
                torch.cuda.empty_cache()
                generation_batch = split_pixel_values_by_grid(generation_batch)
                generation_batch = shuffle_sequence_dict(generation_batch)
                generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                self._buffered_inputs = [unsplit_pixel_values_by_grid(batch) for batch in generation_batches]
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    @profiling_decorator
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):  # Module (no PretrainedModel) for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items() if key != "trainer_state"
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func

    def _generate_single_turn(self, prompts: list[str], images: Optional[list]):
        """Run a single-turn generation pass with the Student model.

        Branches on `use_transformers_paged`:
          - True  : paged-attention `generate_batch` (faster batched gen)
          - False : standard `model.generate()`

        Returns
        -------
        prompt_ids     : list[list[int]] — per-sample prompt token ids
        completion_ids : list[list[int]] — per-sample completion token ids
        logprobs       : None (scored later via _get_per_token_logps_and_entropies)
        forward_kwargs : dict — image-related kwargs to pass through to
                         downstream forwards (empty when images is None)
        """
        device = self.accelerator.device

        # ----- 1. Prompt prep ----------------------------------------------
        # 1.1 Multimodal expansion (no-op for our text-only safety datasets).
        # For multimodal data, expand each conversational user turn from a
        # plain text "content" into a list of {"type": "image"} +
        # {"type": "text"} parts so the chat template renders image tokens
        # correctly.
        kwargs = {}
        if images is not None:
            kwargs = {"images": images}
            for prompt, image_list in zip(prompts, images):
                if isinstance(prompt, list):
                    prepare_multimodal_messages(prompt, num_images=len(image_list))

        # 1.2 Render prompts via the tokenizer's chat template.
        prompts_text = [
            maybe_apply_chat_template({"prompt": prompt}, self.processing_class)["prompt"] for prompt in prompts
        ]

        # 1.3 Extract image tensors (pixel_values etc.) for downstream forward
        # calls. forward_kwargs stays empty in the text-only case.
        if images is not None:
            prompt_inputs = self.processing_class(text=prompts_text, padding=True, return_tensors="pt", **kwargs)
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        else:
            forward_kwargs = {}

        # ----- 2. Generation -----------------------------------------------
        if self.use_transformers_paged:
            # 2.1 Paged generation path (use_transformers_paged=True).
            # Faster batched generation using transformers' paged-attention
            # kernels; the attention impl is temporarily switched on the
            # Student and restored after the call.
            torch.cuda.empty_cache()
            paged_prompt_inputs = self.processing_class(text=prompts_text, **kwargs)
            previous_attn = self.model_wrapped.config._attn_implementation

            # Switch attention impl: paged_attention if FA2 is built, else sdpa_paged.
            if is_flash_attn_2_available():
                self.model_wrapped.config._attn_implementation = "paged_attention"
            else:
                self.model_wrapped.config._attn_implementation = "sdpa_paged"
            with (
                profiling_context(self, "transformers.generate_batch"),
                unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model,
                torch.no_grad(),
            ):
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)
                with torch.inference_mode():
                    all_outputs = unwrapped_model.generate_batch(
                        paged_prompt_inputs.input_ids, generation_config=self.generation_config, progress_bar=False
                    )
                    unwrapped_model.train()  # generate_batch forces eval; restore train mode

            completion_ids = [output.generated_tokens for output in all_outputs.values()]
            prompt_ids = paged_prompt_inputs.input_ids
            # Restore the original attention impl for downstream forward/backward.
            self.model_wrapped.config._attn_implementation = previous_attn
            logprobs = None  # paged generate_batch does not return logprobs

        else:
            # 2.2 Regular generation path (use_transformers_paged=False).
            generate_inputs = self.processing_class(
                text=prompts_text,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                max_length=self.max_prompt_length,
                truncation=True,
                add_special_tokens=False,
                **kwargs,
            )
            generate_inputs = super()._prepare_inputs(generate_inputs)

            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model,
                torch.no_grad(),
            ):
                prompt_completion_ids = unwrapped_model.generate(
                    **generate_inputs, generation_config=self.generation_config, disable_compile=True
                )

            # 2.3 Split out the completion (everything after the prompt).
            prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # 2.4 Mask everything after the first EOS so pad/garbage tokens are dropped.
            is_eos = completion_ids == self.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

            # 2.5 Convert to per-sample lists, dropping pad positions.
            prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool())]
            completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool())]
            logprobs = None  # we score completions later via _get_per_token_logps_and_entropies

        return prompt_ids, completion_ids, logprobs, forward_kwargs

    def _generate(self, prompts: list[str], images: Optional[list]):
        """Generate completions for `prompts` and log generation metrics.

        Thin wrapper around `_generate_single_turn` that adds:
          - aggregating prompt / completion lengths across processes
          - logging length stats, num_tokens, and EOS-vs-truncation ratio

        Returns
        -------
        prompt_ids              : list[list[int]] — per-sample prompt token ids
        completion_ids          : list[list[int]] — per-sample completion token ids
        total_completion_tokens : int — sum of completion tokens across the
                                  global batch (= num_items_in_batch, used
                                  by the DAPO loss scaler)
        logprobs                : None (forwarded from _generate_single_turn)
        forward_kwargs          : dict (forwarded from _generate_single_turn)
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # ----- 1. Run generation -------------------------------------------
        prompt_ids, completion_ids, logprobs, forward_kwargs = self._generate_single_turn(prompts, images)

        # ----- 2. Aggregate lengths across processes -----------------------
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()
        total_completion_tokens = agg_completion_lengths.sum()  # = num_items_in_batch, required for the DAPO loss

        # ----- 3. Logging --------------------------------------------------
        # 3.1 Cumulative input-token count (training only).
        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + total_completion_tokens).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # 3.2 Completion-length distribution.
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # 3.3 Truncated vs EOS-terminated. A completion is "truncated" if its
        # last token is neither EOS nor PAD (i.e., generation hit
        # max_new_tokens before producing EOS).
        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:  # edge case: no terminated sequences in batch
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        return prompt_ids, completion_ids, total_completion_tokens, logprobs, forward_kwargs

    def _generate_with_teacher(self, prompts: list[str], images: Optional[list]):
        """Generate completions from the Teacher (self.ref_model) for debug logging.

        Used by `log_teacher_completions`; never participates in the training
        loss. The ActAdd hook on the Teacher fires automatically during this
        forward pass (since refusal_state["is_active"] is True), so the output
        reflects the *steered* Teacher distribution.

        The Teacher is a raw PreTrainedModel on cuda:1 in .eval() mode, with
        no accelerator wrapping — so we skip `unwrap_model_for_generation`,
        skip the `.train()` restore, skip the dtype cast, and manually move
        inputs to the Teacher's device.

        Returns
        -------
        prompt_ids     : list[list[int]] — per-sample prompt token ids
        completion_ids : list[list[int]] — per-sample completion token ids
        """
        teacher_device = next(self.ref_model.parameters()).device

        # ----- 1. Prompt prep ----------------------------------------------
        # 1.1 Multimodal expansion (no-op for our text-only safety datasets).
        # For multimodal data, expand each conversational user turn from a
        # plain text "content" into a list of {"type": "image"} +
        # {"type": "text"} parts so the chat template renders image tokens
        # correctly.
        kwargs = {}
        if images is not None:
            kwargs = {"images": images}
            for prompt, image_list in zip(prompts, images):
                if isinstance(prompt, list):
                    prepare_multimodal_messages(prompt, num_images=len(image_list))

        # 1.2 Render prompts via the tokenizer's chat template.
        prompts_text = [
            maybe_apply_chat_template({"prompt": prompt}, self.processing_class)["prompt"] for prompt in prompts
        ]

        # 1.3 Tokenize and move all input tensors to the Teacher's device
        # (cuda:1). We bypass super()._prepare_inputs which would put them
        # on the trainer's default device (cuda:0).
        generate_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
            **kwargs,
        )
        generate_inputs = {                                                                                                                                                                                                                                                                                                                                                                                      
            k: (v.to(teacher_device) if isinstance(v, torch.Tensor) else v)
            for k, v in generate_inputs.items()                                                                                                                                                                                                                                                                                                                                                                  
        }

        # ----- 2. Generation -----------------------------------------------
        with torch.no_grad():
            prompt_completion_ids = self.ref_model.generate(
                **generate_inputs,
                generation_config=self.generation_config,
                disable_compile=True,
            )

        # ----- 3. Post-process ---------------------------------------------
        # 3.1 Split out the completion (everything after the prompt).
        prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # 3.2 Mask everything after the first EOS so pad/garbage tokens are dropped.
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1),
                             dtype=torch.long, device=teacher_device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=teacher_device) \
                                .expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # 3.3 Convert to per-sample Python lists, dropping pad positions
        # (matches _generate_single_turn's regular path).
        prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool())]
        completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool())]
        return prompt_ids, completion_ids

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """End-to-end preprocessing of one generation batch into the dict
        consumed by `_compute_loss`.

        Pipeline:
          1. Parse `inputs` into prompts / teacher_prompts / images.
          2. Run generations: Student (always) + optional Teacher (debug).
          3. Re-tokenize both prompt variants for downstream use.
          4. Pad everything into padded tensors and optionally mask
             truncated completions.
          5. Build concatenated (prompt + completion) sequences for the
             scoring forward pass.
          6. Compute reference-model per-token logps
             (old_per_token_logps for importance sampling,
              ref_per_token_logps for KL).
          7. Decode for wandb display + populate the bounded log buffers.
          8. Assemble and return the output dict.
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # ----- 1. Parse inputs ---------------------------------------------
        # 1.1 Prompts (un-steered) and teacher_prompts (steering wrapper).
        prompts = [x["prompt"] for x in inputs]
        teacher_prompts = [x["teacher_prompt"] for x in inputs]

        # 1.2 Images (no-op for our text-only datasets, kept for VLM compat).
        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        # ----- 2. Run generations ------------------------------------------
        # 2.1 Student always generates the training-side completions, using
        # the plain (un-steered) prompts.
        (
            _generation_prompt_ids_list,  # discard — student/teacher prompt IDs are recomputed below
            completion_ids_list,
            num_items_in_batch,
            sampling_per_token_logps_list,
            forward_kwargs,
        ) = self._generate(prompts, images)

        # 2.2 Debug only: have the steered Teacher generate a separate
        # completion from teacher_prompts so we can inspect the effect of
        # refusal-vector / system-prompt injection. Off by default; never
        # affects training.
        if getattr(self.args, "log_teacher_completions", False):
            _, completion_ids_list_teacher = self._generate_with_teacher(teacher_prompts, images)
        else:
            completion_ids_list_teacher = None

        # ----- 3. Re-tokenize prompt variants ------------------------------
        # 3.1 Student prompts (plain, un-steered).
        prompts_text = [
            maybe_apply_chat_template({"prompt": prompt}, self.processing_class)["prompt"] for prompt in prompts
        ]

        student_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
        )
        student_inputs = super()._prepare_inputs(student_inputs)
        student_prompt_ids, student_prompt_mask = student_inputs["input_ids"], student_inputs["attention_mask"]
        prompt_ids_list = [p[m].tolist() for p, m in zip(student_prompt_ids, student_prompt_mask.bool())]

        # 3.2 Teacher prompts (carry the steering wrapper / system prompt).
        teacher_prompts_text = [
            maybe_apply_chat_template({"prompt": prompt}, self.processing_class)["prompt"] for prompt in teacher_prompts
        ]

        teacher_inputs = self.processing_class(
            text=teacher_prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
        )
        teacher_inputs = super()._prepare_inputs(teacher_inputs)

        teacher_prompt_ids, teacher_prompt_mask = teacher_inputs["input_ids"], teacher_inputs["attention_mask"]
        teacher_prompt_ids_list = [p[m].tolist() for p, m in zip(teacher_prompt_ids, teacher_prompt_mask.bool())]

        # ----- 4. Pad everything into padded tensors -----------------------
        # 4.1 Student prompt ids/mask (left-padded).
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")

        # 4.2 Teacher prompt ids/mask (left-padded).
        teacher_prompt_ids = [torch.tensor(ids, device=device) for ids in teacher_prompt_ids_list]
        teacher_prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in teacher_prompt_ids]
        teacher_prompt_ids = pad(teacher_prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        teacher_prompt_mask = pad(teacher_prompt_mask, padding_value=0, padding_side="left")

        # 4.3 Completion ids/mask (right-padded) + optional Teacher debug
        # completion.
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        if completion_ids_list_teacher is not None:
            completion_ids_teacher = [torch.tensor(ids, device=device) for ids in completion_ids_list_teacher]
            completion_ids_teacher = pad(completion_ids_teacher, padding_value=self.pad_token_id, padding_side="right")
        else:
            completion_ids_teacher = None
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")

        # 4.4 Optional: per-token logps from the sampling pass (used by some
        # loss variants for importance-sampling correction).
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None

        # 4.5 Optional: zero out completion_mask on truncated samples so the
        # loss never trains on incomplete generations.
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # ----- 5. Build concatenated sequences for the forward pass --------
        # Build (prompt + completion) for both Student and Teacher prompt
        # variants. The completion segment is shared between them.
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
        teacher_prompt_completion_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)  # (B, P+C)
        teacher_attention_mask = torch.cat([teacher_prompt_mask, completion_mask], dim=1)  # (B, P+C)
        # Extend token_type_ids with zeros over the completion segment (only
        # fires for models that use token_type_ids; no-op otherwise).
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        num_images = [len(img_list) for img_list in images] if images is not None else None

        # ----- 6. Reference-model per-token logps --------------------------
        with torch.no_grad():
            # 6.1 old_per_token_logps for importance sampling.
            # If the generation and optimization steps are misaligned — i.e.,
            # if generation does not occur at the end of a full optimizer
            # step (when gradient_accumulation_steps is not a multiple of
            # generate_every) — then the samples may come from an earlier
            # version of the model. In that case we need to track
            # old_per_token_logps for importance sampling. If the steps are
            # aligned, importance sampling is unnecessary and we leave it None.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0:
                old_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    compute_all_logps=False,
                    **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                )
            else:
                old_per_token_logps = None

            # 6.2 ref_per_token_logps from the Teacher (or, for PEFT setups,
            # the Student with adapter disabled). Used by the KL penalty
            # when beta != 0.
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        compute_all_logps=False,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            compute_all_logps=False,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                        )
            else:
                ref_per_token_logps = None

        # ----- 7. Decode + populate log buffers ----------------------------
        # 7.1 Decode for wandb display.
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        teacher_prompts_text = self.processing_class.batch_decode(teacher_prompt_ids, skip_special_tokens=True)

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        completions_text_teacher = (
            self.processing_class.batch_decode(completion_ids_teacher, skip_special_tokens=True)
            if completion_ids_teacher is not None else None
        )
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # 7.2 Placeholder rewards / advantages — kept as zeros; SafeSteer's
        # loss does not actually consume them, but the buffers still need to
        # be populated for downstream `log()`.
        rewards = torch.zeros_like(completion_ids, dtype=torch.float32)
        advantages = rewards

        # Keep a copy for logging (data is already local to each process, no slicing needed)
        all_process_advantages = advantages.clone()

        # 7.3 Extend the bounded log buffers consumed by `log()`.
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["teacher_prompt"].extend(gather_object(teacher_prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        self._logs["rewards"]["main"].extend(gather_object(rewards.mean(dim=-1).tolist()))
        self._logs["advantages"].extend(gather_object(all_process_advantages.mean(dim=-1).tolist()))
        if completions_text_teacher is not None:
            self._logs["completion_teacher"].extend(gather_object(completions_text_teacher))
        reward_to_log = rewards.clone()
        reward_to_log = reward_to_log[completion_mask.bool()]
        mean_reward = torch.mean(reward_to_log) if reward_to_log.numel() > 0 else torch.tensor(0.0, device=device)
        self._metrics[mode]["rewards"].append(self.accelerator.gather(mean_reward).mean().item())

        if images is not None:
            self._logs["images"].extend(gather_object(images))

        # ----- 8. Build output dict ----------------------------------------
        # Keys here are exactly what `_compute_loss` reads downstream.
        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "teacher_prompt_ids": teacher_prompt_ids,
            "teacher_prompt_mask": teacher_prompt_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if images is not None:
            output["num_images"] = num_images
        return output

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The DistilTrainer does not support returning outputs")
        return self._compute_loss(model, inputs)
    
    def _compute_loss(self, model, inputs):
        """Compute the SafeSteer distillation loss.

        Pipeline:
          1. Unpack inputs from `_generate_and_score_completions`.
          2. Build loss_completion_mask (with skip / keep first N tokens).
          3. Concatenate (prompt + completion) for Student and Teacher.
          4. Forward passes: Student (with entropy) + Teacher (no grad).
          5. Optional entropy mask (keep only top-quantile tokens).
          6. Optional KL-to-init-model penalty (when beta != 0).
          7. Vocabulary selection on logps
             (mode 1: top-K Teacher; mode 2: safe_tokens set).
          8. KL divergence between Student and Teacher
             (alpha = 0 forward, 1 reverse, in-between Generalized JSD).
          9. Reduce to scalar loss + grad-accum scaling.
         10. Log metrics (kl_approx, optional kl_to_base_model, entropy).
        """
        # ----- 1. Unpack inputs --------------------------------------------
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        teacher_prompt_ids, teacher_prompt_mask = inputs["teacher_prompt_ids"], inputs["teacher_prompt_mask"]

        # ----- 2. Build loss_completion_mask -------------------------------
        # completion_mask drives both the attention forward pass and the loss.
        # We keep the original for attention but build a separate mask for the
        # loss so we can optionally skip the leading N tokens and/or keep
        # only the first M tokens.
        loss_completion_mask = completion_mask
        if self.num_loss_tokens_to_skip > 0 or self.num_loss_tokens_to_keep > 0:
            batch_size, seq_len = completion_mask.shape
            token_positions = torch.arange(seq_len, device=completion_mask.device).unsqueeze(0).expand(batch_size, -1)
            if self.num_loss_tokens_to_skip > 0:
                skip_mask = (token_positions >= self.num_loss_tokens_to_skip).int()
                loss_completion_mask = loss_completion_mask * skip_mask
            if self.num_loss_tokens_to_keep > 0:
                keep_mask = (token_positions < self.num_loss_tokens_to_keep).int()
                loss_completion_mask = loss_completion_mask * keep_mask

        # ----- 3. Build concatenated sequences -----------------------------
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        teacher_input_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # ----- 4. Forward passes -------------------------------------------
        # 4.1 Student: per-token logps, full distribution, and entropy.
        per_token_logps, all_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )
        torch.cuda.empty_cache()

        # 4.2 Teacher (no grad). The ActAdd hook auto-steers the distribution
        # since refusal_state["is_active"] is True.
        with torch.no_grad():
            teacher_per_token_logps, teacher_all_logps, teacher_entropies = self._get_per_token_logps_and_entropies(
                self.ref_model,
                teacher_input_ids,
                teacher_attention_mask,
                logits_to_keep,
                compute_entropy=True,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                num_images=inputs.get("num_images"),
                pixel_attention_mask=inputs.get("pixel_attention_mask"),
                image_sizes=inputs.get("image_sizes"),
                token_type_ids=inputs.get("token_type_ids"),
            )

        # ----- 5. Optional entropy mask ------------------------------------
        # Keep only positions whose Student entropy falls in the top quantile.
        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, loss_completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # ----- 6. Optional KL-to-init-model penalty ------------------------
        # When beta != 0, add a per-token KL between the current Student and
        # the reference logps captured in `_generate_and_score_completions`.
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # ----- 7. Vocabulary selection on logps ----------------------------
        # Narrow the KL comparison to a token subset so the Student is steered
        # only on "safe" tokens instead of the full vocab.
        if self.args.voca_selection_mode == 1:
            # Mode 1: top-K of the Teacher's distribution at each position.
            k = self.args.voca_selection_num
            _, topk_indices = teacher_all_logps.topk(k, dim=-1)
            all_logps = all_logps.gather(-1, topk_indices)
            teacher_all_logps = teacher_all_logps.gather(-1, topk_indices)
            if getattr(self.args, "renormalize_selected_tokens", False):
                # Ablation branch: compare only the relative distribution
                # within the safe set (empirically over-refuses).
                all_logps = all_logps - all_logps.logsumexp(dim=-1, keepdim=True)
                teacher_all_logps = teacher_all_logps - teacher_all_logps.logsumexp(dim=-1, keepdim=True)
        elif self.args.voca_selection_mode == 2:
            # Mode 2: a fixed safe_tokens set (precomputed at init and
            # optionally refreshed by the safe-token callback).
            safe_tokens = self.refusal_state["safe_tokens"].to(all_logps.device)  # [k]
            k = safe_tokens.size(0)
            safe_indices = safe_tokens.unsqueeze(0).unsqueeze(0).expand(
                all_logps.size(0), all_logps.size(1), k
            )
            all_logps = all_logps.gather(-1, safe_indices)
            teacher_all_logps = teacher_all_logps.gather(-1, safe_indices)

            if getattr(self.args, "renormalize_selected_tokens", False):
                # Ablation branch: compare only the relative distribution
                # within the safe set (empirically over-refuses).
                all_logps = all_logps - all_logps.logsumexp(dim=-1, keepdim=True)
                teacher_all_logps = teacher_all_logps - teacher_all_logps.logsumexp(dim=-1, keepdim=True)

        # ----- 8. KL divergence loss ---------------------------------------
        # alpha controls the direction:
        #   alpha = 0  -> forward KL  (KL(Teacher || Student))
        #   alpha = 1  -> reverse KL  (KL(Student || Teacher))
        #   0 < alpha < 1 -> Generalized Jensen-Shannon Divergence
        # Note: PyTorch's F.kl_div argument order differs from the standard
        # mathematical definition, so the order of the two distributions is
        # swapped relative to the paper.
        if self.alpha == 0: #Forward KL
            kl_loss = kl_div(all_logps, teacher_all_logps, reduction="none", log_target=True)
        elif self.alpha == 1: #Reverse KL
            kl_loss = kl_div(teacher_all_logps, all_logps, reduction="none", log_target=True)
        else:
            # Compute the log of the mixture distribution
            # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
            alpha = torch.tensor(self.alpha, dtype=all_logps.dtype)
            mixture_log_probs = torch.logsumexp(
                torch.stack([all_logps + torch.log(1 - alpha), teacher_all_logps + torch.log(alpha)]),
                dim=0,
            )

            kl_teacher = kl_div(mixture_log_probs, teacher_all_logps, reduction="none", log_target=True)
            kl_student = kl_div(mixture_log_probs, all_logps, reduction="none", log_target=True)

            # Compute the Generalized Jensen-Shannon Divergence
            kl_loss = alpha * kl_teacher + (1 - alpha) * kl_student
        per_token_loss = kl_loss.sum(-1)

        # Free the big vocab-sized tensors before the final reduction.
        del all_logps
        del teacher_all_logps
        if 'mixture_log_probs' in locals():
            del mixture_log_probs
        torch.cuda.empty_cache()

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        # ----- 9. Reduce to scalar loss ------------------------------------
        # Per-sample mean over valid completion positions, then batch mean,
        # then divide by gradient accumulation steps (we set
        # compute_loss_func to a non-None value in __init__ to disable the
        # parent's automatic gradient-accumulation scaling).
        loss = ((per_token_loss * loss_completion_mask).sum(-1) / loss_completion_mask.sum(-1).clamp(min=1.0)).mean()
        loss = loss / self.current_gradient_accumulation_steps

        # ----- 10. Log metrics ---------------------------------------------
        mode = "train" if self.model.training else "eval"

        # 10.1 kl_approx: unbiased estimator of KL(Student || Teacher) per
        # http://joschu.net/blog/kl-approx.html (k3 form).
        with torch.no_grad():
            kl_approx = (per_token_logps - teacher_per_token_logps) + torch.exp(teacher_per_token_logps - per_token_logps) - 1
            kl_approx_mean = (kl_approx * loss_completion_mask).sum() / loss_completion_mask.sum()
        self._metrics[mode]["kl_approx"].append(self.accelerator.gather(kl_approx_mean).nanmean().item())

        loss_completion_token_count = loss_completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * loss_completion_mask).sum() / loss_completion_token_count

        # 10.2 KL to the init-model reference (only when beta != 0).
        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl_to_base_model"].append(self.accelerator.gather(mean_kl).nanmean().item())

        # 10.3 Student entropy averaged over valid positions.
        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        return loss


    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """Aggregate buffered metrics and emit them to the parent log + wandb.

        Steps:
          1. Average all scalars in `_metrics[mode]` and (in eval mode)
             prefix with "eval_" to match HF's convention.
          2. Forward to `super().log()` and clear the buffer.
          3. If log_completions is set, pretty-print a sample of
             prompts / completions to the console (main process only).
          4. If wandb is configured, upload a per-step table of prompts /
             completions / rewards / (optional) teacher completions and
             images.
        """
        # ----- 1. Aggregate scalar metrics ---------------------------------
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        # ----- 2. Forward to parent log() + clear buffer -------------------
        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

        # Sections 3 and 4 below are completion-table logging; main process only.
        if self.accelerator.is_main_process and self.log_completions:
            # ----- 3. Console pretty-print sample --------------------------
            if is_rich_available():
                print_prompt_completions_sample(
                    self._logs["prompt"],
                    self._logs["completion"],
                    self._logs["completion_teacher"] if self.args.log_teacher_completions else [""] * len(self._logs["prompt"]),
                    self._logs["rewards"],
                    self._logs["advantages"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            # ----- 4. Wandb completions table ------------------------------
            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                # 4.1 Build the table columns. teacher_completions and images
                # are optional and are conditionally appended.
                table = {
                    "step": [str(self.state.global_step)] * len(self._logs["prompt"]),
                    "prompt": self._logs["prompt"],
                    "teacher_prompt": self._logs["teacher_prompt"],
                    "completion": self._logs["completion"],
                    **self._logs["rewards"],
                    "advantage": self._logs["advantages"],
                }
                if self.args.log_teacher_completions and len(self._logs["completion_teacher"]) == len(self._logs["prompt"]):
                    table["completion_teacher"] = self._logs["completion_teacher"]

                if self._logs["images"]:
                    table["images"] = []
                    for image_list in self._logs["images"]:
                        # Convert images to wandb Image objects for proper visualization
                        table["images"].append([wandb.Image(image) for image in image_list])

                # 4.2 Build DataFrame, optionally dedupe by prompt, push to wandb.
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                wandb.log({"completions": wandb.Table(dataframe=df)})

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)

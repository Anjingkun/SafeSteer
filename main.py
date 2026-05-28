import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from distil_trainer import DistilTrainer
from distil_config import DistilConfig
from utils.main_utils import UploadRunConfigCallback, parse_args, validate_training_arguments, construct_output_dir, load_safety_dataset

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAFE_SYSTEM_PROMPT = None

if __name__ == "__main__":
    args = parse_args(_PROJECT_ROOT)

    # =====================================================================
    # 1. Strict Constraint Checks & Logical Validations
    # =====================================================================

    validate_training_arguments(args)    

    # Define the system prompt for prompt-based steering early so it can be saved in config
    SYSTEM_PROMPT = None
    if not args.use_refusal_vector:
        SYSTEM_PROMPT = "You are a safety-conscious assistant. Never produce harmful, unsafe, or disallowed content"

    # =====================================================================
    # 2. Output Directory Construction
    # =====================================================================

    args.output_dir = construct_output_dir(args, _PROJECT_ROOT)

    # =====================================================================
    # 3. Model & Tokenizer Initialization
    # =====================================================================
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Hotfix for Llama-3 specific chat templates to remove date string hardcoding
    if "llama-3" in args.model_name.lower() and tokenizer.chat_template:
        ct = tokenizer.chat_template
        ct = ct.replace('{{- "Cutting Knowledge Date: December 2023\\n" }}\n', '')
        ct = ct.replace('{{- "Today Date: " + date_string + "\\n\\n" }}\n', '')
        tokenizer.chat_template = ct

    # =====================================================================
    # 4. Configuration Setup
    # =====================================================================
    
    config = DistilConfig(
        alpha=args.alpha, 
        seed=args.seed,
        use_transformers_paged=True,
        use_vllm=False,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1, 
        vllm_gpu_memory_utilization=0.3,
        vllm_enable_sleep_mode=True, 
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.num_prompts_per_batch,
        max_prompt_length=512,
        max_completion_length=512,
        model_name=args.model_name,
        num_train_epochs=args.num_train_epochs,
        save_steps=100,
        max_grad_norm=1,
        report_to="wandb",
        run_name=os.path.basename(args.output_dir),
        use_refusal_vector=args.use_refusal_vector,
        safe_system_prompt = SYSTEM_PROMPT,
        update_refusal_vector=args.update_refusal_vector,
        freeze_safe_token=args.freeze_safe_token,
        output_dir=args.output_dir,
        log_completions=True, # Set True for default debugging
        log_teacher_completions=args.log_teacher_completions,
        sync_ref_model=not args.freeze_teacher,
        ref_model_sync_steps=1,
        ref_model_mixup_alpha=args.ref_model_mixup_alpha,
        vllm_importance_sampling_correction=True,
        num_loss_tokens_to_skip=args.num_loss_tokens_to_skip,
        num_loss_tokens_to_keep=args.num_loss_tokens_to_keep,
        voca_selection_mode=args.voca_selection_mode, 
        voca_selection_num=args.voca_selection_num, 
        safe_token_horizon=args.safe_token_horizon,  
        selection_method=args.selection_method,
        vote_top_k_inner=args.vote_top_k_inner,
        num_samples_per_prompt=args.num_samples_per_prompt,
        safe_token_temperature=args.safe_token_temperature,
        safe_token_top_p=args.safe_token_top_p,
        exclude_special_tokens=args.exclude_special_tokens,
        min_steered_prob=args.min_steered_prob,
        renormalize_safe_tokens=args.renormalize_selected_tokens, # Mapped from argparse
    )
    
    config._n_gpu = 1

    # =====================================================================
    # 5. Save Run Config & Execute Training
    # =====================================================================
    
    run_info = {
        "args": vars(args),
        "config": config.to_dict() if hasattr(config, "to_dict") else {k: v for k, v in vars(config).items() if not k.startswith("_")},
        "safe_system_prompt": config.safe_system_prompt,
    }
    
    run_config_path = os.path.join(args.output_dir, "run_config.json")
    with open(run_config_path, "w") as f:
        json.dump(run_info, f, indent=2, default=str, ensure_ascii=False)

    dataset = load_safety_dataset(args.seed, args.train_path, config)
    
    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[UploadRunConfigCallback(run_config_path)],
    )
    
    trainer.train()
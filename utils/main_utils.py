import os
import json
from datasets import Dataset
import argparse
from transformers import TrainerCallback

class UploadRunConfigCallback(TrainerCallback):
    """
    A Hugging Face Trainer callback that automatically uploads a configuration 
    JSON file to Weights & Biases (wandb) at the beginning of the training run.
    """
    def __init__(self, file_path: str):
        """
        Args:
            file_path (str): The local path to the configuration file (e.g., JSON) to be uploaded.
        """
        self.file_path = file_path

    def on_train_begin(self, args, state, control, **kwargs):
        """
        Triggered automatically when the training begins.
        Attempts to log the specified config file and update the wandb run configuration.
        """
        try:
            import wandb
            
            # Exit early if wandb is not initialized or the target file does not exist
            if wandb.run is None or not os.path.exists(self.file_path):
                return
            
            # Upload the physical file to the wandb cloud for backup/reference
            wandb.save(self.file_path, base_path=os.path.dirname(self.file_path), policy="now")
            
            # Parse the JSON file and merge its contents into the current wandb run config
            with open(self.file_path) as f:
                wandb.config.update(json.load(f), allow_val_change=True)
                
        except Exception as e:
            # Catch and print exceptions to prevent logging failures from crashing the training loop
            print(f"[UploadRunConfig] failed: {e}")

def str2bool(v):
    """
    Converts a string representation of truth to True or False.
    Highly useful for `argparse`, as the default `bool` type converts any non-empty string (like "False") to True.
    
    Args:
        v (str or bool): The value to convert.
        
    Returns:
        bool: The parsed boolean value.
        
    Raises:
        argparse.ArgumentTypeError: If the string cannot be interpreted as a valid boolean.
    """
    # If the input is already a boolean, return it directly
    if isinstance(v, bool):
        return v
        
    # Check against common string representations of True
    if v.lower() in ("true", "1", "yes", "y", "t"):
        return True
        
    # Check against common string representations of False
    if v.lower() in ("false", "0", "no", "n", "f"):
        return False
        
    # Raise a specific argparse error if the input is unrecognized
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")

def parse_args(project_root):
    parser = argparse.ArgumentParser(description="Distil Trainer configuration for guided alignment and distillation.")

    # -------------------------------------------------------------- #
    # 1. Basic Training Dynamics & Environment
    # -------------------------------------------------------------- #
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", 
                        help="Specifies the HuggingFace model ID or local directory path for the base model.")
    parser.add_argument("--train_path", type=str, default=os.path.join(project_root, 'data', 'PKU_UnSafeRLHF_100.json'), 
                        help="Specifies the file path to the training dataset.")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Specifies the directory path where model checkpoints and training logs will be saved.")
    parser.add_argument("--learning_rate", type=float, default=1e-6, 
                        help="Sets the initial learning rate for the optimizer.")
    parser.add_argument("--num_train_epochs", type=int, default=1, 
                        help="Defines the total number of training epochs to execute.")
    parser.add_argument("--num_prompts_per_batch", type=int, default=32, 
                        help="Sets the number of input prompts processed per training batch.")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Sets the random seed for environment to ensure reproducibility.")

    # -------------------------------------------------------------- #
    # 2. Distillation & KL Divergence Configuration
    # -------------------------------------------------------------- #
    parser.add_argument("--alpha", type=float, default=1.0, 
                        help="Sets the alpha weighting parameter for the KL divergence loss. A value of 1.0 represents Reverse KL, while 0.0 represents Forward KL.")
    parser.add_argument("--num_loss_tokens_to_skip", type=int, default=0, 
                        help="Defines the number of initial completion tokens to exclude from the loss computation.")
    parser.add_argument("--num_loss_tokens_to_keep", type=int, default=0,
                        help="Defines the maximum number of completion tokens to include in the loss computation. "
                             "If set to 0, all tokens are evaluated. If greater than 0, must be strictly larger than --num_loss_tokens_to_skip.")

    # -------------------------------------------------------------- #
    # 3. Teacher Model Steering & State
    # -------------------------------------------------------------- #
    parser.add_argument("--freeze_teacher", type=str2bool, default=True,
                        help="If True, fully freezes the Teacher (reference model) at its initial state. "
                             "This disables weight synchronization and prevents any mid-training refreshes of refusal directions or safe tokens.")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01, 
                        help="Sets the Exponential Moving Average (EMA) alpha for synchronizing reference model weights. "
                             "Only active if --freeze_teacher is False.")
    parser.add_argument("--use_refusal_vector", type=str2bool, default=True, 
                        help="If True, utilizes a dynamically computed refusal vector to steer the Teacher model's refusal behavior. "
                             "If False, relies on predefined system prompts for behavioral steering.")
    parser.add_argument("--update_refusal_vector", type=str2bool, default=True,
                        help="If True, recomputes the refusal direction and safe tokens at every synchronization step. "
                             "If False, locks the initial refusal direction for the entire training run."
                             "Only active if --use_refusal_vector is True.")
    parser.add_argument("--log_teacher_completions", type=str2bool, default=False, 
                        help="If True, actively generates and logs the Teacher model's text completions to tracking tools (e.g., Weights & Biases) for debugging.")

    # -------------------------------------------------------------- #
    # 4. Vocabulary Selection Strategy (Modes 0, 1, 2)
    # -------------------------------------------------------------- #
    parser.add_argument("--voca_selection_mode", type=int, default=0,
                        help="Defines the overarching strategy for selecting tokens during KL computation "
                             "(e.g., 0 = Full vocabulary, 1 = Top-K filtering, 2 = Advanced safe-token filtering).")
    parser.add_argument("--voca_selection_num", type=int, default=50, 
                        help="Sets the number of tokens to select when --voca_selection_mode is 1 or 2. "
                             "In mode 1, it acts as a strict Top-K limit. In mode 2, it specifies the exact number of safe tokens to extract and retain for the distillation process.")
    parser.add_argument("--renormalize_selected_tokens", type=str2bool, default=False,
                        help="If True, locally renormalizes the Student and Teacher log-probabilities over the filtered vocabulary subset. "
                             "If False, preserves the original, absolute full-vocabulary log-probabilities. "
                             "Only active when --voca_selection_mode != 0.")

    # -------------------------------------------------------------- # 
    # 5. Advanced Safe-Token Mechanics (Mode 2 Specifics)
    # -------------------------------------------------------------- # 
    parser.add_argument("--selection_method", type=str, default="vote", choices=["mean", "vote"],
                        help="Determines the mathematical aggregation method ('mean' or 'vote') used to derive safe tokens across multiple generated trajectories.")
    parser.add_argument("--vote_top_k_inner", type=int, default=200,
                        help="Sets the per-position Top-K threshold used internally during the voting process. "
                             "Only active when --selection_method is 'vote'.")
    parser.add_argument("--safe_token_horizon", type=int, default=1,
                        help="Specifies the number of leading generated positions to evaluate when extracting safe tokens "
                             "(e.g., a value of 1 restricts evaluation to the very first answer token).")
    parser.add_argument("--num_samples_per_prompt", type=int, default=8,
                        help="Sets the number of independent baseline trajectories to sample for each prompt. "
                             "Values strictly greater than 1 will automatically enable non-deterministic sampling (`do_sample=True`).")
    parser.add_argument("--safe_token_temperature", type=float, default=1.0,
                        help="Sets the sampling temperature used specifically for generating safe-token trajectories. "
                             "Only active if --num_samples_per_prompt > 1.")
    parser.add_argument("--safe_token_top_p", type=float, default=1.0,
                        help="Sets the Top-P (nucleus) sampling threshold used for generating safe-token trajectories. "
                             "Only active if --num_samples_per_prompt > 1.")
    parser.add_argument("--exclude_special_tokens", type=str2bool, default=True,
                        help="If True, aggressively masks all special token IDs (including standard special tokens and `<|...|>` patterns) "
                             "to prevent them from being selected as safe tokens.")
    parser.add_argument("--min_steered_prob", type=float, default=1e-6,
                        help="Sets the absolute minimum probability threshold a token must achieve under the steered Teacher "
                             "to be considered a valid safe token. Set to 0.0 to disable this threshold.")
    parser.add_argument("--freeze_safe_token", type=str2bool, default=True,
                        help="If True, safe tokens are only selected at their init state. "
                             "If False, safe tokens will be recalculated when the teacher model is updated. "
                             "Only has effect when voca_selection_mode=2 and freeze_teacher=False.")
    
    
    return parser.parse_args()

def load_safety_dataset(seed=42, train_path=None, config=None) -> Dataset:
    """Load and prepare tooluse dataset with formatted prompts."""
    train_dataset = Dataset.from_json(train_path)

    def format_example(example):

        if config.use_refusal_vector == True:
            return {
                "prompt": [{"role": "user", "content": example['instruction']}],
                "teacher_prompt": [{"role": "user", "content": example['instruction']}],
            }
        else:
            return {
            # Student gets the raw instruction
            "prompt": [
                {"role": "system", "content": ""},
                {"role": "user", "content": example['instruction']}
            ],
            # Teacher gets the safety system prompt + the instruction
            "teacher_prompt": [
                {"role": "system", "content": config.safe_system_prompt},
                {"role": "user", "content": example['instruction']}
            ],
            }   
    
    train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.shuffle(seed=seed)
    return train_dataset


def validate_training_arguments(args):
    """
    Validates all runtime arguments for logical conflicts and mathematical
    boundaries. Mirrors the 4-case decision tree in DistilTrainer.__init__:

        L1: use_refusal_vector
            True  -> "vector" branch (RV steering)
            False -> "prompt" branch (system-prompt steering)
        L2: voca_selection_mode == 2  ->  whether safe_tokens are computed
        L3: freeze_teacher            ->  whether dynamic refresh callbacks run

    Four leaf cases:
        (1) RV=True,  mode==2  : direction + safe_tokens
        (2) RV=True,  mode!=2  : direction only
        (3) RV=False, mode==2  : safe_tokens via system prompt
        (4) RV=False, mode!=2  : nothing to steer (plain SFT-like)

    Raises ValueError on hard conflicts; prints WARNINGs for arguments that
    are silently ignored under the current configuration.
    """
    # ----- [0] Universal sanity checks (apply to every case) -----------
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError(f"alpha must be between 0.0 and 1.0. Got: {args.alpha}")

    if args.num_loss_tokens_to_skip < 0:
        raise ValueError(f"num_loss_tokens_to_skip must be >= 0. Got: {args.num_loss_tokens_to_skip}")
    if args.num_loss_tokens_to_keep < 0:
        raise ValueError(f"num_loss_tokens_to_keep must be >= 0. Got: {args.num_loss_tokens_to_keep}")
    if args.num_loss_tokens_to_keep > 0 and args.num_loss_tokens_to_keep <= args.num_loss_tokens_to_skip:
        raise ValueError(
            f"num_loss_tokens_to_keep ({args.num_loss_tokens_to_keep}) must be strictly "
            f"larger than num_loss_tokens_to_skip ({args.num_loss_tokens_to_skip})."
        )

    if args.voca_selection_mode not in [0, 1, 2]:
        raise ValueError(f"voca_selection_mode must be 0, 1, or 2. Got: {args.voca_selection_mode}")

    # ----- L1 dispatch --------------------------------------------------
    if args.use_refusal_vector:
        # =====================================================================
        # RV branch (use_refusal_vector=True)
        # =====================================================================
        if args.voca_selection_mode == 2:
            # ----- Case (1): RV + mode==2 ---------------------------------
            print("\n[Case 1] RV steering + safe_tokens (use_refusal_vector=True, voca_selection_mode=2)")
            _validate_mode2_params(args)
            if args.freeze_teacher:
                print("[WARNING] Teacher is frozen (--freeze_teacher=True).")
                print("          -> --ref_model_mixup_alpha is ignored.")
                print("          -> --update_refusal_vector is ignored (no callback registered).")
                print("          -> --freeze_safe_token is ignored (safe_tokens are implicitly frozen).")
        else:
            # ----- Case (2): RV + mode!=2 ---------------------------------
            print(f"\n[Case 2] RV steering only (use_refusal_vector=True, voca_selection_mode={args.voca_selection_mode})")
            if args.voca_selection_mode == 1:
                if args.voca_selection_num is None or args.voca_selection_num <= 0:
                    raise ValueError(f"Mode 1 requires voca_selection_num > 0. Got: {args.voca_selection_num}")
            else:  # mode == 0
                print("[WARNING] Mode 0 (full vocab): --voca_selection_num and --renormalize_selected_tokens are ignored.")
            print("[WARNING] Mode!=2 exclusive parameters (selection_method, num_samples_per_prompt, "
                  "safe_token_*, vote_top_k_inner, exclude_special_tokens, min_steered_prob, "
                  "safe_token_horizon, freeze_safe_token) are all ignored.")
            if args.freeze_teacher:
                print("[WARNING] Teacher is frozen (--freeze_teacher=True).")
                print("          -> --ref_model_mixup_alpha is ignored.")
                print("          -> --update_refusal_vector is ignored (no callback registered).")
    else:
        # =====================================================================
        # Prompt branch (use_refusal_vector=False)
        # =====================================================================
        # --update_refusal_vector is structurally meaningless without RV
        if args.update_refusal_vector:
            print("\n[WARNING] use_refusal_vector=False → --update_refusal_vector is ignored "
                  "(no refusal direction exists).")

        if args.voca_selection_mode == 2:
            # ----- Case (3): prompt + mode==2 -----------------------------
            print(f"\n[Case 3] Prompt steering + safe_tokens (use_refusal_vector=False, voca_selection_mode=2)")
            _validate_mode2_params(args)
            if args.freeze_teacher:
                print("[WARNING] Teacher is frozen (--freeze_teacher=True).")
                print("          -> --ref_model_mixup_alpha is ignored.")
                print("          -> --freeze_safe_token is ignored (safe_tokens are implicitly frozen).")
        else:
            # ----- Case (4): prompt + mode!=2 -----------------------------
            print(f"\n[Case 4] Prompt steering only (use_refusal_vector=False, voca_selection_mode={args.voca_selection_mode})")
            if args.voca_selection_mode == 1:
                if args.voca_selection_num is None or args.voca_selection_num <= 0:
                    raise ValueError(f"Mode 1 requires voca_selection_num > 0. Got: {args.voca_selection_num}")
            else:  # mode == 0
                print("[WARNING] Mode 0 (full vocab): --voca_selection_num and --renormalize_selected_tokens are ignored.")
            print("[WARNING] Mode!=2 exclusive parameters (selection_method, num_samples_per_prompt, "
                  "safe_token_*, vote_top_k_inner, exclude_special_tokens, min_steered_prob, "
                  "safe_token_horizon, freeze_safe_token) are all ignored.")
            if args.freeze_teacher:
                print("[WARNING] Teacher is frozen (--freeze_teacher=True).")
                print("          -> --ref_model_mixup_alpha is ignored.")


def _validate_mode2_params(args):
    """Shared Mode-2 (safe-token selection) parameter validation. Used by
    Case (1) and Case (3) in `validate_training_arguments`."""
    if args.voca_selection_num is None or args.voca_selection_num <= 0:
        raise ValueError(f"Mode 2 requires voca_selection_num > 0. Got: {args.voca_selection_num}")
    if not (0.0 <= args.min_steered_prob <= 1.0):
        raise ValueError(f"min_steered_prob must be in [0.0, 1.0]. Got: {args.min_steered_prob}")
    if args.safe_token_horizon <= 0:
        raise ValueError(f"safe_token_horizon must be > 0. Got: {args.safe_token_horizon}")

    # Selection method specifics
    if args.selection_method == "vote":
        if args.num_samples_per_prompt <= 0:
            raise ValueError("'vote' selection requires --num_samples_per_prompt > 0.")
        if args.vote_top_k_inner <= 0:
            raise ValueError("'vote' selection requires --vote_top_k_inner > 0.")
    elif args.selection_method == "mean":
        print("[WARNING] selection_method='mean' → --vote_top_k_inner is ignored.")
    else:
        raise ValueError(f"selection_method must be 'mean' or 'vote'. Got: {args.selection_method}")

    # Sampling specifics
    if args.num_samples_per_prompt > 1:
        if args.safe_token_temperature <= 0.0:
            raise ValueError("Multi-sampling (num_samples_per_prompt > 1) requires safe_token_temperature > 0.0.")
        if not (0.0 < args.safe_token_top_p <= 1.0):
            raise ValueError(f"safe_token_top_p must be in (0.0, 1.0]. Got: {args.safe_token_top_p}")
    else:
        print("[WARNING] num_samples_per_prompt=1 → --safe_token_temperature and --safe_token_top_p are ignored.")

def construct_output_dir(args, project_root):
    """
    Constructs a unique output directory name reflecting the training config.
    Follows the 4-case decision tree in DistilTrainer.__init__:

        (1) RV=True,  mode==2  : vector{RV_upd}-mode2-top{N}-horizon{H}{select}{ST_upd}{renorm}
        (2) RV=True,  mode!=2  : vector{RV_upd}-mode{0|1}[-top{N}][{renorm}]
        (3) RV=False, mode==2  : prompt{V}-mode2-top{N}-horizon{H}{select}{ST_upd}{renorm}
        (4) RV=False, mode!=2  : prompt{V}-mode{0|1}[-top{N}][{renorm}]

    Tags only appear when they actually affect runtime behavior:
      - {RV_upd}  : only when not freeze_teacher (else RV cannot refresh)
      - {ST_upd}  : only when not freeze_teacher (else safe_tokens cannot refresh)
      - {frozenT} : only when freeze_teacher (global modifier on case_tag)
      - {select}  : selection_method-specific params (T/Tp only if multi-sample,
                    Vk only if vote)

    Final layout:
        {root}/{model}-{case_tag}{frozenT}-alpha{α}-{loss_tag}-{data}
    """
    # If the user explicitly provided an output directory, just use it directly
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        return args.output_dir

    model_base = os.path.basename(args.model_name)
    data_base = os.path.basename(args.train_path).split('.')[0]
    root = os.path.join(project_root, "model_weights")

    # ----- Global loss-token tag (skip + keep, both reflected) ----------
    if args.num_loss_tokens_to_skip > 0:
        loss_tag = f"skip{args.num_loss_tokens_to_skip}-keep{args.num_loss_tokens_to_keep}"
    else:
        loss_tag = f"keep{args.num_loss_tokens_to_keep}"
    # Prepend learning rate so LR sweeps don't collide on the same dir.
    loss_tag = f"lr{args.learning_rate}-{loss_tag}"

    # ----- Global teacher-freeze tag (modifier on case_tag) -------------
    # When frozen: just mark as frozen (EMA alpha is ignored at runtime).
    # When synced: include EMA alpha so sweeps over it don't collide.
    if args.freeze_teacher:
        teacher_tag = "-frozenT"
    else:
        teacher_tag = f"-emaA{args.ref_model_mixup_alpha}"

    # ----- Per-case tag construction (mirrors trainer's decision tree) --
    if args.use_refusal_vector:
        # RV branch: refresh tag exists only when teacher can be synced.
        rv_upd_tag = ""
        if not args.freeze_teacher:
            rv_upd_tag = "-updRV" if args.update_refusal_vector else "-fixRV"

        if args.voca_selection_mode == 2:
            # ----- Case (1): RV + mode==2 -----------------------------
            case_tag = (
                f"vector{rv_upd_tag}-mode2"
                f"-top{args.voca_selection_num}"
                f"-horizon{args.safe_token_horizon}"
                f"{_build_select_tag(args)}"
                f"{_build_st_upd_tag(args)}"
                f"{_build_renorm_tag(args)}"
            )
        else:
            # ----- Case (2): RV + mode!=2 -----------------------------
            case_tag = f"vector{rv_upd_tag}-mode{args.voca_selection_mode}"
            if args.voca_selection_mode == 1:
                case_tag += f"-top{args.voca_selection_num}{_build_renorm_tag(args)}"
    else:
        # Prompt branch: system-prompt steering (no refusal direction).
        prompt_tag = "prompt"

        if args.voca_selection_mode == 2:
            # ----- Case (3): prompt + mode==2 -------------------------
            case_tag = (
                f"{prompt_tag}-mode2"
                f"-top{args.voca_selection_num}"
                f"-horizon{args.safe_token_horizon}"
                f"{_build_select_tag(args)}"
                f"{_build_st_upd_tag(args)}"
                f"{_build_renorm_tag(args)}"
            )
        else:
            # ----- Case (4): prompt + mode!=2 -------------------------
            case_tag = f"{prompt_tag}-mode{args.voca_selection_mode}"
            if args.voca_selection_mode == 1:
                case_tag += f"-top{args.voca_selection_num}{_build_renorm_tag(args)}"

    # ----- Assemble final path -----------------------------------------
    # Seed is appended last so multi-seed reproducibility sweeps don't collide.
    final_dir = (
        f"{root}/{model_base}-{case_tag}{teacher_tag}"
        f"-alpha{args.alpha}-{loss_tag}-{data_base}-seed{args.seed}"
    )
    os.makedirs(final_dir, exist_ok=True)
    return final_dir


def _build_select_tag(args):
    """Mode-2 selection-method tag. Method-specific sampling/vote knobs are
    appended here; the top-K count itself stays in the case_tag."""
    select_tag = (
        f"-{args.selection_method}"
        f"-samples{args.num_samples_per_prompt}"
        f"-excl{int(args.exclude_special_tokens)}"
        f"-minP{args.min_steered_prob}"
    )
    # Sampling-only knobs (effective only when num_samples_per_prompt > 1).
    if args.num_samples_per_prompt > 1:
        select_tag += f"-T{args.safe_token_temperature}-Tp{args.safe_token_top_p}"
    # Vote-only knob.
    if args.selection_method == "vote":
        select_tag += f"-Vk{args.vote_top_k_inner}"
    return select_tag


def _build_st_upd_tag(args):
    """Safe-token refresh tag, present only when teacher is not frozen."""
    if args.freeze_teacher:
        return ""
    return "-fixST" if args.freeze_safe_token else "-updST"


def _build_renorm_tag(args):
    """Local-renormalization tag (mode 1 and mode 2 only)."""
    return "-renorm" if args.renormalize_selected_tokens else ""
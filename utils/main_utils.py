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
    parser.add_argument("--freeze_teacher", type=str2bool, default=False,
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
    Validates all runtime arguments for logical conflicts and mathematical boundaries.
    Throws ValueError for fatal constraints and prints warnings for silent invalidations.
    """
    # [1.1] Math & Boundary Checks (Fatal Errors)
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError(f"alpha must be between 0.0 and 1.0. Got: {args.alpha}")
        
    if args.num_loss_tokens_to_keep > 0 and args.num_loss_tokens_to_keep <= args.num_loss_tokens_to_skip:
        raise ValueError(f"num_loss_tokens_to_keep ({args.num_loss_tokens_to_keep}) " 
                         f"must be strictly larger than num_loss_tokens_to_skip ({args.num_loss_tokens_to_skip}).")

    if args.voca_selection_mode not in [0, 1, 2]:
        raise ValueError(f"voca_selection_mode must be exactly 0, 1, or 2. Got: {args.voca_selection_mode}")

    # [1.2] Warnings for useless Teacher parameters (Silent Invalidation)
    if args.freeze_teacher:
        print("\n[WARNING] Teacher is frozen (--freeze_teacher=True).")
        print("          -> --ref_model_mixup_alpha is ignored.")
        print("          -> --freeze_safe_token is ignored (safe tokens are implicitly frozen).")
        if not args.use_refusal_vector:
            print("          -> --update_refusal_vector is ignored.")
            
    if not args.use_refusal_vector and args.update_refusal_vector:
        print("\n[WARNING] Not using refusal vector (--use_refusal_vector=False).")
        print("          -> --update_refusal_vector is ignored.")

    # [1.3] Vocabulary Selection Modes Isolation
    
    # --- MODE 0 ---
    if args.voca_selection_mode == 0:
        print("\n[WARNING] Mode 0 (Full Vocab): --voca_selection_num, --renormalize_selected_tokens, "
              "and all Mode 2 exclusive parameters are completely ignored.")

    # --- MODE 1 ---
    elif args.voca_selection_mode == 1:
        if args.voca_selection_num is None or args.voca_selection_num <= 0:
            raise ValueError(f"Fatal: voca_selection_num must be > 0 for Mode 1. Got: {args.voca_selection_num}")
        print("\n[WARNING] Mode 1 (Top-K): All Mode 2 exclusive parameters (e.g., min_steered_prob, "
              "selection_method, num_samples_per_prompt) are completely ignored.")

    # --- MODE 2 ---
    elif args.voca_selection_mode == 2:
        # Base Requirements
        if args.voca_selection_num is None or args.voca_selection_num <= 0:
            raise ValueError(f"Fatal: voca_selection_num must be > 0 for Mode 2. Got: {args.voca_selection_num}")
        if not (0.0 <= args.min_steered_prob <= 1.0):
            raise ValueError(f"Fatal: min_steered_prob must be between 0.0 and 1.0. Got: {args.min_steered_prob}")

        # Method Specifics: Vote vs Mean
        if args.selection_method == "vote":
            if args.num_samples_per_prompt <= 0:
                raise ValueError("Fatal: 'vote' selection method requires --num_samples_per_prompt > 0.")
            if args.vote_top_k_inner <= 0:
                raise ValueError("Fatal: 'vote' selection method requires --vote_top_k_inner > 0.")
        elif args.selection_method == "mean":
            print("\n[WARNING] Mode 2 using 'mean': --vote_top_k_inner is ignored.")

        # Sampling Specifics: Single vs Multi
        if args.num_samples_per_prompt > 1:
            if args.safe_token_temperature <= 0.0:
                raise ValueError("Fatal: Multi-sampling (num_samples_per_prompt > 1) requires "
                                 "--safe_token_temperature > 0.0 to generate diverse trajectories.")
            if not (0.0 < args.safe_token_top_p <= 1.0):
                raise ValueError(f"Fatal: safe_token_top_p must be strictly greater than 0.0 and less than or equal to 1.0. "
                                 f"Got: {args.safe_token_top_p}")
        elif args.num_samples_per_prompt == 1:
            print("\n[WARNING] Mode 2 using 1 sample: --safe_token_temperature and --safe_token_top_p are ignored.")

def construct_output_dir(args, project_root):
    """
    Constructs and creates the appropriate output directory path based on the training configuration.
    Dynamically includes or excludes tags based on active parameters to keep folder names clean.
    
    Returns:
        str: The absolute path to the finalized output directory.
    """
    # If the user explicitly provided an output directory, just use it directly
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        return args.output_dir

    model_base = os.path.basename(args.model_name)
    data_base = os.path.basename(args.train_path).split('.')[0]
    
    # [1] Base keep tag for loss tokens
    keep_tag = f"keep{args.num_loss_tokens_to_keep}"
    if args.freeze_teacher:
        keep_tag = f"{keep_tag}-frozenT"
        
    root = os.path.join(project_root, "model_weights")

    # [2] Method tag: strictly binds update_refusal_vector to use_refusal_vector
    if args.use_refusal_vector:
        upd_tag = "-updRV" if args.update_refusal_vector else "-fixRV"
        method_tag = f"vector{upd_tag}"
    else:
        method_tag = "prompt"
    
    # [3] Mode 0 (Full Vocab Baseline)
    if args.voca_selection_mode == 0:
        final_dir = f"{root}/{model_base}-{method_tag}-mode0-alpha{args.alpha}-{keep_tag}-{data_base}"
        
    # [4] Mode 1 (Top-K Selection)
    elif args.voca_selection_mode == 1:
        renorm_tag = "-renorm" if args.renormalize_selected_tokens else ""
        final_dir = (
            f"{root}/{model_base}-{method_tag}-mode{args.voca_selection_mode}"
            f"-top{args.voca_selection_num}-alpha{args.alpha}-{keep_tag}{renorm_tag}-{data_base}"
        )
        
    # [5] Mode 2 (Advanced Safe-Token Filtering)
    elif args.voca_selection_mode == 2:
        select_tag = (
            f"-{args.selection_method}"
            f"-samples{args.num_samples_per_prompt}"
            f"-excl{int(args.exclude_special_tokens)}"
            f"-minP{args.min_steered_prob}"
        )
        
        # Dynamically inject sampling parameters only if multi-sampling is active
        if args.num_samples_per_prompt > 1:
            select_tag += f"-T{args.safe_token_temperature}-Tp{args.safe_token_top_p}"
            
        # Dynamically inject vote parameters only if voting is active
        if args.selection_method == "vote":
            select_tag += f"-Vk{args.vote_top_k_inner}"

        # Safe Token Update Tag ONLY if the teacher is actually capable of updating
        if not args.freeze_teacher:
            st_tag = "-fixST" if args.freeze_safe_token else "-updST"
            select_tag += st_tag
            
        renorm_tag = "-renorm" if args.renormalize_selected_tokens else ""
        
        final_dir = (
            f"{root}/{model_base}-{method_tag}-mode{args.voca_selection_mode}"
            f"-top{args.voca_selection_num}-horizon{args.safe_token_horizon}"
            f"-alpha{args.alpha}-{keep_tag}{select_tag}{renorm_tag}-{data_base}"
        )
    else:
        raise ValueError(f"Unexpected voca_selection_mode: {args.voca_selection_mode}")

    # Ensure the generated path exists
    os.makedirs(final_dir, exist_ok=True)
    return final_dir
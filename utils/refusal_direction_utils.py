import os
import json
import torch
import einops
from torch import Tensor
from jaxtyping import Int, Float
from utils.select_direction import get_refusal_scores, select_direction
from utils.generate_directions import generate_directions

SPLITS = ['train', 'val', 'test']
HARMTYPES = ['harmless', 'harmful']

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DATASET_FILENAME = os.path.join(_PROJECT_ROOT, 'data', 'refusal_direction_splits', '{harmtype}_{split}.json')

def get_orthogonalized_matrix(matrix: Float[Tensor, '... d_model'], vec: Float[Tensor, 'd_model']) -> Float[Tensor, '... d_model']:
    vec = vec / torch.norm(vec)
    vec = vec.to(matrix)

    proj = einops.einsum(matrix, vec.unsqueeze(-1), '... d_model, d_model single -> ... single') * vec
    return matrix - proj

def load_dataset_split(harmtype: str, split: str, instructions_only: bool=False):
    assert harmtype in HARMTYPES
    assert split in SPLITS

    file_path = SPLIT_DATASET_FILENAME.format(harmtype=harmtype, split=split)

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    if instructions_only:
        dataset = [d['instruction'] for d in dataset]

    return dataset

def filter_data(model_base, harmful_train, harmless_train, harmful_val, harmless_val, is_main_process):
    """
    Filter datasets based on refusal scores.

    Returns:
        Filtered datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]


    harmful_train_scores = get_refusal_scores(model_base.model, harmful_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
    harmless_train_scores = get_refusal_scores(model_base.model, harmless_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
    harmful_train_filted = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
    harmless_train_filted = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)

    harmful_val_scores = get_refusal_scores(model_base.model, harmful_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
    harmless_val_scores = get_refusal_scores(model_base.model, harmless_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
    harmful_val_filted = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
    harmless_val_filted = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)
    
    if is_main_process:
        print(f"Filtered harmful train samples: {len(harmful_train_filted)}")
        print(f"Filtered harmless train samples: {len(harmless_train_filted)}")
        print(f"Filtered harmful val samples: {len(harmful_val_filted)}")
        print(f"Filtered harmless val samples: {len(harmless_val_filted)}")
    
    return harmful_train_filted, harmless_train_filted, harmful_val_filted, harmless_val_filted

def generate_and_save_candidate_directions(model_base, harmful_train, harmless_train):
    """Generate and save candidate directions."""
    mean_diffs = generate_directions(
        model_base,
        harmful_train,
        harmless_train,
    )

    return mean_diffs

def select_and_save_direction(model_base, harmful_val, harmless_val, candidate_directions, artifact_dir, is_main_process):
    """Select and save the direction."""

    pos, layer, direction = select_direction(
        model_base,
        harmful_val,
        harmless_val,
        candidate_directions,
        artifact_dir=os.path.join(artifact_dir, "select_direction"),
        is_main_process=is_main_process
    )
    if is_main_process:
        with open(f'{artifact_dir}/direction_metadata.json', "w") as f:
            json.dump({"pos": pos, "layer": layer}, f, indent=4)

        torch.save(direction, f'{artifact_dir}/direction.pt')

    return pos, layer, direction
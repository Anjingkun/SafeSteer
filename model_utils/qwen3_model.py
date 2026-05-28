import torch
import functools

from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from jaxtyping import Int, Float

from utils.refusal_direction_utils import get_orthogonalized_matrix
from model_utils.model_base import ModelBase

# Qwen3-Instruct (e.g. Qwen3-4B-Instruct-2507): plain ChatML, no <think> segment.
QWEN3_CHAT_TEMPLATE_WITH_SYSTEM = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
QWEN3_CHAT_TEMPLATE = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

# Note: Always verify these Token IDs for the Qwen 3 vocab!
QWEN3_REFUSAL_TOKS = [40, 2121, 2753, 2132] 

def format_instruction_qwen3_chat(
    instruction: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True,
):
    if system is not None:
        formatted_instruction = QWEN3_CHAT_TEMPLATE_WITH_SYSTEM.format(instruction=instruction, system=system)
    else:
        formatted_instruction = QWEN3_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()
    
    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_qwen3_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True,
):
    if outputs is not None:
        prompts = [
            format_instruction_qwen3_chat(
                instruction=instruction, output=output, system=system, 
                include_trailing_whitespace=include_trailing_whitespace
            )
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_qwen3_chat(
                instruction=instruction, system=system, 
                include_trailing_whitespace=include_trailing_whitespace
            )
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        padding_side = "left",
        truncation=False,
        return_tensors="pt",
    )
    return result

def orthogonalize_qwen3_dense_weights(model, direction: Float[Tensor, "d_model"]):
    # ⚠️ Note: This code is only applicable to Qwen 3 Dense models (e.g., Qwen3-8B).
    # If you are using an MoE model (e.g., Qwen3-30B-A3B), the mlp module here will be replaced by an expert routing mechanism!
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(model.model.embed_tokens.weight.data, direction)

    for block in model.model.layers:
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(block.self_attn.o_proj.weight.data.T, direction).T
        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(block.mlp.down_proj.weight.data.T, direction).T

def act_add_qwen3_dense_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    module = model.model.layers[layer-1].mlp.down_proj
    dtype = module.weight.dtype
    device = module.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)

    if module.bias is None:
        module.bias = torch.nn.Parameter(bias)
    else:
        module.bias.data += bias


class Qwen3Model(ModelBase):

    def _load_model(self, model_path, dtype=torch.float16):
        # ⚠️ Note: Qwen 3 requires transformers version 4.51.0 or higher to prevent a KeyError: 'qwen3'
        pass

    def _load_tokenizer(self, model_path):
        pass

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_qwen3_chat, tokenizer=self.tokenizer, system=None, include_trailing_whitespace=True)

    def _get_eoi_toks(self):
        # ⚠️ Note: Qwen 3 requires transformers version 4.51.0 or higher to prevent a KeyError: 'qwen3'
        return self.tokenizer.encode(QWEN3_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False)
        
    def _get_refusal_toks(self):
        return QWEN3_REFUSAL_TOKS

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block_module.self_attn for block_module in self.model_block_modules])
    
    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_qwen3_dense_weights, direction=direction)
    
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_qwen3_dense_weights, direction=direction, coeff=coeff, layer=layer)
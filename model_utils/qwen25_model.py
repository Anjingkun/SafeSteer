import torch
import functools

from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from jaxtyping import Int, Float

from utils.refusal_direction_utils import get_orthogonalized_matrix
from model_utils.model_base import ModelBase

# Qwen2.5 uses strictly standard ChatML format
QWEN2_CHAT_TEMPLATE_WITH_SYSTEM = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
QWEN2_CHAT_TEMPLATE = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

# Refusal tokens. For Qwen2.5 (vocab size 151936), you might want to verify these.
# 40 = "I", 2121 = "As", 7931 = "Sorry", 11200 = "I'm"
QWEN2_REFUSAL_TOKS = [40, 2121, 7931, 11200] 

def format_instruction_qwen2_chat(
    instruction: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True,
):
    if system is not None:
        formatted_instruction = QWEN2_CHAT_TEMPLATE_WITH_SYSTEM.format(instruction=instruction, system=system)
    else:
        formatted_instruction = QWEN2_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()
    
    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_qwen2_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True,
):
    if outputs is not None:
        prompts = [
            format_instruction_qwen2_chat(instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_qwen2_chat(instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
        padding_side = 'left'
    )

    return result

def orthogonalize_qwen2_weights(model, direction: Float[Tensor, "d_model"]):
    # Qwen2/Llama architecture: embed_tokens instead of wte
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(model.model.embed_tokens.weight.data, direction)

    for block in model.model.layers:
        # self_attn.o_proj replaces attn.c_proj
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(block.self_attn.o_proj.weight.data.T, direction).T
        # mlp.down_proj replaces mlp.c_proj
        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(block.mlp.down_proj.weight.data.T, direction).T

def act_add_qwen2_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    # Inject into mlp.down_proj
    module = model.model.layers[layer-1].mlp.down_proj
    dtype = module.weight.dtype
    device = module.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)

    # Qwen2 linear layers typically do NOT have bias by default. 
    # We dynamically create or add to it.
    if module.bias is None:
        module.bias = torch.nn.Parameter(bias)
    else:
        module.bias.data += bias


class Qwen25Model(ModelBase):

    def _load_model(self, model_path, dtype=torch.float16):
        pass

    def _load_tokenizer(self, model_path):
        pass

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_qwen2_chat, tokenizer=self.tokenizer, system=None, include_trailing_whitespace=True)

    def _get_eoi_toks(self):
        # End of Instruction tokens (extracting the transition from user to assistant)
        return self.tokenizer.encode(QWEN2_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False)

    def _get_refusal_toks(self):
        return QWEN2_REFUSAL_TOKS

    def _get_model_block_modules(self):
        # Qwen2 architecture
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block_module.self_attn for block_module in self.model_block_modules])
    
    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_qwen2_weights, direction=direction)
    
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_qwen2_weights, direction=direction, coeff=coeff, layer=layer)
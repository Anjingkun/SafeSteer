from model_utils.model_base import ModelBase
from transformers import AutoModelForCausalLM, AutoTokenizer

def construct_model_base(ori_model: AutoModelForCausalLM, ori_tokenier: AutoTokenizer, model_path: str) -> ModelBase:

    if 'qwen2.5' in model_path.lower():
        from model_utils.qwen25_model import Qwen25Model
        return Qwen25Model(ori_model, ori_tokenier)
    elif 'qwen3' in model_path.lower():
        from model_utils.qwen3_model import Qwen3Model
        return Qwen3Model(ori_model, ori_tokenier)
    elif 'llama-3' in model_path.lower():
        from model_utils.llama3_model import Llama3Model
        return Llama3Model(ori_model, ori_tokenier)
    else:
        raise ValueError(f"Unknown model family: {model_path}")

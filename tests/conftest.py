import json
from pathlib import Path

import pytest

QWEN_TEMPLATE = "{% if enable_thinking %}<think>{% endif %}{{ messages }}"

CONFIGS = {
    "qwen3-0.6b": {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "num_hidden_layers": 28,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "vocab_size": 151936,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "max_position_embeddings": 40960,
        "tie_word_embeddings": True,
    },
    "qwen3-8b": {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "num_hidden_layers": 36,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "vocab_size": 151936,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 40960,
    },
    "llama-3.2-1b": {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "num_hidden_layers": 16,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "vocab_size": 128256,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 131072,
        "tie_word_embeddings": True,
    },
    "gemma-vl": {
        "architectures": ["Gemma3ForConditionalGeneration"],
        "model_type": "gemma3",
        "text_config": {
            "model_type": "gemma3_text",
            "num_hidden_layers": 34,
            "hidden_size": 2560,
            "intermediate_size": 10240,
            "vocab_size": 262208,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "max_position_embeddings": 131072,
        },
        "vision_config": {"hidden_size": 1152},
    },
    "qwen3-30b-a3b": {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "num_hidden_layers": 48,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "moe_intermediate_size": 768,
        "num_experts": 128,
        "vocab_size": 151936,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "max_position_embeddings": 40960,
    },
    "mla-custom": {
        "architectures": ["DeepseekV3ForCausalLM"],
        "model_type": "deepseek_v3",
        "auto_map": {"AutoConfig": "configuration_deepseek.DeepseekV3Config"},
        "num_hidden_layers": 27,
        "hidden_size": 2048,
        "intermediate_size": 10944,
        "kv_lora_rank": 512,
        "n_routed_experts": 64,
        "vocab_size": 102400,
        "num_attention_heads": 16,
        "max_position_embeddings": 163840,
        "quantization_config": {"quant_method": "fp8"},
    },
}

# bf16 checkpoint sizes, in GB, close to the real models
WEIGHTS_GB = {
    "qwen3-0.6b": 1.4,
    "qwen3-8b": 15.3,
    "llama-3.2-1b": 2.3,
    "gemma-vl": 8.0,
    "qwen3-30b-a3b": 56.9,
    "mla-custom": 15.0,
}


def write_model(root: Path, name: str, *, chat_template: str | None = QWEN_TEMPLATE) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(CONFIGS[name]))
    if chat_template is not None:
        (d / "tokenizer_config.json").write_text(json.dumps({"chat_template": chat_template}))
    total = int(WEIGHTS_GB[name] * 1024**3)
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": {}})
    )
    return d


@pytest.fixture
def model_dir(tmp_path):
    def make(name: str, **kw) -> Path:
        return write_model(tmp_path / "models", name, **kw)

    return make

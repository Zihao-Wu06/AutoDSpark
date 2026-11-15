"""Known upstream problems for specific architectures, surfaced as plan warnings.

Each entry links to the upstream issue so users can check whether it is fixed.
Keep this list short and factual; remove entries once upstream ships a fix.
"""

from __future__ import annotations

from .model_info import ModelInfo


def known_issues(model: ModelInfo) -> list[str]:
    notes: list[str] = []
    if model.uses_mla:
        notes.append(
            "MLA attention (DeepSeek/GLM style): Speculators' default draft config builds a "
            "full-MHA draft for MLA verifiers, which works but is larger than needed. "
            "See vllm-project/speculators#1175."
        )
    if model.is_hybrid:
        notes.append(
            "Hybrid/linear-attention or Mamba layers detected. Drafter training on these "
            "verifiers has had loader bugs upstream (vllm-project/speculators#1156); run "
            "`autodspark run --preset quick` first to confirm the pipeline works."
        )
    if model.model_type.startswith("gpt_oss"):
        notes.append(
            "gpt-oss: serving DFlash-family drafters has known vLLM bugs (FA3 crash, CUDA graph "
            "capture). See vllm-project/speculators#1227."
        )
    if model.is_multimodal:
        notes.append(
            "Vision-language model: the drafter is trained on text only, so image prompts will "
            "see lower acceptance."
        )
    if model.quant_method:
        notes.append(
            f"Quantized checkpoint ({model.quant_method}). The drafter learns from this quantized "
            "model's outputs, so serve it with the same quantized checkpoint."
        )
    if not model.has_chat_template:
        notes.append(
            "No chat template found. Response regeneration uses the chat API, so a base "
            "(non-instruct) model will produce poor training data."
        )
    return notes

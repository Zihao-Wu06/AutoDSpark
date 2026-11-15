---
name: autodspark
description: Train a DSpark speculative-decoding drafter for a user's own LLM with the AutoDSpark CLI, so the model generates faster in vLLM. Use when the user wants to speed up inference of a local or fine-tuned model with speculative decoding, train a DSpark / draft / speculator model, or asks about AutoDSpark runs, plans, logs or reports.
---

# Train a DSpark drafter with AutoDSpark

AutoDSpark plans and runs the whole pipeline (regenerate data with the target model, train the
drafter with vLLM Speculators, benchmark, report). Your job is to drive it safely, explain what it
is doing in plain words, and fix problems when a stage fails.

## Ground rules

- Training uses GPUs for hours. **Show the plan and get the user's go-ahead before `autodspark run`**
  (anything other than `--dry-run`). Mention the preset, sample count and GPU layout.
- Start with `--preset quick` on a new machine or model unless the user asks otherwise. It checks
  the pipeline in minutes; then offer `standard`.
- Never delete a run folder, checkpoints or the `~/.cache/autodspark` environments without asking.
- Run long commands in the background and follow the logs; don't block on them.
- Only pass `--trust-remote-code` (or accept a plan with `trust_remote_code: true`) when the user
  trusts that model's code.

## Workflow

1. **Check the machine.**
   `autodspark doctor --model <model>`
   - No NVIDIA GPU: stop and tell the user training needs a Linux machine with NVIDIA GPUs.
   - Environments missing: ask, then run `autodspark setup` (needs `uv`; downloads vLLM
     and torch, several GB, and the Speculators scripts from GitHub). Safe to re-run after a
     network failure; existing environments are reused.
   - Work folder on a network filesystem: suggest `--workdir` on a local disk.

2. **Make the plan.**
   `autodspark plan --model <model> --preset quick [--data <file.jsonl>]`
   Explain the table: GPU layout (shared GPU / split / offline), samples, sequence length, drafter
   depth. Read every `note:` and `blocker:` line to the user in plain words. Blockers must be fixed
   (fewer samples, smaller preset, more disk) before running. The plan is saved to
   `runs/<model>-<preset>/plan.yaml`; edit it there if the user wants changes.

3. **Run.** After the user agrees:
   `autodspark run --workdir runs/<name> --yes --quiet` in the background.
   Track progress with `autodspark status --workdir runs/<name>` and the newest file in
   `runs/<name>/logs/`. Stages: regen, prepare, (hidden), train, eval, report.

4. **Finish.** Read `runs/<name>/report/report.md`. Tell the user the speedup and mean acceptance
   length first, then give the serve command from `autodspark serve --workdir runs/<name>`.
   If the speedup is below ~1.3x after `quick`, that is expected; suggest `standard`, or their own
   data via `--data`.

## When a stage fails

The CLI prints the failing step, the last log lines and the log path. Read the log, then:

| Symptom in the log | Likely cause | Fix |
|---|---|---|
| `Port 81xx is already in use` | another server running | stop it, or change `ports` in plan.yaml |
| vLLM exits at startup, `max_model_len` / KV cache errors | not enough memory for the context | lower `data.seq_len` in plan.yaml |
| `No such file or directory: 'ninja'`, or "FlashInfer kernels are disabled" | no precompiled FlashInfer kernels and nothing to compile them with | `<vllm env>/bin/vllm download-kernels`, or install the CUDA toolkit (nvcc) |
| vLLM: model architecture not supported | vLLM too old for this model | `autodspark setup --vllm 'vllm==<newer>'` (upgrades the existing vLLM env in place; untested, so `doctor` will warn) |
| `doctor`: "CUDA in vLLM env" fails, torch cannot use the GPU | NVIDIA driver too old for the installed torch build | update the driver (`nvidia-smi` shows the newest CUDA it supports) |
| 401/403 from the Hub | gated model | `huggingface-cli login` or `HF_TOKEN` |
| regen produced 0 rows | no chat template, or prompts longer than the context | use an instruct model; raise `seq_len` |
| `OutOfMemoryError` during train after retries | drafter too big for the GPU | lower `draft.num_layers` (min 3) or `data.seq_len` |
| `No space left on device` (offline) | hidden-state cache too large | lower `data.max_samples`, free disk, or use more GPUs for online mode |
| `KeyError: 'embed_tokens.weight'` or other loader errors | architecture not yet supported upstream | search vllm-project/speculators issues; report it |

After fixing, resume with `autodspark run --workdir runs/<name> --yes`; the saved plan.yaml is
reused and finished stages are skipped. If the user edited plan.yaml, only affected stages re-run
and their old outputs are moved to `<folder>.stale-<time>`; tell the user they can delete those
folders once the new run works. Use `--replan` only when the user wants a fresh plan.

## Useful facts

- The target must be in Hugging Face safetensors format. GGUF (Ollama, llama.cpp) can't be used.
- Serve with the same checkpoint and quantization the drafter was trained against.
- Preview other hardware without GPUs: `autodspark plan --model <m> --gpus 4x80`.
- Dry run prints every command without running anything: `autodspark run --model <m> --dry-run --gpus 1x80`.

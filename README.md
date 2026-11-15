<div align="center">

# AutoDSpark

### Make your own LLM generate up to 2.8x faster, with one command.

**AutoDSpark trains a [DSpark](https://arxiv.org/abs/2607.05147) speculative-decoding drafter for any local model and serves it with vLLM.<br>Same answers, much less waiting.**

[![Speedup](https://img.shields.io/badge/Qwen3--4B-1.83x_faster-brightgreen.svg)](#-results)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-orange.svg)](pyproject.toml)
[![vLLM](https://img.shields.io/badge/vLLM-0.33.0-green.svg)](https://github.com/vllm-project/vllm)
[![Speculators](https://img.shields.io/badge/Speculators-0.9.0-green.svg)](https://github.com/vllm-project/speculators)
[![Status](https://img.shields.io/badge/Status-alpha-yellow.svg)](#-roadmap)

[Quick start](#-quick-start) | [Results](#-results) | [How it works](#-how-it-works) | [Claude Code skill](#-use-it-with-claude-code)

</div>

<p align="center">
  <img src="docs/assets/qwen3-4b-speedup.svg" alt="Qwen3-4B generates 1.83x faster overall and 2.83x faster on math with a drafter trained by AutoDSpark" width="720">
</p>

<p align="center"><sub>Qwen3-4B on 2× RTX 5090: <b>165 → 303 tokens/s</b> overall, <b>165 → 467 tokens/s</b> on math.<br>The drafter was trained by AutoDSpark end to end in under 3 hours.</sub></p>

```bash
autodspark run --model /models/my-finetuned-qwen3-8b   # trains, benchmarks, prints the serve command
```

## 🎉 News

- **[2026/11]** First real result: a `medium` drafter for **Qwen3-4B** makes single-stream
  generation **1.83x faster** overall and **2.83x faster on math**, trained end to end on
  2× RTX 5090 in about 2 h 50 min. See [Results](#-results).
- **[2026/11]** Pipeline check: Qwen3-0.6B with `--preset quick` runs end to end on one
  RTX 5090 in about 15 minutes.

## ⚡ Faster inference, same answers

Every token an LLM writes costs one full forward pass of the model, so long answers are slow no
matter how big your GPU is. Speculative decoding fixes that: a small drafter guesses the next
several tokens, and your model checks all of them in a single pass. When the guesses are right,
you get several tokens for the price of one.

- **No quality loss.** Your model verifies every drafted token, so the output is what it would
  have written on its own.
- **No changes to your model.** Same weights, same precision, no quantization or distillation.
- **Lower latency.** Users wait less for every answer; the biggest wins are on structured
  output like math, code and extraction.

The catch is that the drafter has to match your model. Official DSpark drafters exist for a
handful of popular models; if you fine-tuned a model, or use one nobody has trained a drafter
for, an off-the-shelf drafter matches it poorly and the speedup disappears. **AutoDSpark trains
the matching drafter for you.**

The training code already exists: [vLLM Speculators](https://github.com/vllm-project/speculators),
[DeepSpec](https://github.com/deepseek-ai/DeepSpec), [SpecForge](https://github.com/sgl-project/SpecForge)
and others implement DSpark well. What they leave to you is the per-model setup, and that is the
layer AutoDSpark automates. It reads your model's config and your GPUs, writes a training plan, and
runs the whole pipeline: regenerate training data with your model, train the drafter, benchmark it
against the plain model, and print the `vllm serve` command. It does not reimplement training: it
drives Speculators, so improvements upstream reach you directly.

| | Speculators (directly) | AutoDSpark |
|---|---|---|
| New model | copy an example script, edit ~15 values | `autodspark run --model ...` |
| Target layer ids | kept in sync by hand between vLLM and training (a mismatch trains a broken drafter with no error) | generated once, used everywhere |
| Single GPU | work out the memory split yourself | vLLM and training share it automatically |
| Out of memory | edit and restart | retries with smaller settings |
| Crash halfway | some steps resume | every stage resumes |
| Did it help? | run a separate benchmark | baseline vs DSpark report at the end |

## ✨ Highlights

- **Faster, and you can see it.** Every run ends with a baseline vs DSpark benchmark on your own
  hardware, so you know the speedup before you deploy.
- **One command, whole pipeline.** Plan, regenerate data, prepare, train, benchmark and report.
- **Your model, your data.** Mix your own prompts with public datasets. Your model rewrites every
  answer, so the drafter learns what *your* model actually says, which is what makes it fast.
- **Fits the GPUs you have.** One GPU shared by vLLM and training, or several GPUs with
  data-parallel generation and multi-GPU training. Offline mode when memory is tight.
- **Hard to break.** Every stage resumes; editing the plan reruns only what changed; out-of-memory
  errors retry with smaller settings.
- **Agent-ready.** A Claude Code skill runs the workflow for you and fixes common failures.

## 📊 Results

Speedups are measured by AutoDSpark's own `eval` stage, which serves the model twice with vLLM
(without and with the drafter) and sends the same prompts one at a time.

| Target model | GPUs | Preset | Training data | End-to-end time | Acceptance length | Speedup |
|---|---|---|---|---|---|---|
| Qwen3-4B | 2× RTX 5090 (32 GB) | `medium` | 10k open-perfectblend | 2 h 50 min | 2.25 | **1.83x** |

By task category:

| Category | Baseline tok/s | With DSpark tok/s | Speedup |
|---|---|---|---|
| math | 165.3 | 467.5 | **2.83x** |
| extraction | 165.1 | 306.3 | 1.86x |
| code | 165.0 | 299.1 | 1.81x |
| reasoning | 165.9 | 287.7 | 1.73x |
| writing | 165.8 | 270.1 | 1.63x |
| chat | 165.4 | 249.7 | 1.51x |
| **overall** (34 prompts) | 165.4 | 302.6 | **1.83x** |

Acceptance length is the mean number of tokens produced per target-model forward pass
(1.0 = no speculation). Structured, predictable output (math, extraction, code) gains the most.
The drafter only saw the public English mix it was trained on; prompts unlike that data gain
less, so for production train on your own prompts with `--data`.

<details>
<summary>Setup and how to reproduce</summary>

- **Benchmark:** the 34 built-in prompts of the six categories above, from
  [`eval_prompts.jsonl`](src/autodspark/eval_prompts.jsonl), up to 512 new tokens each,
  temperature 0, one request at a time, vLLM 0.33.0, `num_speculative_tokens: 8`, default
  Qwen3 chat template.
- **Drafter:** DSpark, 5 layers, block size 8, target layers `[2, 18, 33]`, draft vocabulary
  32k. Trained with Speculators 0.9.0 for 3 epochs at sequence length 2048, lr 3e-4.
- **Time:** regenerating 10k answers with Qwen3-4B took 36 min (both GPUs, data-parallel),
  training took 2 h 05 min, the benchmark and report 6 min. Environment setup and the model
  download are not included.
- **Layout:** both GPUs each ran one vLLM replica (36% of memory) and one training rank (DDP).
  The planner does not choose this shared layout on multi-GPU machines yet (it gives vLLM one
  GPU and training the other), so it was set in `plan.yaml`:

  ```yaml
  draft:
    max_anchors: 768
  train:
    extra_args: ["--gradient-checkpointing", "--num-workers", "2"]
  layout:
    shared_gpu: true
    vllm_gpus: [0, 1]
    vllm_mem_util: 0.36
    train_gpus: [0, 1]
  ```

  ```bash
  autodspark plan --model Qwen/Qwen3-4B --preset medium   # writes runs/Qwen3-4B-medium/plan.yaml
  # edit plan.yaml as above, then:
  autodspark run --workdir runs/Qwen3-4B-medium
  ```

</details>

## 🧩 Supported models

AutoDSpark works with decoder-only models in Hugging Face format that vLLM can serve and
Speculators can train a DSpark drafter for.

| Model | GPUs | Preset | Status |
|---|---|---|---|
| Qwen3-4B | 2× RTX 5090 | `medium` | ✅ 1.83x speedup ([Results](#-results)) |
| Qwen3-0.6B | 1× RTX 5090 | `quick` | ✅ pipeline runs end to end (`quick` is a smoke test, too little training for a speedup) |
| Llama 3.x, Gemma, other Qwen3 sizes | | | planned, untested |

The plan warns you up front about known upstream issues for some architectures, with links:
MLA attention (DeepSeek/GLM style), hybrid or Mamba layers, gpt-oss, vision-language models,
quantized checkpoints and models without a chat template.

## 📦 Installation

**Requirements**

- Linux with NVIDIA GPU(s). One GPU works for small and mid-size models.
- The target model in **Hugging Face format** (safetensors). GGUF files from Ollama or llama.cpp
  can't be used, because training reads the model's hidden states.
- Python 3.10+ and [uv](https://docs.astral.sh/uv/) (`pip install uv`), used by `autodspark setup`.
- Disk: a few GB for online mode; much more for offline mode (the plan tells you).

**Install**

```bash
git clone https://github.com/Zihao-Wu06/AutoDSpark.git && cd AutoDSpark
pip install -e .                # not on PyPI yet
autodspark setup                # builds the Speculators and vLLM environments
```

`setup` creates two virtualenvs under `~/.cache/autodspark` (one for Speculators, one for vLLM, so
their torch versions never clash) and fetches the pinned Speculators scripts.

<details>
<summary>Slow or blocked access to PyPI, GitHub or Hugging Face</summary>

Everything AutoDSpark starts inherits your environment, so mirrors and proxies just work:

```bash
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/   # PyPI mirror for setup
export HF_ENDPOINT=https://hf-mirror.com                         # Hugging Face mirror
export HTTPS_PROXY=http://your-proxy:port                        # for the GitHub download in setup
```

Downloading the model ahead of time and passing the local folder to `--model` avoids slow
first starts: `HF_HUB_ENABLE_HF_TRANSFER=1 hf download Qwen/Qwen3-8B --local-dir ./Qwen3-8B`
(needs `pip install "huggingface_hub[hf_transfer]"`).

</details>

## 🚀 Quick start

```bash
autodspark doctor --model Qwen/Qwen3-8B                 # check GPUs, environments and the model
autodspark plan   --model Qwen/Qwen3-8B --preset quick  # preview the plan, nothing runs
autodspark run    --model Qwen/Qwen3-8B --preset quick  # run the whole pipeline
```

When the run finishes, `runs/Qwen3-8B-quick/report/report.md` shows the speedup. Serve the model
with the drafter:

```bash
vllm serve Qwen/Qwen3-8B --speculative-config \
  '{"method": "dspark", "model": "runs/Qwen3-8B-quick/checkpoints/checkpoint_best", "num_speculative_tokens": 8}'
```

`autodspark serve --workdir runs/Qwen3-8B-quick` prints this command for your run. Serve the same
checkpoint (same quantization) you trained against.

`quick` only checks that the pipeline works on your machine. For a drafter that actually speeds
things up, use `medium` or `standard`.

## 📖 Usage

### Presets

| Preset | Samples | Epochs | Seq len | Use it to |
|---|---|---|---|---|
| `quick` | 1,000 | 1 | 2048 | check the pipeline works on your machine |
| `medium` | 10,000 | 3 | 2048 | a first real drafter in a few GPU-hours |
| `standard` | 20,000 | 5 | 4096 | get a useful drafter (default) |
| `full` | 100,000 | 5 | 8192 | get closer to published recipes |

Override anything with `--samples`, `--epochs`, `--seq-len`, `--draft-layers`, or edit
`plan.yaml`.

### Train on your own data

Acceptance is highest when the drafter has seen prompts like the ones you will serve.

```bash
autodspark run --model ./my-model --data my_prompts.jsonl            # mine first, then public data
autodspark run --model ./my-model --data my_prompts.jsonl --public-data none
```

Each line is one conversation, in any of these shapes (see [examples/my_prompts.jsonl](examples/my_prompts.jsonl)):

```json
{"messages": [{"role": "user", "content": "..."}]}
{"conversations": [{"from": "human", "value": "..."}]}
{"prompt": "..."}
```

Answers in your file are ignored: your model regenerates every answer. Public datasets to fill
the rest of the sample budget: `open-perfectblend` (default), `ultrachat`, `sharegpt`, `magpie`,
`nemotron`, `gsm8k`.

### Commands

| Command | What it does |
|---|---|
| `autodspark setup` | Builds the Speculators and vLLM environments |
| `autodspark doctor` | Checks GPUs, driver, disk, environments, ports and the model before you start |
| `autodspark plan` | Writes `plan.yaml` without running anything (`--gpus 4x80` previews other machines) |
| `autodspark run` | Plans if needed and runs the pipeline; run it again to resume |
| `autodspark status` | Shows which stages are done |
| `autodspark serve` | Prints the `vllm serve` command for a finished run |

## 🔧 How it works

```
plan ──> regen ──> prepare ──> [hidden] ──> train ──> eval ──> report
```

| Stage | What runs |
|---|---|
| plan | Reads `config.json` and GPU info, writes `plan.yaml` |
| regen | vLLM serves your model; `speculators regenerate-responses` answers every prompt |
| prepare | `speculators prepare-data` tokenizes and builds the reduced draft vocabulary |
| hidden | Offline mode only: caches target hidden states on disk |
| train | vLLM streams hidden states of the chosen layers; `speculators train --speculator-type dspark` |
| eval | Serves the model without and with the drafter; measures tokens/s and acceptance |
| report | Writes `report/report.md` and the serve command |

Everything a run produces lives in its work folder:

```
runs/Qwen3-8B-standard/
  plan.yaml        the plan; edit and re-run to change it
  state.json       which stages are done
  logs/            one log per step and per vLLM server
  data/            regenerated and prepared data
  checkpoints/     drafter checkpoints (checkpoint_best is used)
  report/          report.md and eval.json
```

**Resuming.** Run the same command again (or `autodspark run --workdir runs/...`). The saved
`plan.yaml` is reused (pass `--replan` to rebuild it) and finished stages are skipped. If you edit
`plan.yaml`, only the stages that depend on what you changed run again, starting from clean
folders; the old outputs are moved to `<folder>.stale-<time>` rather than deleted.
`--from-stage train` forces a stage and everything after it.

**GPU layout.** With one GPU, vLLM and the trainer share it when both fit; otherwise the plan
switches to offline mode. With several GPUs, vLLM gets about half (in tensor-parallel groups) and
training the rest. Preview any machine with `autodspark plan --model ... --gpus 4x80`.

**Out of memory.** If training runs out of GPU memory, AutoDSpark halves `max_anchors` and retries,
up to three times.

## 🤖 Use it with Claude Code

[`skills/autodspark/SKILL.md`](skills/autodspark/SKILL.md) teaches Claude Code to run the workflow
for you: check the machine, explain the plan, start the run, watch the logs and fix common
failures. Copy it into your skills folder:

```bash
mkdir -p ~/.claude/skills && cp -r skills/autodspark ~/.claude/skills/
```

Then ask Claude Code something like "train a DSpark drafter for /models/my-model".

## ❓ FAQ

<details>
<summary><b>Does the drafter change my model's answers?</b></summary>

No. The target model verifies every drafted token, so the output follows the target model; the
drafter only changes how fast it comes out.

</details>

<details>
<summary><b>Why is the speedup lower than the acceptance numbers during training?</b></summary>

Validation during training runs on held-out samples of the training data, which look like what
the drafter learned from. The `eval` stage uses different prompts, and the drafter's own compute
is part of every step. Trust the `eval` number.

</details>

<details>
<summary><b>My drafter makes generation slower. What now?</b></summary>

More data and epochs raise acceptance (`quick` is never meant to be fast). Training on prompts
like the ones you serve (`--data`) helps most. Very small target models also gain little,
because they are already fast compared with the drafter.

</details>

<details>
<summary><b>Does it help when serving many requests at once?</b></summary>

The reported speedup is for one request at a time. Speculative decoding helps most at low
concurrency; as the batch grows, the GPU is busier and the gain shrinks.

</details>

## 🧭 Roadmap

- [x] Single GPU shared by vLLM and training
- [x] Data-parallel response regeneration and multi-GPU training
- [x] Built-in baseline vs DSpark benchmark and report
- [x] Claude Code skill
- [ ] Planner picks the shared multi-GPU layout automatically
- [ ] Published results for larger Qwen3 models, Llama 3.x and Gemma, `standard` preset
- [ ] Calibrated memory and time estimates
- [ ] Install the Claude Code skill as a plugin
- [ ] Optional SpecForge backend for SGLang
- [ ] Multi-node training

## ⚠️ Limitations

- Single machine only (one or more GPUs).
- vLLM serving only. SGLang users can train with SpecForge instead.
- Memory and time estimates are rough; they have been checked on Qwen3 runs only.

## 🛠️ Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests && ruff format --check src tests
```

Issues and pull requests are welcome, especially results from models and GPUs we haven't
tested.

## 📃 Citation

If AutoDSpark helps your work, please cite this repository and the DSpark paper
([arXiv:2607.05147](https://arxiv.org/abs/2607.05147)):

```bibtex
@software{autodspark2026,
  title  = {AutoDSpark: One-Command DSpark Drafter Training for Your Own Model},
  author = {{AutoDSpark contributors}},
  year   = {2026},
  url    = {https://github.com/Zihao-Wu06/AutoDSpark}
}
```

## 🙏 Acknowledgements

DSpark is by DeepSeek-AI and Peking University ([paper](https://arxiv.org/abs/2607.05147),
[DeepSpec](https://github.com/deepseek-ai/DeepSpec)). Training and serving are done by
[vLLM Speculators](https://github.com/vllm-project/speculators) and [vLLM](https://github.com/vllm-project/vllm).
AutoDSpark is an independent project and is not affiliated with them.

## License

[Apache-2.0](LICENSE)

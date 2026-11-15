# AutoDSpark report: Qwen3-0.6B

Generated 2026-11-08 00:17 by AutoDSpark 0.1.0 (Speculators 0.9.0).

## Result

**0.88x the baseline speed**: with this drafter, generation is slower than without it.

| | Baseline | With DSpark |
|---|---|---|
| Output tokens/s (one request at a time) | 593.79 | 520.07 |
| Mean acceptance length (tokens per verify step) | - | 1.388 |
| Draft token acceptance rate | - | 0.049 |

Measured on 40 prompts, up to 512 new tokens each, temperature 0.0.

> This was a `quick` run: it checks that the pipeline works on your machine, with too little training for a useful drafter. Run `--preset standard` for real gains.

> The drafter doesn't speed generation up yet. More training data and epochs raise acceptance; training on prompts like the ones you serve (`--data`) helps most. Small target models also gain little from speculative decoding, because they are already fast relative to the drafter.

| Category | Baseline tok/s | DSpark tok/s | Speedup |
|---|---|---|---|
| chat | 592.14 | 517.91 | 0.87x |
| code | 589.9 | 568.29 | 0.96x |
| extraction | 591.47 | 527.11 | 0.89x |
| math | 594.04 | 638.81 | 1.08x |
| reasoning | 592.35 | 553.39 | 0.93x |
| writing | 601.89 | 505.31 | 0.84x |
| zh | 597.65 | 390.6 | 0.65x |

## Serve it

Serve your model with the drafter:

```bash
vllm serve /root/autodl-tmp/models/Qwen3-0.6B --speculative-config '{"method": "dspark", "model": "/root/autodl-tmp/runs/qwen3-0.6b-quick/checkpoints/checkpoint_best", "num_speculative_tokens": 8}'
```

## Training setup

- Target model: `/root/autodl-tmp/models/Qwen3-0.6B` (28 layers, hidden 1024, vocab 151936, 1.4 GB)
- GPUs: 1 x 31.8 GB, online mode (vLLM and training share one GPU)
- Data: 1,000 samples, sequence length 2048, sources: open-perfectblend (1,000)
- Drafter: 3 layers, block size 8, target layers [2, 14, 25], draft vocab 32000
- Training: 1 epochs, lr 0.0003, preset `quick`

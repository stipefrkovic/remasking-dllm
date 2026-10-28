# 🎭 Re-evaluating Confidence Remasking in Masked Diffusion Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2606.12232-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.12232) [![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg)](https://neurips.cc/virtual/2026/loc/sydney/poster/155851)

*Masked diffusion language models (dLLMs) have recently emerged as a competitive alternative to autoregressive language models, with the promise of faster inference via parallel token generation. A notable limitation of the masked formulation, however, is that once a token has been unmasked it can no longer be revised, leaving dLLMs vulnerable to early sampling mistakes. Confidence-based remasking methods aim to address this in a training-free manner by revising low-confidence tokens, with encouraging early results. In this work, we compare the accuracy–compute Pareto frontiers of three such methods (WINO, Saber, and COVER) to those of confidence-based unmasking alone (Fast-dLLM). We find that under greedy decoding, all three perform on par with Fast-dLLM at commonly used shorter block lengths; gains of remasking appear only at longer block lengths, where confidence-based unmasking is known to degrade. Under stochastic decoding, WINO reduces generation diversity beyond that of unmasking alone, while it helps recover from errors introduced by learned Bernoulli unmasking policies. Our results show that the benefits of confidence-based remasking are highly setting-dependent, underscoring the need for more comprehensive evaluation. Finally, our analyses point to the dLLM, not the remasking rule, as the bottleneck, suggesting that effective remasking may require changes to training, not only to inference.*

This repository contains the code for the paper. It is adapted from [Feng-Hong/WINO-DLLM](https://github.com/Feng-Hong/WINO-DLLM), the official WINO implementation; Saber and COVER are adapted from their official code ([zhaoyMa/Saber](https://github.com/zhaoyMa/Saber), [xyzCS/COVER](https://github.com/xyzCS/COVER)).

## Usage

```bash
# install
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run WINO on GSM8K with LLaDA, block length 128, unmasking threshold 0.7
python src/eval.py --config configs/llada/gsm8k/wino.yaml \
  --set generation_args.block_length=128 method_args.wino.threshold=0.7
```

Configs are at `configs/<model>/<benchmark>/<method>.yaml`, and `--set` overrides any of their values:

- **Models**: LLaDA-8B-Instruct (`llada`), Dream-v0-Instruct-7B (`dream`)
- **Benchmarks**: GSM8K (`gsm8k`), MATH-500 (`math500`), HumanEval (`humaneval`), MBPP (`mbpp`)
- **Methods**: Fast-dLLM (`fastdllm`), WINO (`wino`), Saber (`saber`), COVER (`cover`)

Results go to `results/`, with the response, correctness and NFEs for each problem.
HumanEval and MBPP execute model-generated code, so run them in a sandboxed environment.

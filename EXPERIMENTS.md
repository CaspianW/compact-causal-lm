# Later experiments

The current experimental candidate is `batch3/qknorm`. It scored **1.5040508770 validation BPB** with temperature 1.15; its uncalibrated score was 1.5167488117. These are full-validation CUDA FP32 measurements. This candidate has not received a new test score or CPU resource-budget measurement.

The original coursework submission remains available at [commit d3a7c97](https://github.com/CaspianW/compact-causal-lm/tree/d3a7c97d2f6ba1cbca6a0f8bc68fb076a2041b20). Its matching checkpoint and reported test score are described in the original documentation. README and REPORT still describe that submission. This file records the newer experiments.

## Code and reproduction

`student.py` now accepts the optional normalization and temperature settings through `experimental_models.py`. Its original factory behavior is preserved for the original configuration. QKNorm normalizes queries and keys within each head and uses a learned scale. The optional NormFormer experiment adds head scaling and extra LayerNorm operations; its SwiGLU hidden normalization is an adaptation of the paper's GELU architecture.

Use the installation instructions in `code/README.md`. From `code/`, run:

```bash
python train_normalization.py --variant qknorm --run-dir runs/experimental-reproduction
python evaluate.py --checkpoint runs/experimental-reproduction/checkpoint-calibrated.pt --device cuda --precision fp32 --split validation
```

The trainer uses seed 17, batches of 32 sequences, 9,600 updates, CUDA BF16, fused AdamW, learning rate 0.001, weight decay 0.1, 100 warmup steps, and the original cosine schedule ending near 10% of the peak learning rate. It selects the checkpoint using complete CUDA FP32 validation and compares the same 13 predefined temperatures. Runs start from random initialization. Exact weights can differ across training runs and hardware.

`train_regularization.py` performs two independent dropout forward passes per batch. It averages their cross-entropy losses and adds the selected coefficient times the mean bidirectional KL. Coefficient zero is a paired control with the same two forward passes. It processes 78,643,200 sampled targets and 157,286,400 forward targets per 9,600-update run.

Numeric results, training cost, and source hashes for the current candidate are recorded in [`code/experimental_result.json`](code/experimental_result.json). Checkpoints, optimizer states, virtual environments, and machine-specific logs are excluded from Git. The experimental checkpoint has not replaced the original submission artifact.

## Attribution and limits

Methods follow the [QKNorm paper](https://aclanthology.org/2020.findings-emnlp.379/), [NormFormer paper](https://arxiv.org/abs/2110.09456), [temperature-calibration paper](https://proceedings.mlr.press/v70/guo17a.html), and [R-Drop paper and author implementation](https://github.com/dropreg/R-Drop). The starter data, tokenizer, and evaluator are unchanged. Codex assisted with implementation, validation checks, experiment execution, and documentation, as disclosed in README.

Comparisons cover one seed. Selecting a structure uses validation only. Earlier runs used different training lengths and starting weights, so their score differences cannot isolate architecture or regularization effects. Further improvements published here remain experimental until test evaluation and the CPU limits have been checked for a frozen candidate.

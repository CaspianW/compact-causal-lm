# Later experiments

The final selected candidate is `batch8/rdrop-0.5`. It scored **1.4720205239 validation BPB** with temperature 1.1; its uncalibrated score was 1.4816362302. These are full-validation CUDA FP32 measurements. After freezing the predictor, complete CPU FP32 test evaluation gave **1.4899984302 BPB**. Its paired CPU scoring-time ratio was 4.05, peak evaluator working set was 1.82 GiB, and checkpoint size was 22.07 MiB. Full measurements are in [`code/final_result.json`](code/final_result.json).

The original coursework submission remains available at [commit d3a7c97](https://github.com/CaspianW/compact-causal-lm/tree/d3a7c97d2f6ba1cbca6a0f8bc68fb076a2041b20), with its original documentation and matching checkpoint. The current REPORT and code README describe the final selected predictor. This file records its experiment history.

## Code and reproduction

`student.py` now accepts the optional normalization and temperature settings through `experimental_models.py`. Its original factory behavior is preserved for the original configuration. QKNorm normalizes queries and keys within each head and uses a learned scale. The optional NormFormer experiment adds head scaling and extra LayerNorm operations; its SwiGLU hidden normalization is an adaptation of the paper's GELU architecture.

Use the installation instructions in `code/README.md`. From `code/`, run:

```bash
python reproduce_final.py --recipe configs/latest_training_recipe.json --run-dir runs/experimental-reproduction
python evaluate.py --checkpoint runs/experimental-reproduction/checkpoint-calibrated.pt --device cuda --precision fp32 --split validation
```

The candidate continues selected seed-17 weights. Each branch starts from the same weights, removes temperature scaling during training, resets AdamW and the dropout random stream, and retains R-Drop coefficient 0.5. It adds 2,400 updates with batches of 32 sequences, CUDA BF16, fused AdamW, weight decay 0.1, initial learning rate 0.0001, and the `cosine-zero` schedule. Complete CUDA FP32 validation runs every 1,200 updates. The selected ancestor contains 117,964,800 sampled targets; this branch processes 19,660,800 additional sampled targets and 39,321,600 additional forward targets. Reported training-loop seconds cover this continuation only. Optimizer state is not restored. It selects weights by raw validation BPB and compares the same 13 predefined temperatures. The commands above first reproduce the ancestor training and then the continuation. Exact weights can differ across training runs and hardware.

`train_regularization.py` performs two independent dropout forward passes per batch. It averages their cross-entropy losses and adds the selected coefficient times the mean bidirectional KL. Coefficient zero is a paired control with the same two forward passes. It processes 78,643,200 sampled targets and 157,286,400 forward targets per 9,600-update run.

Validation-search results, training cost, and source hashes for the candidate are recorded in [`code/experimental_result.json`](code/experimental_result.json). The frozen predictor's CPU measurements are in [`code/final_result.json`](code/final_result.json). Checkpoints, optimizer states, virtual environments, and machine-specific logs are excluded from Git. The final checkpoint bundle is supplied separately; the original submission artifact remains available for its original code revision.

## Attribution and limits

Methods follow the [QKNorm paper](https://aclanthology.org/2020.findings-emnlp.379/), [NormFormer paper](https://arxiv.org/abs/2110.09456), [temperature-calibration paper](https://proceedings.mlr.press/v70/guo17a.html), and [R-Drop paper and author implementation](https://github.com/dropreg/R-Drop). The starter data, tokenizer, and evaluator are unchanged. Codex assisted with implementation, validation checks, experiment execution, and documentation, as disclosed in README.

Comparisons cover one seed. Selecting a structure uses validation only. Earlier runs used different training lengths and starting weights, so their score differences cannot isolate architecture or regularization effects. The matched 9,600-update coefficient-zero ablation is described in REPORT.md. The frozen final candidate has completed test evaluation and CPU resource checks on this machine.

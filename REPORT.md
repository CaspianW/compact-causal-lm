# MP1 report: small causal language model

## Method and selection

The submitted predictor is a five-block causal Transformer trained from random initialization on the supplied WikiText-2 training split. It uses width 256, four attention heads, RoPE, SwiGLU, pre-layer normalization, tied token embedding/output weights, residual dropout 0.1, and attention dropout 0.1. AdamW uses an initial learning rate of 0.001, weight decay 0.1, 100 warmup steps, and cosine decay. Training used seed 17, batches of 32 sequences with 256 targets each, and 9,600 updates (78,643,200 sampled training targets). BF16 CUDA was used for training; all reported validation and test scores use FP32 evaluation. Validation alone selected the final checkpoint.

The selected model has 4,465,040 parameters. The checkpoint was saved at step 9,600. Its SHA-256 is `4121726b57a6528b2f83197e7fbc36ca46c90bf513413cffcf8b1630b9b08c90`. Full-test CPU FP32 evaluation gave **1.5510079794 BPB**, on 428,405 scored targets and 1,292,013 UTF-8 bytes. The test set was evaluated after the method and checkpoint were frozen; it was not used to choose model settings.

## Comparisons and ablation

| Run | Parameters | Training targets | Validation BPB | Test BPB |
| --- | ---: | ---: | ---: | ---: |
| Supplied baseline, 1,200 updates | 1,088,256 | 9,830,400 | 2.07108 | 2.10126 |
| Same architecture without attention dropout, 9,600 updates | 4,465,040 | 78,643,200 | 1.53755 | Not used for selection |
| Submitted model, attention dropout 0.1, 9,600 updates | 4,465,040 | 78,643,200 | **1.52665** | **1.55101** |

The last two runs use the same seed, parameter count, batch size, number of updates, and sampled training target count. Their only configuration difference is attention dropout. The validation improvement is **0.01090 BPB**. This paired ablation supports a small regularization benefit in this training setup. It does not isolate the contribution of RoPE, SwiGLU, increased width/depth, or the longer training schedule relative to the original baseline; the original baseline row provides context rather than a controlled estimate of those changes.

Additional validation-only screens covered RMSNorm, partial RoPE, different dropout rates, width/depth/head counts, GQA/MQA, local and gated attention, Muon, MoE, MLA, DSA, linear attention, a DeltaNet hybrid, and a simplified mHC. The last seven structure screens used smaller width-128/depth-4 models for 1,200 updates each. Simplified mHC and MoE were promising in that short screen, but the evidence was insufficient to replace the fully trained, validated submitted checkpoint. The DSA and mHC screens implement core course ideas rather than full paper systems.

## Cost, limits, and interpretation

Across 53 completed runs with saved metrics, cumulative measured training time was about **185 minutes** (training loops only, including CPU and GPU runs and short screens). The submitted run itself took 279 seconds of measured GPU training time. On the same four-thread Intel Core i5-12600KF, full-test CPU scoring took 16.69 seconds for the submitted model and 6.82 seconds for the supplied baseline, a ratio of 2.45 against the 5-times limit. Observed peak process working set was 1.81 GiB, below 4 GiB. The checkpoint is 19.05 MiB, below the 64 MiB inference-asset limit.

The selected model is larger and slower to score than the supplied baseline, but substantially lowers BPB. The attention-dropout gain is modest relative to the architecture and training-budget changes, and the result comes from one seed; it should not be interpreted as a general effect size. Short screens are useful for rejecting clearly worse ideas, but they cannot establish the best architecture at full training scale. GPU training may not reproduce bit-for-bit across hardware, so the matching checkpoint is supplied for exact score verification.

## Reproduction and attribution

The installation, training, and exact checkpoint-evaluation commands are in [`code/README.md`](code/README.md). The supplied data, tokenizer, evaluator, and causal-window protocol are unchanged. RoPE, SwiGLU, and the other screened architecture ideas were covered by the course lectures. An optional Muon experiment used a compact adaptation of the [public Muon implementation](https://github.com/KellerJordan/Muon); Muon is not part of the submitted predictor. AI assistance helped with implementation, debugging, experiment organization, and drafting. The recorded measurements were verified with the supplied evaluator and correctness tests.

# MP1 report: compact causal language model

## Predictor and training

The final predictor is selected from the supplied WikiText-2 training and validation splits. Its configuration is recorded in `code/final_result.json`. It has width 256, depth 6, four attention heads, pre-LayerNorm, RoPE, SwiGLU, tied embedding/output weights, residual dropout 0.1, and attention dropout 0.1. QKNorm applies L2 normalization to queries and keys after RoPE and replaces the usual square-root head scaling with one learned scale per block. It has 5,253,094 parameters.

Training uses seed 17, batches of 32 sequences with 256 targets each, CUDA BF16, fused AdamW, weight decay 0.1, and gradient clipping at 1.0. R-Drop uses two independent dropout passes, the mean of their cross-entropies, and a coefficient times the mean bidirectional KL. Fresh training uses learning rate 0.001, 100 warmup steps, and cosine decay with a 10% floor. Continuation phases reset AdamW and the dropout random stream, continue the seed-17 sampling sequence after the selected checkpoint, and follow the recorded schedule. They do not restore optimizer state. The complete recipe is:

1. `train_capacity.py`: alpha=0.5, steps=9600, eval_every=2400
2. `train_continuation.py`: alpha=0.5, schedule=cosine-zero, learning_rate=0.0002, steps=4800, eval_every=1200, snapshot_every=600, snapshot_start=2400
3. `train_continuation.py`: alpha=0.5, schedule=cosine-zero, learning_rate=0.0001, steps=2400, eval_every=1200

Late-checkpoint averages of the last three or five saved weights were screened on validation. The selected predictor uses an individual checkpoint. The final temperature is 1.1. Raw checkpoints are selected on complete CUDA FP32 validation; each is compared using the same 13 predefined temperatures. Model, training, and averaging choices use validation only. The final predictor was frozen before its new test evaluation.

## Results and paired comparison

| Run | Parameters | Sampled training targets | Validation BPB | Test BPB |
| --- | ---: | ---: | ---: | ---: |
| Supplied baseline, 1,200 updates | 1,088,256 | 9,830,400 | 2.07108 | 2.10126 |
| Six-layer model, R-Drop coefficient 0, 9,600 updates | 5,253,094 | 78,643,200 | 1.49120 | — |
| Six-layer model, R-Drop coefficient 0.5, 9,600 updates | 5,253,094 | 78,643,200 | 1.47914 | — |
| Final selected predictor | 5,253,094 | 137,625,600 | 1.47202 | 1.49000 |

The two 9,600-update six-layer runs used the same initialization, training windows, seed, batch size, architecture, learning-rate schedule, and two dropout forward passes. Only the bidirectional KL coefficient changed, from 0 to 0.5. Calibrated validation BPB was 1.49120 and 1.47914, respectively. Each run processed 78,643,200 sampled targets and 157,286,400 forward targets. This paired comparison tests the KL term within this setup; it covers one seed.

The final checkpoint scored **1.4899984302 full-test CPU FP32 BPB**, over 428,405 targets and 1,292,013 UTF-8 bytes. Its SHA-256 is `5dda862e738655ccd3406c055a8f8025b8fb5be8b5a4d9c2e5a38f2b45faa4f8`. GPU validation BPB was 1.4720205239; CPU validation BPB was 1.4720204364. Both validation measurements use the same split. The reported test score uses the separate test split.

The supplied baseline is smaller and trained for fewer targets; its difference from the final score cannot isolate any single mechanism. The paired six-layer comparison controls training targets and the architecture. The final recipe uses more training than those paired runs, so its improvement also includes continuation and validation selection. The table counts targets processed by the selected training trajectory; its validation-selected checkpoint can come from before the end of that trajectory.

## Search and cost

The earlier course-method screens compared normalization, positional encodings, dropout, widths/depths, head counts, grouped/multi-query attention, local/gated attention, Muon, and small structural prototypes. Later experiments compared EMA, temperature calibration, QKNorm, a SwiGLU adaptation of NormFormer, R-Drop, model capacity, learning-rate continuation, and late-checkpoint averaging. Larger/deeper models and short continuations were selected by validation. Simplified DSA and mHC screens test class-taught ideas rather than complete paper architectures.

The original report recorded 53 completed earlier runs and about 185 minutes of measured training loops. The later local metric files record 36 training loops, including short function checks and any partial runs, totalling 226.47 measured training-loop minutes. These durations are summed across runs; concurrent runs overlap, so the total differs from elapsed wall time. Calibration, scoring, averaging and packaging add time. Averaged candidates reuse their source trajectory and are not counted as new training runs. Training ancestry and the final recipe include the cost of earlier training phases.

## Evaluation resources and limits

On the Intel Core i5-12600KF with four CPU threads, the frozen predictor's full-test scoring time was 22.31 seconds, versus 5.50 seconds for the supplied baseline. The median paired ratio was **4.05**, below 5. Peak evaluator-process working set was **1.82 GiB**, below 4 GiB. The checkpoint was **22.07 MiB**, below the 64 MiB inference-asset limit. Only one predictor is used for inference. Timing varies with Windows background load; per-run measurements are in `code/final_result.json`.

CPU validation was used to check resource feasibility before the new test score was read. Test was used for the frozen predictor's final scoring and resource measurement, not to choose settings. Results cover one seed, and the validation search has many comparisons; the observed gains need not transfer to other datasets or training budgets. The target of about 1.45 test BPB was an experiment goal, not a guaranteed result.

## Reproduction and attribution

See `code/README.md` for installation and direct CPU scoring of the downloadable checkpoint. `code/configs/final_training_recipe.json` and `code/reproduce_final.py` replay the training recipe; exact trained weights can differ across hardware. The supplied data, tokenizer, evaluator and independent causal-window protocol are unchanged. The original submission remains at commit `d3a7c97d2f6ba1cbca6a0f8bc68fb076a2041b20` with its matching checkpoint and 1.5510079794 test BPB.

RoPE, SwiGLU and the original architecture screens follow course lectures. QKNorm follows [Henry et al.](https://aclanthology.org/2020.findings-emnlp.379/); R-Drop follows the [paper and author implementation](https://github.com/dropreg/R-Drop); temperature calibration follows [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html). The NormFormer screen adapts the [paper](https://arxiv.org/abs/2110.09456) to SwiGLU; late averaging borrows the [SWA idea](https://arxiv.org/abs/1803.05407). The optional Muon screen adapts [Keller Jordan's implementation](https://github.com/KellerJordan/Muon) and is not used by the final predictor. Codex assisted with implementation, debugging, experiment execution, checks and drafting. Detailed AI-assistance disclosure is in [`README.md`](README.md).

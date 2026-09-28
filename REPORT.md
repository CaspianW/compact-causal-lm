# MP1 report: small causal language model

## Model and training

The submitted model is a five-block causal Transformer trained from scratch on the supplied WikiText-2 training split. It has width 256 and four attention heads. Each block uses pre-layer normalization, RoPE, SwiGLU, residual dropout of 0.1, and attention dropout of 0.1. The token embedding and output projection share weights.

I trained for 9,600 updates with seed 17 and batches of 32 sequences, each contributing 256 next-token targets. That amounts to 78,643,200 sampled training targets. AdamW started at a learning rate of 0.001 with weight decay 0.1; the schedule had 100 warmup steps followed by cosine decay. Training ran in CUDA BF16, while validation and test evaluation used FP32. I chose the checkpoint using validation BPB alone.

The model has 4,465,040 parameters. The selected checkpoint is from step 9,600 and has SHA-256 `4121726b57a6528b2f83197e7fbc36ca46c90bf513413cffcf8b1630b9b08c90`. With CPU FP32 evaluation on the full test split, it scored **1.5510079794 BPB** across 428,405 targets and 1,292,013 UTF-8 bytes. I evaluated the test split only after fixing the method and checkpoint, and did not use it to choose settings.

## Baseline and ablation

| Run | Parameters | Training targets | Validation BPB | Test BPB |
| --- | ---: | ---: | ---: | ---: |
| Supplied baseline, 1,200 updates | 1,088,256 | 9,830,400 | 2.07108 | 2.10126 |
| Same architecture without attention dropout, 9,600 updates | 4,465,040 | 78,643,200 | 1.53755 | Not used for selection |
| Submitted model, attention dropout 0.1, 9,600 updates | 4,465,040 | 78,643,200 | **1.52665** | **1.55101** |

The two 9,600-update runs used the same seed, model size, batch size, and number of sampled training targets. I changed only attention dropout: 0 in the ablation and 0.1 in the submitted run. Validation BPB fell from 1.53755 to 1.52665, a difference of **0.01090 BPB**. This gives evidence for a small regularization benefit under this training setup. The original baseline used a smaller model and fewer training targets, so its score cannot isolate the effects of RoPE, SwiGLU, model size, or training length.

I also compared RMSNorm, partial RoPE, dropout rates, model widths and depths, head counts, GQA/MQA, local and gated attention, and Muon on validation. A separate set of seven structural screens covered MoE, MLA, DSA, linear attention, a DeltaNet hybrid, simplified mHC, and a common attention baseline. Those screens used width-128, depth-4 models trained for 1,200 updates each. MoE and simplified mHC gave the best results in that set, but I did not run either at the submitted model's scale. The DSA and mHC implementations test the core ideas taught in class, not the complete paper architectures.

## Training cost and evaluation limits

Saved metrics cover 53 completed runs, including CPU and GPU training and the short screens. Their measured training loops total about **185 minutes**; the submitted run accounts for 279 seconds of GPU training. In a paired full-test run with the supplied scorer and four CPU threads on an Intel Core i5-12600KF, scoring took 54.67 seconds for the submitted model and 18.94 seconds for the baseline. The ratio, 2.89, is below the 5-times limit. Absolute times varied between runs on Windows. The observed peak process working set was 1.81 GiB, below the 4 GiB limit, and the checkpoint was 19.05 MiB, below the 64 MiB inference-asset limit.

The model uses more parameters and CPU time than the baseline and achieves a lower BPB. The attention-dropout comparison covers one seed, so the 0.01090 BPB difference should be read as an observation from these two runs. The smaller structural screens helped rule out weaker candidates, but do not establish which architecture would perform best after full-scale training. GPU training may also differ slightly across hardware; the matching checkpoint allows the reported test score to be checked directly.

## Reproduction and attribution

Installation, training, and checkpoint-evaluation commands are in [`code/README.md`](code/README.md). The supplied data, tokenizer, evaluator, and causal-window protocol are unchanged. RoPE, SwiGLU, and the other screened architecture ideas came from the course lectures. An optional Muon experiment used a compact adaptation of the [public Muon implementation](https://github.com/KellerJordan/Muon); the submitted checkpoint does not use Muon. The recorded measurements were checked with the supplied evaluator and correctness tests.

A detailed disclosure of AI assistance is in the [repository README](README.md).

# MP1 code — installation and usage

## Final checkpoint for 30 September 2026

The final checkpoint scores **1.4899984302 full-test CPU FP32 BPB**. Its SHA-256 is `5dda862e738655ccd3406c055a8f8025b8fb5be8b5a4d9c2e5a38f2b45faa4f8`. Use Python 3.12 and the installation instructions below. Download the matching release bundle, extract `checkpoint.pt` to `code/runs/final-submission/`, and evaluate from `code/`:

```bash
python evaluate.py --checkpoint runs/final-submission/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test
```

No training is needed to verify this score. To replay the recorded CUDA training recipe instead:

```bash
python reproduce_final.py --recipe configs/final_training_recipe.json --run-dir runs/reproduced-final
```

The CPU time ratio is 4.05, peak evaluator working set is 1.82 GiB, and checkpoint size is 22.07 MiB. Full measurements and the recipe are in `final_result.json`; model selection used validation before the final test score was read. Exact weights can vary across training runs and hardware.

The older reproduction section below records the original 1.5510079794-BPB submission. It is retained as historical documentation; use the commands above for the final predictor.

Read [the project guide](../GUIDE.md) for the assignment, assessment, deadlines and peer review. This README contains the running instructions and technical rules. The submission report is [REPORT.md](../REPORT.md).

All commands below run from **code/**. Data and the tokenizer are included. No API key, pretrained weights or additional dataset download is needed; after installing dependencies, training and evaluation work offline.

## 1. Install

Use **Python 3.12**. From the extracted package directory:

```bash
cd code
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

Install PyTorch for **one** device:

```bash
# Linux/Windows CPU: recommended; no GPU needed
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

For an NVIDIA GPU with a compatible driver, use this command **instead**:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

For macOS, install `torch==2.7.1` from the default PyPI index and run on CPU. After installing PyTorch, install the remaining dependencies and check the model:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Linux CPU commands were verified with Python 3.12 and PyTorch 2.7.1+cpu. Windows/macOS timings have not been measured.

## 2. Train and evaluate

**Quick installation check** — 10 training steps, then full-test evaluation:

```bash
python train.py --implementation model --steps 10 --run-dir runs/smoke
python evaluate.py --checkpoint runs/smoke/checkpoint.pt --split test
```

This checks that the pipeline works; its score is **not** the full baseline. Each training run needs a new output directory.

**Full baseline** — 1,200 training updates, then evaluation:

```bash
python train.py --implementation model --device cpu --threads 4 --seed 17 --run-dir runs/baseline
python evaluate.py --checkpoint runs/baseline/checkpoint.pt --device cpu --precision fp32 --split test
```

The baseline has four GPT blocks, width 128, four attention heads and **1,088,256 parameters**, and achieves approximately **2.10 test BPB**. On the reference four-thread Xeon Platinum 8457C, measured training took about **311 seconds** and scoring **5.92 seconds**, excluding installation and loading. These are reference measurements, not laptop guarantees or a fixed time allowance.

**Your model** — edit `student.py` and supporting files, then:

```bash
python train.py --implementation student --seed 17 --eval-every 300 --run-dir runs/my-model
python evaluate.py --checkpoint runs/my-model/checkpoint.pt --split validation
# Freeze the final method before testing:
python evaluate.py --checkpoint runs/my-model/checkpoint.pt --split test
```

Training writes `checkpoint.pt` and `metrics.json`. Evaluation writes `test_cpu_fp32.json` (or the corresponding device/split name) and per-window losses. Submit the **bpb** value from the complete-test JSON, not token perplexity or validation BPB. Default evaluation is FP32. Add `--device cuda` for GPU runs; training can use BF16, but ranked evaluation must use FP32 and remain reproducible on CPU. The supplied CUDA runner caps PyTorch allocation at 20 GB; driver overhead is additional.

### Reproduce this submission

The selected model is `student.py` with the configuration in `configs/rope_swiglu_w256_d5_attndropout01.json`. From `code/`, train it with:

```bash
python train.py --implementation student --config configs/rope_swiglu_w256_d5_attndropout01.json --run-dir runs/reproduction --device cuda --precision bf16 --threads 4 --seed 17 --steps 9600 --batch-size 32 --eval-every 2400 --save-best
```

The published score uses the separately supplied `checkpoint-best.pt` (SHA-256 `4121726b57a6528b2f83197e7fbc36ca46c90bf513413cffcf8b1630b9b08c90`). Place it at `runs/final/checkpoint-best.pt`, or change the path below. No retraining is needed to evaluate it:

```bash
python evaluate.py --checkpoint runs/final/checkpoint-best.pt --device cpu --precision fp32 --threads 4 --split test
```

This produced **1.5510079794 test BPB** on 428,405 targets. The matching `student.py` SHA-256 is `ff07948b888b9a3dda01fcc02c8144834a0c58be470b84e4e9bc23779b78bdf5`. The checkpoint is a separate submission artifact and is intentionally excluded from Git; the code revision and checkpoint must both be linked in the final website submission.

## 3. Files and model interface

| Files | Use |
|---|---|
| `model.py`, `configs/baseline.json` | Runnable baseline; preserve for comparisons. |
| `student.py`, `train.py` | Your model factory and training recipe; add supporting code as needed. |
| `common.py`, `evaluate.py` | Fixed data checks, windows and scorer; keep unchanged. |
| `data/` | Supplied splits, tokenizer and dataset hashes; keep unchanged. |
| `tests/test_contract.py` | Checks your model's causality, normalization, independence and gradients. |
| `RUN_LOG_TEMPLATE.csv` | Optional experiment-log template. |
| `PACKAGE_MANIFEST.json` | Release hashes; paths are relative to the package root containing code/ and guide/. |

- `build_model(config)` returns a PyTorch model with `context=256`.
- The supplied trainer calls `forward(ids)` for unnormalized logits; the scorer calls `predict_log_probs(ids)` for finite, normalized natural-log probabilities. Both outputs have shape `[batch, time, 2048]`.
- A prediction at position t may use only the observed prefix through t. Reset temporary state between independent windows, examples and scoring passes. Compact training-derived assets may be reused across windows; evaluation-prefix state may not.
- Checkpoints record the implementation module and configuration. Include that module and every required asset so the evaluator can reconstruct the submitted predictor. No optimizer state is required for direct evaluation.
- Training length, architecture, optimizer, regularization, self-trained weight averaging and ensembles may change within the guide's constraints. Log all seeds, processed training targets, checkpoint ancestry and search costs; reusing a checkpoint does not erase its training cost. No particular seed or score improvement is mandated.

## 4. Benchmark and resource measurements

**Fixed score.** Protocol `7506-mp1-wt2-v2`: WikiText-2 raw text, train-fitted BPE-2048, independent windows of 256 targets, including the final short window. Every target except the first token of each split is scored once. Input windows share a boundary token but carry no state. BPB is summed negative log-base-2 next-token probability divided by the split's entire raw UTF-8 byte length, including the first token's bytes.

| Split | Scored targets | UTF-8 bytes |
|---|---:|---:|
| Validation | 376,599 | 1,148,007 |
| Test | 428,405 | 1,292,013 |

Use validation for all development and checkpoint/mixture selection. Weights, statistics and retrieval entries must derive only from training text. The public test text enables reproduction; it must not be used to tune the method. Once frozen, the same predictor may be evaluated repeatedly for timing or reproduction. Token perplexity is not directly comparable with published word-level perplexity.

Measure all three limits for the same frozen predictor:

- **CPU time ≤5× baseline:** submitted model 54.67 s versus baseline 18.94 s in a paired run with the original scorer on the same four-thread CPU (2.89×). Windows background load made absolute timings vary between runs.
- **Peak RAM ≤4 GiB:** observed 1.81 GiB process working set.
- **Inference assets ≤64 MiB uncompressed:** submitted checkpoint 19.05 MiB.

## 5. Prepare your submission and reproduce a peer

The [guide](../GUIDE.md) specifies the deadline and website workflow. Include the following in your immutable code repository:

- **Report, at most 10 pages including figures, tables and references** 
- **Reproduction instructions**

Your final website submission must link to this code and the matching complete checkpoint bundle. The website generates the Issue JSON automatically. Keep all inference assets downloadable for verification.

To check a peer, obtain their exact code version and checkpoint, follow their installation instructions, and run their frozen model with the supplied evaluator:

```bash
python evaluate.py --checkpoint /path/to/peer-checkpoint.pt --device cpu --precision fp32 --split test --output peer-test.json
```

Compare reproduced BPB with the reported score. Submit **Peer Review Report** with the reproduced score; optionally include the command, environment, difference and evidence/log link.  The instructor adjudicates discrepancies. Confirmed discrepancies during the seven-day review earn bonus credit under the announced marking policy.

## 6. Data attribution

WikiText-2 was introduced by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher in [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843). The text is by Wikipedia contributors. The [upstream dataset](https://huggingface.co/datasets/Salesforce/wikitext) identifies [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and the [GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html); retain these notices when redistributing the data.

The supplied `wikitext-2-raw-v1` splits preserve revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Rows are joined with newlines and encoded as UTF-8; the tokenizer is fitted only to training text. Dataset hashes are in `data/manifest.json`. These dataset notices do not assign a new license to the surrounding classroom code.

## 7. Reused work and AI assistance

The course starter provided the baseline model, data, tokenizer, loader, scorer, and contract tests. These fixed files remain unchanged. The submitted model lives in `student.py`, while `train.py` contains the added configuration and validation-checkpoint options. `course_models.py` holds exploratory architecture tests. RoPE, SwiGLU, and the other screened architecture ideas were covered in the course lectures. The optional `muon.py` experiment adapts the Newton-Schulz update from [Keller Jordan's Muon implementation](https://github.com/KellerJordan/Muon); the submitted checkpoint uses AdamW.

Codex assisted with implementing and debugging model variants and training options, setting up and comparing CPU and GPU experiments, checking the final score and resource use, and drafting the README and report. The final checkpoint was chosen by validation BPB and evaluated with the supplied scorer. The same disclosure appears in the [repository README](../README.md).

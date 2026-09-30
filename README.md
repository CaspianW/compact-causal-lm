# DASE7506 MP1

The final method, score, comparisons and reproduction instructions are in [REPORT.md](REPORT.md) and [code/README.md](code/README.md). This file discloses AI assistance and reused work.

## AI assistance and reused work

The course starter supplied the baseline model, data, tokenizer, data loader, evaluator, and contract tests. The optional `code/muon.py` experiment adapts the Newton-Schulz update from [Keller Jordan's Muon implementation](https://github.com/KellerJordan/Muon). The submitted checkpoint uses AdamW.

Codex assisted with the following work:

- Partially implementing and debugging the submitted Transformer in `code/student.py`, including RoPE, SwiGLU, tied weights, and dropout.
- Adding training configuration and validation-based checkpoint selection in `code/train.py`, and implementing exploratory model variants in `code/course_models.py`.
- Planning and running CPU and CUDA experiments, comparing validation BPB across model, dropout, attention, and optimizer settings, and organizing the saved results.
- Running the supplied correctness tests and evaluator on the selected checkpoint, then checking its full-test BPB, CPU inference time, memory use, and file size.
- Drafting and revising the repository documentation and `REPORT.md` from the recorded experiments.
- Implementing and checking the later QKNorm, NormFormer, temperature-calibration, R-Drop, and continued-training experiments.
- Preparing the final validation selection, matched R-Drop ablation, checkpoint-averaging checks, CPU resource measurements, training-recipe replay, and submission artifacts.

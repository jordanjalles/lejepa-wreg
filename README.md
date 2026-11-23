# Projector-free Wasserstein vs LeJEPA

Can my Wasserstein regularizer keep JEPA features spread out directly, with no projector? ViT-S/8 on Imagenette: LeJEPA as published (MLP + BatchNorm projector + SIGReg) vs the Wasserstein regularizer with no projector vs SIGReg with no projector, scored by a frozen linear probe.

Part of a series of repos from my 2026 machine-learning sabbatical: a designer learning ML by building models from scratch and running experiments.

## What's here

- `lejepa_wreg/`: model, data, training, evaluations (linear probe, kNN, retrieval, spectrum) and diagnostics
- `tools/`: data download, run queue, smoke test
- `queue.txt`: the runs

## Running it

`python tools/fetch_data.py`, then `python tools/smoke.py`; full runs are listed in `queue.txt`.

## Results

In progress; no verdict yet.

**Code:** written and run by Claude (Anthropic) as a coding agent, to my experiment specs. I chose the questions, the controls and the pass/fail rules, and read the results.

**Requirements:** Python 3.11+, PyTorch 2.x, torchvision, matplotlib, a CUDA GPU for anything beyond MNIST.

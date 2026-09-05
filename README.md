# CylinderFlow stride-8: NVIDIA MeshGraphNets adaptation

This independent private copy preserves NVIDIA Modulus `v0.1.0`, commit `307e446d43741d3c9db2432483f585da175ba3e4`, and its Apache-2.0 license. Use [installation and commands](CYLINDERFLOW.md), [the common data/evaluation contract](DATA_CONTRACT.md), and [current alignment verification](ALIGNMENT_VERIFICATION.json) and [prior prefix65 acceptance](ACCEPTANCE.json).

The adapter uses NVIDIA's original 15-block MeshGraphNet with hidden width 128 (2,332,419 parameters). It learns dt=0.08 transitions from scratch: current velocity and four node-type indicators are the six node inputs; directed displacement/length are three edge inputs; outputs are normalized velocity increments and next-frame pressure. Only velocity is recurrent state. All Train trajectories supply 74 adjacent pairs in frames 0..74. The budget is 25 epochs × 74,000 pairs, batch 1, Adam 1e-4, MSE, exponential LR multiplier 0.9999991 per update and normal-node velocity input noise 0.02. Actual updates and exposure are retained in structured logs.

After installation and preparation, with `DATA` and `MANIFEST` set as in the guide:

```bash
git remote add upstream https://github.com/NVIDIA/modulus.git
python -m cylinderflow train --dataset "$DATA" --manifest "$MANIFEST" --prepared runs/prepared --device cuda --output-dir runs/train
python -m cylinderflow resume --dataset "$DATA" --manifest "$MANIFEST" --prepared runs/prepared --checkpoint runs/train/last.pt --device cuda --output-dir runs/train
```

Use train for a new run or resume for an existing run. The model implementation remains unchanged; independent dataset, training, recovery, physical evaluation and rendering are in `cylinderflow/`. No original Modulus training/data entry point is imported. The new path does not require Apex, Hydra, W&B, TensorFlow, a host-specific CUDA library directory, or another private repository.

CPU acceptance uses the full network, exact resume, 64-step prediction and real media. Target GPU/mixed-precision/maximum-mesh acceptance is provided as a separate explicit command; no formal long training was launched. Original upstream documentation follows for provenance.

---

# Modulus (Beta)

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![GitHub](https://img.shields.io/github/license/NVIDIA/modulus)](https://github.com/NVIDIA/modulus/blob/master/LICENSE.txt)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Modulus is a PyTorch based deep-learning toolkit for developing deep learning models for physical systems. This package aims to provide useful utilities for physics-constrained and data-driven workflows.

**This is an early-access beta release**

<p align="center">
  <img src="./docs/img/modulus-pipes.jpg" alt="NVIDIA Modulus"/>
</p>

## Modulus Packages

- [Modulus (Beta)](https://github.com/NVIDIA/modulus)
- [Modulus Launch (Beta)](https://github.com/NVIDIA/modulus-launch)
- [Modulus Symbolic (Beta)](https://github.com/NVIDIA/modulus-sym)

## Installing 

Modulus is coming to PyPi soon! In the mean time the best way is to install from source:

```Bash
git clone git@github.com:NVIDIA/modulus.git && cd modulus

pip install --upgrade pip
pip install .
```

## Docker

To build Modulus docker image:
```
docker build -t modulus:deploy --target deploy -f Dockerfile .
```
Alternatively, you can run `make container-deploy`

To build CI image:
```
docker build -t modulus:ci --target ci -f Dockerfile .
```
Alternatively, you can run `make container-ci`


## Contributing

Modulus is in an open-source beta. We are not accepting external contributions at this time.

## Contact

Reach out to Modulus team members and user community on the [NVIDIA developer forums](https://forums.developer.nvidia.com/c/physics-simulation/modulus-physics-ml-model-framework).

## License
Modulus is provided under the Apache License 2.0, please see [LICENSE.txt](./LICENSE.txt) for full license text.

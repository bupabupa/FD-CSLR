# FD-XMP on VAC-CSLR

This repository is based on
[`VIPL-SLP/VAC_CSLR`](https://github.com/VIPL-SLP/VAC_CSLR) and contains the
FD-XMP continuous sign language recognition extensions.

## Included extensions

- CTC-aligned visual prototypes and prototype recombination (PRA).
- Latent temporal prototype reconstruction with signer control (RDB).
- Weighted CTC, FDGC calibration, semantic text prototypes and multichannel
  calibration.
- Real and pseudo-signer protocols, leave-one-signer utilities and
  signer-invariance metrics.
- Split manifests, candidate locking, one-time Test access and evaluation
  utilities.
- Frequency-aware metrics, retrieval diagnostics, cost profiling and a
  non-GPU regression suite.

## Main entry points

```text
fd_xmp_network.py                 FD-XMP model
modules/rdb.py                    latent temporal prototype reconstruction
modules/prototype.py              CTC prototype head
modules/prototype_recombination.py PRA ablation
main.py                           training/evaluation entry point
configs/fdxmp_rdb_split.yaml      full RDB configuration
RDB_DESIGN_20261115.md            RDB design and failure criteria
```

The original VAC setup, PHOENIX-2014 data preparation, `ctcdecode`, SCTK and
training instructions remain applicable. Dataset files, checkpoints,
generated logs and local software binaries are intentionally excluded from
version control.

## Verification

The latest non-GPU regression suite passes `58/58` in the experiment
environment. The focused model tests can be run with:

```bash
python -m pytest -q \
  tests/test_rdb.py \
  tests/test_prototype_recombination.py \
  tests/test_fdxmp_module_switches.py \
  tests/test_split_configs.py
```

The RDB GPU smoke test requires CUDA and prepared PHOENIX-2014 data:

```bash
python scripts/smoke_rdb.py
```

Code completion does not imply a performance gain. Formal RDB training and
the matched ablations described in `RDB_DESIGN_20261115.md` still need to be
run.

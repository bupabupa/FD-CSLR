# RDB Design: Latent Temporal Prototype Reconstruction

## 1. Scope

RDB means **latent temporal prototype reconstruction**. It reconstructs or
counterfactually perturbs the latent CTC feature sequence. It does not generate
RGB pixels, videos, or new gloss labels. Any paper claim must therefore use
"latent feature reconstruction" rather than "video synthesis" or "generated
samples".

The existing PRA remains available as an ablation. RDB is disabled by default,
has an independent configuration, and does not alter the inference fusion path.
This keeps old checkpoints and the PRA baseline intact.

## 2. Inputs

For a feature sequence `F in R^(T x B x D)`:

- `mu in R^(C x D)` is the EMA visual prototype table.
- `support in R^C` is the accumulated prototype support.
- `gamma in R^(T x B x C)` is CTC posterior class occupancy.
- `confidence in R^(T x B)` is the top-two posterior margin.
- `L in Z^B` contains valid feature lengths.
- `S_a = [gamma_signer,a, beta_signer,a] in R^(2D)` is the signer FiLM
  style context. It is optional when signer FiLM is disabled.

The class content sequence excludes blank and unsupported prototypes:

```text
M_t = sum_c normalized(gamma_t,c) * mu_c,
      c in supported non-blank classes
```

The short-time motion residual is:

```text
Z_t = F_t - Smooth_k(F_t)
```

`Smooth_k` is a length-aware average over `k` adjacent frames. It is computed
from valid frames only, so padded frames do not create artificial motion.

## 3. Reconstruction

The decoder receives content, motion, style, posterior mass, and confidence:

```text
U_t = Decoder([F_t, M_t, Z_t, S_a, mass_t, confidence_t])
F_hat_t = F_t + alpha_t * a_t * tanh(U_t)
```

Here `a_t` is one only when all of the following hold:

- the frame is inside the sample length;
- supported non-blank posterior mass is at least
  `rdb_min_content_mass`;
- the frame confidence is at least `rdb_confidence_threshold`.

The learned scalar `alpha` is bounded by `rdb_max_strength`. The decoder uses a
pointwise projection, a depthwise temporal convolution, and a zero-initialized
output projection. Thus:

```text
F_hat = F
```

at initialization. The gate is deliberately non-zero at initialization so
reconstruction gradients can reach the zero output projection. Padded frames
are exactly unchanged and are excluded from reconstruction loss.

The reconstruction error itself is exactly zero at initialization. The
original-label CTC branch bootstraps the decoder from that identity point; the
reconstruction term then regularizes the learned latent update.
`RDBCTC` is therefore mandatory whenever RDB is enabled. A reconstruction-only
configuration is rejected instead of silently leaving the decoder at zero.

## 4. Label-Preserving Closure

### 4.1 Reconstruction loss

```text
L_recon =
  masked_MSE(F_hat, stopgrad(F))
  + lambda_cos * masked_cosine_distance(F_hat, stopgrad(F))
```

The mask is the intersection of valid length and the RDB activation gate. The
cosine term is divided by a configured positive temperature. Stop-gradient
prevents the reconstruction target from moving toward the augmenter.

### 4.2 Original-label CTC

```text
L_rdb_ctc = CTC(Classifier(F_hat), Y)
```

`Y` is exactly the original gloss sequence. A reconstruction is not accepted
as useful if it cannot still map to the original label sequence.

### 4.3 Counterfactual signer style

For an available alternate signer `b`, RDB reuses the same `F`, `M`, and `Z`
and replaces only the style context:

```text
F_hat_b = F + alpha_t * a_t * tanh(Decoder([F_t, M_t, Z_t, S_b, ...]))

L_style =
  CTC(Classifier(F_hat_b), Y)
  + lambda_sep * relu(margin - relative_RMS(F_hat_b - F_hat))
```

The CTC term preserves content and labels. The separation term prevents the
style branch from collapsing to an identity. Both original and counterfactual
updates are structurally bounded by `alpha * tanh(.)`.

`RDBStyle` is skipped for a batch if no alternate seen signer exists. It is
never enabled when `signer_film=False`.

## 5. Full Objective

The new configuration uses:

```text
L = L_seq_ctc + L_conv_ctc + L_proto_align + L_signer
    + 0.05 * L_recon
    + 0.05 * L_rdb_ctc
    + 0.02 * L_style
```

`L_recon` has a five-epoch linear warmup. `L_rdb_ctc` and `L_style` keep the
model anchored to the original label sequence throughout training.

The RDB branch is auxiliary. It does not replace `sequence_logits`, is not
decoded at inference, and cannot change the existing PRA path when disabled.

## 6. Code Interface

The implementation is in `modules/rdb.py` and is integrated by
`fd_xmp_network.FDXMPModel`.

Main switches:

```text
rdb_enabled: false
rdb_max_strength: 0.20
rdb_hidden_size: 64
rdb_kernel_size: 3
rdb_min_content_mass: 0.05
rdb_min_prototype_support: 0.05
rdb_confidence_threshold: 0.1
rdb_recon_warmup_epochs: 5
```

Loss keys:

```text
RDBRecon
RDBCTC
RDBStyle
```

The full experimental configuration is
`configs/fdxmp_rdb_split.yaml`.

## 7. Required Ablations

1. Existing PRA only.
2. RDB without style swap: `RDBRecon + RDBCTC`.
3. RDB without reconstruction: `RDBCTC` only.
4. Full RDB: `RDBRecon + RDBCTC + RDBStyle`.
5. No-signer RDB.
6. Different `rdb_max_strength` values, for example `0.10`, `0.20`, and
   `0.30`.

All RDB variants must use the same split manifest, seed, checkpoint selector,
and class prior as their corresponding baseline.

## 8. Validation and Failure Criteria

Report at least:

- Dev/Test WER and per-class recall, especially tail classes;
- CTC loss for the original and RDB branches;
- reconstruction MSE and cosine distance;
- mean latent shift, active fraction, and bounded strength;
- signer-swap relative latent shift;
- RDB branch label preservation;
- retrieval or prototype-alignment metrics where applicable.

RDB should be rejected or redesigned if any of the following holds:

- full RDB does not outperform PRA under the matched protocol;
- reconstruction loss improves while Dev WER does not;
- signer style swap changes the decoded gloss sequence on validation data;
- tail-class recall does not improve;
- the style branch collapses to zero or produces unbounded/outlier features;
- only reconstruction or retrieval improves, without CTC or WER gain;
- gains disappear under a different seed or split manifest.

## 9. Methodological References

- Min et al., "Visual Alignment Constraint for Continuous Sign Language
  Recognition", ICCV 2021.
- Graves et al., "Connectionist Temporal Classification: Labelling
  Unsegmented Sequence Data with Recurrent Neural Networks", ICML 2006.
- Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer",
  AAAI 2018.
- Chen and He, "Exploring Simple Siamese Representation Learning", CVPR 2021.
- Snell et al., "Prototypical Networks for Few-shot Learning", NeurIPS 2017.

These references motivate the CTC alignment, signer conditioning, stop-gradient
target, and prototype content representation. They do not by themselves
establish that this particular RDB variant improves CSLR; that remains an
experimental question.

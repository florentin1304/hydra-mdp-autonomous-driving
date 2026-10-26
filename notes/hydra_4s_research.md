# Hydra 4 s research log

Plan: 4 s @ 10 Hz vocab from 0.5 s-strided re-centered windows, flatten-xy L2
imitation, xy-only MLP, paper Eq. 6 self-attn once over K, train windows
class-capped to ~48k. Full epoch before judging.

## 2026-10-21 — implement plan (no off-plan changes)

- Built `scripts/build_hydra_vocab.py --horizon 40 --stride 5 --snap-real --cap 12`.
- `distance: flatten_l2`, tokenizer `40*2` xy only, `vocab_self_attn_layers: 2`.
- Train rows are `(scene, t0)`; val stays one origin per packed scene.
- Artifacts: `data/artifacts/hydra_vocab_4s.npy`, `*_windows.npz`, `hydra_metrics_4s/`.
- Did **not** overwrite `hydra_vocab.npy` / 8 s metric cache.

## 2026-10-21 — vocab built (on-plan)

- 536855 re-centered 4 s windows from 48805 train scenes (stride 5).
- k-means k=4096 on flattened xy; snap to unique real windows.
- Wrote `data/artifacts/hydra_vocab_4s.npy` shape `(4096, 40, 3)`.
- Class cap 12: **48977** windows, **4087 / 4096** occupied classes.
- Did not touch `hydra_vocab.npy` or `hydra_metrics/`.

## 2026-10-21 — train metric cache done (on-plan)

- `hydra_metrics_4s/train.h5`: N=48977 × K=4096 × 8, version=3.
- Means: nc 60%, dac 98%, ddc 82%, tlc 90%, ttc 63%, lk 92%, hc 93%, progress 68%.
- Next: unwindowed val cache, then `checkpoints/hydra_4s` full epoch.

## 2026-10-21 — val metric cache done (on-plan)

- `hydra_metrics_4s/val.h5`: N=4383 × K=4096 × 8 (one origin per scene).
- Starting `PYTHONPATH=src python -m womd_il.train decoder=hydra ckpt_dir=checkpoints/hydra_4s wandb.name=hydra_4s lr=1e-4`.
- Judge only after a **full epoch** + val.

## 2026-10-21 — first train finished (on-plan)

- Early stop epoch **18** (patience 4). `best.pt` = val **argmaxADE@80 = 0.571 m** (ep14; also tied ep18).
- Path: ep0 1.41 → ep4 0.66 → ep9 0.58 → best 0.57. Oracle ADE 0.164. Vocab minADE@40 ~0.5.
- Train `im` left random ~8.3, settled ~5–6 (`exp(im)` ~150–400 mode blob). Occasional NaN skips.
- Closed-loop on `best.pt` (n=64): **drives** (~44 m progress, matches log ~43 m) — not park/floor.
  - log replay: collision 11%, offroad 0%, log_ade 2.16 m
  - IDM: collision 33%, offroad 0%, log_ade 2.36 m
- Open-loop pick is much better than old 8 s Hydra argmax (~3–4 m). CL still collides under IDM.
- Next (one change): improve infer/PDM ranking or harden the soft blob — not another tokenizer.

## 2026-10-22 — metric v4 recache + LDiP stochastic train

- DAC = road-edge segment crossing; TLC uses future TL; DDC no 4 m gate; cache version=4.
- Recache train+val → `hydra_metrics_4s/` then train `checkpoints/hydra_4s_ldip`.
- LDiP prune: `prune_to=[256,16]`, `gumbel_tau=1.0`, `nms_radius=0` (Gumbel-Top-k, not geo-NMS),
  `keep_winner=false`. Equal CE at each stage. Infer = deterministic top-k.

# WOMD Hydra imitation learning

Research stack for the
[Waymo Open Motion Dataset](https://waymo.com/open/data/motion/) (WOMD): a
Wayformer-style encoder with a **Hydra-MDP** discrete trajectory decoder
(vocab classification + optional EPO / rule metric heads).

Closed-loop evaluation drives **only the SDC**. Other agents are either
**log-replayed** or controlled by a simple **IDM** route policy. Simulation uses
a packed-scenario kinematic backend (bicycle + IDM).

## License / data

WOMD is **not** redistributed here. You must accept the
[Waymo Open Dataset license](https://waymo.com/open/terms/) and download the
data yourself. Full scenario TFRecords are on the order of **~500 GB**.

This repo targets **Motion Dataset v1.3.1** (includes `sdc_paths` for route
metrics).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

WOMD parsing does **not** need TensorFlow. Scenario protos are vendored in
`src/womd_il/protos/`.

## Download

Requires a Waymo-authorized GCS account (`gcloud auth login`).

```bash
# Smoke: one validation shard
bash scripts/download_womd.sh --split validation --smoke --out data/womd/raw

# Research-scale (~10% of WOMD)
bash scripts/download_womd.sh --split validation --max-shards 15 --out data/womd/raw
bash scripts/download_womd.sh --split training --max-shards 100 --out data/womd/raw
```

## Preprocess

```bash
python scripts/preprocess.py \
  --raw-dir data/womd/raw/validation \
  --out data/womd/packed/val.h5

python scripts/preprocess.py \
  --raw-dir data/womd/raw/training \
  --out data/womd/packed/train.h5 \
  --max-scenarios 50000
```

## Artifacts (4 s vocab)

```bash
python scripts/build_hydra_vocab.py \
  --h5 data/womd/packed/train.h5 \
  --out data/artifacts/hydra_vocab_4s.npy \
  --k 4096 --horizon 40

python scripts/cache_hydra_metrics.py \
  --h5 data/womd/packed/train.h5 \
  --vocab data/artifacts/hydra_vocab_4s.npy \
  --out data/artifacts/hydra_metrics_4s
```

(`build_hydra_vocab.py` also writes `hydra_vocab_4s_windows.npz` when configured.)

## Train

Defaults match the published Hydra stack (SDE ego, factorized social pool,
triple-cross decoder, mix imitation+EPO, kinematic channel dropout 0.5):

```bash
python -m womd_il.train \
  data.train_file=data/womd/packed/train.h5 \
  data.val_file=data/womd/packed/val.h5 \
  ckpt_dir=checkpoints/hydra_4s
```

Useful overrides:

```bash
# Resume
python -m womd_il.train resume=true ckpt_dir=checkpoints/hydra_4s

# Ablate SDE dropout / ego-on-query
python -m womd_il.train model.ego_sde_dropout=0.0
python -m womd_il.train decoder.ego_on_query=false
```

## Evaluate

```bash
python -m womd_il.eval_open_loop ckpt=checkpoints/hydra_4s/best.pt

python -m womd_il.eval_closed_loop \
  ckpt=checkpoints/hydra_4s/best.pt \
  neighbors=log mode_select=geomean

python -m womd_il.eval_closed_loop \
  ckpt=checkpoints/hydra_4s/best.pt \
  neighbors=idm mode_select=geomean
```

Diverse closed-log videos:

```bash
python scripts/render_diverse_videos.py \
  --ckpt checkpoints/hydra_4s/best.pt \
  --out checkpoints/hydra_4s/videos_diverse_geomean \
  --select geomean --h-exec 1
```

## Layout

```
configs/          Hydra configs (data, model, hydra decoder, eval)
src/womd_il/      Encoder, Hydra decoder, train, eval, sim, metrics
scripts/          Download, preprocess, vocab, metric cache, videos
notes/            Research notes
tests/            Unit / integration tests
```

## References

- Nayakanti et al., *Wayformer*, ICRA 2023
- Ettinger et al., *Waymo Open Motion Dataset*, ICCV 2021
- Li et al., *Hydra-MDP*, CVPR 2024
- Cheng et al., *PlanTF* (StateAttentionEncoder / SDE)

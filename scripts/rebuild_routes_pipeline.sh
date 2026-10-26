#!/usr/bin/env bash
# Pack routed H5 (synthetic routes if sdc_paths missing) → vocab → metric cache v5 → EPO train.
# Assumes data/womd/raw/{validation,training} already downloaded.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
mkdir -p logs checkpoints/hydra_4s_epo_v2

verify_routes() {
  echo "[$(date -Is)] verifying routes via parse (sdc_paths or synthetic)..."
  PYTHONPATH=src python -u <<'PY'
from womd_il.data.tfrecord import iter_tfrecord_file, list_tfrecord_shards
from womd_il.data.parse_scenario import parse_tfrecord_bytes

for split in ("validation", "training"):
    shards = list_tfrecord_shards(f"data/womd/raw/{split}")
    ok = total = 0
    for sh in shards[:2]:
        for i, b in enumerate(iter_tfrecord_file(sh)):
            if i >= 40:
                break
            sample = parse_tfrecord_bytes(b)
            total += 1
            if int(sample["route_valid"][0].sum()) >= 2:
                ok += 1
            if total == 1:
                print(
                    f"  {split} first route0_pts={int(sample['route_valid'][0].sum())} "
                    f"on_route={bool(sample['route_on_route'][0])} "
                    f"n_paths={int((sample['route_valid'].sum(1) >= 2).sum())}"
                )
    print(f"  {split}: {ok}/{total} usable routes ({100 * ok / max(total, 1):.0f}%)")
    if ok < total or ok == 0:
        raise SystemExit(f"FATAL: incomplete synthetic/official routes in {split}")
print("route verification OK")
PY
}

pack_split() {
  local split="$1" out="$2"
  echo "[$(date -Is)] packing $split -> $out"
  mkdir -p "$(dirname "$out")"
  rm -f "$out"
  PYTHONPATH=src python -u scripts/preprocess.py \
    --raw-dir "data/womd/raw/${split}" \
    --out "$out" \
    --require-route
  PYTHONPATH=src python -u <<PY
import h5py
import numpy as np
p = "$out"
with h5py.File(p, "r") as f:
    n = int(f.attrs["n"])
    if "map_on_route" not in f:
        raise SystemExit(f"FATAL: {p} missing map_on_route (repack with current parse)")
    rv = f["route_valid"][:]
    counts = rv.sum(axis=-1).sum(axis=-1)
    frac = float((counts >= 2).mean())
    # sample corridor coverage (full scan is cheap enough for val; subsample train)
    step = max(1, n // 2000)
    on = f["map_on_route"][::step]
    frac_any = float(on.any(axis=-1).mean())
    mean_on = float(on.sum(axis=-1).mean())
    print(
        f"  packed {p}: N={n}  frac_with_route={frac:.4f}  mean_pts={counts.mean():.1f}  "
        f"map_on_route frac_any={frac_any:.3f} mean_lanes={mean_on:.1f}"
    )
    if frac < 0.999:
        raise SystemExit(f"FATAL: only {frac:.2%} scenarios have routes")
    if frac_any < 0.5:
        raise SystemExit(f"FATAL: map_on_route too sparse ({frac_any:.2%})")
print("pack route + map_on_route check OK")
PY
}

SKIP_DL="${SKIP_DL:-1}"
if [[ "$SKIP_DL" != "1" ]]; then
  echo "ERROR: this script expects raw shards already present (SKIP_DL=1)." >&2
  exit 1
fi

# Free GPU + packs before rebuild (try run reads old H5).
echo "[$(date -Is)] killing hydra_4s_epo_try if running..."
pkill -f 'ckpt_dir=checkpoints/hydra_4s_epo_try' || true
pkill -f 'womd_il.train.*hydra_4s_epo_try' || true
sleep 4

verify_routes

echo "[$(date -Is)] freeing old packs for disk headroom..."
mkdir -p data/womd/packed_legacy_nopaths
# Drop previous full packs (raw remains; can re-pack anytime). Keep tiny leftovers out.
rm -f data/womd/packed/train.h5 data/womd/packed/val.h5
rm -f data/womd/packed/train_routes.h5 data/womd/packed/val_routes.h5
df -h . | tail -1

pack_split validation data/womd/packed/val.h5
pack_split training data/womd/packed/train.h5

echo "[$(date -Is)] verifying train batches see route + on-route corridor..."
PYTHONPATH=src python -u scripts/verify_route_in_training.py \
  --h5 data/womd/packed/train.h5 --n 64

echo "[$(date -Is)] rebuilding vocab+windows..."
PYTHONPATH=src python -u scripts/build_hydra_vocab.py \
  --h5 data/womd/packed/train.h5 \
  --out data/artifacts/hydra_vocab_4s.npy \
  --windows-out data/artifacts/hydra_vocab_4s_windows.npz \
  --k 4096 --horizon 40 --stride 5 --cap 12

echo "[$(date -Is)] metric cache val (v5)..."
mkdir -p data/artifacts/hydra_metrics_4s_v4_backup
if [[ -f data/artifacts/hydra_metrics_4s/train.h5 ]]; then
  mv -f data/artifacts/hydra_metrics_4s/train.h5 data/artifacts/hydra_metrics_4s/val.h5 \
    data/artifacts/hydra_metrics_4s_v4_backup/ 2>/dev/null || true
fi
rm -rf data/artifacts/hydra_metrics_4s_v5_build
mkdir -p data/artifacts/hydra_metrics_4s_v5_build data/artifacts/hydra_metrics_4s
PYTHONPATH=src python -u scripts/cache_hydra_metrics.py \
  --h5 data/womd/packed/val.h5 \
  --vocab data/artifacts/hydra_vocab_4s.npy \
  --out-dir data/artifacts/hydra_metrics_4s_v5_build \
  --device cuda --batch-size 8 \
  2>&1 | tee logs/cache_metrics_val_v5.log

echo "[$(date -Is)] metric cache train (windowed, v5)..."
PYTHONPATH=src python -u scripts/cache_hydra_metrics.py \
  --h5 data/womd/packed/train.h5 \
  --vocab data/artifacts/hydra_vocab_4s.npy \
  --out-dir data/artifacts/hydra_metrics_4s_v5_build \
  --windows data/artifacts/hydra_vocab_4s_windows.npz \
  --device cuda --batch-size 8 \
  2>&1 | tee logs/cache_metrics_train_v5.log

mv -f data/artifacts/hydra_metrics_4s_v5_build/train.h5 data/artifacts/hydra_metrics_4s_v5_build/val.h5 \
  data/artifacts/hydra_metrics_4s/
rmdir data/artifacts/hydra_metrics_4s_v5_build 2>/dev/null || true
rm -f data/artifacts/hydra_winner_pdms_4s.npz

echo "[$(date -Is)] starting EPO train hydra_4s_epo_v2 (τ=1.0 ent=0.05)..."
mkdir -p checkpoints/hydra_4s_epo_v2
stdbuf -oL -eL nohup python -u -m womd_il.train \
  decoder=hydra \
  model.factorized=true \
  model.agent_pool=lstm \
  model.agent_lstm_layers=2 \
  model.use_route_tokens=false \
  decoder.dual_cross=true \
  decoder.main=epo \
  decoder.policy_head=epo \
  decoder.prune_to=[] \
  decoder.metric_stopgrad=true \
  decoder.metric_loss_scale=1.0 \
  decoder.epo_loss_scale=1.0 \
  decoder.epo_entropy=0.05 \
  decoder.epo_temperature=1.0 \
  decoder.metric_cache=data/artifacts/hydra_metrics_4s \
  decoder.windows_path=data/artifacts/hydra_vocab_4s_windows.npz \
  decoder.winner_cache=data/artifacts/hydra_winner_pdms_4s.npz \
  decoder.vocab_self_attn_layers=0 \
  ckpt_dir=checkpoints/hydra_4s_epo_v2 \
  resume=null \
  lr=1e-4 \
  epochs=20 \
  early_stop.patience=5 \
  early_stop.min_epochs=3 \
  wandb.enabled=false \
  > checkpoints/hydra_4s_epo_v2/train.log 2>&1 &
echo "train_pid=$!"
nohup bash scripts/monitor_epo_soup.sh >> logs/monitor_epo_soup.log 2>&1 &
echo "[$(date -Is)] pipeline launched train; monitor checkpoints/hydra_4s_epo_v2/train.log"

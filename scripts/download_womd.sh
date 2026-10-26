#!/usr/bin/env bash
# Download Waymo Open Motion Dataset scenario TFRecords (v1.3.1).
# Requires a licensed Waymo Open Dataset GCS account:
#   gcloud auth login
#   gcloud auth application-default login
set -euo pipefail

BUCKET="${WOMD_BUCKET:-gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario}"
OUT="data/womd/raw"
SPLIT="validation"
SMOKE=0
MAX_SHARDS=""

usage() {
  cat <<'EOF'
Usage: bash scripts/download_womd.sh [options]

  --out DIR          Destination root (default: data/womd/raw)
  --split NAME       training | validation | all  (default: validation)
  --smoke            Download a single shard of the chosen split
  --max-shards N     Cap the number of shards copied per split
  --bucket URI       Override GCS bucket prefix
  -h, --help         Show this help

Full training+validation scenario protos are on the order of ~500 GB.
Use --smoke or --max-shards for research-scale experiments.

If v1.3.1 is unavailable, retry with:
  WOMD_BUCKET=gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --max-shards) MAX_SHARDS="$2"; shift 2 ;;
    --bucket) BUCKET="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

ensure_gcloud() {
  if command -v gsutil >/dev/null 2>&1 || command -v gcloud >/dev/null 2>&1; then
    return 0
  fi
  local sdk_bin="${HOME}/google-cloud-sdk/bin"
  if [[ -x "${sdk_bin}/gsutil" || -x "${sdk_bin}/gcloud" ]]; then
    export PATH="${sdk_bin}:${PATH}"
    return 0
  fi
  echo "Need gsutil or gcloud. Install Google Cloud SDK and authenticate:" >&2
  echo "  curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz" >&2
  echo "  tar -xzf google-cloud-cli-linux-x86_64.tar.gz && ./google-cloud-sdk/install.sh" >&2
  echo "  gcloud auth login" >&2
  echo "  gcloud auth application-default login" >&2
  echo "Your Google account must be licensed for the Waymo Open Dataset." >&2
  exit 1
}

copy_split() {
  local split="$1"
  local dest="${OUT}/${split}"
  mkdir -p "${dest}"
  local src="${BUCKET}/${split}"
  echo "Listing ${src} ..."
  ensure_gcloud


  local listing
  if command -v gsutil >/dev/null 2>&1; then
    listing="$(gsutil ls "${src}/" | grep -E 'tfrecord' || true)"
  else
    listing="$(gcloud storage ls "${src}/" | grep -E 'tfrecord' || true)"
  fi
  if [[ -z "${listing}" ]]; then
    echo "No tfrecord shards found under ${src}" >&2
    exit 1
  fi

  mapfile -t shards <<< "${listing}"
  local n="${#shards[@]}"
  echo "Found ${n} shards in ${split}"

  if [[ "${SMOKE}" -eq 1 ]]; then
    shards=("${shards[0]}")
  elif [[ -n "${MAX_SHARDS}" ]]; then
    shards=("${shards[@]:0:${MAX_SHARDS}}")
  fi

  local missing=()
  for shard in "${shards[@]}"; do
    local dest_file="${dest}/$(basename "${shard}")"
    if [[ -f "${dest_file}" ]]; then
      echo "skip existing $(basename "${dest_file}")"
    else
      missing+=("${shard}")
    fi
  done
  echo "Copying ${#missing[@]} new shard(s) (${#shards[@]} listed) -> ${dest}"
  if [[ ${#missing[@]} -eq 0 ]]; then
    return 0
  fi
  # Parallel by default. Prefer gcloud storage; fall back to gsutil -m + boto.
  if [[ -z "${BOTO_CONFIG:-}" && -f "$(cd "$(dirname "$0")/.." && pwd)/configs/gsutil.boto" ]]; then
    export BOTO_CONFIG="$(cd "$(dirname "$0")/.." && pwd)/configs/gsutil.boto"
  fi
  if [[ "${WOMD_USE_GSUTIL:-0}" == "1" ]] || ! command -v gcloud >/dev/null 2>&1; then
    gsutil -m cp "${missing[@]}" "${dest}/"
  else
    gcloud storage cp "${missing[@]}" "${dest}/"
  fi
}

mkdir -p "${OUT}"
case "${SPLIT}" in
  training|validation) copy_split "${SPLIT}" ;;
  all)
    copy_split training
    copy_split validation
    ;;
  *)
    echo "split must be training, validation, or all" >&2
    exit 1
    ;;
esac

echo "Done. Raw shards in ${OUT}"

#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CMD="${SCRIPT_DIR}/train_x.sh"

loss="all"
version="ablations"
gpu="0"
session="x_train_$(date +%H%M%S)"
dataset="shapes"
base_beta_s="1.0"
base_beta_t="1.0"
base_l_nce="0.1"
modes="all,no_kl_t,no_kl_s,no_nce"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --loss) loss="$2"; shift 2 ;;
    --version) version="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --session) session="$2"; shift 2 ;;
    --dataset) dataset="$2"; shift 2 ;;
    --beta_s) base_beta_s="$2"; shift 2 ;;
    --beta_t) base_beta_t="$2"; shift 2 ;;
    --l_nce) base_l_nce="$2"; shift 2 ;;
    --modes) modes="$2"; shift 2 ;;
    --) shift; break ;;
    -*) echo "Option inconnue: $1" >&2; exit 2 ;;
    *) echo "Arg ignoré: $1" >&2; shift ;;
  esac
done

IFS=',' read -r -a mode_list <<<"${modes}"
declare -a windows=()

for i in "${!mode_list[@]}"; do
  mode="${mode_list[i]}"
  bs="${base_beta_s}"
  bt="${base_beta_t}"
  lnce="${base_l_nce}"

  case "${mode}" in
    all) ;; # keep defaults
    no_kl_t) bt="0" ;;
    no_kl_s) bs="0" ;;
    no_nce) lnce="0" ;;
    *) echo "Mode inconnu: ${mode}" >&2; exit 2 ;;
  esac

  win="${mode}"
  windows+=("${win}")

  if (( i == 0 )); then
    tmux new-session -d -s "$session" -n "$win"
  else
    tmux new-window -d -t "$session" -n "$win"
  fi

  tmux send-keys -t "$session:${win}" "${CMD} --dataset ${dataset} --beta_s ${bs} --beta_t ${bt} --l_nce ${lnce} --loss ${loss} --version ${version} --gpu ${gpu} --key mode=${mode}" C-m
done

echo "tmux a -t ${session}"

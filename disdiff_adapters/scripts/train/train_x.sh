#!/usr/bin/env bash
set -Eeuo pipefail

# Generic entry to launch XFactors training.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# Optional venv auto-activation
if [[ -d "${REPO_ROOT}/.venv" ]]; then
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/.venv/bin/activate"
fi

# Defaults
dataset="shapes"
beta_s="100"
beta_t="100"
l_nce_by_factors="0.1"
loss_type="all"
version_model="debug"
gpus="0"
batch_size="32"
max_epochs="50"
latent_dims_s="126"
dims_by_factors="2"
warm_up="False"
lr="1e-4"
arch="res"
l_cov="0.0"
l_anti_nce="0.0"
key=""
experience_override="no_kl"
kl_weight_scale="0.0"
wandb_enable="True"
wandb_entity="thomasboyer"
wandb_project="XFactors"
wandb_run_name="no_kl"
wandb_tags=""
wandb_mode=""
compute_metrics="True"
metric_interval="1"
metric_n_iter="153600"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) dataset="$2"; shift 2 ;;
        --beta_s) beta_s="$2"; shift 2 ;;
        --beta_t) beta_t="$2"; shift 2 ;;
        --l_nce) l_nce_by_factors="$2"; shift 2 ;;
        --loss) loss_type="$2"; shift 2 ;;
        --version) version_model="$2"; shift 2 ;;
        --gpu) gpus="$2"; shift 2 ;;
        --batch) batch_size="$2"; shift 2 ;;
        --epochs) max_epochs="$2"; shift 2 ;;
        --latent_s) latent_dims_s="$2"; shift 2 ;;
        --dims_by_factors) dims_by_factors="$2"; shift 2 ;;
        --arch) arch="$2"; shift 2 ;;
        --warm_up) warm_up="$2"; shift 2 ;;
        --lr) lr="$2"; shift 2 ;;
        --l_cov) l_cov="$2"; shift 2 ;;
        --l_anti_nce) l_anti_nce="$2"; shift 2 ;;
        --key) key="$2"; shift 2 ;;
        --experience) experience_override="$2"; shift 2 ;;
        --kl_weight_scale) kl_weight_scale="$2"; shift 2 ;;
        --wandb) wandb_enable="$2"; shift 2 ;;
        --wandb_project) wandb_project="$2"; shift 2 ;;
        --wandb_entity) wandb_entity="$2"; shift 2 ;;
        --wandb_run_name) wandb_run_name="$2"; shift 2 ;;
        --wandb_tags) wandb_tags="$2"; shift 2 ;;
        --wandb_mode) wandb_mode="$2"; shift 2 ;;
        --compute_metrics) compute_metrics="$2"; shift 2 ;;
        --metric_interval) metric_interval="$2"; shift 2 ;;
        --metric_n_iter) metric_n_iter="$2"; shift 2 ;;
        --) shift; break ;;
        -*) echo "Unknown option: $1" >&2; exit 2 ;;
        *) echo "Unexpected arg: $1" >&2; exit 2 ;;
    esac
done

LOG_DIR_SHELL="${LOG_DIR:-${REPO_ROOT}/logs}"
IFS=' ' read -r -a latent_list <<<"${latent_dims_s}"

for latent_dim_s in "${latent_list[@]}"; do
    experience=${experience_override:-"batch${batch_size}_dim_s${latent_dim_s}"}

    if [[ "${experience_override}" != "" ]]; then
        experience="${experience_override}"
    else
        experience="loss=${loss_type}_batch${batch_size}_dim_s${latent_dim_s}_x_epoch=${max_epochs}_beta=(${beta_s},${beta_t})_latent=(${latent_dim_s},${dims_by_factors})_batch=${batch_size}_warm_up=${warm_up}_lr=${lr}_arch=${arch}+l_cov=${l_cov}+l_nce=${l_nce_by_factors}+l_anti_nce=${l_anti_nce}+kl_scale=${kl_weight_scale}_${key}"
    fi

    log_dir="${LOG_DIR_SHELL}/${version_model}/${dataset}/${experience}"
    mkdir -p "${log_dir}"

    # Pass the exact log directory (including version/experience) to Python
    export LOG_DIR="${log_dir}"

    python -m disdiff_adapters.arch.multi_distillme.train_x \
        --max_epochs "${max_epochs}" \
        --dataset "${dataset}" \
        --beta_s "${beta_s}" \
        --beta_t "${beta_t}" \
        --latent_dim_s "${latent_dim_s}" \
        --dims_by_factors "${dims_by_factors}" \
        --batch_size "${batch_size}" \
        --warm_up "${warm_up}" \
        --lr "${lr}" \
        --arch "${arch}" \
        --gpus "${gpus}" \
        --key "${key}" \
        --loss_type "${loss_type}" \
        --l_cov "${l_cov}" \
        --l_nce_by_factors "${l_nce_by_factors}" \
        --l_anti_nce "${l_anti_nce}" \
        --experience "${experience}" \
        --version_model "${version_model}" \
        --kl_weight_scale "${kl_weight_scale}" \
        --wandb "${wandb_enable}" \
        --wandb_project "${wandb_project}" \
        --wandb_entity "${wandb_entity}" \
        --wandb_run_name "${wandb_run_name}" \
        --wandb_tags "${wandb_tags}" \
        --wandb_mode "${wandb_mode}" \
        --compute_metrics "${compute_metrics}" \
        --metric_interval "${metric_interval}" \
        --metric_n_iter "${metric_n_iter}"

    exitcode=${PIPESTATUS[0]}
    echo "$(date '+%Y-%m-%d %H:%M:%S') - EXIT CODE = ${exitcode}" >> "${log_dir}/log.out"
done

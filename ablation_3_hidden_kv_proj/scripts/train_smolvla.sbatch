#!/usr/bin/env bash

# Cluster-specific resource flags are supplied when submitting the job.
# The variables below must point to the patched LeRobot checkout, persistent
# storage, and the Apptainer image on the target system.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

: "${LEROBOT_ROOT:?Set LEROBOT_ROOT to the patched LeRobot checkout}"
: "${STORAGE_ROOT:?Set STORAGE_ROOT to persistent storage}"
: "${IMAGE_PATH:?Set IMAGE_PATH to the Apptainer image}"

ARCH="${1:?Pass CA, SA, or CA-SA}"
ADAPTER="${2:?Pass true or false}"
ROPE="${3:?Pass true or false}"
VLM_INIT="${4:?Pass true or false}"
SEED="${5:-0}"

case "$ARCH" in
    CA)
        ATTENTION_MODE=cross_attn
        SELF_ATTN_EVERY=-1
        ;;
    SA)
        ATTENTION_MODE=self_attn
        SELF_ATTN_EVERY=-1
        ;;
    CA-SA)
        ATTENTION_MODE=cross_attn
        SELF_ATTN_EVERY=2
        ;;
    *)
        echo "Architecture must be CA, SA, or CA-SA" >&2
        exit 2
        ;;
esac

if [[ "$ADAPTER" != true && "$ADAPTER" != false ]]; then
    echo "Adapter must be true or false" >&2
    exit 2
fi

if [[ "$ARCH" == CA && "$ADAPTER" == true ]]; then
    echo "CA has no self-attention layers, so the KV adapter cannot be used." >&2
    exit 2
fi

if [[ "$ROPE" != true && "$ROPE" != false ]]; then
    echo "RoPE must be true or false" >&2
    exit 2
fi

if [[ "$VLM_INIT" != true && "$VLM_INIT" != false ]]; then
    echo "VLM init must be true or false" >&2
    exit 2
fi

DATASET_REVISION=a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4

ARCH_TAG="${ARCH,,}"
RUN_NAME="${ARCH_TAG}-saproj${ADAPTER}-rope${ROPE}-initvlm${VLM_INIT}-seed${SEED}"
RUN_DIR="/mnt/storage/runs/$RUN_NAME"

verify_image() {
    local checksum_file="${IMAGE_CHECKSUM_FILE:-$IMAGE_PATH.sha256}"

    if [[ ! -f "$checksum_file" ]]; then
        echo "image_checksum=not_checked"
        return
    fi

    local expected actual
    expected="$(awk 'NR == 1 {print $1}' "$checksum_file")"
    actual="$(sha256sum "$IMAGE_PATH" | awk '{print $1}')"

    if [[ "$actual" != "$expected" ]]; then
        echo "Container checksum mismatch" >&2
        exit 1
    fi

    echo "image_checksum=verified"
}

cd "$LEROBOT_ROOT"

echo "run=$RUN_NAME"
echo "source_commit=$(git rev-parse HEAD)"
echo "dataset_revision=$DATASET_REVISION"
verify_image

srun apptainer exec \
    --nv \
    --cleanenv \
    --bind "$LEROBOT_ROOT:/mnt/project/lerobot" \
    --bind "$STORAGE_ROOT:/mnt/storage" \
    --env "PYTHONPATH=/mnt/project/lerobot/src" \
    --env "PYTHONNOUSERSITE=1" \
    --env "PYTHONUNBUFFERED=1" \
    --env "HF_HOME=/mnt/storage/cache/huggingface" \
    --env "HF_LEROBOT_HOME=/mnt/storage/cache/huggingface/lerobot" \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
    "$IMAGE_PATH" \
    /lerobot/.venv/bin/lerobot-train \
        --policy.type=smolvla \
        --policy.device=cuda \
        --policy.load_vlm_weights=true \
        --policy.train_expert_only=true \
        --policy.expert_width_multiplier=0.75 \
        --policy.num_vlm_layers=16 \
        --policy.chunk_size=50 \
        --policy.n_action_steps=1 \
        --policy.num_steps=10 \
        --policy.attention_mode="$ATTENTION_MODE" \
        --policy.self_attn_every_n_layers="$SELF_ATTN_EVERY" \
        --policy.self_attn_kv_adapter="$ADAPTER" \
        --policy.expert_prefix_rope="$ROPE" \
        --policy.init_hidden_proj_with_vlm_weights="$VLM_INIT" \
        --policy.optimizer_lr=1e-4 \
        --policy.scheduler_warmup_steps=100 \
        --policy.scheduler_decay_steps=100000 \
        --policy.scheduler_decay_lr=2.5e-6 \
        --policy.push_to_hub=false \
        --accelerator.mixed_precision=bf16 \
        --dataset.repo_id=lerobot/libero \
        --dataset.revision="$DATASET_REVISION" \
        --dataset.video_backend=torchcodec \
        --steps=100000 \
        --batch_size=64 \
        --seed="$SEED" \
        --env_eval_freq=0 \
        --save_freq=20000 \
        --output_dir="$RUN_DIR" \
        --job_name="$RUN_NAME"

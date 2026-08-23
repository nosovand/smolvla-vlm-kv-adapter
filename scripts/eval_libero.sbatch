#!/usr/bin/env bash

# Cluster-specific resource flags are supplied when submitting the job.
# The variables below must point to the patched LeRobot checkout, persistent
# storage, and the Apptainer image on the target system.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

: "${LEROBOT_ROOT:?Set LEROBOT_ROOT to the patched LeRobot checkout}"
: "${STORAGE_ROOT:?Set STORAGE_ROOT to persistent storage}"
: "${IMAGE_PATH:?Set IMAGE_PATH to the Apptainer image}"

RUN_NAME="${1:?Pass the training run directory name}"
CHECKPOINT_STEP="${2:?Pass the numeric checkpoint step}"
EVAL_SEED="${3:-2000}"

CHECKPOINT_HOST="$STORAGE_ROOT/runs/$RUN_NAME/checkpoints/$CHECKPOINT_STEP/pretrained_model"
CHECKPOINT_CONTAINER="/mnt/storage/runs/$RUN_NAME/checkpoints/$CHECKPOINT_STEP/pretrained_model"

OUTPUT_HOST="$STORAGE_ROOT/evaluations/$RUN_NAME/checkpoint-$CHECKPOINT_STEP/eval-seed-$EVAL_SEED"
OUTPUT_CONTAINER="/mnt/storage/evaluations/$RUN_NAME/checkpoint-$CHECKPOINT_STEP/eval-seed-$EVAL_SEED"

LIBERO_ROOT="$STORAGE_ROOT/libero"
LIBERO_ASSETS="$LIBERO_ROOT/assets"
LIBERO_CONFIG_DIR="${LIBERO_CONFIG_DIR:-$HOME/.libero}"
ASSETS_REVISION_FILE="$LIBERO_ROOT/assets-revision.txt"
ASSETS_READY="$LIBERO_ROOT/assets-ready"

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

echo "Starting LIBERO evaluation"
echo "run_name=$RUN_NAME"
echo "checkpoint_step=$CHECKPOINT_STEP"
echo "evaluation_seed=$EVAL_SEED"
echo "checkpoint=$CHECKPOINT_CONTAINER"
echo "output=$OUTPUT_CONTAINER"

if [[ ! -f "$CHECKPOINT_HOST/config.json" ]]; then
    echo "Missing checkpoint config" >&2
    exit 1
fi

if [[ ! -f "$CHECKPOINT_HOST/model.safetensors" ]]; then
    echo "Missing checkpoint weights" >&2
    exit 1
fi

if [[ -e "$OUTPUT_HOST" ]]; then
    echo "Evaluation output already exists" >&2
    echo "Choose another evaluation seed or move the previous output." >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_HOST")"
mkdir -p "$LIBERO_ROOT" "$LIBERO_ASSETS" "$LIBERO_CONFIG_DIR"

verify_image

echo "Preparing LIBERO assets"

if [[ ! -s "$ASSETS_REVISION_FILE" ]]; then
    apptainer exec \
        --cleanenv \
        --bind "$STORAGE_ROOT:/mnt/storage" \
        --env "HF_HOME=/mnt/storage/cache/huggingface" \
        "$IMAGE_PATH" \
        /lerobot/.venv/bin/python -c \
        'from huggingface_hub import HfApi; print(HfApi().dataset_info("lerobot/libero-assets").sha)' \
        > "$ASSETS_REVISION_FILE"
fi

ASSETS_REVISION="$(<"$ASSETS_REVISION_FILE")"
echo "libero_assets_revision=$ASSETS_REVISION"

if [[ ! -f "$ASSETS_READY" ]]; then
    echo "Downloading LIBERO assets"

    apptainer exec \
        --cleanenv \
        --bind "$STORAGE_ROOT:/mnt/storage" \
        --env "HF_HOME=/mnt/storage/cache/huggingface" \
        --env "ASSETS_REVISION=$ASSETS_REVISION" \
        "$IMAGE_PATH" \
        /lerobot/.venv/bin/python -c \
        'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id="lerobot/libero-assets", repo_type="dataset", revision=os.environ["ASSETS_REVISION"], local_dir="/mnt/storage/libero/assets")'

    touch "$ASSETS_READY"
else
    echo "Reusing existing LIBERO assets"
fi

printf '%s\n' \
    "assets: /mnt/storage/libero/assets" \
    "bddl_files: /lerobot/.venv/lib/python3.12/site-packages/libero/libero/bddl_files" \
    "datasets: /lerobot/.venv/lib/python3.12/site-packages/libero/datasets" \
    "init_states: /lerobot/.venv/lib/python3.12/site-packages/libero/libero/init_files" \
    > "$LIBERO_CONFIG_DIR/config.yaml"

cd "$LEROBOT_ROOT"
echo "source_commit=$(git rev-parse HEAD)"

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
    --env "MUJOCO_GL=egl" \
    --env "PYOPENGL_PLATFORM=egl" \
    --env "TOKENIZERS_PARALLELISM=false" \
    "$IMAGE_PATH" \
    /lerobot/.venv/bin/lerobot-eval \
        --policy.path="$CHECKPOINT_CONTAINER" \
        --policy.device=cuda \
        --policy.num_steps=10 \
        --policy.n_action_steps=10 \
        --env.type=libero \
        --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
        --env.control_mode=relative \
        --env.init_states=true \
        --env.hard_reset=true \
        --env.max_parallel_tasks=1 \
        --eval.n_episodes=50 \
        --eval.batch_size=20 \
        --eval.use_async_envs=true \
        --eval.recording=false \
        --seed="$EVAL_SEED" \
        --output_dir="$OUTPUT_CONTAINER" \
        --job_name="${RUN_NAME}-step${CHECKPOINT_STEP}-eval${EVAL_SEED}"

echo "Evaluation complete"
echo "Metrics: $OUTPUT_CONTAINER/eval_info.json"

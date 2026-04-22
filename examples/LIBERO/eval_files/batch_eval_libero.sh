#!/bin/bash

# ============================================================
# Batch Evaluation for LIBERO - 4 task suites sequentially on 1 GPU
# ============================================================

###########################################################################################
# === Please modify the following paths according to your environment ===

MODEL_PYTHON=/path/to/python
LIBERO_PYTHON=/path/to/python

export LIBERO_HOME=/path/to/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME}
export PYTHONPATH=$(pwd):${PYTHONPATH}

CKPT_DEFAULT="/path/to/checkpoint/pytorch_model.pt"

# === End of environment variable configuration ===
###########################################################################################

PORT=5694
NUM_TRIALS_PER_TASK=50
GPU_DEFAULT="0"

# Parse command-line arguments
CKPT_PATH=${1:-$CKPT_DEFAULT}
GPU_ID=${2:-$GPU_DEFAULT}

folder_name=$(echo "$CKPT_PATH" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

echo "=== LIBERO Sequential Evaluation Configuration ==="
echo "Checkpoint Path      : ${CKPT_PATH}"
echo "GPU ID               : ${GPU_ID}"
echo "Port                 : ${PORT}"
echo "Num Trials Per Task  : ${NUM_TRIALS_PER_TASK}"
echo "==================================================="

# ============================================================
# Task Suite List
# ============================================================

TASK_SUITES=(
    libero_spatial
    libero_object
    libero_goal
    libero_10
)

# ============================================================
# Setup Log Directory
# ============================================================

# LOG_DIR="logs/libero_batch_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${CKPT_PATH}.log/eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "=== Launching Single-GPU Sequential Evaluation ==="
echo "Log Directory   : ${LOG_DIR}"

# ============================================================
# Step 1: Launch Policy Server on the single GPU
# ============================================================

echo "Starting policy server | GPU ${GPU_ID} | Port ${PORT}"

CUDA_VISIBLE_DEVICES=${GPU_ID} \
${MODEL_PYTHON} deployment/model_server/server_policy.py \
    --ckpt_path "${CKPT_PATH}" \
    --port "${PORT}" \
    > "${LOG_DIR}/server_gpu${GPU_ID}_port${PORT}.log" 2>&1 &

SERVER_PID=$!

echo "Waiting 40s for server to initialize..."
sleep 40

# ============================================================
# Step 2: Run Evaluations Sequentially (one task suite at a time)
# ============================================================

for i in $(seq 0 3); do
    TASK_SUITE=${TASK_SUITES[$i]}
    VIDEO_OUT="results/${TASK_SUITE}/${folder_name}"

    echo ""
    echo "=== [$((i+1))/4] Evaluating: ${TASK_SUITE} ==="

    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    ${LIBERO_PYTHON} ./examples/LIBERO/eval_files/eval_libero.py \
        --args.pretrained-path "${CKPT_PATH}" \
        --args.host "127.0.0.1" \
        --args.port "${PORT}" \
        --args.task-suite-name "${TASK_SUITE}" \
        --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
        --args.video-out-path "${VIDEO_OUT}" \
        2>&1 | tee "${LOG_DIR}/eval_${TASK_SUITE}_gpu${GPU_ID}.log"

    echo "=== [$((i+1))/4] Finished: ${TASK_SUITE} ==="
done

echo ""
echo "--- All evaluations done ---"

# ============================================================
# Step 3: Cleanup - Kill Policy Server
# ============================================================

echo ""
echo "Shutting down policy server..."
kill "${SERVER_PID}" 2>/dev/null && echo "Killed server PID ${SERVER_PID}"

echo "=== LIBERO Sequential Evaluation Finished ==="

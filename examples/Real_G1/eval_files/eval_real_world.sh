#!/bin/bash

# Evaluation script for DiT4DiT models trained with run_real.sh
# Usage: bash eval_real.sh

export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0

# ============== Configuration ==============
# Dataset configuration
data_root=/path/to/dataset/real_robot
dataset_name=stack_cups
# pnp_eggplant_lh_200ep_26_1_28
# pnp_spoon


# Evaluation parameters
action_horizon=16
num_trajs=1           # Number of trajectories to evaluate
start_traj=0          # Starting trajectory index
steps_per_traj=300    # Max steps per trajectory

# Output configuration
output_dir=./eval_plot
mkdir -p ${output_dir}



# ============== Run Evaluation ==============
python utils/eval_policy.py \
    --model_path /path/to/checkpoint/pytorch_model.pt \
    --dataset_path "${data_root}/${dataset_name}" \
    --data_config g1_body29_aloha_arms_only \
    --embodiment_tag new_embodiment \
    --action_horizon ${action_horizon} \
    --trajs ${num_trajs} \
    --start_traj ${start_traj} \
    --steps ${steps_per_traj} \
    --save_plot_path ${output_dir}/stack_cups_video_pretrained_id.png \
    --modality_keys left_arm right_arm left_gripper right_gripper \
    --video_backend decord \
    --lerobot_version v2.0

export WANDB_API_KEY=your_wandb_key
export PYTHONPATH=$(pwd)

Framework_name=DiT4DiT
base_model=/path/to/Cosmos-Predict2.5-2B
freeze_module_list="backbone_interface.extractor.text_encoder,backbone_interface.extractor.vae"
DIT_TYPE="DiT-B"
data_root_dir=/path/to/dataset/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
data_mix=fourier_gr1_unified_1000


run_root_dir=./playground/Checkpoints_robocasa
run_id=dit4dit_robocasa_gr1

export WANDB_MODE=online

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

accelerate launch \
  --config_file DiT4DiT/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 16 \
  DiT4DiT/training/train.py \
  --config_yaml ./DiT4DiT/config/robocasa/dit4dit_robocasa_gr1.yaml \
  --framework.name ${Framework_name} \
  --framework.cosmos25.base_model ${base_model} \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 4 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 200000 \
  --trainer.save_interval 50000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100 \
  --trainer.learning_rate.backbone_interface 1e-5 \
  --trainer.learning_rate.action_model 1e-4 \
  --trainer.num_warmup_steps 5000 \
  --framework.cosmos25.extract_layer 17 \
  --framework.cosmos25.flow_matching.time_distribution uniform \
  --framework.cosmos25.flow_matching.high_sigma_ratio null \
  --framework.cosmos25.flow_matching.high_sigma_min null \
  --trainer.framework.cosmos25.conditional_frame_timestep 0.0001 \
  --datasets.vla_data.action_video_freq_ratio 2 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project DiT4DiT_robocasa \
  --wandb_entity your_wandb_entity \




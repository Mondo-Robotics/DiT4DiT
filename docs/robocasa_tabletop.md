# RoboCasa-GR1 Tabletop: Training & Evaluation

This guide covers training and evaluation for DiT4DiT on the RoboCasa tabletop simulation benchmark.

## Prepare Dataset

Download the GR00T-X simulation dataset from Hugging Face:

```bash
python examples/Robocasa_tabletop/train_files/download_gr00t_ft_data.py
```

This downloads 24 task folders from `nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim` to `./playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/`.

## Environment Setup
Please first follow the [official RoboCasa installation guide](https://github.com/robocasa/robocasa-gr1-tabletop-tasks?tab=readme-ov-file#getting-started) to install the base `robocasa-gr1-tabletop-tasks` environment.  

Then pip soceket support

```bash
pip install tyro
```

## Configure Training

The training config is defined in `DiT4DiT/config/robocasa/dit4dit_robocasa_gr1.yaml`. Key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `framework.cosmos25.base_model` | Path to Cosmos-Predict2.5-2B | - |
| `framework.action_model.action_model_type` | DiT variant | `DiT-B` |
| `datasets.vla_data.data_root_dir` | Dataset root directory | - |
| `datasets.vla_data.data_mix` | Dataset mixture name | `fourier_gr1_unified_1000` |
| `datasets.vla_data.per_device_batch_size` | Batch size per GPU | `4` |
| `num_processes` | Num of GPUs | `16` |
| `trainer.max_train_steps` | Total training steps | `200000` |
| `trainer.learning_rate.backbone_interface` | Video DiT learning rate | `1e-5` |
| `trainer.learning_rate.action_model` | Action DiT learning rate | `1e-4` |
| `trainer.freeze_modules` | Modules to freeze | `"backbone_interface.extractor.text_encoder,backbone_interface.extractor.vae"` |

## Launch Training


```bash
bash examples/Robocasa_tabletop/train_files/run_robocasa.sh
```

Checkpoints will be saved to `{run_root_dir}/{run_id}/`. Training supports:
- DeepSpeed ZeRO Stage 2/3
- Gradient checkpointing
- Mixed precision (bf16)
- Wandb logging
- Resume from checkpoint

## Inference


### Option A: Single Evaluation

**Step 1: Start the policy server**

```bash
CUDA_VISIBLE_DEVICES=0 python deployment/model_server/server_policy.py \
  --ckpt_path /path/to/checkpoint.pt \
  --port 6398 \
  --use_bf16
```

**Step 2: Run evaluation against the server**

```bash
python examples/Robocasa_tabletop/eval_files/simulation_env.py \
  --args.env_name "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env" \
  --args.port 6398 \
  --args.pretrained_path /path/to/checkpoint.pt \
  --args.n_episodes 50
```

### Option B: Batch Evaluation (Multi-GPU, recommended)

Run all 24 evaluation environments across multiple GPUs:

```bash
bash examples/Robocasa_tabletop/eval_files/batch_eval_args.sh \
  /path/to/checkpoint.pt \   # Checkpoint path
  1 \                         # Number of parallel envs
  720 \                       # Max episode steps
  12 \                        # Action chunk length
  "0,1,2,3"                   # GPU IDs
```

This script automatically:
1. Launches a policy server on each GPU
2. Distributes environments across GPUs
3. Runs 50 episodes per environment
4. Saves videos and logs


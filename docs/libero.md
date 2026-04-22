# LIBERO: Training & Evaluation

This guide covers training and evaluation for DiT4DiT on the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) benchmark.

## Prepare Dataset

Download the following LIBERO datasets in LeRobot format to your data directory:

- [LIBERO-spatial](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot)
- [LIBERO-object](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot)
- [LIBERO-goal](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot)
- [LIBERO-10](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot)

## Environment Setup

Please first follow the [official LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO) to install the base LIBERO environment.

Then install additional dependencies inside the LIBERO environment:

```bash
pip install tyro imageio matplotlib websockets msgpack
```

## Launch Training

**Single-node:**

```bash
bash examples/LIBERO/train_files/run_libero.sh
```

**Multi-node (SLURM):**

```bash
sbatch examples/LIBERO/train_files/submit_libero_training.sh
```

> **Note:** Adjust `#SBATCH -N` (number of nodes) and `#SBATCH --gres=gpu:` (GPUs per node) in the script to control total GPU count. The total number of processes is computed automatically. Please ensure that you specify the correct paths in the script.

Checkpoints will be saved to `{run_root_dir}/{run_id}/`. Training supports:
- DeepSpeed ZeRO Stage 2/3
- Gradient checkpointing
- Mixed precision (bf16)
- Wandb logging
- Resume from checkpoint

## Inference

The evaluation runs from the **repository root** using **two separate conda environments**:

- **DiT4DiT environment**: runs the inference server.
- **LIBERO environment**: runs the simulation.

### Download Pretrained Checkpoint

You can download our pretrained DiT4DiT-LIBERO checkpoint from Hugging Face to directly run evaluation:

```bash
huggingface-cli download mondo-robotics/dit4dit-model --include "dit4dit_libero/*" --local-dir /path/to/dit4dit-model
```

See the [Model Zoo](../README.md#model-zoo) for all available checkpoints.

> **Note:** After downloading, remember to update **line 46** of `config.yaml` in the checkpoint directory to point to your local Cosmos-Predict2.5-2B path.


### Evaluation

Run all 4 LIBERO task suites sequentially on a single GPU:

```bash
bash examples/LIBERO/eval_files/batch_eval_libero.sh \
  /path/to/checkpoint.pt \   # Checkpoint path
  0                           # GPU ID
```

> **Note:** Please ensure that `MODEL_PYTHON` and `LIBERO_PYTHON` in `batch_eval_libero.sh` point to the correct Python executables in your DiT4DiT and LIBERO conda environments respectively.

> **Note:** We evaluated and reported results on NVIDIA RTX 5880 GPUs.

This script automatically:
1. Launches a policy server on the specified GPU
2. Evaluates all 4 task suites sequentially (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`)
3. Runs 50 episodes per task
4. Saves videos and logs

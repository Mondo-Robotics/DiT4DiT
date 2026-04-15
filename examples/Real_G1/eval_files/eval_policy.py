#### NOTE !!!!!!!!!!!!!!####
# state[-1] = 0.0
# state[-2] = 0.0
# 这两行操作时针对 bug 临时加的


import warnings
from dataclasses import dataclass, field
from typing import List, Literal
from pathlib import Path

import numpy as np
import tyro
import torch
import matplotlib
import matplotlib.pyplot as plt

warnings.simplefilter("ignore", category=FutureWarning)

# numpy print precision settings
np.set_printoptions(precision=3, suppress=True)


@dataclass
class ArgsConfig:
    """Configuration for evaluating a DiT4DiT policy."""

    model_path: str = None
    """Path to the model checkpoint (.pt file)."""

    dataset_path: str = "demo_data/robot_sim.PickNPlace/"
    """Path to the dataset."""

    data_config: str = "g1_body29_aloha_arms_only"
    """
    Data config to use. Available options:
    - g1_body29_aloha_arms_only
    - oxe_bridge
    - oxe_rt1
    - libero_franka
    - fourier_gr1_arms_waist
    See DiT4DiT/dataloader/gr00t_lerobot/data_config.py for more.
    """

    embodiment_tag: str = "new_embodiment"
    """Embodiment tag to use for the dataset."""

    unnorm_key: str = None
    """Key for unnormalization statistics. If None, uses the first key in norm_stats."""

    steps: int = 150
    """Number of steps to evaluate per trajectory."""

    trajs: int = 1
    """Number of trajectories to evaluate."""

    start_traj: int = 0
    """Start trajectory index to evaluate."""

    action_horizon: int = 16
    """Action horizon to evaluate."""

    save_plot_path: str = None
    """Path to save the plot. If None, displays the plot."""

    plot: bool = False
    """Whether to display the plot interactively."""

    modality_keys: List[str] = field(
        default_factory=lambda: [
            "left_arm",
            "right_arm",
            # "left_gripper",
            # "right_gripper",
        ]
    )
    """Modality keys for action concatenation in plotting."""

    device: str = "cuda"
    """Device to run inference on."""

    video_backend: str = "torchcodec"
    """Video backend to use: decord, torchvision_av, or torchcodec."""

    lerobot_version: str = "v2.0"
    """LeRobot dataset version: v2.0 or v3.0."""


def load_model(model_path: str, device: str = "cuda"):
    """Load the DiT4DiT model from checkpoint."""
    from DiT4DiT.model.framework.base_framework import baseframework

    print(f"Loading model from {model_path}")
    model = baseframework.from_pretrained(model_path)
    model = model.to(device).eval()
    return model


def load_dataset(args: ArgsConfig):
    """Load the evaluation dataset."""
    from DiT4DiT.dataloader.gr00t_lerobot.datasets import (
        LeRobotSingleDataset,
        ModalityConfig,
    )
    from DiT4DiT.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP

    # Get data config
    if args.data_config in ROBOT_TYPE_CONFIG_MAP:
        data_config = ROBOT_TYPE_CONFIG_MAP[args.data_config]
    else:
        raise ValueError(
            f"Unknown data config: {args.data_config}. "
            f"Available: {list(ROBOT_TYPE_CONFIG_MAP.keys())}"
        )

    # Build modality configs
    modality_configs = data_config.modality_config()

    # Create data_cfg dict for dataset
    data_cfg = {
        "lerobot_version": args.lerobot_version,
    }

    # Create dataset
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality_configs,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,
        embodiment_tag=args.embodiment_tag,
        data_cfg=data_cfg,
    )

    return dataset, data_config


def get_action_from_model(
    model, obs_data: dict, state_norm_stats: dict, device: str = "cuda"
) -> np.ndarray:
    """
    Get action prediction from the model.

    Args:
        model: The DiT4DiT model
        obs_data: Dictionary containing observation data from dataset
        device: Device to run inference on

    Returns:
        np.ndarray: Predicted actions shape (action_horizon, action_dim)
    """
    from PIL import Image

    # Extract images from observation data
    images = []
    video_keys = [k for k in obs_data.keys() if k.startswith("video.")]
    for video_key in video_keys:
        # obs_data[video_key] has shape (T, H, W, C), we take the first frame
        img_array = obs_data[video_key][0]
        img = Image.fromarray(img_array).resize((224, 224))
        images.append(img)

    # Get language instruction
    lang_keys = [k for k in obs_data.keys() if k.startswith("annotation.")]
    if lang_keys:
        instruction = obs_data[lang_keys[0]][0]
    else:
        instruction = ""

    # Get state if available
    # Note: The model expects state_dim=64 (from config), so we need to pad
    state_keys = [k for k in obs_data.keys() if k.startswith("state.")]
    # state = None
    if state_keys:
        state_list = []
        for state_key in state_keys:
            state_list.append(obs_data[state_key][0])
        # # sin, cos 归一化
        #     sin_state = np.sin(obs_data[state_key][0])
        #     cos_state = np.cos(obs_data[state_key][0])
        #     state_list.append(sin_state)
        #     state_list.append(cos_state)
        state = np.concatenate(state_list, axis=-1)

        # Using "min" "max" to normalize state
        min_state = np.array(state_norm_stats["min"])
        max_state = np.array(state_norm_stats["max"])
        # Normalize to [-1, 1]
        state = (state - min_state) / (max_state - min_state) * 2 - 1

        # Set gripper states to 0 (left_gripper, right_gripper are constant 0 in collected data)
        state[-1] = 0.0
        state[-2] = 0.0

        state = state.reshape(1, -1)  # (1, state_dim)
        # Pad state to 64 dimensions (model's expected state_dim)
        max_state_dim = 64
        if state.shape[1] < max_state_dim:
            padding = np.zeros((1, max_state_dim - state.shape[1]), dtype=state.dtype)
            state = np.concatenate([state, padding], axis=-1)

    # Build input example
    example = {
        "image": images,
        "lang": instruction,
    }
    if state is not None:
        example["state"] = state
    # Run inference
    with torch.no_grad():
        output = model.predict_action([example])

    normalized_actions = output["normalized_actions"][0]  # (action_horizon, action_dim)
    return normalized_actions


def unnormalize_actions(
    normalized_actions: np.ndarray, action_norm_stats: dict
) -> np.ndarray:
    """
    Unnormalize actions using the dataset statistics.

    Args:
        normalized_actions: Normalized actions shape (T, model_action_dim)
        action_norm_stats: Dictionary with q01, q99, mask keys (real_action_dim)

    Returns:
        np.ndarray: Unnormalized actions (T, real_action_dim)
    """
    action_high = np.array(action_norm_stats["max"])
    action_low = np.array(action_norm_stats["min"])
    real_action_dim = len(action_high)

    # Model outputs padded action_dim (e.g., 32), but real actions are smaller (e.g., 16)
    # Only take the first real_action_dim dimensions
    normalized_actions = normalized_actions[:, :real_action_dim]
    normalized_actions = np.clip(normalized_actions, -1, 1)

    mask = action_norm_stats.get("mask", np.ones(real_action_dim, dtype=bool))

    # Unnormalize: map [-1, 1] to [action_low, action_high]
    # actions = np.where(
    #     mask,
    #     0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
    #     normalized_actions,
    # )
    actions = 0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low

    return actions


def calc_mse_for_single_trajectory(
    model,
    dataset,
    traj_id: int,
    action_norm_stats: dict,
    state_norm_stats: dict,
    modality_keys: list,
    steps: int = 300,
    action_horizon: int = 16,
    plot: bool = False,
    save_plot_path: str = None,
    device: str = "cuda",
):
    """
    Calculate MSE for a single trajectory.

    Args:
        model: The DiT4DiT model
        dataset: The evaluation dataset
        traj_id: Trajectory ID to evaluate
        action_norm_stats: Action normalization statistics
        modality_keys: List of modality keys for action concatenation
        steps: Number of steps to evaluate
        action_horizon: Action horizon
        plot: Whether to display plot
        save_plot_path: Path to save plot
        device: Device for inference

    Returns:
        float: MSE value
    """
    gt_action_across_time = []
    pred_action_across_time = []

    # Get trajectory length
    traj_idx = np.where(dataset.trajectory_ids == traj_id)[0][0]
    traj_length = dataset.trajectory_lengths[traj_idx]
    actual_steps = min(steps, traj_length)

    for step_count in range(actual_steps):
        if step_count % action_horizon == 0:
            # Get observation data
            data_point = dataset.get_step_data(traj_id, step_count)

            print(f"Inferencing at step: {step_count}")

            # Get predicted actions (normalized)
            pred_normalized = get_action_from_model(
                model, data_point, state_norm_stats, device
            )
            # Unnormalize predicted actions
            pred_actions = unnormalize_actions(pred_normalized, action_norm_stats)
            # pred_actions = pred_normalized

            # Get ground truth actions
            for j in range(min(action_horizon, actual_steps - step_count)):
                # Get GT action for this step
                gt_data = dataset.get_step_data(traj_id, step_count + j)
                gt_action_list = []
                for action_key in dataset.modality_keys.get("action", []):
                    key_name = action_key.replace("action.", "")
                    if key_name in modality_keys:
                        gt_action_list.append(gt_data[action_key][0])  # First timestep
                gt_action = np.concatenate(gt_action_list, axis=-1)
                gt_action_across_time.append(gt_action)

                # Predicted action for this step
                pred_action_across_time.append(pred_actions[j])

    # Convert to arrays
    gt_action_across_time = np.array(gt_action_across_time)
    pred_action_across_time = np.array(pred_action_across_time)

    # Ensure shapes match
    min_len = min(len(gt_action_across_time), len(pred_action_across_time))
    gt_action_across_time = gt_action_across_time[:min_len]
    pred_action_across_time = pred_action_across_time[:min_len]

    # Handle dimension mismatch
    if gt_action_across_time.shape[1] != pred_action_across_time.shape[1]:
        min_dim = min(gt_action_across_time.shape[1], pred_action_across_time.shape[1])
        print(
            f"Warning: Dimension mismatch. GT: {gt_action_across_time.shape[1]}, "
            f"Pred: {pred_action_across_time.shape[1]}. Using first {min_dim} dimensions."
        )
        gt_action_across_time = gt_action_across_time[:, :min_dim]
        pred_action_across_time = pred_action_across_time[:, :min_dim]

    # Calculate MSE
    mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    print(f"Unnormalized Action MSE across single traj: {mse}")

    print(f"gt_action_joints vs time: {gt_action_across_time.shape}")
    print(f"pred_action_joints vs time: {pred_action_across_time.shape}")

    # Check for NaN
    if np.isnan(pred_action_across_time).any():
        raise ValueError("Pred action has NaN")

    # Plot if requested
    if plot or save_plot_path is not None:
        info = {
            "gt_action_across_time": gt_action_across_time,
            "pred_action_across_time": pred_action_across_time,
            "modality_keys": modality_keys,
            "traj_id": traj_id,
            "mse": mse,
            "action_dim": gt_action_across_time.shape[1],
            "action_horizon": action_horizon,
            "steps": min_len,
        }
        plot_trajectory(info, save_plot_path)

    return mse


def plot_trajectory(info: dict, save_plot_path: str = None):
    """Plot the trajectory comparison between GT and predicted actions."""

    if save_plot_path is not None:
        matplotlib.use("Agg")

    action_dim = info["action_dim"]
    gt_action_across_time = info["gt_action_across_time"]
    pred_action_across_time = info["pred_action_across_time"]
    modality_keys = info["modality_keys"]
    traj_id = info["traj_id"]
    mse = info["mse"]
    action_horizon = info["action_horizon"]
    steps = info["steps"]

    # Create figure
    fig, axes = plt.subplots(
        nrows=action_dim, ncols=1, figsize=(10, 4 * action_dim + 2)
    )

    if action_dim == 1:
        axes = [axes]

    plt.subplots_adjust(top=0.92, left=0.1, right=0.96, hspace=0.4)

    print("Creating visualization...")

    # Title
    modality_string = ", ".join(modality_keys[:4])
    if len(modality_keys) > 4:
        modality_string += "..."
    title_text = f"Trajectory Analysis - ID: {traj_id}\nModalities: {modality_string}\nUnnormalized MSE: {mse:.6f}"
    fig.suptitle(title_text, fontsize=14, fontweight="bold", color="#2E86AB", y=0.98)

    # Plot each action dimension
    for i, ax in enumerate(axes):
        ax.plot(gt_action_across_time[:, i], label="GT action", linewidth=2)
        ax.plot(pred_action_across_time[:, i], label="Pred action", linewidth=2)

        # Mark inference points
        for j in range(0, steps, action_horizon):
            if j == 0:
                ax.plot(
                    j,
                    gt_action_across_time[j, i],
                    "ro",
                    label="Inference point",
                    markersize=6,
                )
            else:
                ax.plot(j, gt_action_across_time[j, i], "ro", markersize=4)

        ax.set_title(f"Action Dimension {i}", fontsize=12, fontweight="bold", pad=10)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time Step", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)

    if save_plot_path:
        print(f"Saving plot to {save_plot_path}")
        plt.savefig(save_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def main(args: ArgsConfig):
    """Main evaluation function."""

    if args.model_path is None:
        raise ValueError("Please provide --model_path")

    # Load model
    model = load_model(args.model_path, args.device)

    # Get action normalization stats
    # Note: get_action_stats is incorrectly decorated as @classmethod in base_framework.py
    # So we access norm_stats directly from the instance
    norm_stats = model.norm_stats
    unnorm_key = args.unnorm_key
    if unnorm_key is None:
        assert len(norm_stats) == 1, (
            f"Your model was trained on more than one dataset, "
            f"please pass --unnorm_key from: {list(norm_stats.keys())}"
        )
        unnorm_key = next(iter(norm_stats.keys()))
    action_norm_stats = norm_stats[unnorm_key]["action"]
    state_norm_stats = norm_stats[unnorm_key]["state"]

    # Load dataset
    dataset, data_config = load_dataset(args)
    # transforms = data_config.transform()

    print(f"Dataset: {dataset.dataset_name}")
    print(f"Total trajectories: {len(dataset.trajectory_lengths)}")
    print(f"Trajectory lengths: {dataset.trajectory_lengths[:10]}...")
    print(f"Running evaluation with modality keys: {args.modality_keys}")

    # Evaluate trajectories
    all_mse = []
    for traj_id in range(args.start_traj, args.start_traj + args.trajs):
        print(f"\n{'='*50}")
        print(f"Running trajectory: {traj_id}")

        # Determine save path for this trajectory
        if args.save_plot_path:
            base_path = Path(args.save_plot_path)
            if args.trajs > 1:
                save_path = str(
                    base_path.parent
                    / f"{base_path.stem}_traj{traj_id}{base_path.suffix}"
                )
            else:
                save_path = args.save_plot_path
        else:
            save_path = None

        mse = calc_mse_for_single_trajectory(
            model=model,
            dataset=dataset,
            traj_id=dataset.trajectory_ids[traj_id],
            action_norm_stats=action_norm_stats,
            state_norm_stats=state_norm_stats,
            modality_keys=args.modality_keys,
            steps=args.steps,
            action_horizon=args.action_horizon,
            plot=args.plot,
            save_plot_path=save_path,
            device=args.device,
        )

        print(f"Trajectory {traj_id} MSE: {mse}")
        all_mse.append(mse)

    print(f"\n{'='*50}")
    print(f"Average MSE across {len(all_mse)} trajectories: {np.mean(all_mse):.6f}")
    print(f"Std MSE: {np.std(all_mse):.6f}")
    print("Done")


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)

# ZMQ version adapted for DiT4DiT gr00t_wbc integration.

import argparse
import logging
import os
import socket

import numpy as np
import torch

from deployment.model_server.tools.zmq_policy_server import ZmqPolicyServer
from DiT4DiT.model.framework.base_framework import baseframework
from DiT4DiT.model.framework.share_tools import read_mode_config
from DiT4DiT.model.tools import get_default_device


class UnnormalizedPolicyWrapper:
    """Wraps a policy to add action unnormalization as a post-processing step."""

    def __init__(self, policy, action_norm_stats, state_norm_stats):
        self.policy = policy
        self.action_norm_stats = action_norm_stats
        self.state_norm_stats = state_norm_stats

    def normalize_state(self, state, max_state_dim=64):
        # Apply SinCosTransform: concat([sin(state), cos(state)])
        sin_val = np.sin(state)
        cos_val = np.cos(state)
        state = np.concatenate([sin_val, cos_val], axis=-1)

        if state.shape[1] < max_state_dim:
            padding = np.zeros((1, max_state_dim - state.shape[1]), dtype=state.dtype)
            state = np.concatenate([state, padding], axis=-1)

        return state

    def predict_action(self, **kwargs):
        # Normalize and pad state to expected dimension (64)
        if "examples" in kwargs and kwargs["examples"]:
            for ex in kwargs["examples"]:
                if "state" in ex and ex["state"] is not None:
                    ex["state"] = self.normalize_state(ex["state"])

        output = self.policy.predict_action(**kwargs)
        normalized_actions = output["normalized_actions"][0]  # [T, D]
        normalized_actions = normalized_actions[
            : self.policy.action_model.action_horizon, :23
        ]
        unnormalized_actions = baseframework.unnormalize_actions(
            normalized_actions, self.action_norm_stats
        )
        output["unnormalized_actions"] = unnormalized_actions
        return output


def main(args) -> None:
    """Start ZMQ policy server with the specified model checkpoint."""
    # Load model
    logging.info(f"Loading model from {args.ckpt_path}")
    vla = baseframework.from_pretrained(args.ckpt_path)

    if args.use_bf16:
        logging.info("Converting model to bfloat16")
        vla = vla.to(torch.bfloat16)

    device = torch.device(args.device) if args.device else get_default_device()
    vla = vla.to(device).eval()
    logging.info("Model loaded and ready")

    # Load dataset statistics for action unnormalization
    _, norm_stats = read_mode_config(args.ckpt_path)
    unnorm_key = baseframework._check_unnorm_key(norm_stats, args.unnorm_key)
    action_norm_stats = norm_stats[unnorm_key]["action"]
    state_norm_stats = norm_stats[unnorm_key]["state"]
    logging.info(f"Loaded action norm stats (unnorm_key={unnorm_key})")

    # Wrap policy with unnormalization
    wrapped_policy = UnnormalizedPolicyWrapper(vla, action_norm_stats, state_norm_stats)

    # Get hostname info
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        local_ip = "unknown"
    logging.info(f"Creating ZMQ server (host: {hostname}, ip: {local_ip})")

    # Start ZMQ server
    server = ZmqPolicyServer(
        policy=wrapped_policy,
        host="0.0.0.0",
        port=args.port,
        metadata={"env": "gr00t_wbc", "protocol": "zmq"},
    )
    logging.info(f"ZMQ server running on port {args.port}...")
    server.run()


def build_argparser():
    """Build argument parser for server configuration."""
    parser = argparse.ArgumentParser(description="DiT4DiT ZMQ Inference Server")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to model checkpoint (pytorch_model.pt)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5556,
        help="ZMQ server port (default: 5556)",
    )
    parser.add_argument(
        "--use_bf16",
        action="store_true",
        help="Use bfloat16 precision for model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda, xpu or cpu (default: first available of cuda, xpu, cpu)",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="g1_body29_aloha_arms_only",
        help="Data configuration name",
    )
    parser.add_argument(
        "--unnorm_key",
        type=str,
        default=None,
        help="Dataset key for action unnormalization stats. Auto-detected if only one dataset in statistics.",
    )
    return parser


def start_debugpy_once():
    """Start debugpy server for remote debugging (optional)."""
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10095))
    print("Waiting for VSCode attach on 0.0.0.0:10095 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()

    if os.getenv("DEBUG", False):
        print("DEBUGPY is enabled")
        start_debugpy_once()

    main(args)

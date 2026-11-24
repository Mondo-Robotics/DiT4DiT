import logging
import socket
import argparse
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from DiT4DiT.model.framework.base_framework import baseframework
from DiT4DiT.model.tools import get_default_device
import torch, os
def main(args) -> None:
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()

    vla = baseframework.from_pretrained(
        args.ckpt_path,
    )

    if args.use_bf16: # False
        vla = vla.to(torch.bfloat16)
    elif args.use_fp16:
        vla = vla.to(torch.float16)
    device = torch.device(args.device) if args.device else get_default_device()
    vla = vla.to(device).eval()
    logging.info("Model running on device: %s", device)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # start websocket server
    server = WebsocketPolicyServer(
        policy=vla,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={"env": "simpler_env"},
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--use_fp16", action="store_true", help="Use float16 precision for model (instead of --use_bf16)")
    parser.add_argument("--device", type=str, default=None, help="cuda, xpu or cpu (default: first available of cuda, xpu, cpu)")
    parser.add_argument("--idle_timeout" , type=int, default=1800, help="Idle timeout in seconds, -1 means never close")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    main(args)
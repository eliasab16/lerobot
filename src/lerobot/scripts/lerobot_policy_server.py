"""CLI entry point for the remote inference server.

Run on a GPU box:

    lerobot-policy-server --host 0.0.0.0 --port 8000

Then point ``lerobot-rollout`` at it from the robot machine:

    lerobot-rollout --inference.type=remote \\
        --inference.server_url=ws://<gpu-box-ip>:8000 ...

Auth is intentionally absent. Deploy behind a VPN or SSH tunnel only.
"""

import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="LeRobot remote inference WebSocket server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    uvicorn.run(
        "lerobot.rollout.remote.server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()

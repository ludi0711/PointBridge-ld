#!/usr/bin/env python3
"""Standalone CTM2F110 gripper test through the xArm Python SDK."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _add_xarm_sdk_to_path() -> None:
    project_dir = Path(__file__).resolve().parents[2]
    candidates = []

    env_path = os.environ.get("XARM_SDK_DIR", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates += [
        project_dir / "xArm_Python_SDK_master",
        project_dir / "third_party" / "xArm_Python_SDK_master",
        Path("/home/gxai/Desktop/embodied-ai-project/modules/control/scripts/third_party/xArm_Python_SDK_master"),
        Path("/home/gxai/Desktop/zjj/embodied-ai-project/modules/control/scripts/third_party/xArm_Python_SDK_master"),
        Path("/home/gxai/Desktop/test_pickandprice/embodied-ai-project/modules/control/scripts/third_party/xArm_Python_SDK_master"),
        Path("/home/gxai/Desktop/task_node_neo/embodied-ai-project/modules/control/scripts/third_party/xArm_Python_SDK_master"),
    ]

    for path in candidates:
        if (path / "xarm").exists():
            sys.path.insert(0, str(path))
            print(f"[SDK] using xArm SDK: {path}")
            return

    print("[SDK] xArm SDK not found. Set XARM_SDK_DIR=/path/to/xArm_Python_SDK_master")


_add_xarm_sdk_to_path()

from xarm.wrapper import XArmAPI
from ctm2f110_gripper import CTM2F110Gripper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test CTM2F110 gripper on the xArm tool port.")
    parser.add_argument("--ip", type=str, default="192.168.73.229")
    parser.add_argument(
        "--cmd",
        choices=("open", "close", "set", "stop", "feedback", "status", "cycle", "interactive"),
        required=True,
    )
    parser.add_argument("--pos", type=int, default=50, help="Target position for --cmd set, 4=closed, 100=open.")
    parser.add_argument("--speed", type=int, default=80, help="Gripper speed, 1..100.")
    parser.add_argument("--torque", type=int, default=80, help="Gripper torque, 0..100.")
    parser.add_argument("--slave_id", type=int, default=1)
    parser.add_argument("--host_id", type=int, default=9)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--sleep_s", type=float, default=1.0, help="Pause between open/close in cycle mode.")
    parser.add_argument("--no_wait", action="store_true")
    return parser.parse_args()


def _run_interactive(gripper: CTM2F110Gripper, args: argparse.Namespace) -> None:
    wait = not args.no_wait
    print("Interactive CTM2F110 gripper control")
    print("  o/open      open gripper")
    print("  c/close     close gripper")
    print("  m/mid       move to --pos")
    print("  p/set POS   move to POS, 4=closed, 100=open")
    print("  f/feedback  print finger feedback")
    print("  t/status    print status")
    print("  s/stop      stop motion")
    print("  q/quit      exit")

    while True:
        try:
            line = input("gripper> ").strip().lower()
        except EOFError:
            print()
            break
        if not line:
            continue

        parts = line.split()
        cmd = parts[0]
        if cmd in ("q", "quit", "exit"):
            gripper.stop()
            break
        if cmd in ("o", "open"):
            ret = gripper.open(speed=args.speed, torque=args.torque, wait=wait, timeout=args.timeout)
            print(f"[open] feedback={ret}")
        elif cmd in ("c", "close"):
            ret = gripper.close(speed=args.speed, torque=args.torque, wait=wait, timeout=args.timeout)
            print(f"[close] feedback={ret}")
        elif cmd in ("m", "mid"):
            ret = gripper.set_position(args.pos, speed=args.speed, torque=args.torque, wait=wait, timeout=args.timeout)
            print(f"[set pos={args.pos}] feedback={ret}")
        elif cmd in ("p", "set"):
            if len(parts) < 2:
                print("usage: p POS")
                continue
            try:
                pos = int(parts[1])
            except ValueError:
                print(f"invalid position: {parts[1]}")
                continue
            ret = gripper.set_position(pos, speed=args.speed, torque=args.torque, wait=wait, timeout=args.timeout)
            print(f"[set pos={pos}] feedback={ret}")
        elif cmd in ("f", "feedback"):
            print(f"[feedback] finger1={gripper.feedback(finger=1)}")
            print(f"[feedback] finger2={gripper.feedback(finger=2)}")
        elif cmd in ("t", "status"):
            print(f"[status] {gripper.status()}")
        elif cmd in ("s", "stop"):
            gripper.stop()
            print("[stop] done")
        else:
            print(f"unknown command: {line}")


def main() -> None:
    args = parse_args()

    arm = XArmAPI(args.ip)
    gripper = CTM2F110Gripper(arm, slave_id=args.slave_id, host_id=args.host_id)

    try:
        arm.clean_error()
        arm.clean_warn()
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(0.2)

        wait = not args.no_wait
        if args.cmd == "open":
            ret = gripper.open(speed=args.speed, torque=args.torque, wait=wait, timeout=args.timeout)
            print(f"[open] feedback={ret}")
        elif args.cmd == "close":
            ret = gripper.close(speed=args.speed, torque=args.torque, wait=wait, timeout=args.timeout)
            print(f"[close] feedback={ret}")
        elif args.cmd == "set":
            ret = gripper.set_position(
                args.pos,
                speed=args.speed,
                torque=args.torque,
                wait=wait,
                timeout=args.timeout,
            )
            print(f"[set pos={args.pos}] feedback={ret}")
        elif args.cmd == "stop":
            gripper.stop()
            print("[stop] done")
        elif args.cmd == "feedback":
            print(f"[feedback] finger1={gripper.feedback(finger=1)}")
            print(f"[feedback] finger2={gripper.feedback(finger=2)}")
        elif args.cmd == "status":
            print(f"[status] {gripper.status()}")
        elif args.cmd == "interactive":
            _run_interactive(gripper, args)
        elif args.cmd == "cycle":
            print("[cycle] open")
            print(f"  feedback={gripper.open(speed=args.speed, torque=args.torque, timeout=args.timeout)}")
            time.sleep(args.sleep_s)
            print("[cycle] close")
            print(f"  feedback={gripper.close(speed=args.speed, torque=args.torque, timeout=args.timeout)}")
            time.sleep(args.sleep_s)
            print("[cycle] stop")
            gripper.stop()
    finally:
        try:
            arm.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()

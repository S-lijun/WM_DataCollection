#!/usr/bin/env python3
"""Export humanoid dataset episode visual frames (.pth) to MP4.

Each ``obses/episode_{i}.pth`` is a uint8 tensor ``[T, H, W, 3]`` saved by
``collect_humanoid_dataset.py``. This script stacks those frames into a video.

Examples
--------
  python data/export_episode_video.py --dataset_dir data/humanoid_g1_test --episode 0
  python data/export_episode_video.py --dataset_dir data/humanoid_g1_test --all
  python data/export_episode_video.py --input obses/episode_3.pth --output ep3.mp4
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import imageio
import torch


def _episode_index(path: Path) -> int:
    match = re.search(r"episode_(\d+)\.pth$", path.name)
    if not match:
        raise ValueError(f"Not an episode file: {path}")
    return int(match.group(1))


def export_episode(
    input_path: Path,
    output_path: Path,
    *,
    fps: float,
    max_frames: int | None,
) -> None:
    frames = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(frames, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor in {input_path}, got {type(frames)}")

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected shape [T, H, W, 3], got {tuple(frames.shape)}")

    if frames.dtype != torch.uint8:
        frames = frames.clamp(0, 255).to(torch.uint8)

    arr = frames.numpy()
    if max_frames is not None:
        arr = arr[:max_frames]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), arr, fps=fps, macro_block_size=1)
    print(f"[OK] {input_path.name} ({arr.shape[0]} frames) -> {output_path}")


def _list_episodes(obses_dir: Path) -> list[Path]:
    paths = sorted(obses_dir.glob("episode_*.pth"), key=_episode_index)
    if not paths:
        raise FileNotFoundError(f"No episode_*.pth under {obses_dir}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export obses/episode_*.pth to MP4.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--dataset_dir",
        type=Path,
        help="Dataset root (reads obses/episode_*.pth inside).",
    )
    src.add_argument(
        "--input",
        type=Path,
        help="Single episode .pth file.",
    )

    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Episode index when using --dataset_dir (default: 0). Ignored with --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export every episode in obses/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .mp4 path (only for single --input or one --episode).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for videos (default: dataset_dir, or parent of --input).",
    )
    parser.add_argument(
        "--obs_subdir",
        type=str,
        default="obses",
        help="Subfolder under dataset_dir with episode_*.pth (e.g. obses or obses_15fps).",
    )
    parser.add_argument("--fps", type=float, default=None, help="Video frame rate (default: 15 for obses_15fps, else 60).")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional cap on number of frames exported per episode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.input is not None:
        out = args.output or args.input.with_suffix(".mp4")
        fps = args.fps if args.fps is not None else 60.0
        export_episode(args.input, out, fps=fps, max_frames=args.max_frames)
        return 0

    dataset_dir = args.dataset_dir.resolve()
    obses_dir = dataset_dir / args.obs_subdir
    output_dir = (args.output_dir or dataset_dir).resolve()
    fps = args.fps
    if fps is None:
        fps = 15.0 if args.obs_subdir == "obses_15fps" else 60.0

    if args.all:
        for ep_path in _list_episodes(obses_dir):
            idx = _episode_index(ep_path)
            out = output_dir / f"episode_{idx}.mp4"
            export_episode(ep_path, out, fps=fps, max_frames=args.max_frames)
        return 0

    episode = 0 if args.episode is None else args.episode
    ep_path = obses_dir / f"episode_{episode}.pth"
    if not ep_path.is_file():
        raise FileNotFoundError(ep_path)

    if args.output is not None:
        out = args.output
    else:
        out = output_dir / f"episode_{episode}.mp4"

    export_episode(ep_path, out, fps=fps, max_frames=args.max_frames)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

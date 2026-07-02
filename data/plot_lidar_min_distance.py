#!/usr/bin/env python3
"""Plot LiDAR min distance (states.pth last dim) vs control step per episode.

The label matches DataCollection_test ``lidar_min_range_nonzero_m`` (see meta.pth
``state_lidar_min_field``).

Examples
--------
  python data/plot_lidar_min_distance.py --dataset_dir data/humanoid_g1 --all
  python data/plot_lidar_min_distance.py --dataset_dir data/humanoid_g1 --episodes 0 1 --output lidar_min.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot lidar min distance vs step from humanoid dataset states.pth.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "humanoid_g1",
        help="Dataset root containing states.pth and seq_lengths.pth.",
    )
    ep_group = parser.add_mutually_exclusive_group()
    ep_group.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Episode indices to plot.",
    )
    ep_group.add_argument(
        "--all",
        action="store_true",
        help="Plot every episode in states.pth.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save figure to this path (default: show interactively).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title.",
    )
    return parser.parse_args()


def load_lidar_min_series(states: torch.Tensor, seq_len: int) -> tuple[list[int], list[float]]:
    """Return (steps, distances) for one episode; skip NaN padding tail."""
    traj = states[:seq_len, -1].detach().cpu().float().numpy()
    steps = list(range(seq_len))
    distances = [float(x) for x in traj]
    return steps, distances


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    states_path = dataset_dir / "states.pth"
    seq_path = dataset_dir / "seq_lengths.pth"

    if not states_path.is_file():
        raise FileNotFoundError(states_path)
    if not seq_path.is_file():
        raise FileNotFoundError(seq_path)

    states = torch.load(states_path, map_location="cpu", weights_only=False)
    seq_lengths = torch.load(seq_path, map_location="cpu", weights_only=False)

    if states.ndim != 3:
        raise ValueError(f"Expected states shape [N, T, D], got {tuple(states.shape)}")

    n_ep = int(states.shape[0])
    if args.all or args.episodes is None:
        episode_ids = list(range(n_ep))
    else:
        episode_ids = args.episodes
    for ep in episode_ids:
        if ep < 0 or ep >= n_ep:
            raise ValueError(f"Episode {ep} out of range [0, {n_ep - 1}]")

    meta_path = dataset_dir / "meta.pth"
    lidar_field = "lidar_min_range_nonzero_m"
    if meta_path.is_file():
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        if isinstance(meta, dict):
            lidar_field = meta.get("state_lidar_min_field", lidar_field)

    fig, ax = plt.subplots(figsize=(10, 4))
    for ep in episode_ids:
        seq_len = int(seq_lengths[ep].item())
        steps, distances = load_lidar_min_series(states[ep], seq_len)
        ax.plot(steps, distances, linewidth=1.0, label=f"episode {ep} (T={seq_len})")

    ax.set_xlabel("step")
    ax.set_ylabel("lidar min distance (m)")
    ax.set_title(args.title or f"LiDAR min distance ({lidar_field})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if args.output is not None:
        out = args.output.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"[OK] Saved {out}")
    else:
        default_out = dataset_dir / "lidar_min_distance.png"
        fig.savefig(default_out, dpi=150)
        print(f"[OK] Saved {default_out}")

    plt.close(fig)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

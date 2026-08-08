#!/usr/bin/env python3
"""Hold the impact state for a few extra publication frames."""

import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageSequence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=Path, required=True)
    parser.add_argument("--impact-frame", type=int, required=True)
    parser.add_argument("--hold-frames", type=int, default=4)
    args = parser.parse_args()

    frames_dir = args.group / "frames"
    impact_path = frames_dir / f"frame_{args.impact_frame:04d}.png"
    for offset in range(1, args.hold_frames + 1):
        shutil.copy2(
            impact_path,
            frames_dir / f"frame_{args.impact_frame + offset:04d}.png",
        )

    gif_path = args.group / "animation.gif"
    with Image.open(gif_path) as source:
        duration = source.info.get("duration", 100)
        loop = source.info.get("loop", 0)
        gif_frames = [frame.convert("RGBA") for frame in ImageSequence.Iterator(source)]
    gif_frames.extend([gif_frames[-1].copy() for _ in range(args.hold_frames)])
    temporary_path = gif_path.with_suffix(".tmp.gif")
    gif_frames[0].save(
        temporary_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration,
        loop=loop,
        disposal=2,
    )
    temporary_path.replace(gif_path)
    print(f"Extended {gif_path} to {len(gif_frames)} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

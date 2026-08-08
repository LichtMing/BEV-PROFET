#!/usr/bin/env python3
"""Create a safe intersection XML by replacing only the ID 160 obstacle."""

import argparse
import re
from pathlib import Path


def obstacle_block(xml: str, obstacle_id: int) -> re.Match:
    pattern = re.compile(
        rf"  <dynamicObstacle id=\"{obstacle_id}\">.*?"
        rf"  </dynamicObstacle>\n",
        re.DOTALL,
    )
    match = pattern.search(xml)
    if match is None:
        raise ValueError(f"dynamic obstacle {obstacle_id} not found")
    return match


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--follower-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = args.base.read_text(encoding="utf-8")
    follower_source = args.follower_source.read_text(encoding="utf-8")
    base_match = obstacle_block(base, 160)
    follower_match = obstacle_block(follower_source, 160)
    output = base[:base_match.start()] + follower_match.group() + base[base_match.end():]

    # This is the controlled scenario exception: all XML outside obstacle 160
    # must remain byte-for-byte identical to the collision scenario.
    output_match = obstacle_block(output, 160)
    if base[:base_match.start()] + base[base_match.end():] != (
        output[:output_match.start()] + output[output_match.end():]
    ):
        raise ValueError("safe variant changed XML outside obstacle 160")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

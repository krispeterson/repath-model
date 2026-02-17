#!/usr/bin/env python3
import argparse


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert pixel bounding box coordinates to a YOLO label line.")
    parser.add_argument("--class-id", type=int, required=True)
    parser.add_argument("--image-width", type=float, required=True)
    parser.add_argument("--image-height", type=float, required=True)
    parser.add_argument("--x1", type=float, required=True)
    parser.add_argument("--y1", type=float, required=True)
    parser.add_argument("--x2", type=float, required=True)
    parser.add_argument("--y2", type=float, required=True)
    parser.add_argument("--precision", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.image_width <= 0 or args.image_height <= 0:
        raise SystemExit("Image width and height must be > 0.")
    if args.class_id < 0:
        raise SystemExit("class-id must be >= 0.")

    x1, x2 = sorted([args.x1, args.x2])
    y1, y2 = sorted([args.y1, args.y2])

    x1n = clamp01(x1 / args.image_width)
    x2n = clamp01(x2 / args.image_width)
    y1n = clamp01(y1 / args.image_height)
    y2n = clamp01(y2 / args.image_height)

    w = x2n - x1n
    h = y2n - y1n
    if w <= 0 or h <= 0:
        raise SystemExit("Box width/height must be > 0 after normalization.")

    xc = x1n + (w / 2)
    yc = y1n + (h / 2)

    p = max(0, args.precision)
    fmt = f"{{:.{p}f}}"
    line = f"{args.class_id} {fmt.format(xc)} {fmt.format(yc)} {fmt.format(w)} {fmt.format(h)}"
    print(line)


if __name__ == "__main__":
    main()

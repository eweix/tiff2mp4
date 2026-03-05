import argparse
import os
from glob import glob

import cv2
import numpy as np
import tifffile
from rich.progress import track
from skimage.transform import resize


def _parse_args():
    parser = argparse.ArgumentParser(
        description="""Export a timeseries image from a zarr array as an mp4 video"""
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        default=None,
        help="Input files",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory",
    )
    parser.add_argument(
        "-c",
        "--channel",
        type=int,
        default=0,
        help="Channel to export as movie (default 0)",
    )
    parser.add_argument(
        "--bar",
        type=float,
        default=None,
        help="Scale bar length in micrometers per pixel (e.g., 0.5 for 0.5 µm/pixel)",
    )
    parser.add_argument(
        "--bar-length",
        type=float,
        default=None,
        help="Desired length of the scale bar in micrometers (e.g., 10 for a 10 µm bar)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Frames per second (default 60)",
    )
    parser.add_argument(
        "--upsample",
        type=int,
        default=None,
        help="Size to upsample (default None)",
    )
    parser.add_argument(
        "-t",
        "--timestamp",
        type=float,
        default=None,
        help="Add timestamp to each frame (default None)",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=1.0,
        help="Contrast scaling factor (default 1.0)",
    )
    parser.add_argument(
        "--brightness",
        type=int,
        default=0,
        help="Brightness offset (default 0)",
    )

    return parser.parse_args()


# Normalize array to 0-255 uint8
def _normalize(arr: np.ndarray) -> np.ndarray:
    min_val = arr.min()
    max_val = arr.max()
    if max_val - min_val == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = (arr - min_val) / (max_val - min_val)
    return (norm * 255).astype(np.uint8)


def _adjust_contrast_brightness(
    arr: np.ndarray, contrast: float = 1.0, brightness: int = 0
) -> np.ndarray:
    adjusted = arr.astype(np.float32) * contrast + brightness
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def _shorten(filepath: str) -> str:
    filename = filepath.split("/")[-1]
    short_name = filename.split(".")[0]
    return short_name


# generate video writer and save to mp4
def write_mp4(
    output_path: str,
    arr: np.ndarray,
    fps: int,
    timestamp: float | None = None,
    bar_factor: float | None = None,
    bar_length_microns: float | None = None,
) -> None:
    t, y, x = arr.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (x, y), isColor=False)
    font = cv2.FONT_HERSHEY_DUPLEX
    for i in range(t):
        frame = arr[i]
        if timestamp is not None:
            time_sec = i * timestamp
            text = f"{time_sec:.1f} sec"
            cv2.putText(frame, text, (30, 60), font, 0.8, 255, 2)
        if bar_factor is not None and bar_length_microns is not None:
            y, x = frame.shape
            length_px = int(bar_factor * bar_length_microns)
            end_pt = (x - 20, y - 20)
            start_pt = (end_pt[0] - length_px, end_pt[1])
            cv2.line(frame, start_pt, end_pt, 255, 3)
            label = f"{bar_length_microns:.1f} µm"
            cv2.putText(frame, label, (start_pt[0], start_pt[1] - 5), font, 0.6, 255, 1)
        writer.write(frame)
    writer.release()


def main() -> None:
    args = _parse_args()
    u = args.upsample

    if args.input is None:
        exit()

    if os.path.isdir(args.input[0]):
        input_files = [
            os.path.join(args.input[0], f)
            for f in glob("*.tif", root_dir=args.input[0], recursive=True)
        ]
    else:
        input_files = args.input

    for f in track(input_files):
        arr = tifffile.imread(f)
        assert len(arr.shape) == 4, "array must have dimensions TCXY"
        t, c, y, x = arr.shape
        arr = arr[:, args.channel, :, :]  # dimensions (T, Y, X)

        # Upsample
        if u is not None:
            arr = resize(arr, (t, y * u, x * u), anti_aliasing=None)

        # Apply contrast and brightness adjustments
        arr = _normalize(arr)
        arr = _adjust_contrast_brightness(arr, args.contrast, args.brightness)

        # Ensure output directory exists
        output_path = os.path.join(args.output, f"{_shorten(f)}.mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Video writer
        write_mp4(
            output_path,
            arr,
            args.fps,
            timestamp=args.timestamp,
            bar_factor=args.bar,
            bar_length_microns=args.bar_length,
        )


if __name__ == "__main__":
    main()

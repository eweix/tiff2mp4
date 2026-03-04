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
    timestamp: float | int | None = None,
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
        writer.write(frame)
    writer.release()


def main() -> None:
    args = _parse_args()
    u = args.upsample

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
        write_mp4(output_path, arr, args.fps, timestamp=args.timestamp)


if __name__ == "__main__":
    main()

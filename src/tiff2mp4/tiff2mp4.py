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
    return parser.parse_args()


# Normalize array to 0-255 uint8
def _normalize(frame: np.ndarray) -> np.ndarray:
    min_val = frame.min()
    max_val = frame.max()
    if max_val - min_val == 0:
        return np.zeros_like(frame, dtype=np.uint8)
    norm = (frame - min_val) / (max_val - min_val)
    return (norm * 255).astype(np.uint8)


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
        frame = _normalize(arr[i])

        # Add timestamp if requested
        if timestamp is not None:
            time_sec = i * timestamp
            text = f"{time_sec:.1f} sec"

            # Add text to frame
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

        # Ensure output directory exists
        output_path = os.path.join(args.output, f"{_shorten(f)}.mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Video writer
        write_mp4(output_path, arr, args.fps, timestamp=args.timestamp)


if __name__ == "__main__":
    main()

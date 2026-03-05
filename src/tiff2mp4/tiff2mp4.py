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
        type=int,
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
        type=str,
        default="1.0",
        help="Contrast scaling factor (default 1.0, or 'auto' to auto-adjust)",
    )
    parser.add_argument(
        "--brightness",
        type=str,
        default="0",
        help="Brightness offset (default 0, or 'auto' to auto-adjust)",
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
    arr: np.ndarray,
    contrast: float = 1.0,
    brightness: int = 0,
) -> np.ndarray:
    return np.clip(arr.astype(np.float32) * contrast + brightness, 0, 255).astype(
        np.uint8
    )


def _auto_brightness_contrast(arr: np.ndarray) -> tuple[int, float]:
    low = np.percentile(arr, 2)
    high = np.percentile(arr, 98)
    if high - low == 0:
        opt_contrast = 1.0
        opt_brightness = 0
    else:
        opt_contrast = 255.0 / (high - low)
        opt_brightness = int(-low * opt_contrast)
    return opt_brightness, opt_contrast


def _shorten(filepath: str) -> str:
    filename = filepath.split("/")[-1]
    short_name = filename.split(".")[0]
    return short_name


def _place_scalebar(
    frame_shape: tuple[int, int],
    bar_factor: float,
    bar_length_microns: float,
) -> dict:
    y, x = frame_shape
    margin = 20
    max_length_px = x - 2 * margin
    length_px = int(bar_factor * bar_length_microns)
    if length_px > max_length_px:
        length_px = max_length_px
    end_pt = (x - margin, y - margin)
    start_pt = (end_pt[0] - length_px, end_pt[1])
    label = f"{bar_length_microns:.0f} um"
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
    text_x = start_pt[0]
    if text_x + text_w + margin > x:
        text_x = end_pt[0] - text_w - margin
    text_pos = (text_x, start_pt[1] - 5)
    return {"start": start_pt, "end": end_pt, "text_pos": text_pos, "label": label}


def _make_scalebar(
    frame: np.ndarray,
    coords: dict,
) -> None:
    cv2.line(frame, coords["start"], coords["end"], 255, 3)
    cv2.putText(
        frame,
        coords["label"],
        coords["text_pos"],
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        255,
        1,
    )


# generate video writer and save to mp4
def write_mp4(
    output_path: str,
    arr: np.ndarray,
    fps: int,
    timestamp: float | None = None,
    bar_factor: float | None = None,
    bar_length_microns: int | None = None,
) -> None:
    t, y, x = arr.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (x, y), isColor=False)
    font = cv2.FONT_HERSHEY_DUPLEX
    if bar_factor is not None and bar_length_microns is not None:
        scalebar_coords = _place_scalebar((y, x), bar_factor, bar_length_microns)
    for i in range(t):
        frame = arr[i]
        if timestamp is not None:
            time_sec = i * timestamp
            text = f"{time_sec:.1f} sec"
            cv2.putText(frame, text, (30, 60), font, 0.8, 255, 2)
        if bar_factor is not None and bar_length_microns is not None:
            _make_scalebar(frame, scalebar_coords)
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
        if args.contrast == "auto" or args.brightness == "auto":
            brightness_value, contrast_value = _auto_brightness_contrast(arr)
        else:
            contrast_value = float(args.contrast)
            brightness_value = int(args.brightness)
        arr = _adjust_contrast_brightness(arr, contrast_value, brightness_value)

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

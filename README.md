# tiff2mp4

`tiff2mp4.py` converts a time‑series image stack stored in a 4‑D TIFF file (TCXY) into an MP4 video.

## Usage

With python:

```bash
python -m tiff2mp4 -i /path/to/input/files  -o /path/to/output/directory [options]
```

With uv (recommended):

```bash
# run script
uv run tiff2mp4 -i /path/to/input/files -o /path/to/output/directory [options]

# install as tool
uv tool install git+https://github.com/eweix/tiff2mp4

# run as tool
uvx tiff2mp4 -i /path/to/input/files -o /path/to/output/directory [options]
```

### Required arguments

- `-i, --input`  
  One or more TIFF files or a directory containing them.

- `-o, --output`  
  Directory where the MP4 files will be written. The script creates sub‑directories as needed.

### Optional arguments

| Flag              | Description                                                        |
| ----------------- | ------------------------------------------------------------------ |
| `-c, --channel`   | Channel index to export (default 0).                               |
| `--bar`           | Scale bar length in µm per pixel.                                  |
| `--bar-length`    | Desired physical length of the scale bar in µm.                    |
| `--fps`           | Frames per second for the output video (default 60).               |
| `--upsample`      | Upsample factor applied to each frame.                             |
| `-t, --timestamp` | Time interval between frames in seconds; adds a timestamp overlay. |
| `--contrast`      | Contrast factor (float) or `"auto"` to auto‑adjust.                |
| `--brightness`    | Brightness offset (int) or `"auto"` to auto‑adjust.                |

## Features

- Handles multi‑channel TIFF stacks; selects a single channel.
- Normalizes pixel values to 0–255 and applies optional contrast/brightness adjustments.
- Upsamples frames if requested.
- Adds an optional scale bar and timestamp overlay.
- Generates a separate MP4 per input file.

## Dependencies

```bash
pip install numpy tifffile opencv-python scikit-image rich
```

#!/usr/bin/env python3
"""Crop deployment rollout panels and environment maps from the master rollout.jpg.

Uses ffmpeg under the hood (Pillow is unavailable in the sandbox) to pre-render
each tile so the website can lazy-load small assets instead of slicing one huge
13163x7182 JPEG with CSS background tricks.

Run from the repository root:
    python3 static/images/rollouts/crop_panels.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

SRC_PATH = "static/images/rollout.jpg"
OUT_DIR = "static/images/rollouts"
SOURCE_W = 13163
SOURCE_H = 7182
TARGET_PANEL_WIDTH = 720
TARGET_MAP_WIDTH = 520


@dataclass(frozen=True)
class CropSpec:
    """Description of a single rectangular crop in the source image.

    Attributes
    ----------
    slug : str
        Output filename stem (lowercase, hyphenated).
    x_pct : float
        Left edge of the crop expressed as a percentage of the source width.
    y_pct : float
        Top edge of the crop expressed as a percentage of the source height.
    w_pct : float
        Width of the crop expressed as a percentage of the source width.
    h_pct : float
        Height of the crop expressed as a percentage of the source height.
    target_width : int
        Output width in pixels after rescaling.
    """

    slug: str
    x_pct: float
    y_pct: float
    w_pct: float
    h_pct: float
    target_width: int


# Photo-only crop coordinates measured directly from rollout.jpg.
# Each row of panels shares the same vertical strip, and the five panels in a
# row share the same horizontal step (~15.95% of source width). Panel widths
# are ~13.9% of source width, photo heights ~13.3% of source height.
_PANEL_X = (18.5, 34.45, 50.4, 66.35, 82.3)
_PANEL_W = 13.45

_ROW_Y = {
    "lab-1": 3.7,
    "office": 30.0,
    "lab-2": 54.3,
    "public-area": 79.7,
}
_ROW_H = {
    "lab-1": 12.4,
    "office": 13.0,
    "lab-2": 13.0,
    "public-area": 13.0,
}


def _panel(slug: str, row: str, col_index: int) -> CropSpec:
    """Build a panel CropSpec from row id and column index.

    Parameters
    ----------
    slug : str
        Output filename stem.
    row : str
        Row identifier ("lab-1", "office", "lab-2", "public-area").
    col_index : int
        Zero-based panel column index within the row.

    Returns
    -------
    CropSpec
        Crop description for the requested panel.
    """

    return CropSpec(
        slug=slug,
        x_pct=_PANEL_X[col_index],
        y_pct=_ROW_Y[row],
        w_pct=_PANEL_W,
        h_pct=_ROW_H[row],
        target_width=TARGET_PANEL_WIDTH,
    )


PANELS: tuple[CropSpec, ...] = (
    _panel("safety-cones", "lab-1", 0),
    _panel("teddy-bear", "lab-1", 1),
    _panel("ultimaker-s5-material", "lab-1", 2),
    _panel("volleyball", "lab-1", 3),
    _panel("hammer", "lab-1", 4),
    _panel("calibration-board", "office", 0),
    _panel("fire-extinguisher", "office", 1),
    _panel("floor-fan", "office", 2),
    _panel("tripod", "office", 3),
    _panel("clothes", "office", 4),
    _panel("stand-fan", "lab-2", 0),
    _panel("play-some-sports", "lab-2", 1),
    _panel("trash-bin", "lab-2", 2),
    _panel("controller-box", "lab-2", 3),
    _panel("safety-helmet", "lab-2", 4),
    _panel("presentation-board", "public-area", 0),
    _panel("item-it-mkmx", "public-area", 1),
    _panel("gap", "public-area", 2),
    _panel("four-cross-signs", "public-area", 3),
    _panel("sign", "public-area", 4),
)

# Environment maps live in the leftmost column of each row. We drop the
# row letter (A/B/C/D) on the far left by starting around 4% from the left
# edge, and stop just before the panel column dividers.
MAPS: tuple[CropSpec, ...] = (
    CropSpec("map-lab-1", 4.0, 2.0, 11.5, 14.5, TARGET_MAP_WIDTH),
    CropSpec("map-office", 4.0, 27.5, 11.5, 16.5, TARGET_MAP_WIDTH),
    CropSpec("map-lab-2", 4.0, 51.5, 11.5, 17.0, TARGET_MAP_WIDTH),
    CropSpec("map-public-area", 4.0, 76.0, 11.5, 17.5, TARGET_MAP_WIDTH),
)


def even(value: int) -> int:
    """Return the largest even integer not exceeding ``value``.

    ffmpeg's JPEG encoder needs even pixel dimensions when chroma subsampling
    is in play, so we round down to the nearest even integer for safety.

    Parameters
    ----------
    value : int
        Pixel dimension to be rounded.

    Returns
    -------
    int
        Even integer no larger than ``value``.
    """

    return value if value % 2 == 0 else value - 1


def render_crop(src: str, dst: str, spec: CropSpec) -> None:
    """Render a single crop from ``src`` to ``dst`` using ffmpeg.

    Parameters
    ----------
    src : str
        Path to the source JPEG image.
    dst : str
        Destination path for the cropped JPEG.
    spec : CropSpec
        Crop geometry and target output width.

    Raises
    ------
    RuntimeError
        If ffmpeg fails to execute the requested crop.
    """

    crop_w = even(round(spec.w_pct / 100.0 * SOURCE_W))
    crop_h = even(round(spec.h_pct / 100.0 * SOURCE_H))
    crop_x = even(round(spec.x_pct / 100.0 * SOURCE_W))
    crop_y = even(round(spec.y_pct / 100.0 * SOURCE_H))

    target_w = even(spec.target_width)
    filter_chain = f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={target_w}:-2:flags=lanczos"

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        src,
        "-vf",
        filter_chain,
        "-frames:v",
        "1",
        "-q:v",
        "3",
        dst,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {spec.slug}: {result.stderr.strip() or result.stdout.strip()}"
        )


def main() -> int:
    """Execute the crop pipeline for every panel and environment map.

    Returns
    -------
    int
        Process exit code (0 on success, non-zero on any crop failure).
    """

    if not shutil.which("ffmpeg"):
        print("ffmpeg not found on PATH", file=sys.stderr)
        return 1
    if not os.path.isfile(SRC_PATH):
        print(f"Source image not found: {SRC_PATH}", file=sys.stderr)
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    failures: list[str] = []

    for spec in (*PANELS, *MAPS):
        dst = os.path.join(OUT_DIR, f"{spec.slug}.jpg")
        try:
            render_crop(SRC_PATH, dst, spec)
            size_kb = os.path.getsize(dst) / 1024.0
            print(f"  wrote {dst}  ({size_kb:.0f} KB)")
        except RuntimeError as exc:
            failures.append(str(exc))
            print(str(exc), file=sys.stderr)

    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

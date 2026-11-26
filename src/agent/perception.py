from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from PIL import Image, ImageGrab  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    Image = None
    ImageGrab = None

from agent.guard import WorkspaceGuard


def _require_pil() -> tuple[Any, Any]:
    if Image is None or ImageGrab is None:
        raise RuntimeError(
            "Optional dependency 'Pillow' is not installed. Install it with: pip install Pillow"
        )
    return Image, ImageGrab


@dataclass(frozen=True)
class ScreenRegion:
    left: int
    top: int
    width: int
    height: int

    def to_bbox(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.left + self.width, self.top + self.height)


def compute_ssim_gray(img_a: Any, img_b: Any) -> float:
    """Compute Structural Similarity Index (SSIM) between two images in [0.0, 1.0] range."""
    pil_img, _ = _require_pil()
    gray_a = img_a.convert("L")
    gray_b = img_b.convert("L")
    if gray_a.size != gray_b.size:
        gray_b = gray_b.resize(gray_a.size, pil_img.Resampling.BILINEAR)

    arr_a = np.asarray(gray_a, dtype=np.float64)
    arr_b = np.asarray(gray_b, dtype=np.float64)
    if arr_a.size == 0:
        return 1.0

    # Dynamic range L = 255
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    mu_x = float(np.mean(arr_a))
    mu_y = float(np.mean(arr_b))
    var_x = float(np.var(arr_a))
    var_y = float(np.var(arr_b))
    cov_xy = float(np.mean((arr_a - mu_x) * (arr_b - mu_y)))

    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    if abs(denominator) < 1e-12:
        return 1.0

    ssim = numerator / denominator
    return round(max(-1.0, min(1.0, ssim)), 5)


class PerceptionEngine:
    """Captures screen/region snapshots and compares visual state before/after actions."""

    def __init__(self, guard: WorkspaceGuard | None = None) -> None:
        self.guard = guard

    def capture(self, region: ScreenRegion | None = None) -> Any:
        """Capture full desktop or a specific ScreenRegion bounding box."""
        _, pil_grab = _require_pil()
        bbox = region.to_bbox() if region is not None else None
        return pil_grab.grab(bbox=bbox, all_screens=True)

    def save_snapshot(self, image: Any, relative_path: str) -> dict[str, Any]:
        """Save a PNG snapshot inside WorkspaceGuard if guard is configured."""
        _require_pil()
        if self.guard is None:
            return {"ok": False, "error": "No WorkspaceGuard attached for saving snapshots."}
        target = self.guard.resolve_path(relative_path)
        if target.suffix.lower() != ".png":
            target = target.with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, format="PNG")
        return {"ok": True, "path": str(target), "size": list(image.size)}

    def compare_frames(
        self,
        before: Any,
        after: Any,
        unchanged_threshold: float = 0.992,
    ) -> dict[str, Any]:
        """Compare pre-action and post-action images and classify whether state changed."""
        _require_pil()
        score = compute_ssim_gray(before, after)
        changed = score < unchanged_threshold

        arr_a = np.asarray(before.convert("L").resize((160, 90)), dtype=np.float64)
        arr_b = np.asarray(after.convert("L").resize((160, 90)), dtype=np.float64)
        rms_diff = math.sqrt(float(np.mean((arr_a - arr_b) ** 2)))

        return {
            "ok": True,
            "ssim": score,
            "changed": changed,
            "unchanged_threshold": unchanged_threshold,
            "rms_diff_160x90": round(rms_diff, 3),
        }

    def load_image(self, relative_or_abs_path: str | Path) -> Any:
        pil_img, _ = _require_pil()
        path = self.guard.resolve_path(relative_or_abs_path) if self.guard else Path(relative_or_abs_path)
        return pil_img.open(path)

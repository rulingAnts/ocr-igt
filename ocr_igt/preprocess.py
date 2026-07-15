"""Image loading and cleanup, shared by both OCR engines.

OpenCV is used when available. Two clean-up profiles:

* ``tesseract`` — deskew, denoise, adaptive threshold (binary). Traditional OCR
  wants crisp black-on-white.
* ``vision``    — deskew + contrast (CLAHE), *no* binarization. A vision model
  reads handwriting better from grey strokes than from a lossy black/white mask.

Everything degrades gracefully: if OpenCV isn't installed, we fall back to
sending the raw file bytes to the vision engine, and the Tesseract engine will
ask the user to install OpenCV.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing numpy at module load
    import numpy as np


def have_cv2() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except Exception:
        return False


def _load_gray(path: str | Path):
    import cv2
    import numpy as np

    data = np.fromfile(str(path), dtype=np.uint8)  # unicode-safe on all platforms
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"could not read image: {path}")
    return img


def _deskew(gray):
    import cv2
    import numpy as np

    try:
        inv = cv2.bitwise_not(gray)
        thr = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = cv2.findNonZero(thr)
        if coords is None:
            return gray
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if angle > 45:
            angle -= 90
        if abs(angle) < 0.3 or abs(angle) > 20:
            return gray  # nothing worth rotating, or a bogus estimate
        h, w = gray.shape[:2]
        m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(gray, m, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return gray


def _enhance(gray):
    import cv2

    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7,
                                    searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _binarize(gray):
    import cv2

    gray = cv2.fastNlMeansDenoising(gray, None, h=12)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )


def preprocess_for_tesseract(path: str | Path, do_pre: bool = True):
    """Return a numpy image ready for Tesseract (requires OpenCV)."""
    if not have_cv2():
        raise RuntimeError(
            "OpenCV is required for the Tesseract engine. Install with:\n"
            "  pip install opencv-python-headless"
        )
    gray = _load_gray(path)
    if not do_pre:
        return gray
    return _binarize(_deskew(gray))


def image_for_vision(
    path: str | Path, do_pre: bool = True, max_edge: int = 1568
) -> tuple[str, str]:
    """Return (base64_data, media_type) for a Claude image content block."""
    if not have_cv2():
        # No OpenCV: send the original file untouched.
        raw = Path(path).read_bytes()
        media = mimetypes.guess_type(str(path))[0] or "image/png"
        return base64.standard_b64encode(raw).decode("ascii"), media

    import cv2

    gray = _load_gray(path)
    if do_pre:
        gray = _enhance(_deskew(gray))

    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest > max_edge:
        scale = max_edge / float(longest)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".png", gray)
    if not ok:
        raise ValueError(f"could not encode image: {path}")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii"), "image/png"

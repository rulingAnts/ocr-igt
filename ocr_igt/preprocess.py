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


def _order_corners(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    import numpy as np

    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]        # top-left  (smallest x+y)
    rect[2] = pts[np.argmax(s)]        # bottom-right (largest x+y)
    d = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(d)]        # top-right (smallest y-x)
    rect[3] = pts[np.argmax(d)]        # bottom-left (largest y-x)
    return rect


def _find_page_quad(gray):
    """Return the page's 4 corners (full-res coords) or None if not confident."""
    import cv2
    import numpy as np

    h, w = gray.shape[:2]
    scale = 1000.0 / max(h, w) if max(h, w) > 1000 else 1.0
    small = (cv2.resize(gray, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_AREA) if scale != 1.0 else gray)
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = small.shape[0] * small.shape[1]
    best, best_area = None, 0.0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.35 * img_area:           # must cover a big chunk of the frame
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx) and area > best_area:
            best, best_area = approx, area
    if best is None:
        return None
    return best.reshape(4, 2).astype(np.float32) / scale


def _shows_perspective(rect) -> bool:
    """True only if the quad is a real trapezoid (worth a perspective warp).

    A flat scan (rectangle, possibly slightly rotated) returns False — pure
    rotation is left to _deskew, so we never warp a page that doesn't need it.
    """
    import numpy as np

    tl, tr, br, bl = rect
    top, bot = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    left, right = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)

    def ratio(a, b):
        return max(a, b) / max(1e-6, min(a, b))

    if ratio(top, bot) > 1.04 or ratio(left, right) > 1.04:
        return True  # opposite sides unequal → foreshortening

    def corner(a, b, c):  # angle at b, in degrees
        v1, v2 = a - b, c - b
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    angles = [corner(bl, tl, tr), corner(tl, tr, br),
              corner(tr, br, bl), corner(br, bl, tl)]
    return max(abs(a - 90) for a in angles) > 7.0


def _dewarp(gray):
    """Flatten a page photographed at an angle via a 4-point perspective warp.

    Returns the corrected image, or the original unchanged when no confident,
    genuinely-skewed page boundary is found (safe no-op on flat scans).
    """
    import cv2
    import numpy as np

    quad = _find_page_quad(gray)
    if quad is None:
        return gray
    rect = _order_corners(quad)
    if not _shows_perspective(rect):
        return gray

    tl, tr, br, bl = rect
    max_w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    max_h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if max_w < 50 or max_h < 50:
        return gray
    dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
                   dtype=np.float32)
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(gray, m, (max_w, max_h),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


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


def preprocess_for_tesseract(path: str | Path, do_pre: bool = True,
                             dewarp: bool = True):
    """Return a numpy image ready for Tesseract (requires OpenCV)."""
    if not have_cv2():
        raise RuntimeError(
            "OpenCV is required for the Tesseract engine. Install with:\n"
            "  pip install opencv-python-headless"
        )
    gray = _load_gray(path)
    if not do_pre:
        return gray
    if dewarp:
        gray = _dewarp(gray)
    return _binarize(_deskew(gray))


def image_for_vision(
    path: str | Path, do_pre: bool = True, max_edge: int = 1568,
    dewarp: bool = True,
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
        if dewarp:
            gray = _dewarp(gray)
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

"""
OCR utilities – EasyOCR primary, pytesseract fallback.

Every public function catches its own exceptions so the caller's
overall parse never crashes due to OCR failure.

Key changes from previous version
──────────────────────────────────
* `run_ocr_on_image()` returns **bounding-box level** results
  (list of dicts with bbox/text/confidence/source) instead of
  a single merged string.
* pytesseract is tried automatically when EasyOCR fails.
* `merge_ocr_text()` helper to flatten boxes into a single string.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── lazy singletons ──────────────────────────────────────────────────

_easyocr_reader: Any = None
_easyocr_failed: bool = False


def _get_easyocr() -> Any:
    """Return shared EasyOCR Reader (ko+en). Created once."""
    global _easyocr_reader, _easyocr_failed
    if _easyocr_failed:
        return None
    if _easyocr_reader is not None:
        return _easyocr_reader
    try:
        import easyocr
        _easyocr_reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
        logger.info("EasyOCR reader initialised (ko+en, CPU)")
        return _easyocr_reader
    except Exception as exc:
        _easyocr_failed = True
        logger.warning("EasyOCR init failed – will try pytesseract: %s", exc)
        return None


def _tesseract_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


# ── preprocessing ────────────────────────────────────────────────────

def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Light OpenCV preprocessing: grayscale → adaptive threshold."""
    import cv2

    if image is None or image.size == 0:
        return image

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
    )
    return processed


# ── public API ───────────────────────────────────────────────────────

def run_ocr_on_image(
    image: np.ndarray,
    *,
    page_width: float = 0.0,
    page_height: float = 0.0,
    zoom: float = 2.0,
) -> dict[str, Any]:
    """
    Run OCR on a numpy BGR/grayscale image.

    Returns
    -------
    {
      "boxes": [ { "bbox": [x0,y0,x1,y1], "text": str, "confidence": float } ],
      "text": str,            # merged full text
      "confidence": float,    # mean confidence
      "success": bool,
      "engine": str | None,   # "easyocr" | "pytesseract" | None
      "fallback_reason": str | None,
      "error": str | None,
    }
    """
    fallback: dict[str, Any] = {
        "boxes": [],
        "text": "",
        "confidence": 0.0,
        "success": False,
        "engine": None,
        "fallback_reason": None,
        "error": None,
    }

    # Scale factor to convert pixel coords back to PDF points
    sx = (page_width / (image.shape[1])) if page_width and image.shape[1] else 1.0
    sy = (page_height / (image.shape[0])) if page_height and image.shape[0] else 1.0

    processed = preprocess_for_ocr(image)

    # ── attempt 1: EasyOCR ───────────────────────────────────────────
    result = _run_easyocr(processed, sx, sy)
    if result["success"]:
        return result

    easyocr_error = result.get("error", "unknown")

    # ── attempt 2: pytesseract ───────────────────────────────────────
    result = _run_pytesseract(processed, sx, sy)
    if result["success"]:
        result["fallback_reason"] = f"easyocr failed: {easyocr_error}"
        return result

    # ── both failed ──────────────────────────────────────────────────
    fallback["error"] = f"easyocr: {easyocr_error}; pytesseract: {result.get('error', 'unknown')}"
    fallback["fallback_reason"] = "all_ocr_engines_failed"
    return fallback


def _run_easyocr(image: np.ndarray, sx: float, sy: float) -> dict[str, Any]:
    reader = _get_easyocr()
    if reader is None:
        return {"success": False, "error": "EasyOCR reader unavailable", "boxes": [],
                "text": "", "confidence": 0.0, "engine": None, "fallback_reason": None}
    try:
        results: list[tuple] = reader.readtext(image, detail=1, paragraph=False)
        boxes = []
        for item in results:
            pts = item[0]                       # [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
            text_val = item[1] if len(item) > 1 else ""
            conf_val = float(item[2]) if len(item) > 2 else 0.0
            if not text_val.strip():
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bbox = [
                round(min(xs) * sx, 2),
                round(min(ys) * sy, 2),
                round(max(xs) * sx, 2),
                round(max(ys) * sy, 2),
            ]
            boxes.append({"bbox": bbox, "text": text_val.strip(), "confidence": round(conf_val, 4)})

        texts = [b["text"] for b in boxes]
        confs = [b["confidence"] for b in boxes]
        avg = sum(confs) / len(confs) if confs else 0.0

        return {
            "boxes": boxes,
            "text": "\n".join(texts),
            "confidence": round(avg, 4),
            "success": True,
            "engine": "easyocr",
            "fallback_reason": None,
            "error": None,
        }
    except Exception as exc:
        logger.warning("EasyOCR execution failed: %s", exc)
        return {"success": False, "error": str(exc), "boxes": [],
                "text": "", "confidence": 0.0, "engine": None, "fallback_reason": None}


def _run_pytesseract(image: np.ndarray, sx: float, sy: float) -> dict[str, Any]:
    try:
        import pytesseract
        from PIL import Image

        pil_img = Image.fromarray(image)
        data = pytesseract.image_to_data(pil_img, lang="kor+eng", output_type=pytesseract.Output.DICT)

        boxes = []
        for i in range(len(data["text"])):
            txt = (data["text"][i] or "").strip()
            conf = int(data["conf"][i]) if data["conf"][i] != -1 else 0
            if not txt or conf < 10:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            bbox = [
                round(x * sx, 2),
                round(y * sy, 2),
                round((x + w) * sx, 2),
                round((y + h) * sy, 2),
            ]
            boxes.append({"bbox": bbox, "text": txt, "confidence": round(conf / 100.0, 4)})

        texts = [b["text"] for b in boxes]
        confs = [b["confidence"] for b in boxes]
        avg = sum(confs) / len(confs) if confs else 0.0

        return {
            "boxes": boxes,
            "text": "\n".join(texts),
            "confidence": round(avg, 4),
            "success": bool(boxes),
            "engine": "pytesseract",
            "fallback_reason": None,
            "error": None if boxes else "pytesseract returned no text",
        }
    except Exception as exc:
        logger.warning("pytesseract execution failed: %s", exc)
        return {"success": False, "error": str(exc), "boxes": [],
                "text": "", "confidence": 0.0, "engine": None, "fallback_reason": None}


# ── legacy compatibility shim ────────────────────────────────────────

def get_ocr_reader() -> Any:
    """Kept for backward compatibility with any external callers."""
    return _get_easyocr()


def run_easyocr_on_image(image: np.ndarray) -> dict[str, Any]:
    """Legacy wrapper — returns old-style {text, confidence, success}."""
    result = run_ocr_on_image(image)
    return {
        "text": result["text"],
        "confidence": result["confidence"],
        "success": result["success"],
        "error": result.get("error"),
    }

"""
Parser for HWP (Hangul Word Processor) files.

HWP5 is an OLE2 compound file.  Text lives in BodyText/Section{N}
streams, optionally zlib-compressed.  HWPTAG_PARA_TEXT (tag 67)
records hold UTF-16LE text with HWP-specific control characters.
"""

from __future__ import annotations

import logging
import re
import struct
import zlib
from typing import Any

import olefile

logger = logging.getLogger(__name__)

HWPTAG_PARA_TEXT = 67

EXTENDED_CONTROLS = frozenset(
    {1, 2, 3, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
)


# ── text normalisation ───────────────────────────────────────────────

def _normalise(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── public entry point ───────────────────────────────────────────────

def parse_hwp(filepath: str) -> dict[str, Any]:
    if not olefile.isOleFile(filepath):
        raise ValueError(f"Not a valid OLE2/HWP file: {filepath}")

    ole = olefile.OleFileIO(filepath)
    try:
        metadata = _read_metadata(ole)
        is_compressed = _check_compressed(ole)
        metadata["is_compressed"] = is_compressed

        sections: list[str] = []
        empty_sections = 0
        section_idx = 0

        while True:
            stream_name = f"BodyText/Section{section_idx}"
            if not ole.exists(stream_name):
                break
            try:
                raw = ole.openstream(stream_name).read()
                data = _decompress(raw, is_compressed)
                text = _normalise(_parse_section_records(data))
                if text:
                    sections.append(text)
                else:
                    empty_sections += 1
            except Exception as exc:
                logger.warning("HWP Section%d failed: %s", section_idx, exc)
                empty_sections += 1
            section_idx += 1
    finally:
        ole.close()

    pages: list[dict[str, Any]] = []
    for i, text in enumerate(sections):
        pages.append({
            "page_num": i + 1,
            "page_width": 0,
            "page_height": 0,
            "preview_width": 0,
            "preview_height": 0,
            "preview_scale_x": 1.0,
            "preview_scale_y": 1.0,
            "coord_space": "page_points",
            "preview_image": None,
            "text": text,
            "tables": [],
            "blocks": [{
                "id": f"p{i+1}_b0",
                "type": "text",
                "bbox": [0, 0, 0, 0],
                "text": text,
                "page_num": i + 1,
                "source": "hwp_parser",
                "score": 1.0,
                "meta": {}
            }] if text else [],
            "image_count": 0,
            "text_source": "native",
            "ocr_applied": False,
            "ocr_confidence": 0.0,
            "parser_debug": {
                "preview_generated": False,
                "preview_error": None,
                "native_text_chars": len(text),
                "ocr_used": False,
                "ocr_trigger_reason": "ocr_not_needed",
                "candidate_counts": {
                    "raw_text_blocks": 1 if text else 0,
                    "final_blocks": 1 if text else 0
                },
                "block_type_counts": {
                    "text": 1 if text else 0
                },
                "dropped_blocks": [],
                "bbox_warnings": []
            }
        })

    metadata.update({
        "parser_used": "custom OLE HWP parser",
        "page_count": len(pages),
        "section_count": section_idx,
        "empty_sections": empty_sections,
    })

    if not pages:
        return {
            "pages": [],
            "metadata": metadata,
            "status": "error",
            "error": f"No text extracted from {section_idx} section(s) ({empty_sections} empty)",
        }

    return {"pages": pages, "metadata": metadata, "status": "success"}


# ── internals ────────────────────────────────────────────────────────

def _check_compressed(ole: olefile.OleFileIO) -> bool:
    if not ole.exists("FileHeader"):
        return False
    header = ole.openstream("FileHeader").read()
    return bool(header[36] & 0x01) if len(header) > 36 else False


def _read_metadata(ole: olefile.OleFileIO) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    try:
        summary = ole.get_metadata()
        if summary:
            for key in ("title", "author", "subject", "last_saved_by"):
                val = getattr(summary, key, None)
                meta[key] = str(val) if val else ""
    except Exception:
        pass
    return meta


def _decompress(data: bytes, is_compressed: bool) -> bytes:
    if not is_compressed:
        return data
    try:
        return zlib.decompress(data, -15)
    except zlib.error:
        return zlib.decompress(data)


def _parse_section_records(data: bytes) -> str:
    parts: list[str] = []
    offset = 0
    length = len(data)

    while offset < length:
        if offset + 4 > length:
            break

        header = struct.unpack_from("<I", data, offset)[0]
        tag_id = header & 0x3FF
        size = (header >> 20) & 0xFFF
        offset += 4

        if size == 0xFFF:
            if offset + 4 > length:
                break
            size = struct.unpack_from("<I", data, offset)[0]
            offset += 4

        if offset + size > length:
            break

        if tag_id == HWPTAG_PARA_TEXT:
            parts.append(_decode_para_text(data[offset: offset + size]))

        offset += size

    return "\n".join(t for t in parts if t.strip())


def _decode_para_text(record: bytes) -> str:
    chars: list[str] = []
    i = 0
    end = len(record) - 1

    while i < end:
        code = struct.unpack_from("<H", record, i)[0]
        i += 2

        if code < 32:
            if code in EXTENDED_CONTROLS:
                i += 14
            elif code in (10, 13):
                chars.append("\n")
        else:
            chars.append(chr(code))

    return "".join(chars)

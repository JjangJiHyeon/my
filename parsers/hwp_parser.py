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

# ── parser ─────────────────────────────────────────────────────────

HWPTAG_PARA_TEXT = 67

EXTENDED_CONTROLS = frozenset(
    {1, 2, 3, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
)


# ── text normalisation ───────────────────────────────────────────────

def _analyze_narrative_characteristics(text: str) -> dict[str, Any]:
    import re
    clean = text.replace(" ", "").replace("\n", "")
    num_ratio = len(re.findall(r'\d', clean)) / max(1, len(clean))
    sentences = len(re.findall(r'[가-힣][다음함]\.', text))
    
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    long_paras = sum(1 for p in paragraphs if len(p) > 60 and ('다' in p or '음' in p or '함' in p))
    label_patterns = sum(1 for p in paragraphs if len(p) < 30 and re.search(r'[:\-\u2022\[\]]', p))
    
    return {
        "num_ratio": round(num_ratio, 3),
        "sentence_endings_found": sentences,
        "long_narrative_paragraphs": long_paras,
        "short_label_patterns": label_patterns,
        "is_false_positive_warning_candidate": bool(num_ratio > 0.25 and sentences <= 1 and long_paras > 0),
        "is_truly_dashboard": bool(num_ratio > 0.25 and sentences <= 1 and long_paras == 0)
    }

def _normalise(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── section header splitting ─────────────────────────────────────────

# 줄 전체가 독립 헤더일 때만 매치 (anchored)
_SECTION_HEADER_RE = re.compile(
    r'^\s*'
    r'(?:'
    r'[Ⅰ-Ⅹ]'                           # 로마자 Ⅰ~Ⅹ
    r'|V?I{0,3}'                         # ASCII 로마자 I, II, III, IV, V, VI
    r'|[0-9]{1,2}'                       # 아라비아 숫자 (최대 2자리)
    r'|[가-힣]'                           # 한글 가, 나, 다
    r'|[A-Z]'                            # 영문 대문자 A, B, C
    r')'
    r'[\.\)]\s*'                         # 점 또는 닫는 괄호 + 선택적 공백
    r'(.+)',                             # 뒤에 텍스트가 있어야 함
    re.MULTILINE
)

# 날짜/표성 숫자 라인은 제외
_DATE_LINE_RE = re.compile(
    r'^\s*\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2}'   # 2022.01.19 형태
)
_PURE_NUMBER_RE = re.compile(
    r'^\s*[\d,.\-+%()△▲▽※ ]+\s*$'           # 숫자/부호만으로 구성
)


def _is_doc_section_header(line: str) -> bool:
    """줄 전체가 독립적인 섹션 헤더인지 판단"""
    stripped = line.strip()
    # 너무 긴 라인은 헤더가 아님
    if len(stripped) > 60:
        return False
    # 날짜 라인 제외
    if _DATE_LINE_RE.match(stripped):
        return False
    # 순수 숫자/기호 라인 제외
    if _PURE_NUMBER_RE.match(stripped):
        return False
    # 섹션 패턴 매치
    if _SECTION_HEADER_RE.match(stripped):
        return True
    return False


def _is_data_row(line: str) -> bool:
    """순수 숫자/금액 행인지 판단"""
    stripped = line.strip()
    if not stripped:
        return False
    if _PURE_NUMBER_RE.match(stripped):
        return True
    return False


def _split_into_section_blocks(text: str, page_num: int) -> list[dict[str, Any]]:
    """텍스트를 섹션 헤더 기준으로 title/text 블록으로 분리.
    빈 줄 기준 body 분리 + 순수 숫자 데이터 행 분리.
    원문 순서 무변경, 텍스트 내용 무변경."""
    if not text or not text.strip():
        return []

    lines = text.split('\n')
    blocks: list[dict[str, Any]] = []
    block_idx = 0
    current_body_lines: list[str] = []
    current_data_lines: list[str] = []

    def _flush_body():
        nonlocal block_idx
        body_text = '\n'.join(current_body_lines).strip()
        if body_text:
            blocks.append({
                "id": f"p{page_num}_b{block_idx}",
                "type": "text",
                "bbox": [0, 0, 0, 0],
                "text": body_text,
                "page_num": page_num,
                "source": "hwp_parser",
                "score": 1.0,
                "meta": {}
            })
            block_idx += 1
        current_body_lines.clear()

    def _flush_data():
        nonlocal block_idx
        data_text = '\n'.join(current_data_lines).strip()
        if data_text:
            blocks.append({
                "id": f"p{page_num}_b{block_idx}",
                "type": "text",
                "bbox": [0, 0, 0, 0],
                "text": data_text,
                "page_num": page_num,
                "source": "hwp_parser",
                "score": 1.0,
                "meta": {"block_subtype": "data_rows"}
            })
            block_idx += 1
        current_data_lines.clear()

    for line in lines:
        stripped = line.strip()
        is_header = _is_doc_section_header(line)
        is_data = _is_data_row(line)
        # 목록 기호 시작 판정 (□, ◦, -, 1), Ⅰ., 가. 등)
        is_list_item = len(stripped) < 200 and re.match(r'^\s*([□◦\-\u25e6\u2022\u25a1\u25cb\u2023\u25a0]|[\d]{1,2}[\.\)]|[Ⅰ-Ⅹ][\.\)]|[가-힣][\.\)])', line)

        if is_header:
            _flush_data()
            _flush_body()
            blocks.append({
                "id": f"p{page_num}_b{block_idx}",
                "type": "title",
                "bbox": [0, 0, 0, 0],
                "text": line.strip(),
                "page_num": page_num,
                "source": "hwp_parser",
                "score": 1.0,
                "meta": {}
            })
            block_idx += 1
        elif not stripped:
            # 빈 줄: 블록 단절 생성
            _flush_data()
            _flush_body()
        elif is_list_item:
            # 목록 아이템 시작: 이전 버퍼 비우고 새 블록 유도
            _flush_data()
            _flush_body()
            if is_data:
                current_data_lines.append(line)
            else:
                current_body_lines.append(line)
        elif is_data:
            # 데이터 행: 바디 버퍼가 있으면 비우고 데이터 버퍼로
            if current_body_lines:
                _flush_body()
            current_data_lines.append(line)
        else:
            # 일반 텍스트: 데이터 버퍼가 있으면 비우고 바디 버퍼로
            if current_data_lines:
                _flush_data()
            current_body_lines.append(line)

    _flush_data()
    _flush_body()
    return blocks

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
        section_blocks = _split_into_section_blocks(text, i + 1)
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
            "blocks": section_blocks,
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
                    "final_blocks": len(section_blocks)
                },
                "block_type_counts": {
                    "title": sum(1 for b in section_blocks if b["type"] == "title"),
                    "text": sum(1 for b in section_blocks if b["type"] == "text")
                },
                "dropped_blocks": [],
                "bbox_warnings": [],
                "narrative_analysis": _analyze_narrative_characteristics(text)
            }
        })

    metadata.update({
        "parser_used": "custom OLE HWP parser",
        "page_count": len(pages),
        "section_count": section_idx,
        "empty_sections": empty_sections,
    })
    
    if pages:
        metadata["narrative_analysis"] = pages[0]["parser_debug"]["narrative_analysis"]

    if not pages:
        return {
            "pages": [],
            "metadata": metadata,
            "status": "error",
            "error": f"No text extracted from {section_idx} section(s) ({empty_sections} empty)",
        }

    # ── Document Routing ──────────────────────────────
    from .document_router import route_document
    routing_result = route_document(None, pages, metadata=metadata)
    
    doc_type = routing_result.get("document_type", "text_report")
    metadata.update({
        "document_type": doc_type,
        "routing_signals": routing_result.get("routing_signals"),
        "routing_reason": routing_result.get("routing_reason")
    })
    
    for p in pages:
        p.setdefault("parser_debug", {})["document_type"] = doc_type

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

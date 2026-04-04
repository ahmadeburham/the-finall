import argparse
import json
import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from paddleocr import PaddleOCR

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_OCR = None
AR_NUM_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
TO_AR_NUM_MAP = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
EXPIRY_LABEL = "البطاقه ساريه حتي"
EXPIRY_LABEL_TOKENS = ["البطاقه", "ساريه", "حتي"]
ARABIC_TEXT_RE = re.compile(r"[\u0600-\u06FF]")
RLI = "\u2067"
PDI = "\u2069"

# ── Detailed Tracing Logger ─────────────────────────────────────────────
_tracer_logger = None

def setup_tracer(log_file: str = None) -> logging.Logger:
    global _tracer_logger
    if _tracer_logger is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_file or f"ocr_trace_{timestamp}.log"
        _tracer_logger = logging.getLogger("ocr_tracer")
        _tracer_logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(log_path, encoding="utf-8", mode="w")
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(funcName)-25s | %(message)s",
            datefmt="%H:%M:%S"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        _tracer_logger.addHandler(fh)
        _tracer_logger.addHandler(ch)
        
        _tracer_logger.info("=== OCR TRACING STARTED ===")
    return _tracer_logger

def trace(func_name: str, msg: str, level: str = "debug"):
    logger = setup_tracer()
    getattr(logger, level)(f"[{func_name}] {msg}")

def trace_var(func_name: str, var_name: str, value, level: str = "debug"):
    logger = setup_tracer()
    if isinstance(value, (list, tuple)):
        logger.log(getattr(logging, level.upper()), f"[{func_name}] {var_name}: len={len(value)} | value={str(value)[:200]}")
    elif isinstance(value, dict):
        logger.log(getattr(logging, level.upper()), f"[{func_name}] {var_name}: keys={list(value.keys())[:5]} | items={len(value)}")
    else:
        logger.log(getattr(logging, level.upper()), f"[{func_name}] {var_name}: {str(value)[:200]}")


def build_ocr():
    global _OCR
    if _OCR is None:
        _OCR = PaddleOCR(
            lang="ar",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    return _OCR


def sort_key(item: dict):
    box = item["box"]
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    cy = sum(ys) / len(ys)
    cx = sum(xs) / len(xs)
    return (round(cy / 18), -cx)


def extract_lines_from_predict_result(res_obj):
    data = res_obj
    if isinstance(data, dict) and "res" in data:
        data = data["res"]
    texts = data.get("rec_texts", []) or []
    scores = data.get("rec_scores", []) or []
    polys = data.get("rec_polys", []) or data.get("dt_polys", []) or []

    lines = []
    for i, text in enumerate(texts):
        text = str(text).strip()
        if not text:
            continue
        score = None
        if i < len(scores):
            try:
                score = float(scores[i])
            except Exception:
                score = None
        box = [[0, 0], [0, 0], [0, 0], [0, 0]]
        if i < len(polys):
            try:
                box = [[int(round(p[0])), int(round(p[1]))] for p in polys[i]]
            except Exception:
                pass
        lines.append({"text": text, "score": score, "box": box})

    lines.sort(key=sort_key)
    return lines


def normalize_arabic_text(text: str) -> str:
    text = text.replace("ـ", "")
    text = text.replace("\u200f", " ").replace("\u200e", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def format_text_for_json(text: str) -> str:
    if not isinstance(text, str):
        return text
    if not text or not ARABIC_TEXT_RE.search(text):
        return text
    if text.startswith(RLI) and text.endswith(PDI):
        return text
    return f"{RLI}{text}{PDI}"


def prepare_for_json_output(value):
    if isinstance(value, dict):
        return {str(k): prepare_for_json_output(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [prepare_for_json_output(v) for v in value]
    if isinstance(value, str):
        return format_text_for_json(value)
    return value


def normalize_arabic_for_match(text: str) -> str:
    text = normalize_arabic_text(text)
    text = normalize_digits(text)
    text = (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ة", "ه")
        .replace("ى", "ي")
        .replace("ؤ", "و")
        .replace("ئ", "ي")
    )
    text = re.sub(r"[^\u0600-\u06FF0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_digits(text: str) -> str:
    return text.translate(AR_NUM_MAP)


def to_arabic_digits(text: str) -> str:
    return normalize_digits(text).translate(TO_AR_NUM_MAP)


def only_digits(text: str) -> str:
    return re.sub(r"\D+", "", normalize_digits(text))


def ids_match_strict(front_id: str, back_id: str) -> bool:
    front_digits = only_digits(front_id)
    back_digits = only_digits(back_id)
    return len(front_digits) == 14 and len(back_digits) == 14 and front_digits == back_digits


def _resize_up(image_bgr: np.ndarray, target_short: int = 320) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    short_side = min(h, w)
    if short_side >= target_short:
        return image_bgr
    scale = target_short / max(short_side, 1)
    scale = min(scale, 4.0)
    return cv2.resize(
        image_bgr,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_CUBIC,
    )


def preprocess_text_block(image_bgr: np.ndarray) -> List[np.ndarray]:
    variants = []
    up = _resize_up(image_bgr, target_short=360)
    variants.append(up)

    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    variants.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    blur = cv2.GaussianBlur(gray, (0, 0), 1.5)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
    variants.append(cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR))

    bilateral = cv2.bilateralFilter(gray, 7, 50, 50)
    variants.append(cv2.cvtColor(bilateral, cv2.COLOR_GRAY2BGR))
    return variants




def _ocr_lines_from_image(image_bgr: np.ndarray) -> List[dict]:
    ocr = build_ocr()
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_name = tmp.name
        cv2.imwrite(tmp_name, image_bgr)
        results = list(ocr.predict(tmp_name))

        lines = []
        for i, res in enumerate(results):
            data = None
            try:
                data = res.json
                if callable(data):
                    data = data()
            except Exception:
                data = None

            if data is None:
                try:
                    sidecar = f"{tmp_name}.{i}.json"
                    res.save_to_json(sidecar)
                    data = json.loads(Path(sidecar).read_text(encoding="utf-8"))
                    Path(sidecar).unlink(missing_ok=True)
                except Exception:
                    data = None

            if data is not None:
                lines.extend(extract_lines_from_predict_result(data))

        lines.sort(key=sort_key)
        return lines
    finally:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except Exception:
                pass


def _score_text(lines: List[dict]) -> float:
    joined = normalize_arabic_text(" ".join(x["text"] for x in lines))
    ar_letters = len(re.findall(r"[\u0600-\u06FF]", joined))
    avg_conf = float(np.mean([x["score"] for x in lines if isinstance(x.get("score"), (int, float))])) if lines else 0.0
    return ar_letters + avg_conf * 10.0




def merge_lines(lines: List[dict]) -> str:
    ordered = sorted(lines, key=sort_key)
    joined = " ".join(x["text"] for x in ordered)
    return normalize_arabic_text(joined)


def numeric_field(lines: List[dict], prefer_len: int | None = None) -> str:
    # Extract digit candidates from lines
    joined = normalize_digits(" ".join(x["text"] for x in lines))
    joined = joined.replace("/", " ").replace("-", " ").replace(".", " ").replace("_", " ")
    parts = re.findall(r"\d+", joined)

    candidates = []
    for p in parts:
        if p:
            candidates.append(p)

    merged = only_digits(joined)
    if merged:
        candidates.append(merged)

    # Remove duplicates while preserving order
    seen = set()
    uniq = []
    for x in candidates:
        if x not in seen:
            uniq.append(x)
            seen.add(x)

    # Choose best candidate
    if not uniq:
        return ""
    if prefer_len is not None:
        for c in sorted(uniq, key=len, reverse=True):
            if len(c) == prefer_len:
                return c
        for c in sorted(uniq, key=len, reverse=True):
            m = re.search(rf"\d{{{prefer_len}}}", c)
            if m:
                return m.group(0)
    return sorted(uniq, key=len, reverse=True)[0]


def _best_lines_for_text(image_bgr: np.ndarray) -> Tuple[List[dict], int, float]:
    trace("_best_lines_for_text", "STARTED")
    best_lines = []
    best_score = -1e18
    best_idx = -1
    for idx, variant in enumerate(preprocess_text_block(image_bgr)):
        trace_var("_best_lines_for_text", f"variant_{idx}_shape", variant.shape)
        lines = _ocr_lines_from_image(variant)
        trace_var("_best_lines_for_text", f"variant_{idx}_lines", lines)
        score = _score_text(lines)
        trace_var("_best_lines_for_text", f"variant_{idx}_score", score)
        if score > best_score:
            best_score = score
            best_lines = lines
            best_idx = idx
            trace("_best_lines_for_text", f"New best variant {idx}")
    trace_var("_best_lines_for_text", "best_idx", best_idx)
    trace_var("_best_lines_for_text", "best_score", best_score)
    return best_lines, best_idx, float(best_score)


def _best_numeric_from_crop(image_bgr: np.ndarray, prefer_len: int | None = None) -> Tuple[str, List[dict], int, float]:
    trace("_best_numeric_from_crop", f"STARTED prefer_len={prefer_len}")
    # Use the same text detection pipeline as _best_lines_for_text
    best_lines, best_idx, best_score = _best_lines_for_text(image_bgr)
    trace_var("_best_numeric_from_crop", "best_lines", best_lines)
    trace_var("_best_numeric_from_crop", "best_idx", best_idx)
    trace_var("_best_numeric_from_crop", "best_score", best_score)
    # Extract numeric result from those lines
    best_text = numeric_field(best_lines, prefer_len=prefer_len)
    trace_var("_best_numeric_from_crop", "best_text", best_text)
    return best_text, best_lines, best_idx, float(best_score)


def infer_birth_date_from_id(id_number: str) -> str:
    id_number = only_digits(id_number)
    if len(id_number) < 7:
        return ""
    century = id_number[0]
    yy = id_number[1:3]
    mm = id_number[3:5]
    dd = id_number[5:7]
    if century not in {"2", "3"}:
        return ""
    if not (1 <= int(mm) <= 12 and 1 <= int(dd) <= 31):
        return ""
    year = (1900 if century == "2" else 2000) + int(yy)
    return f"{year:04d}{mm}{dd}"


def is_valid_yyyymmdd(digits: str) -> bool:
    digits = only_digits(digits)
    if len(digits) != 8:
        return False
    year = int(digits[:4])
    month = int(digits[4:6])
    day = int(digits[6:8])
    if year < 1900 or year > 2100:
        return False
    if month < 1 or month > 12:
        return False
    if day < 1 or day > 31:
        return False
    return True


def _extract_yyyymmdd_ascii(text: str) -> str:
    text = normalize_digits(text)
    patterns = [
        r"(\d{4})\s*[/\\\-.]\s*(\d{2})\s*[/\\\-.]\s*(\d{2})",
        r"(\d{4})\s*(\d{2})\s*(\d{2})",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            value = "".join(m.groups())
            if is_valid_yyyymmdd(value):
                return value
    digits = only_digits(text)
    if len(digits) >= 8:
        for i in range(0, len(digits) - 7):
            chunk = digits[i:i + 8]
            if is_valid_yyyymmdd(chunk):
                return chunk
    return ""


def _line_height(line: dict) -> float:
    box = line.get("box") or []
    if not box:
        return 0.0
    ys = [p[1] for p in box]
    return float(max(ys) - min(ys))


def _line_center_y(line: dict) -> float:
    box = line.get("box") or []
    if not box:
        return 0.0
    ys = [p[1] for p in box]
    return float(sum(ys) / len(ys))


def _expiry_label_score(text: str) -> int:
    norm = normalize_arabic_for_match(text)
    score = 0
    for token in EXPIRY_LABEL_TOKENS:
        if token in norm:
            score += 1
    return score


def _lines_for_candidate(lines: List[dict], idx: int) -> List[dict]:
    selected = [lines[idx]]
    current_y = _line_center_y(lines[idx])
    current_h = max(_line_height(lines[idx]), 1.0)
    for j in range(idx + 1, min(idx + 3, len(lines))):
        next_y = _line_center_y(lines[j])
        if abs(next_y - current_y) <= current_h * 1.3:
            selected.append(lines[j])
        else:
            break
    return selected


def extract_expiry_date_from_lines(lines: List[dict]) -> Dict:
    best = {
        "matched": False,
        "expiry_date_ascii": "",
        "expiry_date": "",
        "matched_text": "",
        "matched_lines": [],
        "label_score": 0,
    }
    for idx, line in enumerate(lines):
        label_score = _expiry_label_score(line.get("text", ""))
        if label_score <= 0:
            continue
        candidate_lines = _lines_for_candidate(lines, idx)
        candidate_text = merge_lines(candidate_lines)
        expiry_ascii = _extract_yyyymmdd_ascii(candidate_text)
        candidate = {
            "matched": label_score >= 2 and bool(expiry_ascii),
            "expiry_date_ascii": expiry_ascii,
            "expiry_date": to_arabic_digits(expiry_ascii) if expiry_ascii else "",
            "matched_text": candidate_text,
            "matched_lines": candidate_lines,
            "label_score": label_score,
        }
        if (candidate["matched"], candidate["label_score"], len(candidate["expiry_date_ascii"])) > (
            best["matched"],
            best["label_score"],
            len(best["expiry_date_ascii"]),
        ):
            best = candidate
    return best


def debug_expiry_date_from_crop(image_bgr: np.ndarray) -> Dict:
    best_info = {
        "field_name": "expiry_date",
        "best_variant_index": -1,
        "best_score": -1e18,
        "best_text": "",
        "best_lines": [],
        "best_matched_text": "",
        "variants": [],
    }

    for idx, variant in enumerate(preprocess_text_block(image_bgr)):
        lines = _ocr_lines_from_image(variant)
        parsed = extract_expiry_date_from_lines(lines)
        avg_conf = float(np.mean([x["score"] for x in lines if isinstance(x.get("score"), (int, float))])) if lines else 0.0
        score = float(parsed["matched"]) * 100.0 + parsed["label_score"] * 10.0 + len(parsed["expiry_date_ascii"]) + avg_conf * 5.0
        variant_info = {
            "variant_index": idx,
            "score": score,
            "text": parsed["expiry_date"],
            "matched_text": parsed["matched_text"],
            "matched": parsed["matched"],
            "label_score": parsed["label_score"],
            "lines": lines,
            "matched_lines": parsed["matched_lines"],
        }
        best_info["variants"].append(variant_info)

        if score > best_info["best_score"]:
            best_info["best_variant_index"] = idx
            best_info["best_score"] = score
            best_info["best_text"] = parsed["expiry_date"]
            best_info["best_lines"] = parsed["matched_lines"]
            best_info["best_matched_text"] = parsed["matched_text"]

    return best_info


def _best_expiry_from_crop(image_bgr: np.ndarray) -> Tuple[str, List[dict], int, float]:
    info = debug_expiry_date_from_crop(image_bgr)
    return (
        info.get("best_text", ""),
        info.get("best_lines", []),
        int(info.get("best_variant_index", -1)),
        float(info.get("best_score", -1e18)),
    )


def extract_text_from_image(input_path: str) -> dict:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    if path.suffix.lower() not in IMAGE_EXTS:
        raise ValueError(f"Unsupported image type: {path.suffix}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")

    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    lines, _, _ = _best_lines_for_text(image)
    full_text = merge_lines(lines)
    return {"full_text": full_text, "lines": lines}


def extract_fields_from_crops(front_crops: Dict[str, np.ndarray], back_crops: Dict[str, np.ndarray] | None = None) -> Dict:
    trace("extract_fields_from_crops", "STARTED")
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    
    trace_var("extract_fields_from_crops", "front_crops_keys", list(front_crops.keys()))
    trace_var("extract_fields_from_crops", "back_crops_keys", list(back_crops.keys()) if back_crops else [])

    # ── front side ────────────────────────────────────────────────────────────
    trace("extract_fields_from_crops", "Processing front side fields")
    name_lines, _, _ = _best_lines_for_text(front_crops["name"]) if "name" in front_crops else ([], -1, 0.0)
    trace_var("extract_fields_from_crops", "name_lines", name_lines)
    
    address_lines, _, _ = _best_lines_for_text(front_crops["address"]) if "address" in front_crops else ([], -1, 0.0)
    trace_var("extract_fields_from_crops", "address_lines", address_lines)
    
    id_number, id_lines, _, _ = _best_numeric_from_crop(front_crops["id_number"], prefer_len=14) if "id_number" in front_crops else ("", [], -1, 0.0)
    trace_var("extract_fields_from_crops", "id_number_initial", id_number)
    trace_var("extract_fields_from_crops", "id_lines", id_lines)
    
    birth_date, birth_lines, _, _ = _best_numeric_from_crop(front_crops["birth_date"], prefer_len=8) if "birth_date" in front_crops else ("", [], -1, 0.0)
    trace_var("extract_fields_from_crops", "birth_date_initial", birth_date)

    name = merge_lines(name_lines)
    address = merge_lines(address_lines)

    # FIX 1a: century correction — the front card's brown stain makes the first
    # digit unreliable.  If the implied birth year is implausible (> 100 years
    # ago or in the future), try alternative century codes until we find one that
    # places the birth year within a plausible 0-100 year window.
    import datetime as _dt
    _id_partial = only_digits(id_number)
    trace_var("extract_fields_from_crops", "_id_partial", _id_partial)
    
    if 7 <= len(_id_partial):
        _bd0 = infer_birth_date_from_id(_id_partial)
        trace_var("extract_fields_from_crops", "_bd0", _bd0)
        if _bd0:
            _age0 = _dt.datetime.now().year - int(_bd0[:4])
            trace_var("extract_fields_from_crops", "_age0", _age0)
            if not (0 <= _age0 <= 100):
                trace("extract_fields_from_crops", "Age invalid, trying century correction")
                for _c in ("3", "2", "1"):
                    _alt = _c + _id_partial[1:]
                    trace_var("extract_fields_from_crops", f"trying_century_{_c}", _alt)
                    _alt_bd = infer_birth_date_from_id(_alt)
                    if _alt_bd:
                        _alt_age = _dt.datetime.now().year - int(_alt_bd[:4])
                        trace_var("extract_fields_from_crops", f"alt_age_{_c}", _alt_age)
                        if 0 <= _alt_age <= 100:
                            id_number = _alt
                            trace("extract_fields_from_crops", f"Century corrected to {_c}")
                            break
    trace_var("extract_fields_from_crops", "id_number_after_century_fix", id_number)

    front_id_number = only_digits(id_number)

    # ── back side ─────────────────────────────────────────────────────────────
    back_full_text_parts = []
    expiry_date = ""
    expiry_lines: List[dict] = []
    issue_date = ""
    profession = ""
    gender = ""
    religion = ""
    marital_status = ""
    back_id_number = ""
    back_id_lines: List[dict] = []

    if back_crops:
        trace("extract_fields_from_crops", "Processing back side fields")

        # ── Back ID: fully independent extraction ─────────────────────────────
        # Extract the back-side ID number completely on its own (prefer_len=14).
        # This is NOT conditional on what the front extracted.
        if "back_id_number" in back_crops:
            _bi = back_crops["back_id_number"]
            # Full-crop pass first
            back_id_number, back_id_lines, _, _ = _best_numeric_from_crop(_bi, prefer_len=14)
            trace_var("extract_fields_from_crops", "back_id_full_crop", back_id_number)

            # Right-edge pass (avoids Tutankhamun watermark in centre)
            _bw = _bi.shape[1]
            _rs = _bi[:, int(_bw * 0.75):]
            _rs_up = cv2.resize(_rs, (_rs.shape[1] * 3, _rs.shape[0] * 3),
                                 interpolation=cv2.INTER_CUBIC)
            _rs_text, _rs_lines_r, _, _ = _best_numeric_from_crop(_rs_up, prefer_len=14)
            trace_var("extract_fields_from_crops", "back_id_right_edge", _rs_text)

            # Pick whichever pass gave more digits (14-digit wins outright)
            _back_full_d  = only_digits(back_id_number)
            _back_edge_d  = only_digits(_rs_text)
            if len(_back_edge_d) == 14:
                back_id_number = _back_edge_d
                back_id_lines  = _rs_lines_r
            elif len(_back_full_d) == 14:
                pass  # already set
            elif len(_back_edge_d) > len(_back_full_d):
                back_id_number = _back_edge_d
                back_id_lines  = _rs_lines_r

        trace_var("extract_fields_from_crops", "back_id_number_independent", back_id_number)

        # ── Reconcile front vs back ID candidates ─────────────────────────────
        _front_d = front_id_number
        _back_d  = only_digits(back_id_number)
        trace_var("extract_fields_from_crops", "front_id_digits", _front_d)
        trace_var("extract_fields_from_crops", "back_id_digits",  _back_d)

        if len(_back_d) == 14 and len(_front_d) != 14:
            id_number = _back_d
            id_lines  = back_id_lines
            trace("extract_fields_from_crops", "Using back ID (back=14, front!=14)")
        elif len(_front_d) == 14 and len(_back_d) != 14:
            trace("extract_fields_from_crops", "Using front ID (front=14, back!=14)")
        elif len(_back_d) == 14 and len(_front_d) == 14:
            trace("extract_fields_from_crops", "Both sides gave 14 digits — keeping front ID")
        elif len(_back_d) > len(_front_d):
            id_number = _back_d
            id_lines  = back_id_lines
            trace("extract_fields_from_crops", f"Using back ID (longer: {len(_back_d)} vs {len(_front_d)})")
        else:
            trace("extract_fields_from_crops", f"Keeping front ID (longer or equal: {len(_front_d)} vs {len(_back_d)})")

        trace_var("extract_fields_from_crops", "id_number_after_reconcile", id_number)

        # NEW: extract dedicated back-card text fields
        if "back_issue_date" in back_crops:
            issue_date_lines, _, _ = _best_lines_for_text(back_crops["back_issue_date"])
            _raw_issue = normalize_digits(merge_lines(issue_date_lines))
            # primary: actual separator (/, -, \)
            _im = re.search(r"((?:19|20)\d{2})\s*[/\\\-]\s*(\d{1,2})", _raw_issue)
            if not _im:
                # fallback: '/' is commonly OCR'd as '1' on low-contrast crops
                _im = re.search(r"((?:19|20)\d{2})[1l](\d{2})", _raw_issue)
            if _im:
                _iy, _imm = _im.group(1), _im.group(2).zfill(2)
                if 1 <= int(_imm) <= 12:
                    issue_date = to_arabic_digits(_iy) + "/" + to_arabic_digits(_imm)
        if "profession" in back_crops:
            profession_lines, _, _ = _best_lines_for_text(back_crops["profession"])
            profession = merge_lines(profession_lines)
        if "gender" in back_crops:
            gender_lines, _, _ = _best_lines_for_text(back_crops["gender"])
            gender = merge_lines(gender_lines)
            gender = re.sub(r"[دنر]كر", "ذكر", gender)
        if "religion" in back_crops:
            religion_lines, _, _ = _best_lines_for_text(back_crops["religion"])
            religion = merge_lines(religion_lines)
        if "marital_status" in back_crops:
            marital_lines, _, _ = _best_lines_for_text(back_crops["marital_status"])
            marital_status = merge_lines(marital_lines)

        # build back_full_text from back_text fallback crop
        for crop_name, crop in back_crops.items():
            if crop_name != "back_text":
                continue
            lines, _, _ = _best_lines_for_text(crop)
            txt = merge_lines(lines)
            if txt:
                back_full_text_parts.append(txt)

        expiry_source = back_crops.get("expiry_date")
        if expiry_source is None:
            expiry_source = back_crops.get("back_text")
        if expiry_source is not None:
            expiry_date, expiry_lines, _, _ = _best_expiry_from_crop(expiry_source)

    # FIX 3: cross-validate expiry with back_full_text (text pipeline is more
    # reliable than numeric pipeline for mixed Arabic+digit lines like the expiry row).
    back_full_text_str = normalize_arabic_text(" ".join(back_full_text_parts))
    ft_expiry = _extract_yyyymmdd_ascii(back_full_text_str)
    if ft_expiry:
        expiry_date = to_arabic_digits(ft_expiry)

    # FIX 2: if birth_date is absent OR not a valid YYYYMMDD (holographic foil
    # makes the front ROI unreadable), infer it from the national ID number.
    # Fix 1 must run first so we use the corrected 14-digit ID here.
    trace_var("extract_fields_from_crops", "birth_date_before_inference", birth_date)
    if (not birth_date or not is_valid_yyyymmdd(birth_date)) and id_number:
        trace("extract_fields_from_crops", "Birth date invalid/missing, inferring from ID")
        inferred = infer_birth_date_from_id(id_number)
        trace_var("extract_fields_from_crops", "birth_date_inferred", inferred)
        if inferred:
            birth_date = inferred
            trace("extract_fields_from_crops", "Birth date inferred successfully")

    front_full_text = normalize_arabic_text(" ".join(filter(None, [name, address, id_number, birth_date])))
    
    trace("extract_fields_from_crops", "COMPLETED - returning results")
    trace_var("extract_fields_from_crops", "final_id_number", id_number)
    trace_var("extract_fields_from_crops", "final_birth_date", birth_date)

    return {
        "name": name,
        "address": address,
        "id_number": id_number,
        "front_id_number": front_id_number,
        "back_id_number": only_digits(back_id_number),
        "birth_date": birth_date,
        "expiry_date": expiry_date,
        "issue_date": issue_date,
        "profession": profession,
        "gender": gender,
        "religion": religion,
        "marital_status": marital_status,
        "front_full_text": front_full_text,
        "back_full_text": back_full_text_str,
        "raw": {
            "name_lines": name_lines,
            "address_lines": address_lines,
            "id_number_lines": id_lines,
            "birth_date_lines": birth_lines,
            "expiry_date_lines": expiry_lines,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", default="ocr_output")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = extract_text_from_image(args.input)
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "full_text.txt").write_text(result["full_text"], encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

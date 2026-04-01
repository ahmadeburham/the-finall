import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from paddleocr import PaddleOCR

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

_OCR = None
AR_NUM_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


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


def normalize_digits(text: str) -> str:
    return text.translate(AR_NUM_MAP)


def only_digits(text: str) -> str:
    return re.sub(r"\D+", "", normalize_digits(text))


def _resize_up(image_bgr: np.ndarray, target_short: int = 320) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    short_side = min(h, w)
    if short_side >= target_short:
        return image_bgr
    scale = target_short / max(short_side, 1)
    scale = min(scale, 4.0)
    return cv2.resize(image_bgr, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_CUBIC)


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


def preprocess_numeric_block(image_bgr: np.ndarray) -> List[np.ndarray]:
    variants = []

    up = _resize_up(image_bgr, target_short=260)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    variants.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    variants.append(cv2.cvtColor(blackhat, cv2.COLOR_GRAY2BGR))

    _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(th_otsu, cv2.COLOR_GRAY2BGR))

    _, th_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(th_inv, cv2.COLOR_GRAY2BGR))

    adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
    variants.append(cv2.cvtColor(adap, cv2.COLOR_GRAY2BGR))

    adap_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 11)
    variants.append(cv2.cvtColor(adap_inv, cv2.COLOR_GRAY2BGR))

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


def _extract_digit_candidates(lines: List[dict]) -> List[str]:
    joined = normalize_digits(" ".join(x["text"] for x in lines))
    joined = joined.replace("/", " ").replace("-", " ").replace(".", " ").replace("_", " ")
    parts = re.findall(r"\d+", joined)
    out = []
    for p in parts:
        if p:
            out.append(p)
    merged = only_digits(joined)
    if merged:
        out.append(merged)
    # dedupe preserving order
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def _choose_numeric_candidate(candidates: List[str], prefer_len: int | None = None) -> str:
    if not candidates:
        return ""
    if prefer_len is not None:
        for c in sorted(candidates, key=len, reverse=True):
            if len(c) == prefer_len:
                return c
        for c in sorted(candidates, key=len, reverse=True):
            m = re.search(rf"\d{{{prefer_len}}}", c)
            if m:
                return m.group(0)
    return sorted(candidates, key=len, reverse=True)[0]


def _score_numeric(lines: List[dict], prefer_len: int | None = None) -> float:
    candidates = _extract_digit_candidates(lines)
    best = _choose_numeric_candidate(candidates, prefer_len=prefer_len)
    digit_count = len(best)
    avg_conf = float(np.mean([x["score"] for x in lines if isinstance(x.get("score"), (int, float))])) if lines else 0.0
    length_bonus = 10.0 if prefer_len is not None and len(best) == prefer_len else 0.0
    return digit_count * 3.0 + avg_conf * 10.0 + length_bonus


def merge_lines(lines: List[dict]) -> str:
    ordered = sorted(lines, key=sort_key)
    joined = " ".join(x["text"] for x in ordered)
    return normalize_arabic_text(joined)


def numeric_field(lines: List[dict], prefer_len: int | None = None) -> str:
    return _choose_numeric_candidate(_extract_digit_candidates(lines), prefer_len=prefer_len)


def _best_lines_for_text(image_bgr: np.ndarray) -> Tuple[List[dict], int, float]:
    best_lines = []
    best_score = -1e18
    best_idx = -1
    for idx, variant in enumerate(preprocess_text_block(image_bgr)):
        lines = _ocr_lines_from_image(variant)
        score = _score_text(lines)
        if score > best_score:
            best_score = score
            best_lines = lines
            best_idx = idx
    return best_lines, best_idx, float(best_score)


def _best_numeric_from_crop(image_bgr: np.ndarray, prefer_len: int | None = None) -> Tuple[str, List[dict], int, float]:
    best_text = ""
    best_lines = []
    best_score = -1e18
    best_idx = -1
    for idx, variant in enumerate(preprocess_numeric_block(image_bgr)):
        lines = _ocr_lines_from_image(variant)
        score = _score_numeric(lines, prefer_len=prefer_len)
        text = numeric_field(lines, prefer_len=prefer_len)
        if score > best_score:
            best_score = score
            best_text = text
            best_lines = lines
            best_idx = idx
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
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    name_lines, _, _ = _best_lines_for_text(front_crops["name"]) if "name" in front_crops else ([], -1, 0.0)
    address_lines, _, _ = _best_lines_for_text(front_crops["address"]) if "address" in front_crops else ([], -1, 0.0)

    id_number, id_lines, _, _ = _best_numeric_from_crop(front_crops["id_number"], prefer_len=14) if "id_number" in front_crops else ("", [], -1, 0.0)
    birth_date, birth_lines, _, _ = _best_numeric_from_crop(front_crops["birth_date"], prefer_len=8) if "birth_date" in front_crops else ("", [], -1, 0.0)

    name = merge_lines(name_lines)
    address = merge_lines(address_lines)

    if not birth_date and id_number:
        birth_date = infer_birth_date_from_id(id_number)

    front_full_text = normalize_arabic_text(" ".join(filter(None, [name, address, id_number, birth_date])))

    back_full_text_parts = []
    if back_crops:
        for crop in back_crops.values():
            lines, _, _ = _best_lines_for_text(crop)
            txt = merge_lines(lines)
            if txt:
                back_full_text_parts.append(txt)

    return {
        "name": name,
        "address": address,
        "id_number": id_number,
        "birth_date": birth_date,
        "front_full_text": front_full_text,
        "back_full_text": normalize_arabic_text(" ".join(back_full_text_parts)),
        "raw": {
            "name_lines": name_lines,
            "address_lines": address_lines,
            "id_number_lines": id_lines,
            "birth_date_lines": birth_lines,
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

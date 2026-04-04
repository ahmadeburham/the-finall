#!/usr/bin/env python3
"""
Run the full pipeline on front.jpg + back.jpeg,
save crops, aligned images, OCR text, and verification to a timestamped folder.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import arabic_ocr as ao
import template_fields as tf

HERE = Path(__file__).resolve().parent

# ── inputs ────────────────────────────────────────────────────────────────────
FRONT_IMAGE     = HERE / "front.jpg"
BACK_IMAGE      = HERE / "back.jpeg"
FRONT_TEMPLATE  = HERE / "templates" / "front_template.png"
BACK_TEMPLATE   = HERE / "templates" / "back_template.png"
TEMPLATE_CONFIG = HERE / "id_template_config.json"

# selfie: look in this folder first, then parent folder
SELFIE = None
for candidate in [HERE / "selfie.jpg", HERE.parent / "selfie.jpg"]:
    if candidate.exists():
        SELFIE = candidate
        break

# ── output folder ─────────────────────────────────────────────────────────────
OUT_DIR = HERE / f"run_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def _serial(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))
    payload = ao.prepare_for_json_output(data)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_serial), encoding="utf-8")

def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    with (OUT_DIR / "run.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")

# ══════════════════════════════════════════════════════════════════════════════
log(f"Output folder : {OUT_DIR}")
log(f"Front image   : {FRONT_IMAGE}")
log(f"Back image    : {BACK_IMAGE}")
log(f"Selfie        : {SELFIE or 'NOT FOUND – face match skipped'}")

result = {
    "timestamp": datetime.now().isoformat(),
    "inputs": {
        "front":  str(FRONT_IMAGE),
        "back":   str(BACK_IMAGE),
        "selfie": str(SELFIE) if SELFIE else None,
    },
    "alignment": {},
    "text": {},
    "face_match": None,
}

# ── 1. ALIGNMENT ──────────────────────────────────────────────────────────────
log("Aligning front card to template …")
try:
    aligned_front, front_info = tf.align_card_to_template(
        FRONT_IMAGE, FRONT_TEMPLATE, return_candidates=True
    )
    result["alignment"]["front"] = front_info
    log(f"  front aligned: method={front_info['method']}  "
        f"rotation={front_info['rotation']}°  inliers={front_info['inliers']}  "
        f"score={front_info['score']:.1f}")
except Exception as e:
    log(f"  FAILED: {e}")
    aligned_front = None

log("Aligning back card to template …")
try:
    aligned_back, back_info = tf.align_card_to_template(
        BACK_IMAGE, BACK_TEMPLATE, return_candidates=True
    )
    result["alignment"]["back"] = back_info
    log(f"  back  aligned: method={back_info['method']}  "
        f"rotation={back_info['rotation']}°  inliers={back_info['inliers']}  "
        f"score={back_info['score']:.1f}")
except Exception as e:
    log(f"  FAILED: {e}")
    aligned_back = None

save_json(OUT_DIR / "alignment_info.json", result["alignment"])

# ── 2. EXTRACT CROPS ──────────────────────────────────────────────────────────
config = tf.load_config(TEMPLATE_CONFIG)

front_artifacts = {"aligned": None, "cleaned": None, "photo": None, "crops": {}}
back_artifacts  = {"aligned": None, "cleaned": None, "photo": None, "crops": {}}

crops_dir = OUT_DIR / "crops"
crops_dir.mkdir(exist_ok=True)

if aligned_front is not None:
    log("Extracting front crops …")
    front_artifacts = tf.extract_side_artifacts(
        aligned_front,
        config,
        "front",
        subject_image=FRONT_IMAGE,
        align_info=front_info,
    )
    cv2.imwrite(str(OUT_DIR / "front_aligned.png"), aligned_front)
    if front_artifacts.get("cleaned") is not None:
        cv2.imwrite(str(OUT_DIR / "front_cleaned.png"), front_artifacts["cleaned"])
    if front_artifacts.get("photo") is not None:
        cv2.imwrite(str(OUT_DIR / "front_photo_crop.png"), front_artifacts["photo"])
    if front_artifacts.get("outline") is not None:
        cv2.imwrite(str(OUT_DIR / "front_outline.png"), front_artifacts["outline"])
    if front_artifacts.get("projection") is not None:
        cv2.imwrite(str(OUT_DIR / "front_projection.png"), front_artifacts["projection"])
    for name, crop in front_artifacts.get("crops", {}).items():
        if crop is not None and crop.size > 0:
            cv2.imwrite(str(crops_dir / f"front_{name}.png"), crop)
            log(f"  saved front/{name}  {crop.shape[1]}x{crop.shape[0]}px")

if aligned_back is not None:
    log("Extracting back crops …")
    back_artifacts = tf.extract_side_artifacts(
        aligned_back,
        config,
        "back",
        subject_image=BACK_IMAGE,
        align_info=back_info,
    )
    cv2.imwrite(str(OUT_DIR / "back_aligned.png"), aligned_back)
    if back_artifacts.get("cleaned") is not None:
        cv2.imwrite(str(OUT_DIR / "back_cleaned.png"), back_artifacts["cleaned"])
    if back_artifacts.get("outline") is not None:
        cv2.imwrite(str(OUT_DIR / "back_outline.png"), back_artifacts["outline"])
    if back_artifacts.get("projection") is not None:
        cv2.imwrite(str(OUT_DIR / "back_projection.png"), back_artifacts["projection"])
    for name, crop in back_artifacts.get("crops", {}).items():
        if crop is not None and crop.size > 0:
            cv2.imwrite(str(crops_dir / f"back_{name}.png"), crop)
            log(f"  saved back/{name}  {crop.shape[1]}x{crop.shape[0]}px")

# ── 3. OCR ────────────────────────────────────────────────────────────────────
log("Running OCR …")
try:
    fields = ao.extract_fields_from_crops(
        front_crops=front_artifacts.get("crops", {}),
        back_crops=back_artifacts.get("crops", {}),
    )
    result["text"] = fields
    log(f"  name         : {fields.get('name','')}")
    log(f"  id_number    : {fields.get('id_number','')}")
    log(f"  birth_date   : {fields.get('birth_date','')}")
    log(f"  expiry_date  : {fields.get('expiry_date','')}")
    log(f"  address      : {fields.get('address','')}")
    log(f"  front_full   : {fields.get('front_full_text','')[:80]}")
    log(f"  back_full    : {fields.get('back_full_text','')[:80]}")
    save_json(OUT_DIR / "ocr_fields.json", fields)
except Exception as e:
    log(f"  OCR FAILED: {e}")
    import traceback; traceback.print_exc()

# ── 4. FACE MATCH ─────────────────────────────────────────────────────────────
if SELFIE and front_artifacts.get("photo") is not None and front_artifacts["photo"].size > 0:
    log("Running face match …")
    try:
        import face_match_top4 as fm
        face_match, face_logs = fm.verify_top4_from_array(
            selfie_path=str(SELFIE),
            target_face_bgr=front_artifacts["photo"],
            detector_backend="retinaface",
            distance_metric="cosine",
            enforce_detection=False,
        )
        result["face_match"] = {"passed": bool(face_match), "logs": face_logs}
        log(f"  face match result: {'✅ PASSED' if face_match else '❌ FAILED'}")
        save_json(OUT_DIR / "face_match.json", result["face_match"])
    except Exception as e:
        log(f"  face match FAILED: {e}")
        result["face_match"] = {"passed": False, "error": str(e)}
else:
    log("Face match skipped (no selfie or no photo crop)")
    result["face_match"] = {"passed": False, "reason": "no selfie or no photo crop"}

# ── 5. FINAL SUMMARY ──────────────────────────────────────────────────────────
result["overall"] = {
    "front_aligned": aligned_front is not None,
    "back_aligned":  aligned_back  is not None,
    "ocr_name":      bool(result["text"].get("name")),
    "ocr_id_number": bool(result["text"].get("id_number")),
    "ocr_expiry":    bool(result["text"].get("expiry_date")),
    "face_match":    result["face_match"].get("passed", False) if isinstance(result["face_match"], dict) else False,
}

save_json(OUT_DIR / "final_result.json", result)

log("─" * 60)
log("DONE")
log(f"  Output folder: {OUT_DIR}")
log(f"  ├── front_aligned.png")
log(f"  ├── back_aligned.png")
log(f"  ├── front_photo_crop.png")
log(f"  ├── front_cleaned.png / back_cleaned.png")
log(f"  ├── crops/  (all field crops)")
log(f"  ├── ocr_fields.json")
log(f"  ├── alignment_info.json")
log(f"  ├── face_match.json")
log(f"  ├── final_result.json")
log(f"  └── run.log")
log("─" * 60)

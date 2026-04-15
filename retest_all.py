"""
retest_all.py - Blind full-pipeline test on every image in test/.
Runs detection → rotation → alignment → field extraction with NO hints.
Saves debug visuals to test_output/<image>/ and prints summary table.
"""
import os, sys, glob, json, time, traceback
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import template_fields as tf

def safe_imwrite(path: str, img):
    """cv2.imwrite wrapper that handles non-ASCII paths on Windows."""
    try:
        ok = cv2.imwrite(path, img)
        if ok:
            return
    except Exception:
        pass
    # Fallback: encode in memory then write bytes
    ext = os.path.splitext(path)[1] or '.jpg'
    ok, buf = cv2.imencode(ext, img)
    if ok:
        with open(path, 'wb') as f:
            f.write(buf.tobytes())


TEST_DIR = "test"
TMPL_PATH = "templates/front_template.png"
CFG_PATH = "id_template_config.json"
OUT_DIR = "test_output"

EXPECTED_FIELDS = ["name", "address", "id_number", "birth_date"]

# ── collect images ────────────────────────────────────────────────────────
images = sorted(
    glob.glob(os.path.join(TEST_DIR, "*.jpg")) +
    glob.glob(os.path.join(TEST_DIR, "*.png"))
)
print(f"Found {len(images)} test images")
print(f"Template: {TMPL_PATH}")
print(f"Config: {CFG_PATH}\n")

config = tf.load_config(CFG_PATH)
os.makedirs(OUT_DIR, exist_ok=True)

# ── run blind pipeline on each image ──────────────────────────────────────
rows = []
for idx, img_path in enumerate(images):
    fname = os.path.basename(img_path)
    stem = Path(fname).stem
    out_subdir = os.path.join(OUT_DIR, stem)
    os.makedirs(out_subdir, exist_ok=True)

    print(f"[{idx+1}/{len(images)}] {fname}")

    row = {
        "file": fname,
        "card_detected": False,
        "rotation": None,
        "score": None,
        "crops_ok": False,
        "photo_ok": False,
        "error": None,
    }

    try:
        t0 = time.perf_counter()

        # BLIND alignment — no hints, no expected rotation
        aligned, info = tf.align_card_to_template(
            img_path, TMPL_PATH, return_candidates=False
        )
        elapsed = time.perf_counter() - t0

        row["card_detected"] = bool(info.get("card_detected", False))
        row["rotation"] = info.get("rotation")
        row["score"] = round(float(info.get("score", 0)), 4)
        row["time_s"] = round(elapsed, 2)

        # Save aligned
        safe_imwrite(os.path.join(out_subdir, "aligned.jpg"), aligned)

        # Field extraction — blind
        arts = tf.extract_side_artifacts(
            aligned, config, "front",
            subject_image=img_path,
            align_info=info,
        )

        # Save outline + projection
        if arts.get("outline") is not None:
            h, w = arts["outline"].shape[:2]
            if max(h, w) > 2000:
                sc = 2000 / max(h, w)
                arts["outline"] = cv2.resize(arts["outline"], (int(w*sc), int(h*sc)))
            safe_imwrite(os.path.join(out_subdir, "outline.jpg"), arts["outline"])
        if arts.get("projection") is not None:
            h, w = arts["projection"].shape[:2]
            if max(h, w) > 2000:
                sc = 2000 / max(h, w)
                arts["projection"] = cv2.resize(arts["projection"], (int(w*sc), int(h*sc)))
            safe_imwrite(os.path.join(out_subdir, "projection.jpg"), arts["projection"])

        # Save photo
        if arts.get("photo") is not None and arts["photo"].size > 0:
            safe_imwrite(os.path.join(out_subdir, "photo.jpg"), arts["photo"])
            row["photo_ok"] = True

        # Save field crops
        crops = arts.get("crops", {})
        all_fields_ok = True
        for field in EXPECTED_FIELDS:
            crop = crops.get(field)
            if crop is not None and crop.size > 0:
                safe_imwrite(os.path.join(out_subdir, f"crop_{field}.jpg"), crop)
            else:
                all_fields_ok = False
        row["crops_ok"] = all_fields_ok

        # Save info JSON
        info_safe = {}
        for k, v in info.items():
            if isinstance(v, np.ndarray):
                info_safe[k] = v.tolist()
            elif isinstance(v, (list, dict, str, int, float, bool, type(None))):
                info_safe[k] = v
            else:
                info_safe[k] = str(v)
        with open(os.path.join(out_subdir, "info.json"), "w", encoding="utf-8") as f:
            json.dump(info_safe, f, ensure_ascii=False, indent=2)

    except Exception as e:
        row["error"] = str(e)
        traceback.print_exc()

    rows.append(row)

# ── summary table ─────────────────────────────────────────────────────────
print("\n" + "=" * 110)
print(f"{'File':<45} {'Detected':>8} {'Rotation':>8} {'Score':>8} {'Photo':>6} {'Crops':>6} {'Error'}")
print("-" * 110)
for r in rows:
    det = "YES" if r["card_detected"] else "NO"
    rot = str(r["rotation"]) if r["rotation"] is not None else "?"
    sc = f"{r['score']:.3f}" if r["score"] is not None else "?"
    photo = "OK" if r["photo_ok"] else "FAIL"
    crops = "OK" if r["crops_ok"] else "FAIL"
    err = r["error"] or ""
    print(f"{r['file']:<45} {det:>8} {rot:>8} {sc:>8} {photo:>6} {crops:>6}  {err}")

# ── totals ────────────────────────────────────────────────────────────────
total = len(rows)
detected = sum(1 for r in rows if r["card_detected"])
photo_ok = sum(1 for r in rows if r["photo_ok"])
crops_ok = sum(1 for r in rows if r["crops_ok"])
print("-" * 110)
print(f"TOTALS: {detected}/{total} detected,  {photo_ok}/{total} photo OK,  {crops_ok}/{total} all crops OK")
print(f"Output saved to: {OUT_DIR}/")

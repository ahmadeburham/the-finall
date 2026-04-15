"""
generate_features.py
Crops design-only reference regions from the front/back templates and saves
them into features/front/ and features/back/.

Run:
    python generate_features.py [--visualise]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent

FRONT_TEMPLATE = HERE / "templates" / "front_template.png"
BACK_TEMPLATE  = HERE / "templates" / "back_template.png"
FEATURES_FRONT = HERE / "features" / "front"
FEATURES_BACK  = HERE / "features" / "back"

# ---------------------------------------------------------------------------
# Region definitions  [x1_norm, y1_norm, x2_norm, y2_norm]
# All regions are design-only (no personal data).
# ---------------------------------------------------------------------------

FRONT_REGIONS = [
    # name,                  [x1,   y1,   x2,   y2 ]
    # Green Arabic header banner (جمهورية مصر العربية / بطاقة تحقيق الشخصية).
    # This is the single most reliable feature across all mobile photos:
    # high-contrast green text on light background, always visible regardless
    # of how the card is held.  Background artwork (sphinx, pyramids) is too
    # faint in real compressed mobile photos for AKAZE/ORB to match reliably.
    ("front_header",         [0.28, 0.00, 1.00, 0.18]),
]

BACK_REGIONS = [
    # name,                  [x1,   y1,   x2,   y2 ]
    # 2D barcode strip — the single most reliable back feature:
    # extremely high and uniform keypoint density, appears on every real
    # Egyptian ID back regardless of card holder.  Eagle emblem and pharaoh
    # artwork are too small / too faint in mobile photos for reliable matching.
    ("back_barcode",         [0.00, 0.57, 1.00, 1.00]),
]


def crop_norm(img: np.ndarray, box) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = max(0, int(round(box[0] * w)))
    y1 = max(0, int(round(box[1] * h)))
    x2 = min(w, int(round(box[2] * w)))
    y2 = min(h, int(round(box[3] * h)))
    return img[y1:y2, x1:x2].copy()


def write_image_safe(path: Path, img: np.ndarray) -> None:
    success, buf = cv2.imencode(path.suffix or ".png", img)
    if success:
        buf.tofile(str(path))


def draw_regions(img: np.ndarray, regions, color=(0, 220, 60), thickness=3) -> np.ndarray:
    vis = img.copy()
    h, w = vis.shape[:2]
    for name, box in regions:
        x1 = int(round(box[0] * w))
        y1 = int(round(box[1] * h))
        x2 = int(round(box[2] * w))
        y2 = int(round(box[3] * h))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(vis, name, (x1 + 4, max(y1 + 18, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return vis


def generate(visualise: bool = False):
    FEATURES_FRONT.mkdir(parents=True, exist_ok=True)
    FEATURES_BACK.mkdir(parents=True, exist_ok=True)

    for template_path, regions, out_dir, label in [
        (FRONT_TEMPLATE, FRONT_REGIONS, FEATURES_FRONT, "front"),
        (BACK_TEMPLATE,  BACK_REGIONS,  FEATURES_BACK,  "back"),
    ]:
        success, buf = True, None
        raw = np.fromfile(str(template_path), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[ERROR] Cannot read {template_path}")
            continue

        print(f"\n=== {label.upper()} template: {img.shape[1]}x{img.shape[0]} ===")

        for name, box in regions:
            crop = crop_norm(img, box)
            if crop.size == 0:
                print(f"  [SKIP] {name} — empty crop")
                continue
            out_path = out_dir / f"{name}.png"
            write_image_safe(out_path, crop)
            print(f"  saved {name}.png  {crop.shape[1]}x{crop.shape[0]}")

        if visualise:
            vis = draw_regions(img, regions)
            vis_path = HERE / f"feature_regions_{label}.png"
            write_image_safe(vis_path, vis)
            print(f"  visualisation → {vis_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--visualise", action="store_true",
                    help="Save annotated template images showing crop boxes")
    args = ap.parse_args()
    generate(visualise=args.visualise)

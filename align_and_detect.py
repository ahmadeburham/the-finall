import cv2
import numpy as np
import json
from pathlib import Path


FRONT_SRC      = Path(__file__).parent.parent / "front.jpeg"
BACK_SRC       = Path(__file__).parent.parent / "back.jpeg"
FRONT_TEMPLATE = Path(__file__).parent / "templates" / "front_template.png"
BACK_TEMPLATE  = Path(__file__).parent / "templates" / "back_template.png"
OUT_DIR        = Path(__file__).parent


def order_points(pts):
    pts  = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s         = pts.sum(axis=1)
    rect[0]   = pts[np.argmin(s)]
    rect[2]   = pts[np.argmax(s)]
    diff      = np.diff(pts, axis=1)
    rect[1]   = pts[np.argmin(diff)]
    rect[3]   = pts[np.argmax(diff)]
    return rect


def four_point_warp(image, pts, out_w, out_h):
    rect = order_points(pts)
    dst  = np.array([
        [0,         0        ],
        [out_w - 1, 0        ],
        [out_w - 1, out_h - 1],
        [0,         out_h - 1],
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (out_w, out_h))


def _segment_card_contour(image_bgr):
    """Find the largest bright quadrilateral — the card boundary."""
    H, W     = image_bgr.shape[:2]
    img_area = H * W

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # Try multiple threshold strategies and pick best 4-sided contour
    candidates = []

    # Strategy 1: Otsu on blurred gray
    _, th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(th1)

    # Strategy 2: Strong bright-region extraction (card is much lighter than background)
    _, th2 = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY)
    candidates.append(th2)

    # Strategy 3: Canny edges, dilated
    edges = cv2.Canny(blur, 20, 80)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=3)
    candidates.append(edges)

    best_contour = None
    best_score   = -1

    for mask in candidates:
        # Fill holes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts    = sorted(cnts, key=cv2.contourArea, reverse=True)

        for c in cnts[:5]:
            area = cv2.contourArea(c)
            frac = area / img_area
            if frac < 0.10 or frac > 0.97:
                continue
            peri = cv2.arcLength(c, True)
            for eps in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
                approx = cv2.approxPolyDP(c, eps * peri, True)
                if len(approx) != 4:
                    continue
                pts    = approx.reshape(4, 2).astype("float32")
                rect   = order_points(pts)
                wc     = max(np.linalg.norm(rect[1] - rect[0]), np.linalg.norm(rect[2] - rect[3]))
                hc     = max(np.linalg.norm(rect[3] - rect[0]), np.linalg.norm(rect[2] - rect[1]))
                if hc == 0:
                    continue
                ratio      = max(wc, hc) / max(min(wc, hc), 1)
                card_ratio = 85.6 / 54.0          # ~1.585
                err        = abs(ratio - card_ratio) / card_ratio
                score      = frac - err * 0.8
                if score > best_score:
                    best_score   = score
                    best_contour = approx
                break

    return best_contour


def align_to_template(photo_bgr, template_bgr, label="card"):
    """Segment card from photo using contour detection, warp to template size."""
    tH, tW = template_bgr.shape[:2]

    contour = _segment_card_contour(photo_bgr)

    if contour is not None:
        pts    = contour.reshape(4, 2).astype("float32")
        rect   = order_points(pts)
        wc     = max(np.linalg.norm(rect[1] - rect[0]), np.linalg.norm(rect[2] - rect[3]))
        hc     = max(np.linalg.norm(rect[3] - rect[0]), np.linalg.norm(rect[2] - rect[1]))
        # If card is portrait in the photo, out dimensions should be swapped
        if hc > wc:
            out_w, out_h = tH, tW
        else:
            out_w, out_h = tW, tH
        warped = four_point_warp(photo_bgr, pts, out_w, out_h)
        # Ensure landscape orientation
        if warped.shape[0] > warped.shape[1]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        # Resize to exact template dimensions
        aligned = cv2.resize(warped, (tW, tH))
        print(f"  [{label}] Card contour found — warped to {tW}x{tH}")
        return aligned

    print(f"  [{label}] No card contour found — resizing directly")
    return cv2.resize(photo_bgr, (tW, tH))


def detect_text_regions(aligned_bgr):
    H, W  = aligned_bgr.shape[:2]
    gray  = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate horizontally to connect characters into words/lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(W * 0.04), 3))
    dilated  = cv2.dilate(thresh, h_kernel, iterations=2)
    # Close vertically to merge multi-row labels
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, int(H * 0.025)))
    dilated  = cv2.dilate(dilated, v_kernel, iterations=1)
    dilated  = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)))

    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = (W * H) * 0.0003
    max_area = (W * H) * 0.55
    min_w    = W * 0.02
    min_h    = H * 0.015

    boxes = []
    for c in cnts:
        x, y, w, bh = cv2.boundingRect(c)
        area        = w * bh
        aspect      = w / max(bh, 1)
        if area < min_area or area > max_area:
            continue
        if w < min_w or bh < min_h:
            continue
        if aspect < 0.2 or aspect > 40:
            continue
        boxes.append((x, y, x + w, y + bh))

    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def merge_overlapping(boxes, gap=8):
    if not boxes:
        return []
    merged = True
    result = list(boxes)
    while merged:
        merged = False
        out = []
        used = [False] * len(result)
        for i, a in enumerate(result):
            if used[i]:
                continue
            ax1, ay1, ax2, ay2 = a
            for j, b in enumerate(result):
                if i == j or used[j]:
                    continue
                bx1, by1, bx2, by2 = b
                ix1 = max(ax1, bx1) - gap
                iy1 = max(ay1, by1) - gap
                ix2 = min(ax2, bx2) + gap
                iy2 = min(ay2, by2) + gap
                if ix1 < ix2 and iy1 < iy2:
                    ax1 = min(ax1, bx1)
                    ay1 = min(ay1, by1)
                    ax2 = max(ax2, bx2)
                    ay2 = max(ay2, by2)
                    used[j] = True
                    merged   = True
            out.append((ax1, ay1, ax2, ay2))
            used[i] = True
        result = out
    return result


def highlight_and_crop(aligned_bgr, boxes, crop_dir: Path, label: str):
    H, W = aligned_bgr.shape[:2]
    highlighted = aligned_bgr.copy()
    rois = {}
    crop_dir.mkdir(parents=True, exist_ok=True)

    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255),
        (255, 0, 255), (255, 165, 0), (0, 128, 255), (128, 0, 255),
    ]

    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        color = colors[idx % len(colors)]
        cv2.rectangle(highlighted, (x1, y1), (x2, y2), color, 2)
        cv2.putText(highlighted, str(idx),
                    (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        crop = aligned_bgr[y1:y2, x1:x2]
        crop_path = crop_dir / f"region_{idx:02d}.png"
        cv2.imwrite(str(crop_path), crop)
        print(f"  [{label}] region_{idx:02d}  box=({x1},{y1},{x2},{y2})  "
              f"roi=[{x1/W:.4f},{y1/H:.4f},{x2/W:.4f},{y2/H:.4f}]")

        rois[f"region_{idx:02d}"] = [
            round(x1 / W, 4),
            round(y1 / H, 4),
            round(x2 / W, 4),
            round(y2 / H, 4),
        ]

    return highlighted, rois


def process_side(src_path: Path, template_path: Path, label: str):
    print(f"\n=== Processing {label} ===")
    image = cv2.imread(str(src_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read {src_path}")
    template = cv2.imread(str(template_path))
    if template is None:
        raise FileNotFoundError(f"Cannot read template {template_path}")

    print(f"  Original size: {image.shape[1]}x{image.shape[0]}")
    aligned = align_to_template(image, template, label)
    print(f"  Aligned size:  {aligned.shape[1]}x{aligned.shape[0]}")

    aligned_path = OUT_DIR / f"{label}_aligned.png"
    cv2.imwrite(str(aligned_path), aligned)
    print(f"  Saved: {aligned_path.name}")

    boxes = detect_text_regions(aligned)
    boxes = merge_overlapping(boxes, gap=10)
    print(f"  Detected {len(boxes)} text regions")

    crop_dir = OUT_DIR / "crops" / label
    highlighted, rois = highlight_and_crop(aligned, boxes, crop_dir, label)

    highlighted_path = OUT_DIR / f"{label}_highlighted.png"
    cv2.imwrite(str(highlighted_path), highlighted)
    print(f"  Saved: {highlighted_path.name}")

    return rois


def main():
    front_rois = process_side(FRONT_SRC, FRONT_TEMPLATE, "front")
    back_rois  = process_side(BACK_SRC,  BACK_TEMPLATE,  "back")

    detected = {
        "front": {"regions": front_rois},
        "back":  {"regions": back_rois},
    }

    roi_path = OUT_DIR / "detected_rois.json"
    with open(roi_path, "w", encoding="utf-8") as f:
        json.dump(detected, f, indent=2, ensure_ascii=False)
    print(f"\nROIs saved to: {roi_path}")
    print("Done.")


if __name__ == "__main__":
    main()

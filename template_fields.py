import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif"}


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    img = None
    if path.exists():
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.size:
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            img = None
    if img is None:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def load_config(path: str | Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = int(max(0, min(w - 1, x1)))
    y1 = int(max(0, min(h - 1, y1)))
    x2 = int(max(1, min(w, x2)))
    y2 = int(max(1, min(h, y2)))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def norm_box_to_px(box, w, h, pad_px: int = 0):
    x1 = int(round(box[0] * w)) - pad_px
    y1 = int(round(box[1] * h)) - pad_px
    x2 = int(round(box[2] * w)) + pad_px
    y2 = int(round(box[3] * h)) + pad_px
    return clamp_box(x1, y1, x2, y2, w, h)


def crop_norm_box(img: np.ndarray, box, pad_px: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = norm_box_to_px(box, w, h, pad_px=pad_px)
    return img[y1:y2, x1:x2].copy()


def _rotation_matrix_for_image(image: np.ndarray, angle: int) -> Optional[np.ndarray]:
    if angle == 0:
        return None
    if angle not in {90, 180, 270}:
        raise ValueError(f"Unsupported angle: {angle}")
    h, w = image.shape[:2]
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    scale = 1.0
    if angle in {90, 270}:
        scale = min(float(w) / max(float(h), 1.0), float(h) / max(float(w), 1.0))
    return cv2.getRotationMatrix2D(center, -float(angle), scale)


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return image.copy()
    if angle not in {90, 180, 270}:
        raise ValueError(f"Unsupported angle: {angle}")
    h, w = image.shape[:2]
    matrix = _rotation_matrix_for_image(image, angle)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _segment_card_contour(image_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    h, w = image_bgr.shape[:2]
    img_area = float(h * w)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    candidates = []
    _, th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(th1)
    _, th2 = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY)
    candidates.append(th2)
    edges = cv2.Canny(blur, 20, 80)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=3)
    candidates.append(edges)

    best_contour = None
    best_score = -1e18

    for mask in candidates:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        for c in cnts[:5]:
            area = float(cv2.contourArea(c))
            frac = area / max(img_area, 1.0)
            if frac < 0.10 or frac > 0.97:
                continue
            peri = cv2.arcLength(c, True)
            for eps in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
                approx = cv2.approxPolyDP(c, eps * peri, True)
                if len(approx) != 4:
                    continue
                pts = approx.reshape(4, 2).astype("float32")
                rect = order_points(pts)
                wc = max(np.linalg.norm(rect[1] - rect[0]), np.linalg.norm(rect[2] - rect[3]))
                hc = max(np.linalg.norm(rect[3] - rect[0]), np.linalg.norm(rect[2] - rect[1]))
                if hc == 0:
                    continue
                ratio = max(wc, hc) / max(min(wc, hc), 1.0)
                card_ratio = 85.6 / 54.0
                err = abs(ratio - card_ratio) / card_ratio
                score = frac - err * 0.8
                if score > best_score:
                    best_score = float(score)
                    best_contour = approx
                break

    return best_contour, float(best_score)


def _rotate_source_corners(rect: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 0:
        return rect.astype("float32")
    if rotation == 90:
        return np.array([rect[3], rect[0], rect[1], rect[2]], dtype="float32")
    if rotation == 180:
        return np.array([rect[2], rect[3], rect[0], rect[1]], dtype="float32")
    if rotation == 270:
        return np.array([rect[1], rect[2], rect[3], rect[0]], dtype="float32")
    raise ValueError(f"Unsupported rotation: {rotation}")


def _score_alignment_candidate(aligned_bgr: np.ndarray, template_bgr: np.ndarray) -> float:
    if aligned_bgr.shape[:2] != template_bgr.shape[:2]:
        aligned_bgr = cv2.resize(aligned_bgr, (template_bgr.shape[1], template_bgr.shape[0]))

    aligned_gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    aligned_gray = cv2.GaussianBlur(aligned_gray, (5, 5), 0)
    template_gray = cv2.GaussianBlur(template_gray, (5, 5), 0)

    corr_score = float(cv2.matchTemplate(aligned_gray, template_gray, cv2.TM_CCOEFF_NORMED)[0][0])

    aligned_edges = cv2.Canny(aligned_gray, 60, 160)
    template_edges = cv2.Canny(template_gray, 60, 160)
    edge_score = float(cv2.matchTemplate(aligned_edges, template_edges, cv2.TM_CCOEFF_NORMED)[0][0])

    third = max(1, aligned_gray.shape[1] // 3)
    aligned_lr = float(np.mean(aligned_gray[:, :third]) - np.mean(aligned_gray[:, -third:]))
    template_lr = float(np.mean(template_gray[:, :third]) - np.mean(template_gray[:, -third:]))
    lr_score = 1.0 - min(abs(aligned_lr - template_lr) / 255.0, 1.0)

    return float(0.35 * corr_score + 0.50 * edge_score + 0.15 * lr_score)


def _candidate_rotations_for_subject(subject_h: int, subject_w: int) -> Tuple[int, ...]:
    if subject_h >= subject_w:
        return (0, 180)
    return (90, 270)


def _fallback_rotations_for_subject(subject_h: int, subject_w: int) -> Tuple[int, ...]:
    preferred = set(_candidate_rotations_for_subject(subject_h, subject_w))
    return tuple(rotation for rotation in (0, 90, 180, 270) if rotation not in preferred)


def _map_rotated_points_to_source(points: np.ndarray, source_bgr: np.ndarray, rotation: int) -> np.ndarray:
    mapped = np.asarray(points, dtype="float32").reshape(-1, 2)
    if rotation != 0:
        matrix = _rotation_matrix_for_image(source_bgr, rotation)
        inverse_matrix = cv2.invertAffineTransform(matrix)
        mapped = cv2.transform(mapped.reshape(-1, 1, 2), inverse_matrix).reshape(-1, 2)
    h, w = source_bgr.shape[:2]
    mapped[:, 0] = np.clip(mapped[:, 0], 0, max(w - 1, 0))
    mapped[:, 1] = np.clip(mapped[:, 1], 0, max(h - 1, 0))
    return order_points(mapped.astype("float32"))


def _build_alignment_candidate(
    subject_bgr: np.ndarray,
    template_bgr: np.ndarray,
    template_corners: np.ndarray,
    rotation: int,
) -> Dict:
    rotated_subject = rotate_image(subject_bgr, rotation)
    contour, contour_score = _segment_card_contour(rotated_subject)
    candidate = {
        "rotation": int(rotation),
        "valid": False,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "good_matches": 0,
        "score": float(contour_score),
        "orientation_score": 0.0,
    }
    if contour is None:
        return candidate

    template_h, template_w = template_bgr.shape[:2]
    rect = order_points(contour.reshape(4, 2).astype("float32"))
    source_rect = _map_rotated_points_to_source(rect, subject_bgr, rotation)
    source_to_template = cv2.getPerspectiveTransform(source_rect, template_corners)
    template_to_source = cv2.getPerspectiveTransform(template_corners, source_rect)
    aligned = cv2.warpPerspective(subject_bgr, source_to_template, (template_w, template_h))
    orientation_score = _score_alignment_candidate(aligned, template_bgr)
    candidate.update(
        {
            "valid": True,
            "inliers": 4,
            "inlier_ratio": 1.0,
            "good_matches": 4,
            "source_corners": source_rect,
            "source_to_template": source_to_template,
            "template_to_source": template_to_source,
            "aligned": aligned,
            "orientation_score": float(orientation_score),
        }
    )
    return candidate


def _build_full_image_candidate(subject_bgr: np.ndarray, template_bgr: np.ndarray, template_corners: np.ndarray) -> Optional[Dict]:
    template_h, template_w = template_bgr.shape[:2]
    source_h, source_w = subject_bgr.shape[:2]
    if source_w <= 1 or source_h <= 1:
        return None

    card_ratio = 85.6 / 54.0
    source_ratio = float(source_w) / max(float(source_h), 1.0)
    ratio_error = abs(source_ratio - card_ratio) / card_ratio
    if ratio_error > 0.12:
        return None

    source_rect = np.array([[0, 0], [source_w - 1, 0], [source_w - 1, source_h - 1], [0, source_h - 1]], dtype="float32")
    source_to_template = cv2.getPerspectiveTransform(source_rect, template_corners)
    template_to_source = cv2.getPerspectiveTransform(template_corners, source_rect)
    aligned = cv2.warpPerspective(subject_bgr, source_to_template, (template_w, template_h))
    orientation_score = _score_alignment_candidate(aligned, template_bgr)
    return {
        "rotation": 0,
        "valid": True,
        "inliers": 4,
        "inlier_ratio": 1.0,
        "good_matches": 4,
        "score": float(1.0 - ratio_error),
        "source_corners": source_rect,
        "source_to_template": source_to_template,
        "template_to_source": template_to_source,
        "aligned": aligned,
        "orientation_score": float(orientation_score),
    }


def align_card_to_template(subject_image: str | Path, template_image: str | Path, return_candidates: bool = False):
    template_bgr = read_image(template_image)
    subject_bgr = read_image(subject_image)
    template_h, template_w = template_bgr.shape[:2]
    subject_h, subject_w = subject_bgr.shape[:2]
    template_corners = np.array(
        [[0, 0], [template_w - 1, 0], [template_w - 1, template_h - 1], [0, template_h - 1]],
        dtype="float32",
    )

    candidates = []
    best_candidate = None
    for rotation in _candidate_rotations_for_subject(subject_h, subject_w):
        candidate = _build_alignment_candidate(subject_bgr, template_bgr, template_corners, rotation)
        candidates.append(candidate)
        if candidate["valid"] and (best_candidate is None or candidate["orientation_score"] > best_candidate["orientation_score"]):
            best_candidate = candidate

    if best_candidate is None:
        for rotation in _fallback_rotations_for_subject(subject_h, subject_w):
            candidate = _build_alignment_candidate(subject_bgr, template_bgr, template_corners, rotation)
            candidates.append(candidate)
            if candidate["valid"] and (best_candidate is None or candidate["orientation_score"] > best_candidate["orientation_score"]):
                best_candidate = candidate

    full_image_candidate = _build_full_image_candidate(subject_bgr, template_bgr, template_corners)
    if full_image_candidate is not None:
        candidates.append(full_image_candidate)
        if best_candidate is None or float(best_candidate.get("score", 0.0)) < 0.10:
            best_candidate = full_image_candidate

    if best_candidate is None:
        aligned = cv2.resize(subject_bgr, (template_w, template_h))
        info = {
            "method": "contour_template_projection",
            "rotation": 0,
            "orientation_score": 0.0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "score": 0.0,
            "good_matches": 0,
            "card_detected": False,
            "template_size": [int(template_w), int(template_h)],
            "source_size": [int(subject_w), int(subject_h)],
            "card_corners_px": [],
            "card_corners_norm": [],
            "source_to_template_matrix": [],
            "template_to_source_matrix": [],
        }
        if return_candidates:
            info["candidates"] = [
                {
                    "method": "contour_template_projection",
                    "rotation": int(candidate["rotation"]),
                    "valid": bool(candidate["valid"]),
                    "inliers": int(candidate["inliers"]),
                    "inlier_ratio": float(candidate["inlier_ratio"]),
                    "score": float(candidate["score"]),
                    "orientation_score": float(candidate["orientation_score"]),
                }
                for candidate in candidates
            ]
        return aligned, info

    rotation = int(best_candidate["rotation"])
    source_corners = best_candidate["source_corners"]
    source_to_template = best_candidate["source_to_template"]
    template_to_source = best_candidate["template_to_source"]
    aligned = best_candidate["aligned"]

    info = {
        "method": "contour_template_projection",
        "rotation": int(rotation),
        "orientation_score": float(best_candidate["orientation_score"]),
        "inliers": 4,
        "inlier_ratio": 1.0,
        "score": float(best_candidate["score"]),
        "good_matches": 4,
        "card_detected": True,
        "template_size": [int(template_w), int(template_h)],
        "source_size": [int(subject_w), int(subject_h)],
        "card_corners_px": [[float(x), float(y)] for x, y in source_corners],
        "card_corners_norm": [[float(x / max(subject_w, 1)), float(y / max(subject_h, 1))] for x, y in source_corners],
        "source_to_template_matrix": [[float(v) for v in row] for row in source_to_template],
        "template_to_source_matrix": [[float(v) for v in row] for row in template_to_source],
    }
    if return_candidates:
        info["candidates"] = [
            {
                "method": "contour_template_projection",
                "rotation": int(candidate["rotation"]),
                "valid": bool(candidate["valid"]),
                "inliers": int(candidate["inliers"]),
                "inlier_ratio": float(candidate["inlier_ratio"]),
                "score": float(candidate["score"]),
                "orientation_score": float(candidate["orientation_score"]),
            }
            for candidate in candidates
        ]
    return aligned, info


def build_cleaned_image(aligned_bgr: np.ndarray, rois: Dict[str, list]) -> np.ndarray:
    h, w = aligned_bgr.shape[:2]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    for box in rois.values():
        x1, y1, x2, y2 = norm_box_to_px(box, w, h, pad_px=4)
        canvas[y1:y2, x1:x2] = aligned_bgr[y1:y2, x1:x2]
    return canvas


def _tight_bbox_from_mask(mask: np.ndarray, pad: int = 6):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = int(xs.max()) + pad + 1
    y2 = int(ys.max()) + pad + 1
    return x1, y1, x2, y2


def refine_crop_tight(crop_bgr: np.ndarray, mode: str = "text") -> np.ndarray:
    if crop_bgr.size == 0:
        return crop_bgr

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    if mode == "numeric":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.dilate(mask, np.ones((3, 9), np.uint8), iterations=1)
    else:
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, mask = cv2.threshold(blur, 185, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.medianBlur(mask, 3)
        mask = cv2.dilate(mask, np.ones((3, 5), np.uint8), iterations=1)

    bbox = _tight_bbox_from_mask(mask, pad=6)
    if bbox is None:
        return crop_bgr

    x1, y1, x2, y2 = clamp_box(*bbox, crop_bgr.shape[1], crop_bgr.shape[0])
    refined = crop_bgr[y1:y2, x1:x2].copy()

    if refined.shape[0] < max(20, crop_bgr.shape[0] * 0.25) or refined.shape[1] < max(20, crop_bgr.shape[1] * 0.25):
        return crop_bgr
    return refined


def _box_corners_in_template(box, template_w: int, template_h: int, pad_px: int = 0) -> np.ndarray:
    x1, y1, x2, y2 = norm_box_to_px(box, template_w, template_h, pad_px=pad_px)
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype="float32")


def _perspective_crop_from_source(
    source_bgr: np.ndarray,
    box,
    template_w: int,
    template_h: int,
    template_to_source_matrix,
    pad_px: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    src_quad = _box_corners_in_template(box, template_w, template_h, pad_px=pad_px).reshape(-1, 1, 2)
    matrix = np.array(template_to_source_matrix, dtype="float32")
    src_quad = cv2.perspectiveTransform(src_quad, matrix).reshape(4, 2)

    out_w = int(round(max(np.linalg.norm(src_quad[1] - src_quad[0]), np.linalg.norm(src_quad[2] - src_quad[3]))))
    out_h = int(round(max(np.linalg.norm(src_quad[3] - src_quad[0]), np.linalg.norm(src_quad[2] - src_quad[1]))))
    out_w = max(out_w, 1)
    out_h = max(out_h, 1)

    dst_quad = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype="float32",
    )
    warp_matrix = cv2.getPerspectiveTransform(src_quad.astype("float32"), dst_quad)
    crop = cv2.warpPerspective(source_bgr, warp_matrix, (out_w, out_h))
    return crop, src_quad


def _draw_side_visuals(source_bgr: np.ndarray, side_cfg: Dict, align_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
    outline = source_bgr.copy()
    projection = source_bgr.copy()
    corners = np.array(align_info.get("card_corners_px") or [], dtype="float32")
    if corners.shape == (4, 2):
        contour = corners.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(outline, [contour], True, (60, 220, 60), 4)
        cv2.polylines(projection, [contour], True, (60, 220, 60), 4)

    matrix = align_info.get("template_to_source_matrix") or []
    template_size = align_info.get("template_size") or [0, 0]
    if len(matrix) != 3 or not template_size[0] or not template_size[1]:
        return outline, projection

    palette = [
        (231, 76, 60),
        (41, 128, 185),
        (39, 174, 96),
        (243, 156, 18),
        (142, 68, 173),
        (22, 160, 133),
        (211, 84, 0),
        (44, 62, 80),
    ]

    all_boxes = []
    if side_cfg.get("photo_roi") is not None:
        all_boxes.append(("photo", side_cfg["photo_roi"], (80, 180, 255)))
    for idx, (name, box) in enumerate((side_cfg.get("ocr_rois") or {}).items()):
        all_boxes.append((name, box, palette[idx % len(palette)]))

    transform = np.array(matrix, dtype="float32")
    template_w, template_h = int(template_size[0]), int(template_size[1])
    for name, box, color in all_boxes:
        quad = _box_corners_in_template(box, template_w, template_h).reshape(-1, 1, 2)
        quad = cv2.perspectiveTransform(quad, transform).reshape(4, 2)
        poly = quad.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(projection, [poly], True, color, 2)
        label_x = int(poly[0, 0, 0])
        label_y = max(14, int(poly[0, 0, 1]) - 6)
        cv2.putText(projection, name, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    return outline, projection


def extract_side_artifacts(
    aligned_bgr: np.ndarray,
    config: Dict,
    side: str,
    subject_image: str | Path | np.ndarray | None = None,
    align_info: Optional[Dict] = None,
) -> Dict:
    side_cfg = config[side]
    out = {"aligned": aligned_bgr, "cleaned": None, "photo": None, "crops": {}, "outline": None, "projection": None}

    source_bgr = None
    if isinstance(subject_image, np.ndarray):
        source_bgr = subject_image.copy()
    elif subject_image is not None:
        source_bgr = read_image(subject_image)

    template_size = None
    template_to_source_matrix = None
    if align_info and align_info.get("card_detected") and align_info.get("template_size") and align_info.get("template_to_source_matrix"):
        template_size = align_info["template_size"]
        template_to_source_matrix = align_info["template_to_source_matrix"]

    if source_bgr is not None and align_info:
        out["outline"], out["projection"] = _draw_side_visuals(source_bgr, side_cfg, align_info)

    rois = side_cfg.get("ocr_rois", {})
    if rois:
        out["cleaned"] = build_cleaned_image(aligned_bgr, rois)

    if "photo_roi" in side_cfg:
        if source_bgr is not None and template_size is not None and template_to_source_matrix is not None:
            photo, _ = _perspective_crop_from_source(
                source_bgr,
                side_cfg["photo_roi"],
                int(template_size[0]),
                int(template_size[1]),
                template_to_source_matrix,
                pad_px=4,
            )
        else:
            photo = crop_norm_box(aligned_bgr, side_cfg["photo_roi"], pad_px=4)
        out["photo"] = refine_crop_tight(photo, mode="text")

    pad = int(side_cfg.get("roi_pad_px", 4))
    numeric_fields = set(side_cfg.get("numeric_fields", ["id_number", "birth_date"]))
    refine_numeric = bool(side_cfg.get("refine_numeric", True))
    refine_text = bool(side_cfg.get("refine_text", True))

    for name, box in rois.items():
        if source_bgr is not None and template_size is not None and template_to_source_matrix is not None:
            crop, _ = _perspective_crop_from_source(
                source_bgr,
                box,
                int(template_size[0]),
                int(template_size[1]),
                template_to_source_matrix,
                pad_px=pad,
            )
        else:
            crop = crop_norm_box(aligned_bgr, box, pad_px=pad)
        if name in numeric_fields and refine_numeric:
            crop = refine_crop_tight(crop, mode="numeric")
        elif name not in numeric_fields and refine_text:
            crop = refine_crop_tight(crop, mode="text")
        out["crops"][name] = crop

    return out


def write_image(path: str | Path, image: np.ndarray) -> bool:
    path = Path(path)
    suffix = path.suffix or ".png"
    success, buffer = cv2.imencode(suffix, image)
    if not success:
        return False
    try:
        buffer.tofile(str(path))
        return True
    except Exception:
        return False


def save_debug_artifacts(debug_dir: str | Path, prefix: str, artifacts: Dict):
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    if artifacts.get("aligned") is not None:
        write_image(debug_dir / f"{prefix}_aligned.png", artifacts["aligned"])
    if artifacts.get("cleaned") is not None:
        write_image(debug_dir / f"{prefix}_cleaned.png", artifacts["cleaned"])
    if artifacts.get("photo") is not None:
        write_image(debug_dir / f"{prefix}_photo.png", artifacts["photo"])
    if artifacts.get("outline") is not None:
        write_image(debug_dir / f"{prefix}_outline.png", artifacts["outline"])
    if artifacts.get("projection") is not None:
        write_image(debug_dir / f"{prefix}_projection.png", artifacts["projection"])

    for name, crop in artifacts.get("crops", {}).items():
        write_image(debug_dir / f"{prefix}_{name}.png", crop)

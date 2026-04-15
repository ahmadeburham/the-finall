import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import card_detect as _cd


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


_CV2_ROTATE_CODE = {
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return image.copy()
    if angle not in _CV2_ROTATE_CODE:
        raise ValueError(f"Unsupported angle: {angle}")
    return cv2.rotate(image, _CV2_ROTATE_CODE[angle])


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
        src_h, src_w = source_bgr.shape[:2]
        xs, ys = mapped[:, 0].copy(), mapped[:, 1].copy()
        if rotation == 90:
            # cv2.ROTATE_90_CLOCKWISE forward: (sx,sy) -> (src_h-1-sy, sx)
            # Inverse:  (rx,ry) -> source (sx=ry, sy=src_h-1-rx)
            mapped[:, 0] = ys
            mapped[:, 1] = src_h - 1 - xs
        elif rotation == 180:
            # cv2.ROTATE_180 forward: (sx,sy) -> (src_w-1-sx, src_h-1-sy)
            # Inverse:  (rx,ry) -> source (sx=src_w-1-rx, sy=src_h-1-ry)
            mapped[:, 0] = src_w - 1 - xs
            mapped[:, 1] = src_h - 1 - ys
        elif rotation == 270:
            # cv2.ROTATE_90_COUNTERCLOCKWISE forward: (sx,sy) -> (sy, src_w-1-sx)
            # Inverse:  (rx,ry) -> source (sx=src_w-1-ry, sy=rx)
            mapped[:, 0] = src_w - 1 - ys
            mapped[:, 1] = xs
    h, w = source_bgr.shape[:2]
    mapped[:, 0] = np.clip(mapped[:, 0], 0, max(w - 1, 0))
    mapped[:, 1] = np.clip(mapped[:, 1], 0, max(h - 1, 0))
    return order_points(mapped.astype("float32"))


def _strip_orientation_score(img_gray: np.ndarray, tmpl_gray: np.ndarray) -> float:
    """Score how well the top/bottom strips match the template's top/bottom strips.
    Positive = correct orientation, negative = likely flipped."""
    th, tw = tmpl_gray.shape[:2]
    fifth = max(1, th // 5)
    ref_top = tmpl_gray[:fifth, :]
    ref_bot = tmpl_gray[th - fifth:, :]
    img_top = img_gray[:fifth, :]
    img_bot = img_gray[img_gray.shape[0] - fifth:, :]
    # How well does img-top match template-top (and img-bot match template-bot)?
    correct_s = (float(cv2.matchTemplate(img_top, ref_top, cv2.TM_CCOEFF_NORMED)[0][0]) +
                 float(cv2.matchTemplate(img_bot, ref_bot, cv2.TM_CCOEFF_NORMED)[0][0]))
    # How well does img-top match template-bot (and img-bot match template-top)?
    flipped_s = (float(cv2.matchTemplate(img_top, ref_bot, cv2.TM_CCOEFF_NORMED)[0][0]) +
                 float(cv2.matchTemplate(img_bot, ref_top, cv2.TM_CCOEFF_NORMED)[0][0]))
    return correct_s - flipped_s  # positive = correct, negative = flipped


def _best_orientation_correction(aligned_bgr: np.ndarray, template_bgr: np.ndarray) -> int:
    """Return 0 or 180 to correct the aligned image orientation.
    The card detector + perspective warp already produces a landscape crop,
    so only 0° vs 180° ambiguity needs resolving (not 90/270)."""
    th, tw = template_bgr.shape[:2]
    img = cv2.resize(aligned_bgr, (tw, th))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tmpl_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    # Signal 1: Strip matching — top/bottom strips vs template top/bottom
    strip_diff = _strip_orientation_score(img_gray, tmpl_gray)
    # strip_diff > 0 → correct orientation, < 0 → flipped 180°

    # Signal 2: Full-image scoring for 0° vs 180°
    score_0 = _score_alignment_candidate(img, template_bgr)
    img_180 = cv2.resize(cv2.rotate(img, cv2.ROTATE_180), (tw, th))
    score_180 = _score_alignment_candidate(img_180, template_bgr)

    # Signal 3: Photo-position (top-left should be darker than top-right)
    qh, qw = th // 2, tw // 2
    tl = float(np.mean(img_gray[:qh, :qw]))
    tr = float(np.mean(img_gray[:qh, qw:]))
    bl = float(np.mean(img_gray[qh:, :qw]))
    br = float(np.mean(img_gray[qh:, qw:]))
    photo_0   = max(0.0, (tr - tl) / 255.0)   # correct: tl darker (photo)
    photo_180 = max(0.0, (bl - br) / 255.0)   # flipped: br darker

    # Combined vote for 180° correction (positive → flip needed)
    flip_vote = (
        -0.50 * strip_diff +                   # negative strip_diff → flip
         0.35 * (score_180 - score_0) +         # 180 scores better → flip
         0.15 * (photo_180 - photo_0)           # photo position favours flip
    )

    return 180 if flip_vote > 0.02 else 0


def _aspect_ratio_score(rect: np.ndarray) -> float:
    wc = max(float(np.linalg.norm(rect[1] - rect[0])), float(np.linalg.norm(rect[2] - rect[3])))
    hc = max(float(np.linalg.norm(rect[3] - rect[0])), float(np.linalg.norm(rect[2] - rect[1])))
    if hc < 1.0:
        return 0.0
    card_ratio = 85.6 / 54.0
    actual_ratio = wc / hc
    err = abs(actual_ratio - card_ratio) / card_ratio
    return float(max(0.0, 1.0 - err * 2.0))


def _build_alignment_candidate(
    subject_bgr: np.ndarray,
    template_bgr: np.ndarray,
    template_corners: np.ndarray,
    rotation: int,
) -> Dict:
    rotated_subject = rotate_image(subject_bgr, rotation)
    rot_h, rot_w = rotated_subject.shape[:2]

    # Use the benchmark-proven pipeline_fused detector (replaces old _segment_card_contour)
    _det = _cd.detect_card(rotated_subject, "pipeline_fused", template_bgr=template_bgr)
    contour = _det["contour"]
    contour_score = float(_det["score"])

    candidate = {
        "rotation": int(rotation),
        "valid": False,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "good_matches": 0,
        "score": float(contour_score),
        "orientation_score": 0.0,
        "template_match_score": -1.0,
    }
    if contour is None:
        return candidate

    template_h, template_w = template_bgr.shape[:2]
    rect = order_points(contour.reshape(4, 2).astype("float32"))

    # Check if the detected rectangle has inverted aspect ratio (taller than wide).
    # The card is landscape (~1.585:1), so if width < height, rotate corners by 90°.
    top_w = float(np.linalg.norm(rect[1] - rect[0]))
    left_h = float(np.linalg.norm(rect[3] - rect[0]))
    needs_corner_fix = left_h > top_w * 1.1

    # Build list of corner orderings to try — original + 180° flip,
    # plus CCW and CW rotated corners (+ their flips) if aspect ratio is inverted
    rect_variants = [
        rect,                                                           # original
        np.array([rect[2], rect[3], rect[0], rect[1]], dtype="float32"),  # 180° flip
    ]
    if needs_corner_fix:
        ccw = np.array([rect[3], rect[0], rect[1], rect[2]], dtype="float32")
        cw  = np.array([rect[1], rect[2], rect[3], rect[0]], dtype="float32")
        rect_variants.extend([
            ccw,                                                        # CCW rotation
            np.array([ccw[2], ccw[3], ccw[0], ccw[1]], dtype="float32"),  # CCW + 180° flip
            cw,                                                         # CW rotation
            np.array([cw[2], cw[3], cw[0], cw[1]], dtype="float32"),    # CW + 180° flip
        ])

    tmpl_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    _face_cc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    best_vote = -999.0
    best_aligned = None
    best_rect = rect
    for rv in rect_variants:
        M = cv2.getPerspectiveTransform(rv, template_corners)
        a = cv2.warpPerspective(rotated_subject, M, (template_w, template_h))
        a_rs = cv2.resize(a, (template_w, template_h))
        s = _score_alignment_candidate(a_rs, template_bgr)
        st = _strip_orientation_score(cv2.cvtColor(a_rs, cv2.COLOR_BGR2GRAY), tmpl_gray)
        vote = 0.4 * s + 0.6 * st
        # Face detection bonus: photo is in the left ~35% of a correctly oriented card
        a_gray = cv2.cvtColor(a_rs, cv2.COLOR_BGR2GRAY)
        faces = _face_cc.detectMultiScale(a_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
        if len(faces) > 0:
            for (fx, fy, fw, fh) in faces:
                face_cx = fx + fw / 2
                face_cy = fy + fh / 2
                face_large = fh > template_h * 0.15  # at least 15% of card height
                face_left = face_cx < template_w * 0.40
                face_upper = face_cy < template_h * 0.80
                if face_large and face_left and face_upper:
                    vote += 0.5  # strong bonus for correct photo position
                    break
        if vote > best_vote:
            best_vote = vote
            best_aligned = a
            best_rect = rv

    aligned = best_aligned
    rect = best_rect
    orientation_score = _aspect_ratio_score(rect)
    rot_to_template = cv2.getPerspectiveTransform(rect, template_corners)
    template_to_rot = cv2.getPerspectiveTransform(template_corners, rect)
    tmatch = best_vote

    candidate.update(
        {
            "valid": True,
            "inliers": 4,
            "inlier_ratio": 1.0,
            "good_matches": 4,
            "source_corners": rect,
            "source_to_template": rot_to_template,
            "template_to_source": template_to_rot,
            "aligned": aligned,
            "orientation_score": float(orientation_score),
            "template_match_score": float(tmatch),
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
    # Smart rotation set: each candidate already tries 0° vs 180° flip internally,
    # so rot=180 is equivalent to rot=0+flip and rot=270 is equivalent to rot=90+flip.
    # Only add 90° when source is portrait (card likely needs rotation to become landscape).
    smart_rotations = [0]
    if subject_h > subject_w:
        smart_rotations.append(90)
    for rotation in smart_rotations:
        candidate = _build_alignment_candidate(subject_bgr, template_bgr, template_corners, rotation)
        candidates.append(candidate)
        if candidate["valid"] and (
            best_candidate is None
            or candidate["template_match_score"] > best_candidate.get("template_match_score", -1.0)
        ):
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

    # Orientation correction is now embedded in _build_alignment_candidate
    # (each candidate already picks the better of 0° vs 180° alignment)

    # corners are in rotated-image space; compute rotated dims for normalisation
    rot_subject = rotate_image(subject_bgr, rotation)
    rot_h, rot_w = rot_subject.shape[:2]

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
        "source_size": [int(rot_w), int(rot_h)],
        "card_corners_px": [[float(x), float(y)] for x, y in source_corners],
        "card_corners_norm": [[float(x / max(rot_w, 1)), float(y / max(rot_h, 1))] for x, y in source_corners],
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

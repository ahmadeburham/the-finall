"""
card_detect.py — Multi-method card outline detection.

Every method follows the same API:
    detect_card(image_bgr, method, template_bgr=None) -> dict

Template-trained methods use the template to learn the card appearance and
project its exact corners onto the subject for pixel-precise detection.

Use list_methods() to enumerate available detectors.
Use method_needs_template(name) to check if a method requires a template.
Use rank_results(results) to sort by combined quality score.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import cv2
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

CARD_RATIO = 85.6 / 54.0          # ISO/IEC 7810 ID-1 landscape ratio ≈ 1.585
MIN_AREA_FRAC = 0.05              # card must be ≥5 % of image
MAX_AREA_FRAC = 0.97              # card must be <97 % of image
MAX_RATIO_ERR = 0.50              # reject if aspect-ratio error > 50 %

_METHODS: Dict[str, dict] = {}  # populated by @_register


def _register(name: str, needs_template: bool = False):
    """Decorator to register a detection method."""
    def wrapper(fn):
        _METHODS[name] = {"fn": fn, "needs_template": needs_template}
        return fn
    return wrapper


# ── Shared helpers ────────────────────────────────────────────────────────────

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


def _rect_dims(rect: np.ndarray):
    """Return (width, height) of an ordered 4-point rectangle."""
    wc = max(float(np.linalg.norm(rect[1] - rect[0])),
             float(np.linalg.norm(rect[2] - rect[3])))
    hc = max(float(np.linalg.norm(rect[3] - rect[0])),
             float(np.linalg.norm(rect[2] - rect[1])))
    return wc, hc


def _score_rect(rect: np.ndarray, img_area: float) -> float:
    """Score a candidate rectangle: area fraction – aspect-ratio penalty."""
    wc, hc = _rect_dims(rect)
    if hc < 1.0:
        return -1e18
    area = cv2.contourArea(rect.astype(np.float32))
    frac = area / max(img_area, 1.0)
    if frac < MIN_AREA_FRAC or frac > MAX_AREA_FRAC:
        return -1e18
    ratio = max(wc, hc) / max(min(wc, hc), 1.0)
    err = abs(ratio - CARD_RATIO) / CARD_RATIO
    if err > MAX_RATIO_ERR:
        return -1e18
    return float(frac - err * 0.8)


def _best_quad_from_mask(mask: np.ndarray, img_area: float):
    """Find best 4-point quadrilateral from a binary mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    best_rect, best_score = None, -1e18
    for c in cnts[:8]:
        peri = cv2.arcLength(c, True)
        if peri < 10:
            continue
        for eps in (0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4:
                rect = order_points(approx.reshape(4, 2).astype("float32"))
                sc = _score_rect(rect, img_area)
                if sc > best_score:
                    best_score = sc
                    best_rect = rect
                break
    return best_rect, best_score


def _best_quad_from_contours(cnts, img_area: float):
    """Iterate contours, pick the best 4-pt quad."""
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    best_rect, best_score = None, -1e18
    for c in cnts[:10]:
        peri = cv2.arcLength(c, True)
        if peri < 10:
            continue
        for eps in (0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4:
                rect = order_points(approx.reshape(4, 2).astype("float32"))
                sc = _score_rect(rect, img_area)
                if sc > best_score:
                    best_score = sc
                    best_rect = rect
                break
    return best_rect, best_score


def _draw_outline(img: np.ndarray, rect: Optional[np.ndarray],
                  score: float, method: str) -> np.ndarray:
    vis = img.copy()
    h, w = vis.shape[:2]
    if rect is not None:
        pts_int = rect.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts_int], True, (0, 255, 0), max(2, min(h, w) // 200))
        for i, pt in enumerate(rect.astype(int)):
            cv2.circle(vis, tuple(pt), max(4, min(h, w) // 150), (0, 0, 255), -1)
    # label
    font_scale = max(0.5, min(h, w) / 1200.0)
    thick = max(1, int(font_scale * 2))
    label = f"{method}  score={score:.3f}"
    cv2.putText(vis, label, (10, int(30 * font_scale + 10)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thick)
    return vis


def _prep_gray(image_bgr: np.ndarray):
    """Return (gray, blur) standard preprocessing."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    return gray, blur


# ── Method 1: Otsu threshold + contour ────────────────────────────────────────

@_register("otsu_contour")
def _detect_otsu(image_bgr: np.ndarray, **_kw) -> dict:
    gray, blur = _prep_gray(image_bgr)
    img_area = float(image_bgr.shape[0] * image_bgr.shape[1])

    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rect, score = _best_quad_from_mask(th, img_area)

    # also try inverted
    rect_inv, score_inv = _best_quad_from_mask(255 - th, img_area)
    if score_inv > score:
        rect, score = rect_inv, score_inv

    return {"contour": rect, "score": score, "mask": th}


# ── Method 2: Adaptive threshold + contour ────────────────────────────────────

@_register("adaptive_contour")
def _detect_adaptive(image_bgr: np.ndarray, **_kw) -> dict:
    gray, blur = _prep_gray(image_bgr)
    img_area = float(image_bgr.shape[0] * image_bgr.shape[1])

    best_rect, best_score, best_mask = None, -1e18, None

    for block_size in (11, 21, 31, 51):
        for C in (2, 5, 10):
            th = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block_size, C)
            rect, sc = _best_quad_from_mask(th, img_area)
            if sc > best_score:
                best_rect, best_score, best_mask = rect, sc, th
            # inverted
            rect_inv, sc_inv = _best_quad_from_mask(255 - th, img_area)
            if sc_inv > best_score:
                best_rect, best_score, best_mask = rect_inv, sc_inv, 255 - th

    return {"contour": best_rect, "score": best_score, "mask": best_mask}


# ── Method 3: Multi-scale Canny + contour ─────────────────────────────────────

@_register("canny_contour")
def _detect_canny(image_bgr: np.ndarray, **_kw) -> dict:
    gray, blur = _prep_gray(image_bgr)
    img_area = float(image_bgr.shape[0] * image_bgr.shape[1])

    best_rect, best_score, best_mask = None, -1e18, None

    for low, high in [(20, 60), (30, 90), (50, 150), (80, 200)]:
        edges = cv2.Canny(blur, low, high)
        # dilate to close gaps
        for d_iter in (2, 4):
            dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=d_iter)
            rect, sc = _best_quad_from_mask(dilated, img_area)
            if sc > best_score:
                best_rect, best_score, best_mask = rect, sc, dilated

    return {"contour": best_rect, "score": best_score, "mask": best_mask}


# ── Method 4: GrabCut segmentation ────────────────────────────────────────────

@_register("grabcut")
def _detect_grabcut(image_bgr: np.ndarray, **_kw) -> dict:
    h, w = image_bgr.shape[:2]
    img_area = float(h * w)

    # Downscale for speed if large
    max_side = 800
    scale = 1.0
    work = image_bgr
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        work = cv2.resize(image_bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                          interpolation=cv2.INTER_AREA)

    wh, ww = work.shape[:2]
    # Init rect: center 70% of image
    margin_x, margin_y = int(ww * 0.15), int(wh * 0.15)
    gc_rect = (margin_x, margin_y, ww - 2 * margin_x, wh - 2 * margin_y)

    mask_gc = np.zeros((wh, ww), np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(work, mask_gc, gc_rect, bg_model, fg_model, 5,
                    cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return {"contour": None, "score": -1e18, "mask": None}

    fg_mask = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD),
                       255, 0).astype(np.uint8)

    # Scale mask back up
    if scale != 1.0:
        fg_mask = cv2.resize(fg_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    rect, score = _best_quad_from_mask(fg_mask, img_area)
    return {"contour": rect, "score": score, "mask": fg_mask}


# ── Method 5: K-means color clustering ────────────────────────────────────────

@_register("color_cluster")
def _detect_color_cluster(image_bgr: np.ndarray, **_kw) -> dict:
    h, w = image_bgr.shape[:2]
    img_area = float(h * w)

    # Downscale for speed
    max_side = 600
    scale = 1.0
    work = image_bgr
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        work = cv2.resize(image_bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                          interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape(-1, 3).astype(np.float32)

    best_rect, best_score, best_mask = None, -1e18, None

    for k in (3, 4, 5):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3,
                                        cv2.KMEANS_PP_CENTERS)
        labels = labels.reshape(work.shape[:2])

        # Try each cluster as the card
        for ci in range(k):
            cluster_mask = np.where(labels == ci, 255, 0).astype(np.uint8)
            # Scale up
            if scale != 1.0:
                cluster_mask = cv2.resize(cluster_mask, (w, h),
                                          interpolation=cv2.INTER_NEAREST)
            rect, sc = _best_quad_from_mask(cluster_mask, img_area)
            if sc > best_score:
                best_rect, best_score, best_mask = rect, sc, cluster_mask

    return {"contour": best_rect, "score": best_score, "mask": best_mask}


# ── Method 6: Hough lines intersection ────────────────────────────────────────

@_register("hough_lines")
def _detect_hough(image_bgr: np.ndarray, **_kw) -> dict:
    gray, blur = _prep_gray(image_bgr)
    h, w = image_bgr.shape[:2]
    img_area = float(h * w)

    best_rect, best_score = None, -1e18

    for low, high in [(30, 90), (50, 150), (80, 200)]:
        edges = cv2.Canny(blur, low, high)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                                minLineLength=min(h, w) // 8,
                                maxLineGap=min(h, w) // 15)
        if lines is None or len(lines) < 4:
            continue

        lines = lines.reshape(-1, 4)

        # Classify lines as ~horizontal or ~vertical
        horiz, vert = [], []
        for x1, y1, x2, y2 in lines:
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 30 or angle > 150:
                horiz.append((x1, y1, x2, y2))
            elif 60 < angle < 120:
                vert.append((x1, y1, x2, y2))

        if len(horiz) < 2 or len(vert) < 2:
            continue

        # Sort by position
        horiz.sort(key=lambda l: (l[1] + l[3]) / 2)
        vert.sort(key=lambda l: (l[0] + l[2]) / 2)

        # Pick top/bottom horizontal, left/right vertical
        top_line = horiz[0]
        bot_line = horiz[-1]
        left_line = vert[0]
        right_line = vert[-1]

        # Intersect pairs to get 4 corners
        def _line_to_eq(x1, y1, x2, y2):
            a = y2 - y1
            b = x1 - x2
            c = a * x1 + b * y1
            return a, b, c

        def _intersect(l1, l2):
            a1, b1, c1 = _line_to_eq(*l1)
            a2, b2, c2 = _line_to_eq(*l2)
            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-6:
                return None
            x = (c1 * b2 - c2 * b1) / det
            y = (a1 * c2 - a2 * c1) / det
            return (x, y)

        corners = []
        for h_line in (top_line, bot_line):
            for v_line in (left_line, right_line):
                pt = _intersect(h_line, v_line)
                if pt is not None:
                    corners.append(pt)

        if len(corners) == 4:
            pts = np.array(corners, dtype="float32")
            # Clamp to image bounds
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            rect = order_points(pts)
            sc = _score_rect(rect, img_area)
            if sc > best_score:
                best_rect, best_score = rect, sc

    return {"contour": best_rect, "score": best_score, "mask": None}


# ── Method 7: Morphological gradient + contour ────────────────────────────────

@_register("morph_gradient")
def _detect_morph_gradient(image_bgr: np.ndarray, **_kw) -> dict:
    gray, blur = _prep_gray(image_bgr)
    img_area = float(image_bgr.shape[0] * image_bgr.shape[1])

    best_rect, best_score, best_mask = None, -1e18, None

    for ksize in (3, 5, 7):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
        _, th = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Close to fill card interior
        close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, close_k, iterations=4)

        rect, sc = _best_quad_from_mask(closed, img_area)
        if sc > best_score:
            best_rect, best_score, best_mask = rect, sc, closed

    return {"contour": best_rect, "score": best_score, "mask": best_mask}


# ── Method 8: YOLO card detection ─────────────────────────────────────────────

@_register("yolo_card")
def _detect_yolo(image_bgr: np.ndarray, **_kw) -> dict:
    h, w = image_bgr.shape[:2]
    img_area = float(h * w)

    try:
        from ultralytics import YOLO
    except ImportError:
        return {"contour": None, "score": -1e18, "mask": None,
                "error": "ultralytics not installed. pip install ultralytics"}

    # Use pretrained YOLOv8n — detect generic objects, filter for card-like boxes
    try:
        model = YOLO("yolov8n.pt")
        results = model(image_bgr, verbose=False, conf=0.15)
    except Exception as e:
        return {"contour": None, "score": -1e18, "mask": None, "error": str(e)}

    best_rect, best_score = None, -1e18

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])

            bw, bh = x2 - x1, y2 - y1
            area = bw * bh
            frac = area / img_area
            if frac < MIN_AREA_FRAC or frac > MAX_AREA_FRAC:
                continue

            ratio = max(bw, bh) / max(min(bw, bh), 1.0)
            err = abs(ratio - CARD_RATIO) / CARD_RATIO
            if err > MAX_RATIO_ERR:
                continue

            rect = order_points(np.array([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ], dtype="float32"))
            sc = frac - err * 0.8 + conf * 0.3
            if sc > best_score:
                best_rect, best_score = rect, sc

    # If YOLO found a box, try to refine to actual card contour inside it
    if best_rect is not None:
        x_min = int(max(0, best_rect[:, 0].min() - 10))
        y_min = int(max(0, best_rect[:, 1].min() - 10))
        x_max = int(min(w, best_rect[:, 0].max() + 10))
        y_max = int(min(h, best_rect[:, 1].max() + 10))
        roi = image_bgr[y_min:y_max, x_min:x_max]
        if roi.size > 0:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
            edges = cv2.Canny(roi_blur, 30, 100)
            dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
            roi_area = float(roi.shape[0] * roi.shape[1])
            refined, refined_sc = _best_quad_from_mask(dilated, roi_area)
            if refined is not None:
                # Offset back to full image coords
                refined[:, 0] += x_min
                refined[:, 1] += y_min
                final_sc = _score_rect(refined, img_area)
                if final_sc > -1e18:
                    best_rect, best_score = refined, final_sc

    return {"contour": best_rect, "score": best_score, "mask": None}


# ══════════════════════════════════════════════════════════════════════════════
#  TEMPLATE-TRAINED METHODS
# ══════════════════════════════════════════════════════════════════════════════

def _feature_homography(image_bgr: np.ndarray, template_bgr: np.ndarray,
                        detector_name: str) -> dict:
    """
    Core logic shared by SIFT / AKAZE / ORB homography methods.
    1. Detect keypoints+descriptors in both template and subject.
    2. Match descriptors via BFMatcher + Lowe's ratio test.
    3. Find homography via RANSAC.
    4. Project the 4 known template corners onto the subject → exact card outline.
    Tries 4 rotations of the subject to handle any phone orientation.
    """
    if template_bgr is None:
        return {"contour": None, "score": -1e18, "mask": None,
                "error": f"{detector_name}: no template provided"}

    th, tw = template_bgr.shape[:2]
    orig_h, orig_w = image_bgr.shape[:2]
    img_area = float(orig_h * orig_w)

    # Downscale large images to prevent OOM (max 1600px on longest side)
    MAX_SIDE = 1600
    scale_factor = 1.0
    work_img = image_bgr
    if max(orig_h, orig_w) > MAX_SIDE:
        scale_factor = MAX_SIDE / max(orig_h, orig_w)
        new_w = max(1, int(orig_w * scale_factor))
        new_h = max(1, int(orig_h * scale_factor))
        work_img = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Template corners (what we project through the homography)
    tmpl_corners = np.array([
        [0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]
    ], dtype="float32").reshape(-1, 1, 2)

    # Create detector
    if detector_name == "SIFT":
        det = cv2.SIFT_create(nfeatures=3000)
        norm = cv2.NORM_L2
        ratio_thresh = 0.75
    elif detector_name == "AKAZE":
        det = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
        ratio_thresh = 0.80
    elif detector_name == "ORB":
        det = cv2.ORB_create(nfeatures=5000)
        norm = cv2.NORM_HAMMING
        ratio_thresh = 0.80
    else:
        return {"contour": None, "score": -1e18, "mask": None,
                "error": f"Unknown detector: {detector_name}"}

    # Detect on template
    tmpl_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    kp_t, des_t = det.detectAndCompute(tmpl_gray, None)
    if des_t is None or len(kp_t) < 10:
        return {"contour": None, "score": -1e18, "mask": None,
                "error": f"{detector_name}: too few template keypoints"}

    _ROTATE_CODES = {
        0: None,
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }

    best_rect = None
    best_score = -1e18
    best_inliers = 0

    for rot_deg, rot_code in _ROTATE_CODES.items():
        if rot_code is not None:
            subj_rot = cv2.rotate(work_img, rot_code)
        else:
            subj_rot = work_img

        subj_gray = cv2.cvtColor(subj_rot, cv2.COLOR_BGR2GRAY)
        kp_s, des_s = det.detectAndCompute(subj_gray, None)
        if des_s is None or len(kp_s) < 10:
            continue

        # Match
        bf = cv2.BFMatcher(norm)
        raw_matches = bf.knnMatch(des_t, des_s, k=2)

        # Lowe's ratio test
        good = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < ratio_thresh * n.distance:
                    good.append(m)

        if len(good) < 8:
            continue

        # Compute homography
        pts_t = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_s = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, inlier_mask = cv2.findHomography(pts_t, pts_s, cv2.RANSAC, 5.0)
        if H is None:
            continue

        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        inlier_ratio = inliers / max(len(good), 1)

        if inliers < 6 or inlier_ratio < 0.15:
            continue

        # Project template corners to rotated-subject space
        projected = cv2.perspectiveTransform(tmpl_corners, H)
        pts4 = projected.reshape(4, 2).astype("float32")

        # Map back to original (unrotated) image coordinates
        # First map to work_img coords, then scale up
        sh, sw = work_img.shape[:2]
        rh, rw = subj_rot.shape[:2]
        if rot_deg == 90:
            mapped = np.column_stack([pts4[:, 1], rw - 1 - pts4[:, 0]])
        elif rot_deg == 180:
            mapped = np.column_stack([rw - 1 - pts4[:, 0], rh - 1 - pts4[:, 1]])
        elif rot_deg == 270:
            mapped = np.column_stack([rh - 1 - pts4[:, 1], pts4[:, 0]])
        else:
            mapped = pts4.copy()

        mapped = mapped.astype("float32")
        # Scale back to original image size
        if scale_factor != 1.0:
            mapped[:, 0] /= scale_factor
            mapped[:, 1] /= scale_factor
        mapped[:, 0] = np.clip(mapped[:, 0], 0, orig_w - 1)
        mapped[:, 1] = np.clip(mapped[:, 1], 0, orig_h - 1)

        rect = order_points(mapped)
        sc = _score_rect(rect, img_area)

        # Boost score with inlier quality
        sc_boosted = sc + inlier_ratio * 0.5 + min(inliers / 50.0, 0.5)

        if sc_boosted > best_score:
            best_rect = rect
            best_score = sc_boosted
            best_inliers = inliers

    return {"contour": best_rect, "score": best_score, "mask": None,
            "inliers": best_inliers}


# ── Method 9: SIFT homography (template-trained) ─────────────────────────────

@_register("sift_homography", needs_template=True)
def _detect_sift(image_bgr: np.ndarray, template_bgr=None, **_kw) -> dict:
    return _feature_homography(image_bgr, template_bgr, "SIFT")


# ── Method 10: AKAZE homography (template-trained) ───────────────────────────

@_register("akaze_homography", needs_template=True)
def _detect_akaze(image_bgr: np.ndarray, template_bgr=None, **_kw) -> dict:
    return _feature_homography(image_bgr, template_bgr, "AKAZE")


# ── Method 11: ORB homography (template-trained) ─────────────────────────────

@_register("orb_homography", needs_template=True)
def _detect_orb(image_bgr: np.ndarray, template_bgr=None, **_kw) -> dict:
    return _feature_homography(image_bgr, template_bgr, "ORB")


# ── Method 12: Multi-scale template matching ─────────────────────────────────

@_register("template_match", needs_template=True)
def _detect_template_match(image_bgr: np.ndarray, template_bgr=None, **_kw) -> dict:
    """
    Multi-scale, multi-rotation NCC template matching.
    Slides the template (at many scales) over the subject, finds the best
    match location, then estimates the 4-corner bounding box.
    """
    if template_bgr is None:
        return {"contour": None, "score": -1e18, "mask": None,
                "error": "template_match: no template provided"}

    h, w = image_bgr.shape[:2]
    img_area = float(h * w)
    tmpl_h, tmpl_w = template_bgr.shape[:2]

    subj_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    tmpl_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    best_rect = None
    best_val = -1.0

    for rot_code in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180,
                     cv2.ROTATE_90_COUNTERCLOCKWISE):
        if rot_code is not None:
            work_subj = cv2.rotate(subj_gray, rot_code)
        else:
            work_subj = subj_gray

        wh, ww = work_subj.shape[:2]

        for scale in np.linspace(0.15, 1.0, 18):
            sw_t = max(10, int(tmpl_w * scale))
            sh_t = max(10, int(tmpl_h * scale))
            if sw_t >= ww or sh_t >= wh:
                continue
            resized_tmpl = cv2.resize(tmpl_gray, (sw_t, sh_t))
            result = cv2.matchTemplate(work_subj, resized_tmpl,
                                       cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_val:
                x, y = max_loc
                pts_rot = np.array([
                    [x, y],
                    [x + sw_t - 1, y],
                    [x + sw_t - 1, y + sh_t - 1],
                    [x, y + sh_t - 1]
                ], dtype="float32")

                # Map back to original coordinates
                oh, ow = image_bgr.shape[:2]
                rh, rw = work_subj.shape[:2]
                if rot_code == cv2.ROTATE_90_CLOCKWISE:
                    mapped = np.column_stack([pts_rot[:, 1],
                                              rw - 1 - pts_rot[:, 0]])
                elif rot_code == cv2.ROTATE_180:
                    mapped = np.column_stack([rw - 1 - pts_rot[:, 0],
                                              rh - 1 - pts_rot[:, 1]])
                elif rot_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
                    mapped = np.column_stack([rh - 1 - pts_rot[:, 1],
                                              pts_rot[:, 0]])
                else:
                    mapped = pts_rot.copy()

                mapped = mapped.astype("float32")
                mapped[:, 0] = np.clip(mapped[:, 0], 0, ow - 1)
                mapped[:, 1] = np.clip(mapped[:, 1], 0, oh - 1)

                best_rect = order_points(mapped)
                best_val = max_val

    # Convert NCC score to our scoring scale
    if best_rect is not None:
        geo_score = _score_rect(best_rect, img_area)
        if geo_score > -1e17:
            final_score = geo_score + best_val * 0.5
        else:
            best_rect = None
            final_score = -1e18
    else:
        final_score = -1e18

    return {"contour": best_rect, "score": final_score, "mask": None}


# ── Method 13: Ensemble best-of-all (template-trained) ───────────────────────

@_register("ensemble_best", needs_template=True)
def _detect_ensemble(image_bgr: np.ndarray, template_bgr=None, **_kw) -> dict:
    """
    Runs SIFT + AKAZE + ORB homography, picks the result with the
    highest inlier-boosted score. Combines three detectors' strengths.
    """
    results = []
    for det_name in ("SIFT", "AKAZE", "ORB"):
        try:
            r = _feature_homography(image_bgr, template_bgr, det_name)
        except Exception:
            r = {"contour": None, "score": -1e18, "mask": None}
        results.append(r)

    results.sort(key=lambda r: r.get("score", -1e18), reverse=True)
    best = results[0]
    return {
        "contour": best["contour"],
        "score": best["score"],
        "mask": None,
        "inliers": best.get("inliers", 0),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FUSED PIPELINE (chains the top 2 methods for maximum accuracy)
# ══════════════════════════════════════════════════════════════════════════════

def _refine_corners_subpix(image_bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Apply cv2.cornerSubPix for sub-pixel corner refinement."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # cornerSubPix needs (N,1,2) float32
    pts = corners.reshape(-1, 1, 2).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
    try:
        refined = cv2.cornerSubPix(gray, pts, (11, 11), (-1, -1), criteria)
        return refined.reshape(4, 2).astype(np.float32)
    except cv2.error:
        return corners


@_register("pipeline_fused", needs_template=True)
def _detect_pipeline_fused(image_bgr: np.ndarray, template_bgr=None, **_kw) -> dict:
    """
    Fused pipeline — chains the benchmark-proven top 2 methods:
      Stage A (Primary):   ensemble_best  (SIFT+AKAZE+ORB, picks best)
      Stage B (Validator): sift_homography (independent SIFT-only run)
      Stage C (Fallback):  color_cluster   (generic, no template needed)

    Fusion logic:
      - Both A & B succeed and agree (<25px): average corners → sub-pixel refine
      - Both succeed but disagree: trust higher score, shrink 2px inward
      - Only one succeeds: use it, lower confidence
      - Neither: try fallback C
    """
    img_area = float(image_bgr.shape[0] * image_bgr.shape[1])
    h, w = image_bgr.shape[:2]

    # ── Stage A: ensemble_best ──
    try:
        ra = _detect_ensemble(image_bgr, template_bgr=template_bgr)
    except Exception:
        ra = {"contour": None, "score": -1e18}
    # ── Stage B: sift_homography (independent run) ──
    try:
        rb = _feature_homography(image_bgr, template_bgr, "SIFT")
    except Exception:
        rb = {"contour": None, "score": -1e18}

    ca, sa = ra.get("contour"), float(ra.get("score", -1e18))
    cb, sb = rb.get("contour"), float(rb.get("score", -1e18))

    a_ok = ca is not None and sa > -1e17
    b_ok = cb is not None and sb > -1e17

    AGREE_THRESH = 25.0  # px mean corner distance to consider "agreement"

    if a_ok and b_ok:
        # Both succeeded — check agreement
        corner_dist = float(np.mean(np.linalg.norm(ca - cb, axis=1)))

        if corner_dist < AGREE_THRESH:
            # ── AGREE: average corners for maximum precision ──
            fused = (ca + cb) / 2.0
            fused = _refine_corners_subpix(image_bgr, fused)
            fused = order_points(fused)
            fused[:, 0] = np.clip(fused[:, 0], 0, w - 1)
            fused[:, 1] = np.clip(fused[:, 1], 0, h - 1)
            sc = _score_rect(fused, img_area)
            # Boost: agreement bonus
            sc_final = sc + 0.5 + min(corner_dist, 25) / 50.0
            return {"contour": fused, "score": sc_final, "mask": None}
        else:
            # ── DISAGREE: trust higher score, shrink 2px inward ──
            if sa >= sb:
                winner = ca.copy()
            else:
                winner = cb.copy()
            # Shrink inward by 2px on each corner toward centroid
            centroid = winner.mean(axis=0)
            shrink = 2.0
            for i in range(4):
                direction = centroid - winner[i]
                length = np.linalg.norm(direction)
                if length > 0:
                    winner[i] += direction / length * shrink
            winner = order_points(winner)
            winner[:, 0] = np.clip(winner[:, 0], 0, w - 1)
            winner[:, 1] = np.clip(winner[:, 1], 0, h - 1)
            winner = _refine_corners_subpix(image_bgr, winner)
            sc = _score_rect(winner, img_area)
            sc_final = sc + 0.2
            return {"contour": winner, "score": sc_final, "mask": None}

    elif a_ok:
        refined = _refine_corners_subpix(image_bgr, ca)
        refined = order_points(refined)
        sc = _score_rect(refined, img_area)
        return {"contour": refined, "score": sc + 0.1, "mask": None}

    elif b_ok:
        refined = _refine_corners_subpix(image_bgr, cb)
        refined = order_points(refined)
        sc = _score_rect(refined, img_area)
        return {"contour": refined, "score": sc + 0.1, "mask": None}

    else:
        # ── Stage C: fallback to color_cluster (no template needed) ──
        rc = _detect_color_cluster(image_bgr)
        cc = rc.get("contour")
        if cc is not None:
            refined = _refine_corners_subpix(image_bgr, cc)
            refined = order_points(refined)
            sc = _score_rect(refined, img_area)
            return {"contour": refined, "score": sc, "mask": rc.get("mask")}

        return {"contour": None, "score": -1e18, "mask": None}


# ── Public API ────────────────────────────────────────────────────────────────

def list_methods() -> List[str]:
    """Return names of all registered detection methods."""
    return list(_METHODS.keys())


def method_needs_template(method: str) -> bool:
    """Check if a method requires a template image."""
    if method not in _METHODS:
        return False
    return _METHODS[method]["needs_template"]


def detect_card(image_bgr: np.ndarray, method: str,
                template_bgr: Optional[np.ndarray] = None) -> Dict:
    """
    Run a single detection method on an image.

    For template-based methods, pass template_bgr (the front/back template).
    Generic methods ignore it.

    Returns dict with keys:
        contour     : np.ndarray (4,2) float32 or None
        score       : float (-inf if failed)
        mask        : np.ndarray or None (debug visualization)
        debug_img   : np.ndarray  (image with outline drawn)
        method      : str
        time_ms     : float
        aspect_ratio: float or None
        width       : float or None
        height      : float or None
        error       : str or None
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown method: {method!r}. Available: {list_methods()}")

    entry = _METHODS[method]
    fn = entry["fn"]

    t0 = time.perf_counter()
    try:
        if entry["needs_template"]:
            result = fn(image_bgr, template_bgr=template_bgr)
        else:
            result = fn(image_bgr)
    except Exception as e:
        result = {"contour": None, "score": -1e18, "mask": None, "error": str(e)}
    elapsed = (time.perf_counter() - t0) * 1000.0

    rect = result.get("contour")
    score = float(result.get("score", -1e18))

    aspect_ratio = None
    wc_val = hc_val = None
    if rect is not None:
        wc_val, hc_val = _rect_dims(rect)
        if hc_val > 0:
            aspect_ratio = wc_val / hc_val

    # Build debug image — downscale if very large to avoid OOM
    try:
        dbg_h, dbg_w = image_bgr.shape[:2]
        if max(dbg_h, dbg_w) > 2000:
            dbg_scale = 2000 / max(dbg_h, dbg_w)
            dbg_img = cv2.resize(image_bgr,
                (max(1, int(dbg_w * dbg_scale)), max(1, int(dbg_h * dbg_scale))),
                interpolation=cv2.INTER_AREA)
            dbg_rect = rect.copy() * dbg_scale if rect is not None else None
        else:
            dbg_img = image_bgr
            dbg_rect = rect
        debug_vis = _draw_outline(dbg_img, dbg_rect, score, method)
    except Exception:
        debug_vis = None

    return {
        "contour": rect,
        "score": score,
        "mask": result.get("mask"),
        "debug_img": debug_vis,
        "method": method,
        "time_ms": round(elapsed, 1),
        "aspect_ratio": aspect_ratio,
        "width": wc_val,
        "height": hc_val,
        "error": result.get("error"),
    }


def detect_all(image_bgr: np.ndarray,
               template_bgr: Optional[np.ndarray] = None) -> List[Dict]:
    """Run every registered method and return ranked results."""
    results = []
    for name in _METHODS:
        results.append(detect_card(image_bgr, name, template_bgr=template_bgr))
    return rank_results(results)


def rank_results(results: List[Dict]) -> List[Dict]:
    """Sort detection results by score descending. Add 'rank' key."""
    ranked = sorted(results, key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(ranked):
        r["rank"] = i + 1
    return ranked

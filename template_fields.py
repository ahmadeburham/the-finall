import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif"}


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
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


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return image.copy()
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported angle: {angle}")


def _create_detector(method: str):
    if method == "sift":
        if not hasattr(cv2, "SIFT_create"):
            return None, None, None, None
        return cv2.SIFT_create(nfeatures=5000), cv2.NORM_L2, True, 0.75
    if method == "orb":
        return cv2.ORB_create(nfeatures=8000, fastThreshold=5), cv2.NORM_HAMMING, False, 0.85
    if method == "akaze":
        return cv2.AKAZE_create(), cv2.NORM_HAMMING, False, 0.80
    return None, None, None, None


def _match_and_align(template_bgr: np.ndarray, subject_bgr: np.ndarray, method: str):
    tpl_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    sub_gray = cv2.cvtColor(subject_bgr, cv2.COLOR_BGR2GRAY)

    detector, norm, use_flann, ratio = _create_detector(method)
    if detector is None:
        return None

    kp_t, des_t = detector.detectAndCompute(tpl_gray, None)
    kp_s, des_s = detector.detectAndCompute(sub_gray, None)

    if des_t is None or des_s is None or len(kp_t) < 4 or len(kp_s) < 4:
        return None

    if use_flann:
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    else:
        matcher = cv2.BFMatcher(norm, crossCheck=False)

    knn = matcher.knnMatch(des_t, des_s, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 4:
        return None

    pts_t = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_s = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_s, pts_t, cv2.RANSAC, 4.0)
    if H is None or mask is None:
        return None

    mask = mask.ravel().astype(bool)
    inliers = int(mask.sum())
    inlier_ratio = float(inliers / max(len(mask), 1))
    if inliers < 8:
        return None

    h, w = template_bgr.shape[:2]
    aligned = cv2.warpPerspective(subject_bgr, H, (w, h))
    score = float(inliers * inlier_ratio)

    return {
        "aligned": aligned,
        "method": method,
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "score": score,
        "good_matches": len(good),
    }


def align_card_to_template(subject_image: str | Path, template_image: str | Path, return_candidates: bool = False):
    template_bgr = read_image(template_image)
    subject_bgr = read_image(subject_image)

    best = None
    candidates: List[Dict] = []

    for angle in [0, 90, 180, 270]:
        rotated = rotate_image(subject_bgr, angle)
        for method in ["sift", "akaze", "orb"]:
            res = _match_and_align(template_bgr, rotated, method)
            if res is None:
                candidates.append({"method": method, "rotation": angle, "valid": False})
                continue
            res["rotation"] = angle
            candidates.append(
                {
                    "method": method,
                    "rotation": angle,
                    "valid": True,
                    "inliers": res["inliers"],
                    "inlier_ratio": res["inlier_ratio"],
                    "score": res["score"],
                    "good_matches": res["good_matches"],
                }
            )
            if best is None or res["score"] > best["score"]:
                best = res

    if best is None:
        raise RuntimeError("Could not align card to template")

    info = {
        "method": best["method"],
        "rotation": best["rotation"],
        "inliers": best["inliers"],
        "inlier_ratio": best["inlier_ratio"],
        "score": best["score"],
        "good_matches": best["good_matches"],
        "template_size": [template_bgr.shape[1], template_bgr.shape[0]],
    }
    if return_candidates:
        info["candidates"] = candidates
    return best["aligned"], info


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


def extract_side_artifacts(aligned_bgr: np.ndarray, config: Dict, side: str) -> Dict:
    side_cfg = config[side]
    out = {"aligned": aligned_bgr, "cleaned": None, "photo": None, "crops": {}}

    rois = side_cfg.get("ocr_rois", {})
    if rois:
        out["cleaned"] = build_cleaned_image(aligned_bgr, rois)

    if "photo_roi" in side_cfg:
        photo = crop_norm_box(aligned_bgr, side_cfg["photo_roi"], pad_px=4)
        out["photo"] = refine_crop_tight(photo, mode="text")

    pad = int(side_cfg.get("roi_pad_px", 4))
    numeric_fields = set(side_cfg.get("numeric_fields", ["id_number", "birth_date"]))
    refine_numeric = bool(side_cfg.get("refine_numeric", True))
    refine_text = bool(side_cfg.get("refine_text", True))

    for name, box in rois.items():
        crop = crop_norm_box(aligned_bgr, box, pad_px=pad)
        if name in numeric_fields and refine_numeric:
            crop = refine_crop_tight(crop, mode="numeric")
        elif name not in numeric_fields and refine_text:
            crop = refine_crop_tight(crop, mode="text")
        out["crops"][name] = crop

    return out


def save_debug_artifacts(debug_dir: str | Path, prefix: str, artifacts: Dict):
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    if artifacts.get("aligned") is not None:
        cv2.imwrite(str(debug_dir / f"{prefix}_aligned.png"), artifacts["aligned"])
    if artifacts.get("cleaned") is not None:
        cv2.imwrite(str(debug_dir / f"{prefix}_cleaned.png"), artifacts["cleaned"])
    if artifacts.get("photo") is not None:
        cv2.imwrite(str(debug_dir / f"{prefix}_photo.png"), artifacts["photo"])

    for name, crop in artifacts.get("crops", {}).items():
        cv2.imwrite(str(debug_dir / f"{prefix}_{name}.png"), crop)

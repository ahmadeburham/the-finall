import argparse
import csv
import json
import math
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif"}


@dataclass
class MatchResult:
    model_name: str
    feature_name: str
    feature_path: str
    subject_path: str
    success: bool
    reason: str
    kp_feature: int = 0
    kp_subject: int = 0
    raw_matches: int = 0
    good_matches: int = 0
    homography_found: bool = False
    inliers: int = 0
    inlier_ratio: float = 0.0
    confidence_mean: float = 0.0
    quad_area_ratio: float = 0.0
    score: float = 0.0
    found: bool = False
    elapsed_ms: float = 0.0


class BaseMatcher:
    name = "base"

    def match(
        self, feature_bgr: np.ndarray, subject_bgr: np.ndarray
    ) -> Tuple[MatchResult, Optional[List[cv2.KeyPoint]], Optional[List[cv2.KeyPoint]], Optional[np.ndarray], Optional[List[cv2.DMatch]]]:
        raise NotImplementedError


def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def read_image(path: Path, max_side: int = 2048) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    if max_side and max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return img


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def polygon_area(pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def draw_projected_quad(subject_bgr: np.ndarray, quad: Optional[np.ndarray], found: bool) -> np.ndarray:
    vis = subject_bgr.copy()
    if quad is not None and len(quad) >= 4:
        quad = np.int32(quad.reshape(-1, 1, 2))
        color = (0, 255, 0) if found else (0, 0, 255)
        cv2.polylines(vis, [quad], True, color, 3, cv2.LINE_AA)
    return vis


def save_match_viz(
    feature_bgr: np.ndarray,
    subject_bgr: np.ndarray,
    kp0: Optional[List[cv2.KeyPoint]],
    kp1: Optional[List[cv2.KeyPoint]],
    good_matches: Optional[List[cv2.DMatch]],
    quad: Optional[np.ndarray],
    out_path: Path,
    found: bool,
) -> None:
    kp0 = kp0 or []
    kp1 = kp1 or []
    good_matches = good_matches or []

    subject_marked = draw_projected_quad(subject_bgr, quad, found)

    if kp0 and kp1 and good_matches:
        draw_n = min(len(good_matches), 80)
        vis = cv2.drawMatches(
            feature_bgr,
            kp0,
            subject_marked,
            kp1,
            good_matches[:draw_n],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
    else:
        h = max(feature_bgr.shape[0], subject_marked.shape[0])
        w = max(feature_bgr.shape[1], subject_marked.shape[1])
        left = cv2.resize(feature_bgr, (w, h))
        right = cv2.resize(subject_marked, (w, h))
        vis = np.hstack([left, right])

    cv2.imwrite(str(out_path), vis)


def homography_from_matches(
    pts0: np.ndarray,
    pts1: np.ndarray,
    feature_shape: Tuple[int, int, int],
    subject_shape: Tuple[int, int, int],
    ransac_thresh: float = 4.0,
) -> Tuple[bool, Optional[np.ndarray], int, float, float]:
    if len(pts0) < 4 or len(pts1) < 4:
        return False, None, 0, 0.0, 0.0

    H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, ransac_thresh)
    if H is None or mask is None:
        return False, None, 0, 0.0, 0.0

    mask = mask.ravel().astype(bool)
    inliers = int(mask.sum())
    inlier_ratio = float(inliers / max(len(mask), 1))

    h0, w0 = feature_shape[:2]
    corners = np.float32([[0, 0], [w0 - 1, 0], [w0 - 1, h0 - 1], [0, h0 - 1]]).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    h1, w1 = subject_shape[:2]
    quad_area = polygon_area(quad)
    scene_area = max(float(h1 * w1), 1.0)
    quad_area_ratio = float(quad_area / scene_area)

    return True, quad, inliers, inlier_ratio, quad_area_ratio


def score_detection(
    inliers: int,
    inlier_ratio: float,
    quad_area_ratio: float,
    confidence_mean: float = 0.0,
) -> float:
    inlier_term = min(inliers / 30.0, 1.0)
    ratio_term = min(max(inlier_ratio, 0.0), 1.0)
    area_term = min(quad_area_ratio / 0.06, 1.0)
    conf_term = min(max(confidence_mean, 0.0), 1.0)
    return float(0.45 * inlier_term + 0.35 * ratio_term + 0.10 * area_term + 0.10 * conf_term)


def found_from_metrics(inliers: int, inlier_ratio: float, score: float) -> bool:
    if inlier_ratio >= 0.65 and inliers >= 4 and score >= 0.30:
        return True
    return bool(inliers >= 8 and inlier_ratio >= 0.20 and score >= 0.35)


class OpenCVFeatureMatcher(BaseMatcher):
    def __init__(self, name: str, detector, norm_type: int, ratio: float = 0.75, use_flann: bool = False):
        self.name = name
        self.detector = detector
        self.norm_type = norm_type
        self.ratio = ratio
        self.use_flann = use_flann

    def _matcher(self):
        if self.use_flann and self.norm_type == cv2.NORM_L2:
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=64)
            return cv2.FlannBasedMatcher(index_params, search_params)
        return cv2.BFMatcher(self.norm_type, crossCheck=False)

    def match(self, feature_bgr, subject_bgr):
        t0 = time.perf_counter()
        feature_gray = to_gray(feature_bgr)
        subject_gray = to_gray(subject_bgr)

        kp0, des0 = self.detector.detectAndCompute(feature_gray, None)
        kp1, des1 = self.detector.detectAndCompute(subject_gray, None)

        result = MatchResult(
            model_name=self.name,
            feature_name="",
            feature_path="",
            subject_path="",
            success=False,
            reason="",
            kp_feature=0 if kp0 is None else len(kp0),
            kp_subject=0 if kp1 is None else len(kp1),
        )

        if des0 is None or des1 is None or len(kp0) < 4 or len(kp1) < 4:
            result.reason = "not enough descriptors"
            result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return result, kp0 or [], kp1 or [], None, []

        matcher = self._matcher()
        knn = matcher.knnMatch(des0, des1, k=2)

        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                good.append(m)

        result.raw_matches = len(knn)
        result.good_matches = len(good)

        if len(good) < 4:
            result.reason = "not enough good matches"
            result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return result, kp0, kp1, None, good

        pts0 = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts1 = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        ok, quad, inliers, inlier_ratio, quad_area_ratio = homography_from_matches(pts0, pts1, feature_bgr.shape, subject_bgr.shape)

        result.homography_found = ok
        result.inliers = inliers
        result.inlier_ratio = inlier_ratio
        result.quad_area_ratio = quad_area_ratio
        result.score = score_detection(inliers, inlier_ratio, quad_area_ratio)
        result.found = found_from_metrics(inliers, inlier_ratio, result.score)
        result.success = True
        result.reason = "ok" if ok else "homography failed"
        result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return result, kp0, kp1, quad, good


class LoFTRMatcher(BaseMatcher):
    def __init__(self, pretrained: str = "outdoor", max_side: int = 1600):
        self.name = f"loftr_{pretrained}"
        self.pretrained = pretrained
        self.max_side = max_side
        self.available = False
        self.reason = ""

        if torch is None:
            self.reason = "torch not installed"
            return

        try:
            from kornia.feature import LoFTR

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.matcher = LoFTR(pretrained=pretrained).to(self.device).eval()
            self.available = True
        except Exception as e:
            self.reason = str(e)

    def _tensor_from_bgr(self, img_bgr: np.ndarray):
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = img_gray.shape[:2]
        scale = 1.0
        if max(h, w) > self.max_side:
            scale = self.max_side / max(h, w)
            img_gray = cv2.resize(img_gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        ten = torch.from_numpy(img_gray).float() / 255.0
        ten = ten.unsqueeze(0).unsqueeze(0).to(self.device)
        return ten, scale

    def match(self, feature_bgr, subject_bgr):
        t0 = time.perf_counter()
        result = MatchResult(
            model_name=self.name,
            feature_name="",
            feature_path="",
            subject_path="",
            success=False,
            reason=self.reason if not self.available else "",
        )

        if not self.available:
            result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return result, [], [], None, []

        with torch.inference_mode():
            img0, s0 = self._tensor_from_bgr(feature_bgr)
            img1, s1 = self._tensor_from_bgr(subject_bgr)
            out = self.matcher({"image0": img0, "image1": img1})

        kpts0 = out["keypoints0"].detach().cpu().numpy()
        kpts1 = out["keypoints1"].detach().cpu().numpy()
        conf = out.get("confidence", None)
        conf = conf.detach().cpu().numpy() if conf is not None else np.zeros((len(kpts0),), dtype=np.float32)

        if s0 != 1.0:
            kpts0 = kpts0 / s0
        if s1 != 1.0:
            kpts1 = kpts1 / s1

        result.kp_feature = len(kpts0)
        result.kp_subject = len(kpts1)
        result.raw_matches = len(kpts0)
        result.good_matches = len(kpts0)
        result.confidence_mean = float(conf.mean()) if len(conf) else 0.0

        if len(kpts0) < 4 or len(kpts1) < 4:
            result.reason = "not enough matches"
            result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return result, [], [], None, []

        pts0 = kpts0.reshape(-1, 1, 2).astype(np.float32)
        pts1 = kpts1.reshape(-1, 1, 2).astype(np.float32)

        ok, quad, inliers, inlier_ratio, quad_area_ratio = homography_from_matches(pts0, pts1, feature_bgr.shape, subject_bgr.shape)

        result.homography_found = ok
        result.inliers = inliers
        result.inlier_ratio = inlier_ratio
        result.quad_area_ratio = quad_area_ratio
        result.score = score_detection(inliers, inlier_ratio, quad_area_ratio, result.confidence_mean)
        result.found = found_from_metrics(inliers, inlier_ratio, result.score)
        result.success = True
        result.reason = "ok" if ok else "homography failed"
        result.elapsed_ms = (time.perf_counter() - t0) * 1000.0

        kp0 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kpts0]
        kp1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kpts1]
        good = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=float(1.0 - (conf[i] if i < len(conf) else 0.0))) for i in range(len(kpts0))]
        return result, kp0, kp1, quad, good


class LightGlueMatcher(BaseMatcher):
    def __init__(self, feature_type: str, max_keypoints: int = 2048):
        self.name = f"lightglue_{feature_type}"
        self.feature_type = feature_type
        self.max_keypoints = max_keypoints
        self.available = False
        self.reason = ""
        self.device = "cpu"

        if torch is None:
            self.reason = "torch not installed"
            return

        try:
            from lightglue import ALIKED, DISK, LightGlue, SIFT, SuperPoint
            from lightglue.utils import load_image, rbd

            extractor_map = {
                "superpoint": SuperPoint,
                "disk": DISK,
                "aliked": ALIKED,
                "sift": SIFT,
            }

            self.load_image = load_image
            self.rbd = rbd
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.extractor = extractor_map[feature_type](max_num_keypoints=max_keypoints).eval().to(self.device)
            self.matcher = LightGlue(features=feature_type).eval().to(self.device)
            self.available = True
        except Exception as e:
            self.reason = str(e)

    def match(self, feature_bgr, subject_bgr):
        t0 = time.perf_counter()
        result = MatchResult(
            model_name=self.name,
            feature_name="",
            feature_path="",
            subject_path="",
            success=False,
            reason=self.reason if not self.available else "",
        )

        if not self.available:
            result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return result, [], [], None, []

        tmp_dir = Path(tempfile.mkdtemp(prefix="lightglue_tmp_"))
        tmp0 = tmp_dir / "feature.png"
        tmp1 = tmp_dir / "subject.png"

        cv2.imwrite(str(tmp0), feature_bgr)
        cv2.imwrite(str(tmp1), subject_bgr)

        try:
            image0 = self.load_image(str(tmp0)).to(self.device)
            image1 = self.load_image(str(tmp1)).to(self.device)

            with torch.inference_mode():
                feats0 = self.extractor.extract(image0)
                feats1 = self.extractor.extract(image1)
                matches01 = self.matcher({"image0": feats0, "image1": feats1})

            feats0, feats1, matches01 = [self.rbd(x) for x in [feats0, feats1, matches01]]

            matches = matches01["matches"].detach().cpu().numpy()
            scores = matches01.get("scores", None)
            scores = scores.detach().cpu().numpy() if scores is not None else None

            kpts0 = feats0["keypoints"].detach().cpu().numpy()
            kpts1 = feats1["keypoints"].detach().cpu().numpy()

            result.kp_feature = len(kpts0)
            result.kp_subject = len(kpts1)
            result.raw_matches = int(len(matches))
            result.good_matches = int(len(matches))

            if scores is not None and len(scores):
                result.confidence_mean = float(scores.mean())

            if len(matches) < 4:
                result.reason = "not enough matches"
                result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
                kp0 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kpts0]
                kp1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kpts1]
                return result, kp0, kp1, None, []

            pts0 = kpts0[matches[:, 0]].reshape(-1, 1, 2).astype(np.float32)
            pts1 = kpts1[matches[:, 1]].reshape(-1, 1, 2).astype(np.float32)

            ok, quad, inliers, inlier_ratio, quad_area_ratio = homography_from_matches(pts0, pts1, feature_bgr.shape, subject_bgr.shape)

            result.homography_found = ok
            result.inliers = inliers
            result.inlier_ratio = inlier_ratio
            result.quad_area_ratio = quad_area_ratio
            result.score = score_detection(inliers, inlier_ratio, quad_area_ratio, result.confidence_mean)
            result.found = found_from_metrics(inliers, inlier_ratio, result.score)
            result.success = True
            result.reason = "ok" if ok else "homography failed"
            result.elapsed_ms = (time.perf_counter() - t0) * 1000.0

            kp0 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kpts0]
            kp1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kpts1]
            good = []
            for i, (a, b) in enumerate(matches):
                dist = float(1.0 - scores[i]) if scores is not None and i < len(scores) else 0.0
                good.append(cv2.DMatch(_queryIdx=int(a), _trainIdx=int(b), _distance=dist))

            return result, kp0, kp1, quad, good

        finally:
            try:
                for p in tmp_dir.iterdir():
                    p.unlink(missing_ok=True)
                tmp_dir.rmdir()
            except Exception:
                pass


def build_models(include_deep: bool) -> List[BaseMatcher]:
    models: List[BaseMatcher] = []

    if hasattr(cv2, "SIFT_create"):
        models.append(OpenCVFeatureMatcher(name="sift_flann", detector=cv2.SIFT_create(nfeatures=4000), norm_type=cv2.NORM_L2, ratio=0.75, use_flann=True))

    models.append(OpenCVFeatureMatcher(name="akaze_bf", detector=cv2.AKAZE_create(), norm_type=cv2.NORM_HAMMING, ratio=0.80, use_flann=False))
    models.append(OpenCVFeatureMatcher(name="orb_bf", detector=cv2.ORB_create(nfeatures=5000, fastThreshold=5), norm_type=cv2.NORM_HAMMING, ratio=0.85, use_flann=False))

    if include_deep:
        for ft in ["superpoint", "disk", "aliked", "sift"]:
            models.append(LightGlueMatcher(ft))
        models.append(LoFTRMatcher("outdoor"))

    return models


FRONT_PREFERRED_MODELS = [
    "lightglue_sift",
    "lightglue_superpoint",
    "lightglue_aliked",
    "sift_flann",
    "akaze_bf",
    "lightglue_disk",
]

BACK_PREFERRED_MODELS = [
    "lightglue_superpoint",
    "lightglue_aliked",
    "lightglue_disk",
    "akaze_bf",
    "sift_flann",
    "lightglue_sift",
]


def select_preferred_models(include_deep: bool, top_k: int = 4, side_name: str = "front"):
    preferred = FRONT_PREFERRED_MODELS if side_name == "front" else BACK_PREFERRED_MODELS
    all_models = build_models(include_deep=include_deep)
    ready = {m.name: m for m in all_models if not hasattr(m, "available") or bool(m.available)}

    selected = []
    seen = set()
    for name in preferred:
        if name in ready and name not in seen:
            selected.append(ready[name])
            seen.add(name)
        if len(selected) == top_k:
            return selected

    for model in all_models:
        if model.name not in seen and (not hasattr(model, "available") or bool(model.available)):
            selected.append(model)
            seen.add(model.name)
        if len(selected) == top_k:
            break
    return selected


def aggregate_model_results(rows: List[MatchResult], min_feature_hits: int) -> Dict:
    usable = [r for r in rows if r.success]
    found = [r for r in usable if r.found]

    avg_score = float(np.mean([r.score for r in usable])) if usable else 0.0
    avg_inlier_ratio = float(np.mean([r.inlier_ratio for r in usable])) if usable else 0.0
    total_inliers = int(sum(r.inliers for r in usable))
    found_count = len(found)
    valid_id = bool(found_count >= min_feature_hits)

    return {
        "model_name": rows[0].model_name if rows else "unknown",
        "features_total": len(rows),
        "features_ran": len(usable),
        "features_found": found_count,
        "min_feature_hits_required": min_feature_hits,
        "avg_score": avg_score,
        "avg_inlier_ratio": avg_inlier_ratio,
        "total_inliers": total_inliers,
        "valid_id": valid_id,
    }


def stage_vote(results: List[bool]) -> bool:
    true_count = sum(1 for x in results if x)
    false_count = len(results) - true_count
    return true_count > false_count


def sequential_majority(models, vote_fn):
    votes = []
    logs = []

    for idx, model in enumerate(models):
        passed, info = vote_fn(model)
        votes.append(bool(passed))
        logs.append(info)

        if idx == 1 and len(votes) == 2 and votes[0] and votes[1]:
            return True, logs

        remaining = len(models) - len(votes)
        true_count = sum(1 for x in votes if x)
        false_count = len(votes) - true_count

        if true_count > false_count + remaining:
            return True, logs
        if false_count >= true_count + remaining:
            return False, logs

    return stage_vote(votes), logs


def validate_subject_with_top_models(
    subject_path: str,
    features_dir: str,
    include_deep: bool = False,
    min_feature_hits: Optional[int] = None,
    top_k: int = 4,
    side_name: str = "front",
) -> Tuple[bool, List[Dict]]:
    feature_paths = list_images(Path(features_dir))
    if not feature_paths:
        raise ValueError(f"No feature images found in: {features_dir}")

    if min_feature_hits is None:
        min_feature_hits = max(1, math.ceil(len(feature_paths) * 0.40))

    subject_bgr = read_image(Path(subject_path))
    models = select_preferred_models(include_deep=include_deep, top_k=top_k, side_name=side_name)
    if len(models) < 2:
        raise RuntimeError("Need at least 2 ready models for validation")

    def vote_fn(model):
        rows = []
        for feature_path in feature_paths:
            feature_bgr = read_image(feature_path)
            result, _, _, _, _ = model.match(feature_bgr, subject_bgr)
            result.feature_name = feature_path.stem
            result.feature_path = str(feature_path.resolve())
            result.subject_path = str(Path(subject_path).resolve())
            rows.append(result)

        summary = aggregate_model_results(rows, min_feature_hits)
        return bool(summary["valid_id"]), {
            "model": model.name,
            "passed": bool(summary["valid_id"]),
            "features_found": int(summary["features_found"]),
            "features_total": int(summary["features_total"]),
            "avg_score": float(summary["avg_score"]),
            "avg_inlier_ratio": float(summary["avg_inlier_ratio"]),
            "total_inliers": int(summary["total_inliers"]),
        }

    return sequential_majority(models, vote_fn)


def write_csv(path: Path, rows: List[MatchResult]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, help="Path to the test subject image")
    ap.add_argument("--features-dir", required=True, help="Folder with feature crop images")
    ap.add_argument("--output-dir", default="match_runs", help="Root output folder")
    ap.add_argument("--include-deep", action="store_true", help="Also try LightGlue/LoFTR if installed")
    ap.add_argument("--min-feature-hits", type=int, default=0, help="Minimum matched features required for a valid ID; 0 = auto")
    args = ap.parse_args()

    subject_path = Path(args.subject)
    features_dir = Path(args.features_dir)
    output_root = Path(args.output_dir)

    if not subject_path.exists():
        raise FileNotFoundError(subject_path)
    if not features_dir.exists():
        raise FileNotFoundError(features_dir)

    ensure_dir(output_root)

    feature_paths = list_images(features_dir)
    if not feature_paths:
        raise ValueError(f"No feature images found in: {features_dir}")

    min_feature_hits = args.min_feature_hits or max(1, math.ceil(len(feature_paths) * 0.40))
    subject_bgr = read_image(subject_path)
    models = build_models(args.include_deep)

    overall = []

    for model in models:
        model_dir = output_root / model.name
        ensure_dir(model_dir)
        vis_dir = model_dir / "visualizations"
        ensure_dir(vis_dir)

        model_rows: List[MatchResult] = []

        for feature_path in feature_paths:
            feature_bgr = read_image(feature_path)
            result, kp0, kp1, quad, good = model.match(feature_bgr, subject_bgr)

            result.feature_name = feature_path.stem
            result.feature_path = str(feature_path.resolve())
            result.subject_path = str(subject_path.resolve())

            vis_path = vis_dir / f"{feature_path.stem}.jpg"
            try:
                save_match_viz(feature_bgr, subject_bgr, kp0, kp1, good, quad, vis_path, result.found)
            except Exception:
                pass

            model_rows.append(result)

        summary = aggregate_model_results(model_rows, min_feature_hits)
        summary["subject_path"] = str(subject_path.resolve())
        summary["features_dir"] = str(features_dir.resolve())

        write_csv(model_dir / "results.csv", model_rows)
        (model_dir / "results.json").write_text(json.dumps([asdict(r) for r in model_rows], ensure_ascii=False, indent=2), encoding="utf-8")
        (model_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        overall.append(summary)

    overall_sorted = sorted(overall, key=lambda x: (x["valid_id"], x["features_found"], x["avg_score"], x["total_inliers"]), reverse=True)

    (output_root / "overall_summary.json").write_text(json.dumps(overall_sorted, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(overall_sorted, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

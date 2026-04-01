import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace

MODELS = ["Buffalo_L", "Facenet512", "Facenet", "ArcFace"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def facial_area_size(face_obj: dict) -> float:
    area = face_obj.get("facial_area") or {}
    w = area.get("w") or area.get("width") or 0
    h = area.get("h") or area.get("height") or 0
    try:
        return float(w) * float(h)
    except Exception:
        return 0.0


def pick_largest_face(face_objs):
    if not face_objs:
        raise ValueError("No faces returned")
    return sorted(face_objs, key=facial_area_size, reverse=True)[0]


def face_to_bgr(face_obj: dict) -> np.ndarray:
    arr = np.asarray(face_obj["face"])
    if arr.ndim == 2:
        arr = arr.astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def resize_fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    scale = min(w / max(iw, 1), h / max(ih, 1))
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    resized = cv2.resize(
        img,
        (nw, nh),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def make_preview(selfie_face: np.ndarray, id_face: np.ndarray, text_lines):
    left = resize_fit(selfie_face, 320, 320)
    right = resize_fit(id_face, 320, 320)
    top = np.hstack([left, right])

    panel_h = 180
    panel = np.full((panel_h, top.shape[1], 3), 255, dtype=np.uint8)

    y = 30
    for line in text_lines:
        cv2.putText(
            panel,
            str(line),
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        y += 28

    return np.vstack([top, panel])


def save_json(path: Path, data):
    path.write_text(
        json.dumps(sanitize(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_face_crops(img_path: str, detector_backend: str, enforce_detection: bool):
    faces = DeepFace.extract_faces(
        img_path=img_path,
        detector_backend=detector_backend,
        align=True,
        enforce_detection=enforce_detection,
    )
    chosen = pick_largest_face(faces)
    return faces, chosen, face_to_bgr(chosen)


def get_embedding(img_path: str, model_name: str, detector_backend: str, enforce_detection: bool):
    reps = DeepFace.represent(
        img_path=img_path,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=True,
    )
    if isinstance(reps, dict):
        reps = [reps]
    rep = pick_largest_face(reps)
    emb = np.asarray(rep["embedding"], dtype=np.float32)
    return rep, emb


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return float("nan")
    cosine_similarity = float(np.dot(a, b) / denom)
    return float(1.0 - cosine_similarity)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfie", required=True, help="Path to the selfie image")
    ap.add_argument("--id-image", required=True, help="Path to the ID face image or full ID image")
    ap.add_argument("--output-dir", default="face_match_runs", help="Root output folder")
    ap.add_argument("--detector-backend", default="retinaface", help="DeepFace detector backend")
    ap.add_argument(
        "--distance-metric",
        default="cosine",
        choices=["cosine", "euclidean", "euclidean_l2", "angular"],
        help="DeepFace distance metric",
    )
    ap.add_argument(
        "--enforce-detection",
        action="store_true",
        help="Fail if a face is not detected",
    )
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    overall = []

    for model_name in MODELS:
        model_dir = output_dir / model_name
        ensure_dir(model_dir)
        started = time.perf_counter()

        try:
            selfie_faces, _, selfie_face_bgr = get_face_crops(
                args.selfie, args.detector_backend, args.enforce_detection
            )
            id_faces, _, id_face_bgr = get_face_crops(
                args.id_image, args.detector_backend, args.enforce_detection
            )

            cv2.imwrite(str(model_dir / "selfie_face.png"), selfie_face_bgr)
            cv2.imwrite(str(model_dir / "id_face.png"), id_face_bgr)
            save_json(model_dir / "selfie_faces.json", selfie_faces)
            save_json(model_dir / "id_faces.json", id_faces)

            verify_res = DeepFace.verify(
                img1_path=args.selfie,
                img2_path=args.id_image,
                model_name=model_name,
                detector_backend=args.detector_backend,
                distance_metric=args.distance_metric,
                align=True,
                enforce_detection=args.enforce_detection,
            )

            rep1, emb1 = get_embedding(
                args.selfie, model_name, args.detector_backend, args.enforce_detection
            )
            rep2, emb2 = get_embedding(
                args.id_image, model_name, args.detector_backend, args.enforce_detection
            )

            emb_cosine_distance = cosine_distance(emb1, emb2)
            elapsed = time.perf_counter() - started

            result = {
                "status": "ok",
                "model_name": model_name,
                "detector_backend": args.detector_backend,
                "distance_metric": args.distance_metric,
                "verify": verify_res,
                "embedding_selfie": {
                    "embedding_length": int(emb1.shape[0]),
                    "embedding_norm": float(np.linalg.norm(emb1)),
                    "face": rep1,
                },
                "embedding_id": {
                    "embedding_length": int(emb2.shape[0]),
                    "embedding_norm": float(np.linalg.norm(emb2)),
                    "face": rep2,
                },
                "embedding_cosine_distance": emb_cosine_distance,
                "elapsed_seconds": elapsed,
            }

            save_json(model_dir / "result.json", result)

            verified = bool(verify_res.get("verified", False))
            distance = float(verify_res.get("distance", float("nan")))
            threshold = float(verify_res.get("threshold", float("nan")))
            margin = (
                float(threshold - distance)
                if math.isfinite(distance) and math.isfinite(threshold)
                else float("nan")
            )

            preview = make_preview(
                selfie_face_bgr,
                id_face_bgr,
                [
                    f"model: {model_name}",
                    f"verified: {verified}",
                    f"distance: {distance:.6f}" if math.isfinite(distance) else "distance: nan",
                    f"threshold: {threshold:.6f}" if math.isfinite(threshold) else "threshold: nan",
                    f"threshold_margin: {margin:.6f}" if math.isfinite(margin) else "threshold_margin: nan",
                    f"embedding_cosine_distance: {emb_cosine_distance:.6f}" if math.isfinite(emb_cosine_distance) else "embedding_cosine_distance: nan",
                    f"detector: {args.detector_backend}",
                ],
            )
            cv2.imwrite(str(model_dir / "preview.png"), preview)

            overall.append(
                {
                    "model_name": model_name,
                    "status": "ok",
                    "verified": verified,
                    "distance": distance,
                    "threshold": threshold,
                    "threshold_margin": margin,
                    "embedding_cosine_distance": emb_cosine_distance,
                    "elapsed_seconds": elapsed,
                }
            )

        except Exception as e:
            elapsed = time.perf_counter() - started
            err = {
                "status": "error",
                "model_name": model_name,
                "detector_backend": args.detector_backend,
                "distance_metric": args.distance_metric,
                "error": str(e),
                "elapsed_seconds": elapsed,
            }
            save_json(model_dir / "error.json", err)
            overall.append(
                {
                    "model_name": model_name,
                    "status": "error",
                    "verified": False,
                    "distance": None,
                    "threshold": None,
                    "threshold_margin": None,
                    "embedding_cosine_distance": None,
                    "elapsed_seconds": elapsed,
                    "error": str(e),
                }
            )

    def overall_key(x):
        ok = 1 if x.get("status") == "ok" else 0
        verified = 1 if x.get("verified") else 0
        margin = x.get("threshold_margin")
        if margin is None or not isinstance(margin, (int, float)) or not math.isfinite(margin):
            margin = -1e18
        return (ok, verified, margin)

    overall = sorted(overall, key=overall_key, reverse=True)
    save_json(output_dir / "overall_summary.json", overall)
    print(json.dumps(sanitize(overall), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
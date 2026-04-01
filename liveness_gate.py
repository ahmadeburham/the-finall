import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from deepface import DeepFace


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


def pick_largest_face(face_objs: List[dict]) -> Dict:
    if not face_objs:
        raise ValueError("No faces returned")
    return sorted(face_objs, key=facial_area_size, reverse=True)[0]


def _face_summary(face_obj: dict) -> Dict:
    area = face_obj.get("facial_area") or {}
    return {
        "is_real": bool(face_obj.get("is_real", False)),
        "antispoof_score": float(face_obj.get("antispoof_score", 0.0)) if isinstance(face_obj.get("antispoof_score"), (int, float)) else None,
        "confidence": float(face_obj.get("confidence", 0.0)) if isinstance(face_obj.get("confidence"), (int, float)) else None,
        "facial_area": sanitize(area),
        "face_area_size": facial_area_size(face_obj),
    }


def check_selfie_liveness(
    selfie_path: str,
    detector_backend: str = "retinaface",
    enforce_detection: bool = False,
) -> Dict:
    started = time.perf_counter()

    try:
        faces = DeepFace.extract_faces(
            img_path=selfie_path,
            detector_backend=detector_backend,
            align=True,
            enforce_detection=enforce_detection,
            anti_spoofing=True,
        )
    except Exception as e:
        return {
            "status": "error",
            "passed": False,
            "error": str(e),
            "detector_backend": detector_backend,
            "elapsed_seconds": time.perf_counter() - started,
            "face_count": 0,
            "faces": [],
            "selected_face": None,
        }

    if not faces:
        return {
            "status": "error",
            "passed": False,
            "error": "No face detected in selfie",
            "detector_backend": detector_backend,
            "elapsed_seconds": time.perf_counter() - started,
            "face_count": 0,
            "faces": [],
            "selected_face": None,
        }

    selected = pick_largest_face(faces)
    selected_summary = _face_summary(selected)
    faces_summary = [_face_summary(face) for face in faces]

    return {
        "status": "ok",
        "passed": bool(selected_summary["is_real"]),
        "detector_backend": detector_backend,
        "elapsed_seconds": time.perf_counter() - started,
        "face_count": len(faces),
        "faces": faces_summary,
        "selected_face": selected_summary,
    }


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(data), ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfie", required=True)
    ap.add_argument("--output-json", default="liveness_result.json")
    ap.add_argument("--detector-backend", default="retinaface")
    ap.add_argument("--enforce-detection", action="store_true")
    args = ap.parse_args()

    result = check_selfie_liveness(
        selfie_path=args.selfie,
        detector_backend=args.detector_backend,
        enforce_detection=args.enforce_detection,
    )
    save_json(Path(args.output_json), result)
    print(json.dumps(sanitize(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

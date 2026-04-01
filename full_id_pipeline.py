import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import arabic_ocr as ao
import face_match_top4 as fm
import feature_pipeline as fp
import liveness_gate as lg
import template_fields as tf


class StepLogger:
    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path) if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        print(line)

    def exception(self, stage: str, exc: Exception) -> None:
        self.log(f"{stage} FAILED: {exc}")
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(traceback.format_exc() + "\n")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def empty_text():
    return {
        "name": "",
        "address": "",
        "id_number": "",
        "birth_date": "",
        "expiry_date": "",
        "front_full_text": "",
        "back_full_text": "",
    }


def write_result(path: Path, result: dict):
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_dual_validation(front_image, back_image, features_root, include_deep, min_front, min_back):
    root = Path(features_root)
    front_dir = root / "front"
    back_dir = root / "back"

    if not front_dir.exists():
        raise FileNotFoundError(f"Missing front features folder: {front_dir}")
    if not back_dir.exists():
        raise FileNotFoundError(f"Missing back features folder: {back_dir}")

    front_valid, front_logs = fp.validate_subject_with_top_models(
        subject_path=front_image,
        features_dir=str(front_dir),
        include_deep=include_deep,
        min_feature_hits=min_front,
        top_k=4,
        side_name="front",
    )
    back_valid, back_logs = fp.validate_subject_with_top_models(
        subject_path=back_image,
        features_dir=str(back_dir),
        include_deep=include_deep,
        min_feature_hits=min_back,
        top_k=4,
        side_name="back",
    )
    return bool(front_valid), bool(back_valid), bool(front_valid and back_valid), front_logs, back_logs


def run_pipeline(
    selfie: str,
    front_image: str,
    back_image: str,
    features_root: str,
    front_template: str,
    back_template: str,
    template_config: str,
    feature_include_deep: bool = False,
    min_feature_hits_front: int | None = None,
    min_feature_hits_back: int | None = None,
    detector_backend: str = "retinaface",
    distance_metric: str = "cosine",
    enforce_detection: bool = False,
    debug_dir: str = "",
    logger: Optional[StepLogger] = None,
) -> dict:
    logger = logger or StepLogger()
    result = {
        "card_valid": False,
        "face_match": None,
        "liveness_passed": None,
        "overall_passed": False,
        "text": empty_text(),
        "feature_validation": {"front": None, "back": None},
        "face_match_logs": [],
        "liveness": None,
    }

    logger.log("feature validation started")
    try:
        front_valid, back_valid, card_valid, front_logs, back_logs = run_dual_validation(
            front_image=front_image,
            back_image=back_image,
            features_root=features_root,
            include_deep=feature_include_deep,
            min_front=min_feature_hits_front,
            min_back=min_feature_hits_back,
        )
        result["feature_validation"] = {
            "front": {"passed": front_valid, "logs": front_logs},
            "back": {"passed": back_valid, "logs": back_logs},
        }
        result["card_valid"] = bool(card_valid)
        logger.log(f"feature validation finished: front={front_valid}, back={back_valid}, card_valid={card_valid}")
    except Exception as e:
        logger.exception("feature validation", e)
        return result

    if not result["card_valid"]:
        logger.log("stopping after feature validation because the card did not pass")
        return result

    logger.log("alignment started")
    try:
        config = tf.load_config(template_config)
        aligned_front, front_align_info = tf.align_card_to_template(front_image, front_template, return_candidates=False)
        aligned_back, back_align_info = tf.align_card_to_template(back_image, back_template, return_candidates=False)
        front_artifacts = tf.extract_side_artifacts(aligned_front, config, "front")
        back_artifacts = tf.extract_side_artifacts(aligned_back, config, "back")
        if debug_dir:
            tf.save_debug_artifacts(debug_dir, "front", front_artifacts)
            tf.save_debug_artifacts(debug_dir, "back", back_artifacts)
        result["alignment"] = {
            "front": front_align_info,
            "back": back_align_info,
        }
        logger.log("alignment finished")
    except Exception as e:
        logger.exception("alignment", e)
        return result

    photo_crop = front_artifacts.get("photo", None)
    if photo_crop is None or photo_crop.size == 0:
        logger.log("front photo crop is missing; skipping face match, OCR, and liveness")
        return result

    logger.log("face match started")
    try:
        face_match, face_logs = fm.verify_top4_from_array(
            selfie_path=selfie,
            target_face_bgr=photo_crop,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            enforce_detection=enforce_detection,
        )
        result["face_match"] = bool(face_match)
        result["face_match_logs"] = face_logs
        logger.log(f"face match finished: passed={face_match}")
    except Exception as e:
        logger.exception("face match", e)

    logger.log("ocr started")
    try:
        fields = ao.extract_fields_from_crops(
            front_crops=front_artifacts.get("crops", {}),
            back_crops=back_artifacts.get("crops", {}),
        )
        result["text"] = {
            "name": fields.get("name", ""),
            "address": fields.get("address", ""),
            "id_number": fields.get("id_number", ""),
            "birth_date": fields.get("birth_date", ""),
            "expiry_date": fields.get("expiry_date", ""),
            "front_full_text": fields.get("front_full_text", ""),
            "back_full_text": fields.get("back_full_text", ""),
        }
        logger.log("ocr finished")
    except Exception as e:
        logger.exception("ocr", e)

    logger.log("liveness started")
    try:
        liveness = lg.check_selfie_liveness(
            selfie_path=selfie,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
        )
        result["liveness"] = liveness
        result["liveness_passed"] = bool(liveness.get("passed", False))
        logger.log(f"liveness finished: passed={result['liveness_passed']}")
    except Exception as e:
        logger.exception("liveness", e)

    result["overall_passed"] = bool(
        result["card_valid"]
        and bool(result.get("face_match"))
        and bool(result.get("liveness_passed"))
    )
    logger.log(f"pipeline finished: overall_passed={result['overall_passed']}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfie", required=True)
    ap.add_argument("--front-image", required=True)
    ap.add_argument("--back-image", required=True)
    ap.add_argument("--features-root", required=True)

    ap.add_argument("--front-template", required=True)
    ap.add_argument("--back-template", required=True)
    ap.add_argument("--template-config", required=True)

    ap.add_argument("--output-json", default="final_result.json")
    ap.add_argument("--log-file", default="")
    ap.add_argument("--feature-include-deep", action="store_true")
    ap.add_argument("--min-feature-hits-front", type=int, default=None)
    ap.add_argument("--min-feature-hits-back", type=int, default=None)

    ap.add_argument("--detector-backend", default="retinaface")
    ap.add_argument("--distance-metric", default="cosine", choices=["cosine", "euclidean", "euclidean_l2", "angular"])
    ap.add_argument("--enforce-detection", action="store_true")

    ap.add_argument("--debug-dir", default="")
    args = ap.parse_args()

    output_json = Path(args.output_json)
    ensure_parent(output_json)

    logger = StepLogger(args.log_file or None)
    result = run_pipeline(
        selfie=args.selfie,
        front_image=args.front_image,
        back_image=args.back_image,
        features_root=args.features_root,
        front_template=args.front_template,
        back_template=args.back_template,
        template_config=args.template_config,
        feature_include_deep=args.feature_include_deep,
        min_feature_hits_front=args.min_feature_hits_front,
        min_feature_hits_back=args.min_feature_hits_back,
        detector_backend=args.detector_backend,
        distance_metric=args.distance_metric,
        enforce_detection=args.enforce_detection,
        debug_dir=args.debug_dir,
        logger=logger,
    )
    write_result(output_json, result)


if __name__ == "__main__":
    main()

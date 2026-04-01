import argparse
import json
from pathlib import Path

import arabic_ocr as ao
import face_match_top4 as fm
import feature_pipeline as fp
import template_fields as tf


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def empty_text():
    return {
        "name": "",
        "address": "",
        "id_number": "",
        "birth_date": "",
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

    front_valid, _ = fp.validate_subject_with_top_models(
        subject_path=front_image,
        features_dir=str(front_dir),
        include_deep=include_deep,
        min_feature_hits=min_front,
        top_k=4,
        side_name="front",
    )
    back_valid, _ = fp.validate_subject_with_top_models(
        subject_path=back_image,
        features_dir=str(back_dir),
        include_deep=include_deep,
        min_feature_hits=min_back,
        top_k=4,
        side_name="back",
    )
    return bool(front_valid), bool(back_valid), bool(front_valid and back_valid)


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

    result = {
        "card_valid": False,
        "face_match": None,
        "text": empty_text(),
    }

    front_valid, back_valid, card_valid = run_dual_validation(
        front_image=args.front_image,
        back_image=args.back_image,
        features_root=args.features_root,
        include_deep=args.feature_include_deep,
        min_front=args.min_feature_hits_front,
        min_back=args.min_feature_hits_back,
    )

    if not card_valid:
        write_result(output_json, result)
        return

    config = tf.load_config(args.template_config)

    try:
        aligned_front, _ = tf.align_card_to_template(args.front_image, args.front_template, return_candidates=False)
        aligned_back, _ = tf.align_card_to_template(args.back_image, args.back_template, return_candidates=False)
    except Exception:
        write_result(output_json, result)
        return

    front_artifacts = tf.extract_side_artifacts(aligned_front, config, "front")
    back_artifacts = tf.extract_side_artifacts(aligned_back, config, "back")

    if args.debug_dir:
        tf.save_debug_artifacts(args.debug_dir, "front", front_artifacts)
        tf.save_debug_artifacts(args.debug_dir, "back", back_artifacts)

    photo_crop = front_artifacts.get("photo", None)
    if photo_crop is None or photo_crop.size == 0:
        write_result(output_json, result)
        return

    face_match, _ = fm.verify_top4_from_array(
        selfie_path=args.selfie,
        target_face_bgr=photo_crop,
        detector_backend=args.detector_backend,
        distance_metric=args.distance_metric,
        enforce_detection=args.enforce_detection,
    )

    result["card_valid"] = True
    result["face_match"] = bool(face_match)

    if not face_match:
        write_result(output_json, result)
        return

    fields = ao.extract_fields_from_crops(
        front_crops=front_artifacts.get("crops", {}),
        back_crops=back_artifacts.get("crops", {}),
    )

    result["text"] = {
        "name": fields.get("name", ""),
        "address": fields.get("address", ""),
        "id_number": fields.get("id_number", ""),
        "birth_date": fields.get("birth_date", ""),
        "front_full_text": fields.get("front_full_text", ""),
        "back_full_text": fields.get("back_full_text", ""),
    }

    write_result(output_json, result)


if __name__ == "__main__":
    main()

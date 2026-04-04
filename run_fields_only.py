import argparse
import json
from pathlib import Path

import arabic_ocr as ao
import template_fields as tf


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: dict) -> None:
    ensure_parent(path)
    payload = ao.prepare_for_json_output(data)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front-image", required=True)
    ap.add_argument("--back-image", required=True)
    ap.add_argument("--front-template", required=True)
    ap.add_argument("--back-template", required=True)
    ap.add_argument("--template-config", required=True)
    ap.add_argument("--output-json", default="fields_only_result.json")
    ap.add_argument("--debug-dir", default="")
    args = ap.parse_args()

    config = tf.load_config(args.template_config)

    aligned_front, front_align_info = tf.align_card_to_template(
        args.front_image,
        args.front_template,
        return_candidates=True,
    )
    aligned_back, back_align_info = tf.align_card_to_template(
        args.back_image,
        args.back_template,
        return_candidates=True,
    )

    front_artifacts = tf.extract_side_artifacts(
        aligned_front,
        config,
        "front",
        subject_image=args.front_image,
        align_info=front_align_info,
    )
    back_artifacts = tf.extract_side_artifacts(
        aligned_back,
        config,
        "back",
        subject_image=args.back_image,
        align_info=back_align_info,
    )

    if args.debug_dir:
        tf.save_debug_artifacts(args.debug_dir, "front", front_artifacts)
        tf.save_debug_artifacts(args.debug_dir, "back", back_artifacts)

    fields = ao.extract_fields_from_crops(
        front_crops=front_artifacts.get("crops", {}),
        back_crops=back_artifacts.get("crops", {}),
    )

    result = {
        "front_alignment": front_align_info,
        "back_alignment": back_align_info,
        "text": {
            "name": fields.get("name", ""),
            "address": fields.get("address", ""),
            "id_number": fields.get("id_number", ""),
            "birth_date": fields.get("birth_date", ""),
            "front_full_text": fields.get("front_full_text", ""),
            "back_full_text": fields.get("back_full_text", ""),
        },
        "raw": fields.get("raw", {}),
    }

    save_json(Path(args.output_json), result)
    print(json.dumps(ao.prepare_for_json_output(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
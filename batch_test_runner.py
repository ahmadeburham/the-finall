import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import full_id_pipeline as pipeline

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def find_named_image(folder: Path, required_substring: str) -> Path:
    candidates = sorted(
        [p for p in folder.iterdir() if is_image(p) and required_substring.lower() in p.stem.lower()]
    )
    if not candidates:
        raise FileNotFoundError(f"Could not find an image containing '{required_substring}' in {folder}")
    if len(candidates) > 1:
        raise RuntimeError(f"Found multiple images containing '{required_substring}' in {folder}: {[p.name for p in candidates]}")
    return candidates[0]


def discover_workspace(workspace: Path) -> Dict[str, Path]:
    template_root = workspace / "template"
    if not template_root.exists():
        template_root = workspace

    config_path = workspace / "id_template_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing template config: {config_path}")

    features_root = workspace / "features"
    if not features_root.exists():
        raise FileNotFoundError(f"Missing features folder: {features_root}")

    testing_root = workspace / "testing"
    if not testing_root.exists():
        raise FileNotFoundError(f"Missing testing folder: {testing_root}")

    return {
        "selfie": find_named_image(workspace, "selfie"),
        "front_template": find_named_image(template_root, "front_template"),
        "back_template": find_named_image(template_root, "back_template"),
        "template_config": config_path,
        "features_root": features_root,
        "testing_root": testing_root,
    }


def discover_case_images(case_dir: Path) -> Tuple[Path, Path]:
    front_image = find_named_image(case_dir, "front")
    back_image = find_named_image(case_dir, "back")
    return front_image, back_image


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=".")
    ap.add_argument("--output-root", default="test_outputs")
    ap.add_argument("--feature-include-deep", action="store_true")
    ap.add_argument("--min-feature-hits-front", type=int, default=None)
    ap.add_argument("--min-feature-hits-back", type=int, default=None)
    ap.add_argument("--detector-backend", default="retinaface")
    ap.add_argument("--distance-metric", default="cosine", choices=["cosine", "euclidean", "euclidean_l2", "angular"])
    ap.add_argument("--enforce-detection", action="store_true")
    args = ap.parse_args()

    workspace = Path(args.workspace).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    common = discover_workspace(workspace)
    summary = []

    for case_dir in sorted([p for p in common["testing_root"].iterdir() if p.is_dir()]):
        case_output = output_root / case_dir.name
        case_output.mkdir(parents=True, exist_ok=True)
        logger = pipeline.StepLogger(case_output / "run.log")

        try:
            front_image, back_image = discover_case_images(case_dir)
            logger.log(f"case started: {case_dir.name}")
            logger.log(f"front image: {front_image.name}")
            logger.log(f"back image: {back_image.name}")

            result = pipeline.run_pipeline(
                selfie=str(common["selfie"]),
                front_image=str(front_image),
                back_image=str(back_image),
                features_root=str(common["features_root"]),
                front_template=str(common["front_template"]),
                back_template=str(common["back_template"]),
                template_config=str(common["template_config"]),
                feature_include_deep=args.feature_include_deep,
                min_feature_hits_front=args.min_feature_hits_front,
                min_feature_hits_back=args.min_feature_hits_back,
                detector_backend=args.detector_backend,
                distance_metric=args.distance_metric,
                enforce_detection=args.enforce_detection,
                debug_dir=str(case_output / "debug_artifacts"),
                logger=logger,
            )
            save_json(case_output / "result.json", result)
            logger.log(f"case finished: overall_passed={result.get('overall_passed', False)}")
            summary.append(
                {
                    "case": case_dir.name,
                    "front_image": front_image.name,
                    "back_image": back_image.name,
                    "card_valid": result.get("card_valid"),
                    "face_match": result.get("face_match"),
                    "liveness_passed": result.get("liveness_passed"),
                    "overall_passed": result.get("overall_passed"),
                    "result_json": str((case_output / "result.json").resolve()),
                    "log_file": str((case_output / "run.log").resolve()),
                }
            )
        except Exception as e:
            logger.exception(f"case {case_dir.name}", e)
            error_result = {
                "case": case_dir.name,
                "status": "error",
                "error": str(e),
            }
            save_json(case_output / "result.json", error_result)
            summary.append(
                {
                    "case": case_dir.name,
                    "status": "error",
                    "error": str(e),
                    "result_json": str((case_output / "result.json").resolve()),
                    "log_file": str((case_output / "run.log").resolve()),
                }
            )

    save_json(output_root / "summary.json", {"workspace": str(workspace), "cases": summary})
    print(json.dumps({"workspace": str(workspace), "cases": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

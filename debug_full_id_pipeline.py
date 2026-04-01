import argparse
import json
import math
import shutil
import tempfile
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

import arabic_ocr as ao
import face_match_top4 as fm
import feature_pipeline as fp
import template_fields as tf
import liveness_gate as lg


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


def save_json(path: Path, data) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(sanitize(data), ensure_ascii=False, indent=2), encoding="utf-8")


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        ensure_dir(path.parent)
        self.log("debug run started")

    def log(self, msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line)

    def exception(self, stage: str, exc: Exception) -> None:
        tb = traceback.format_exc()
        self.log(f"{stage} FAILED: {exc}")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(tb + "\n")


def copy_file(src: str | Path, dst_dir: Path) -> Path:
    src = Path(src)
    ensure_dir(dst_dir)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst


def copy_code_snapshot(run_dir: Path) -> None:
    snapshot_dir = run_dir / "code_snapshot"
    ensure_dir(snapshot_dir)
    for name in [
        "feature_pipeline.py",
        "face_match_top4.py",
        "arabic_ocr.py",
        "template_fields.py",
        "full_id_pipeline.py",
        "debug_full_id_pipeline.py",
        "liveness_gate.py",
        "batch_test_runner.py",
        "id_template_config.json",
    ]:
        p = Path(name)
        if p.exists():
            shutil.copy2(p, snapshot_dir / p.name)


def sequential_decision(logs: List[Dict]) -> Tuple[bool, List[Dict]]:
    votes = []
    used = []

    for idx, log in enumerate(logs):
        passed = bool(log.get("passed", False))
        votes.append(passed)
        used.append(log)

        if idx == 1 and len(votes) == 2 and votes[0] and votes[1]:
            return True, used

        remaining = len(logs) - len(votes)
        true_count = sum(1 for x in votes if x)
        false_count = len(votes) - true_count

        if true_count > false_count + remaining:
            return True, used
        if false_count >= true_count + remaining:
            return False, used

    true_count = sum(1 for x in votes if x)
    false_count = len(votes) - true_count
    return true_count > false_count, used


def run_feature_side_debug(side_name: str, subject_path: str, features_dir: str, include_deep: bool, min_feature_hits: int | None, out_dir: Path, logger: RunLogger):
    logger.log(f"{side_name}: feature validation started")
    side_dir = out_dir / side_name
    ensure_dir(side_dir)

    feature_paths = fp.list_images(Path(features_dir))
    if not feature_paths:
        raise ValueError(f"No feature images found in: {features_dir}")

    if min_feature_hits is None:
        min_feature_hits = max(1, math.ceil(len(feature_paths) * 0.40))

    subject_bgr = fp.read_image(Path(subject_path))
    models = fp.build_models(include_deep=include_deep)
    preferred_models = fp.select_preferred_models(include_deep=include_deep, top_k=4, side_name=side_name)
    preferred_names = [m.name for m in preferred_models]

    overall = []
    summary_by_name = {}

    for model in models:
        model_dir = side_dir / model.name
        vis_dir = model_dir / "visualizations"
        ensure_dir(vis_dir)

        rows = []
        logger.log(f"{side_name}: running model {model.name}")

        for feature_path in feature_paths:
            feature_bgr = fp.read_image(feature_path)
            try:
                result, kp0, kp1, quad, good = model.match(feature_bgr, subject_bgr)
            except Exception as e:
                result = fp.MatchResult(
                    model_name=model.name,
                    feature_name=feature_path.stem,
                    feature_path=str(feature_path.resolve()),
                    subject_path=str(Path(subject_path).resolve()),
                    success=False,
                    reason=f"exception: {e}",
                )
                kp0, kp1, quad, good = [], [], None, []

            result.feature_name = feature_path.stem
            result.feature_path = str(feature_path.resolve())
            result.subject_path = str(Path(subject_path).resolve())
            rows.append(result)

            try:
                fp.save_match_viz(feature_bgr, subject_bgr, kp0, kp1, good, quad, vis_dir / f"{feature_path.stem}.jpg", result.found)
            except Exception:
                pass

        fp.write_csv(model_dir / "results.csv", rows)
        save_json(model_dir / "results.json", [asdict(r) for r in rows])

        summary = fp.aggregate_model_results(rows, min_feature_hits)
        summary["subject_path"] = str(Path(subject_path).resolve())
        summary["features_dir"] = str(Path(features_dir).resolve())
        summary["is_preferred_top4"] = model.name in preferred_names

        save_json(model_dir / "summary.json", summary)
        overall.append(summary)
        summary_by_name[model.name] = summary

    overall_sorted = sorted(overall, key=lambda x: (x["valid_id"], x["features_found"], x["avg_score"], x["total_inliers"]), reverse=True)
    save_json(side_dir / "overall_summary.json", overall_sorted)

    decision_logs = []
    for name in preferred_names:
        if name not in summary_by_name:
            continue
        s = summary_by_name[name]
        decision_logs.append(
            {
                "model": name,
                "passed": bool(s["valid_id"]),
                "features_found": int(s["features_found"]),
                "features_total": int(s["features_total"]),
                "avg_score": float(s["avg_score"]),
                "avg_inlier_ratio": float(s["avg_inlier_ratio"]),
                "total_inliers": int(s["total_inliers"]),
            }
        )

    passed, used_logs = sequential_decision(decision_logs)
    decision = {
        "side": side_name,
        "passed": bool(passed),
        "min_feature_hits_required": int(min_feature_hits),
        "preferred_model_order": preferred_names,
        "decision_logs": decision_logs,
        "used_logs": used_logs,
    }
    save_json(side_dir / "decision.json", decision)
    logger.log(f"{side_name}: feature validation finished, passed={passed}")
    return bool(passed), decision_logs, overall_sorted


def run_alignment_debug(side_name: str, image_path: str, template_path: str, config: Dict, out_dir: Path, logger: RunLogger):
    logger.log(f"{side_name}: alignment started")
    side_dir = out_dir / side_name
    ensure_dir(side_dir)

    aligned, align_info = tf.align_card_to_template(image_path, template_path, return_candidates=True)
    save_json(side_dir / "align_info.json", align_info)

    if "candidates" in align_info:
        cand_str = ", ".join(
            [
                f"({c.get('method')},{c.get('rotation')},valid={c.get('valid')},inliers={c.get('inliers')},ratio={c.get('inlier_ratio')},score={c.get('score')})"
                for c in align_info["candidates"]
            ]
        )
        logger.log(f"{side_name}: alignment candidates {cand_str}")

    artifacts = tf.extract_side_artifacts(aligned, config, side_name)
    tf.save_debug_artifacts(side_dir, side_name, artifacts)

    manifest = {
        "has_aligned": artifacts.get("aligned") is not None,
        "has_cleaned": artifacts.get("cleaned") is not None,
        "has_photo": artifacts.get("photo") is not None,
        "crop_names": sorted(list(artifacts.get("crops", {}).keys())),
    }
    if artifacts.get("photo") is not None:
        manifest["photo_shape"] = list(artifacts["photo"].shape)
    manifest["crop_shapes"] = {k: list(v.shape) for k, v in artifacts.get("crops", {}).items()}
    save_json(side_dir / "artifact_manifest.json", manifest)

    logger.log(f"{side_name}: alignment finished")
    return artifacts, align_info


def run_face_debug(selfie_path: str, target_face_bgr: np.ndarray, detector_backend: str, distance_metric: str, enforce_detection: bool, out_dir: Path, logger: RunLogger):
    logger.log("face stage started")
    ensure_dir(out_dir)

    target_face_path = out_dir / "target_face.png"
    cv2.imwrite(str(target_face_path), target_face_bgr)

    overall = []

    for model_name in fm.MODELS:
        model_dir = out_dir / model_name
        ensure_dir(model_dir)
        started = time.perf_counter()
        logger.log(f"face stage: running {model_name}")

        try:
            selfie_faces, _, selfie_face_bgr = fm.get_face_crops(selfie_path, detector_backend, enforce_detection)
            target_faces, _, target_face_crop_bgr = fm.get_face_crops(str(target_face_path), detector_backend, enforce_detection)

            cv2.imwrite(str(model_dir / "selfie_face.png"), selfie_face_bgr)
            cv2.imwrite(str(model_dir / "target_detected_face.png"), target_face_crop_bgr)
            fm.save_json(model_dir / "selfie_faces.json", selfie_faces)
            fm.save_json(model_dir / "target_faces.json", target_faces)

            verify_res = fm.DeepFace.verify(
                img1_path=selfie_path,
                img2_path=str(target_face_path),
                model_name=model_name,
                detector_backend=detector_backend,
                distance_metric=distance_metric,
                align=True,
                enforce_detection=enforce_detection,
            )

            rep1, emb1 = fm.get_embedding(selfie_path, model_name, detector_backend, enforce_detection)
            rep2, emb2 = fm.get_embedding(str(target_face_path), model_name, detector_backend, enforce_detection)

            emb_cosine_distance = fm.cosine_distance(emb1, emb2)
            elapsed = time.perf_counter() - started

            result = {
                "status": "ok",
                "model": model_name,
                "passed": bool(verify_res.get("verified", False)),
                "distance": float(verify_res.get("distance", float("nan"))),
                "threshold": float(verify_res.get("threshold", float("nan"))),
                "threshold_margin": float(verify_res.get("threshold", float("nan")) - verify_res.get("distance", float("nan"))) if isinstance(verify_res.get("distance", None), (int, float)) and isinstance(verify_res.get("threshold", None), (int, float)) else float("nan"),
                "embedding_cosine_distance": emb_cosine_distance,
                "elapsed_seconds": elapsed,
            }
            fm.save_json(model_dir / "result.json", result)
            overall.append(result)
        except Exception as e:
            elapsed = time.perf_counter() - started
            err = {"model": model_name, "passed": False, "status": "error", "error": str(e), "elapsed_seconds": elapsed}
            fm.save_json(model_dir / "error.json", err)
            overall.append(err)

    save_json(out_dir / "overall_summary.json", overall)
    passed, used_logs = sequential_decision(overall)
    save_json(out_dir / "decision.json", {"passed": bool(passed), "model_order": fm.MODELS, "decision_logs": overall, "used_logs": used_logs})
    logger.log(f"face stage finished, passed={passed}")
    return bool(passed), overall


def run_single_ocr_field_debug(field_name: str, crop_bgr: np.ndarray, out_dir: Path, logger: RunLogger):
    ensure_dir(out_dir)
    cv2.imwrite(str(out_dir / "original.png"), crop_bgr)

    if field_name == "expiry_date":
        for idx, variant in enumerate(ao.preprocess_text_block(crop_bgr)):
            cv2.imwrite(str(out_dir / f"variant_{idx:02d}.png"), variant)
        best_info = ao.debug_expiry_date_from_crop(crop_bgr)
        for variant in best_info.get("variants", []):
            save_json(out_dir / f"variant_{variant['variant_index']:02d}.json", variant)
        save_json(out_dir / "best.json", best_info)
        logger.log(f"ocr field {field_name}: best_variant={best_info.get('best_variant_index')}, text={best_info.get('best_text', '')}")
        return best_info

    if field_name in {"name", "address", "back_text", "middle_text", "top_number"}:
        variants = ao.preprocess_text_block(crop_bgr)
        score_fn = ao._score_text
        post_text_fn = ao.merge_lines
    else:
        prefer_len = 14 if field_name == "id_number" else 8 if field_name == "birth_date" else None
        variants = ao.preprocess_numeric_block(crop_bgr)
        score_fn = lambda lines: ao._score_numeric(lines, prefer_len=prefer_len)
        post_text_fn = lambda lines: ao.numeric_field(lines, prefer_len=prefer_len)

    best_score = -1e18
    best_idx = -1
    best_lines = []
    variant_summaries = []

    for idx, variant in enumerate(variants):
        variant_path = out_dir / f"variant_{idx:02d}.png"
        cv2.imwrite(str(variant_path), variant)

        try:
            lines = ao._ocr_lines_from_image(variant)
            score = float(score_fn(lines))
            text = post_text_fn(lines)
            variant_info = {"variant_index": idx, "score": score, "text": text, "lines": lines}
        except Exception as e:
            variant_info = {"variant_index": idx, "score": None, "text": "", "lines": [], "error": str(e)}
            score = -1e18
            lines = []

        save_json(out_dir / f"variant_{idx:02d}.json", variant_info)
        variant_summaries.append(variant_info)

        if score > best_score:
            best_score = score
            best_idx = idx
            best_lines = lines

    best_text = post_text_fn(best_lines)
    best_info = {
        "field_name": field_name,
        "best_variant_index": best_idx,
        "best_score": best_score,
        "best_text": best_text,
        "best_lines": best_lines,
        "variants": variant_summaries,
    }
    save_json(out_dir / "best.json", best_info)
    logger.log(f"ocr field {field_name}: best_variant={best_idx}, text={best_text}")
    return best_info


def run_ocr_debug(front_crops: Dict[str, np.ndarray], back_crops: Dict[str, np.ndarray], out_dir: Path, logger: RunLogger):
    logger.log("ocr stage started")
    ensure_dir(out_dir)

    front_dir = out_dir / "front_fields"
    back_dir = out_dir / "back_fields"
    ensure_dir(front_dir)
    ensure_dir(back_dir)

    front_debug = {}
    for field_name, crop in front_crops.items():
        front_debug[field_name] = run_single_ocr_field_debug(field_name, crop, front_dir / field_name, logger)

    back_debug = {}
    for field_name, crop in back_crops.items():
        back_debug[field_name] = run_single_ocr_field_debug(field_name, crop, back_dir / field_name, logger)

    fields = ao.extract_fields_from_crops(front_crops=front_crops, back_crops=back_crops)
    save_json(out_dir / "final_fields.json", fields)
    save_json(out_dir / "summary.json", {"front_debug_fields": sorted(list(front_debug.keys())), "back_debug_fields": sorted(list(back_debug.keys())), "final_fields": fields})
    logger.log("ocr stage finished")
    return fields


def run_liveness_debug(selfie_path: str, detector_backend: str, enforce_detection: bool, out_dir: Path, logger: RunLogger):
    logger.log("liveness stage started")
    ensure_dir(out_dir)
    result = lg.check_selfie_liveness(
        selfie_path=selfie_path,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
    )
    save_json(out_dir / "result.json", result)
    logger.log(f"liveness stage finished, passed={result.get('passed', False)}")
    return bool(result.get("passed", False)), result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfie", required=True)
    ap.add_argument("--front-image", required=True)
    ap.add_argument("--back-image", required=True)
    ap.add_argument("--features-root", required=True)
    ap.add_argument("--front-template", required=True)
    ap.add_argument("--back-template", required=True)
    ap.add_argument("--template-config", required=True)
    ap.add_argument("--run-dir", default="debug_runs")
    ap.add_argument("--feature-include-deep", action="store_true")
    ap.add_argument("--min-feature-hits-front", type=int, default=None)
    ap.add_argument("--min-feature-hits-back", type=int, default=None)
    ap.add_argument("--detector-backend", default="retinaface")
    ap.add_argument("--distance-metric", default="cosine", choices=["cosine", "euclidean", "euclidean_l2", "angular"])
    ap.add_argument("--enforce-detection", action="store_true")
    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) / f"run_{timestamp}"
    ensure_dir(run_dir)

    logger = RunLogger(run_dir / "run.log")

    save_json(
        run_dir / "run_config.json",
        {
            "selfie": args.selfie,
            "front_image": args.front_image,
            "back_image": args.back_image,
            "features_root": args.features_root,
            "front_template": args.front_template,
            "back_template": args.back_template,
            "template_config": args.template_config,
            "run_dir": str(run_dir.resolve()),
            "feature_include_deep": args.feature_include_deep,
            "min_feature_hits_front": args.min_feature_hits_front,
            "min_feature_hits_back": args.min_feature_hits_back,
            "detector_backend": args.detector_backend,
            "distance_metric": args.distance_metric,
            "enforce_detection": args.enforce_detection,
            "resolved_feature_model_order_front": [m.name for m in fp.select_preferred_models(args.feature_include_deep, top_k=4, side_name="front")],
            "resolved_feature_model_order_back": [m.name for m in fp.select_preferred_models(args.feature_include_deep, top_k=4, side_name="back")],
            "resolved_face_model_order": fm.MODELS,
        },
    )

    inputs_dir = run_dir / "inputs"
    copy_file(args.selfie, inputs_dir)
    copy_file(args.front_image, inputs_dir)
    copy_file(args.back_image, inputs_dir)
    copy_file(args.front_template, inputs_dir)
    copy_file(args.back_template, inputs_dir)
    copy_file(args.template_config, inputs_dir)
    copy_code_snapshot(run_dir)

    result = {
        "card_valid": False,
        "face_match": None,
        "liveness_passed": None,
        "overall_passed": False,
        "text": {
            "name": "",
            "address": "",
            "id_number": "",
            "birth_date": "",
            "expiry_date": "",
            "front_full_text": "",
            "back_full_text": "",
        },
    }

    try:
        front_valid, front_decision_logs, front_overall = run_feature_side_debug(
            side_name="front",
            subject_path=args.front_image,
            features_dir=str(Path(args.features_root) / "front"),
            include_deep=args.feature_include_deep,
            min_feature_hits=args.min_feature_hits_front,
            out_dir=run_dir / "01_feature_validation",
            logger=logger,
        )

        back_valid, back_decision_logs, back_overall = run_feature_side_debug(
            side_name="back",
            subject_path=args.back_image,
            features_dir=str(Path(args.features_root) / "back"),
            include_deep=args.feature_include_deep,
            min_feature_hits=args.min_feature_hits_back,
            out_dir=run_dir / "01_feature_validation",
            logger=logger,
        )

        save_json(
            run_dir / "01_feature_validation" / "stage_summary.json",
            {
                "front_valid": front_valid,
                "back_valid": back_valid,
                "card_valid": bool(front_valid and back_valid),
                "front_decision_logs": front_decision_logs,
                "back_decision_logs": back_decision_logs,
                "front_overall": front_overall,
                "back_overall": back_overall,
            },
        )

        if not (front_valid and back_valid):
            save_json(run_dir / "final_result.json", result)
            return

        result["card_valid"] = True

        config = tf.load_config(args.template_config)

        try:
            front_artifacts, front_align = run_alignment_debug("front", args.front_image, args.front_template, config, run_dir / "02_alignment", logger)
            back_artifacts, back_align = run_alignment_debug("back", args.back_image, args.back_template, config, run_dir / "02_alignment", logger)
            save_json(run_dir / "02_alignment" / "stage_summary.json", {"front_align": front_align, "back_align": back_align})
        except Exception as e:
            logger.exception("alignment stage", e)
            save_json(run_dir / "02_alignment" / "error.json", {"error": str(e), "traceback": traceback.format_exc()})
            save_json(run_dir / "final_result.json", result)
            return

        photo_crop = front_artifacts.get("photo", None)
        if photo_crop is None or photo_crop.size == 0:
            save_json(run_dir / "03_face_match" / "error.json", {"error": "front photo crop missing or empty"})
            save_json(run_dir / "final_result.json", result)
            return

        face_match, face_logs = run_face_debug(
            selfie_path=args.selfie,
            target_face_bgr=photo_crop,
            detector_backend=args.detector_backend,
            distance_metric=args.distance_metric,
            enforce_detection=args.enforce_detection,
            out_dir=run_dir / "03_face_match",
            logger=logger,
        )

        result["face_match"] = bool(face_match)
        save_json(run_dir / "03_face_match" / "stage_summary.json", {"face_match": face_match, "logs": face_logs})

        fields = run_ocr_debug(
            front_crops=front_artifacts.get("crops", {}),
            back_crops=back_artifacts.get("crops", {}),
            out_dir=run_dir / "04_ocr",
            logger=logger,
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

        liveness_passed, liveness_result = run_liveness_debug(
            selfie_path=args.selfie,
            detector_backend=args.detector_backend,
            enforce_detection=args.enforce_detection,
            out_dir=run_dir / "05_liveness",
            logger=logger,
        )
        result["liveness_passed"] = bool(liveness_passed)
        result["liveness"] = liveness_result
        result["overall_passed"] = bool(result["card_valid"] and result["face_match"] and result["liveness_passed"])

        save_json(run_dir / "final_result.json", result)
        logger.log("debug run finished")

    except Exception as e:
        logger.exception("full debug pipeline", e)
        save_json(run_dir / "fatal_error.json", {"error": str(e), "traceback": traceback.format_exc()})
        save_json(run_dir / "final_result.json", result)


if __name__ == "__main__":
    main()

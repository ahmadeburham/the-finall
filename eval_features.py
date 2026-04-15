"""
eval_features.py
Evaluates feature validation against all UploadFromMobile front images.

Usage:
    python eval_features.py [--out-dir feature_eval_baseline] [--features-dir features/front]
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir",      default="feature_eval_baseline")
    ap.add_argument("--features-dir", default=str(HERE / "features" / "front"))
    ap.add_argument("--img-dir",      default=str(HERE / "UploadFromMobile"))
    ap.add_argument("--side",         default="front")
    args = ap.parse_args()

    import feature_pipeline as fp

    out_dir = HERE / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    img_dir = Path(args.img_dir)
    images = sorted(p for p in img_dir.glob("*.jpg")
                    if p.name.lower() != "template.jpg")

    summary = []
    for img_path in images:
        try:
            passed, logs = fp.validate_subject_with_top_models(
                subject_path=str(img_path),
                features_dir=args.features_dir,
                include_deep=False,
                top_k=4,
                side_name=args.side,
            )
            row = {
                "image": img_path.name,
                "passed": bool(passed),
                "logs": logs,
            }
            detail = "  |  ".join(
                f"{l['model']}: found={l['features_found']}/{l['features_total']} "
                f"inliers={l['total_inliers']} score={l['avg_score']:.3f} "
                f"ir={l['avg_inlier_ratio']:.3f}"
                for l in logs
            )
            print(f"{'PASS' if passed else 'FAIL'}  {img_path.name}")
            print(f"      {detail}")
        except Exception as e:
            row = {"image": img_path.name, "passed": False, "error": str(e)}
            print(f"ERROR {img_path.name}: {e}")
        summary.append(row)

    passed_count = sum(1 for r in summary if r.get("passed"))
    print(f"\n=== {passed_count}/{len(summary)} passed ===")

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved → {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

"""
debug_feature_match.py
Shows per-crop match results for each test image to identify which crops are matching.
"""
from pathlib import Path
import feature_pipeline as fp
import json

HERE = Path(__file__).resolve().parent
features_dir = HERE / "features" / "front"
img_dir = HERE / "UploadFromMobile"

feature_paths = list(fp.list_images(features_dir))
models = fp.select_preferred_models(include_deep=False, top_k=4, side_name="front")
sift = next(m for m in models if m.name == "sift_flann")

for img_path in sorted(img_dir.glob("*.jpg")):
    if img_path.name.lower() == "template.jpg":
        continue
    print(f"\n--- {img_path.name} ---")
    subject = fp.read_image(img_path)
    for feat_path in feature_paths:
        feat = fp.read_image(feat_path)
        result, _, _, _, _ = sift.match(feat, subject)
        status = "FOUND" if result.found else "miss "
        print(f"  {status}  {feat_path.stem:30s}  inliers={result.inliers:3d}  ir={result.inlier_ratio:.3f}  score={result.score:.3f}")

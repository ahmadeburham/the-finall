"""
liveness_ui.py  â€”  Flask-based liveness + ID verification UI.

Usage:
    python liveness_ui.py
    Then open http://localhost:5000 in your browser.
"""
import base64
import shutil
import socket
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request

import arabic_ocr as ao
import face_match_top4 as fm
import liveness_gate as lg
import template_fields as tf

# â”€â”€ project root on sys.path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# â”€â”€ Flask app â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
app = Flask(__name__, template_folder=str(ROOT / "templates"))
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB

FRONT_TPL = str(ROOT / "templates" / "front_template.png")
BACK_TPL  = str(ROOT / "templates" / "back_template.png")
TPL_CFG   = str(ROOT / "id_template_config.json")

OCR_FIELDS = (
    "name", "address", "id_number", "birth_date",
    "expiry_date", "issue_date", "profession",
    "gender", "religion", "marital_status",
    "front_id_number", "back_id_number",
)


# â”€â”€ Routes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route("/")
def index():
    return render_template("liveness.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "ts": datetime.now().isoformat()})


@app.route("/verify", methods=["POST"])
def verify():
    tmpdir = None
    try:
        tmpdir = tempfile.mkdtemp(prefix="lv_ui_")
        front_path  = _save_upload("front",  tmpdir, "front.jpg")
        back_path   = _save_upload("back",   tmpdir, "back.jpg")
        selfie_path = _save_upload("selfie", tmpdir, "selfie.jpg")

        for name, path in [("front", front_path), ("back", back_path), ("selfie", selfie_path)]:
            if not path:
                return jsonify({"stage": "error", "gate": 0, "overall_passed": False,
                                "error": f"Missing {name} image"}), 400

        config = tf.load_config(TPL_CFG)

        # â•â• GATE 1 â€” Card alignment â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        try:
            aligned_front, front_align_info = tf.align_card_to_template(
                front_path, FRONT_TPL, return_candidates=False
            )
            front_art = tf.extract_side_artifacts(
                aligned_front,
                config,
                "front",
                subject_image=front_path,
                align_info=front_align_info,
            )
        except Exception as exc:
            return jsonify({
                "stage": "alignment_failed", "gate": 1, "overall_passed": False,
                "error": str(exc),
                "message_ar": "ØªØ¹Ø°Ù‘Ø± Ø§Ù„ØªØ¹Ø±Ù Ø¹Ù„Ù‰ Ø§Ù„ÙˆØ¬Ù‡ Ø§Ù„Ø£Ù…Ø§Ù…ÙŠ â€” ØªØ£ÙƒØ¯ Ø£Ù†Ù‡Ø§ Ù‡ÙˆÙŠØ© ÙˆØ·Ù†ÙŠØ© Ù…ØµØ±ÙŠØ©",
            })

        try:
            aligned_back, back_align_info = tf.align_card_to_template(
                back_path, BACK_TPL, return_candidates=False
            )
            back_art = tf.extract_side_artifacts(
                aligned_back,
                config,
                "back",
                subject_image=back_path,
                align_info=back_align_info,
            )
        except Exception as exc:
            return jsonify({
                "stage": "alignment_failed", "gate": 1, "overall_passed": False,
                "error": str(exc),
                "message_ar": "ØªØ¹Ø°Ù‘Ø± Ø§Ù„ØªØ¹Ø±Ù Ø¹Ù„Ù‰ Ø§Ù„ÙˆØ¬Ù‡ Ø§Ù„Ø®Ù„ÙÙŠ â€” ØªØ£ÙƒØ¯ Ø£Ù†Ù‡Ø§ Ù‡ÙˆÙŠØ© ÙˆØ·Ù†ÙŠØ© Ù…ØµØ±ÙŠØ©",
            })

        # â•â• GATE 2 â€” OCR + valid 14-digit Egyptian ID â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        ocr_fields: dict = {}
        try:
            fields = ao.extract_fields_from_crops(
                front_crops=front_art.get("crops", {}),
                back_crops=back_art.get("crops", {}),
            )
            ocr_fields = {k: fields.get(k, "") for k in OCR_FIELDS}
            ocr_fields["id_numbers_match"] = ao.ids_match_strict(
                fields.get("front_id_number", ""),
                fields.get("back_id_number", ""),
            )
        except Exception as exc:
            return jsonify({
                "stage": "invalid_card", "gate": 2, "overall_passed": False,
                "error": str(exc),
                "message_ar": "ÙØ´Ù„ Ø§Ø³ØªØ®Ø±Ø§Ø¬ Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª Ù…Ù† Ø§Ù„Ø¨Ø·Ø§Ù‚Ø©",
            })

        id_digits = "".join(c for c in ocr_fields.get("id_number", "") if c.isdigit())
        if len(id_digits) != 14:
            return jsonify({
                "stage": "invalid_card", "gate": 2, "overall_passed": False,
                "ocr": ocr_fields,
                "message_ar": "Ù„Ø§ ØªØ¨Ø¯Ùˆ Ø§Ù„Ø¨Ø·Ø§Ù‚Ø© Ù‡ÙˆÙŠØ© ÙˆØ·Ù†ÙŠØ© Ù…ØµØ±ÙŠØ© â€” Ù„Ù… ÙŠÙØ³ØªØ®Ø±Ø¬ Ø±Ù‚Ù… Ù‡ÙˆÙŠØ© Ù…ÙƒÙˆÙ‘Ù† Ù…Ù† Ù¡Ù¤ Ø±Ù‚Ù…Ù‹Ø§",
            })


        # â•â• GATE 3 â€” Face match â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        photo_crop = front_art.get("photo")
        if photo_crop is not None and getattr(photo_crop, "size", 0) > 0:
            try:
                fm_passed, fm_logs = fm.verify_top4_from_array(
                    selfie_path=selfie_path,
                    target_face_bgr=photo_crop,
                )
                face_result = {"passed": bool(fm_passed), "logs": lg.sanitize(fm_logs)}
            except Exception as exc:
                face_result = {"passed": False, "error": str(exc)}
        else:
            face_result = {"passed": False, "error": "No photo crop found in ID"}

        if not face_result.get("passed"):
            return jsonify({
                "stage": "face_match_failed", "gate": 3, "overall_passed": False,
                "ocr": ocr_fields,
                "face_match": face_result,
                "message_ar": "Ø§Ù„ÙˆØ¬Ù‡ ØºÙŠØ± Ù…Ø·Ø§Ø¨Ù‚ â€” ÙŠÙØ±Ø¬Ù‰ Ø§Ù„ØªØ£ÙƒØ¯ Ø£Ù† Ø§Ù„Ø´Ø®Øµ Ù‡Ùˆ ØµØ§Ø­Ø¨ Ø§Ù„Ø¨Ø·Ø§Ù‚Ø©",
            })

        # â•â• GATE 4 â€” Liveness â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        try:
            liveness_result = lg.sanitize(lg.check_selfie_liveness(selfie_path=selfie_path))
        except Exception as exc:
            liveness_result = {"passed": False, "error": str(exc)}


        # â•â• ALL GATES PASSED â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        return jsonify({
            "stage": "completed", "gate": None, "overall_passed": True,
            "ocr": ocr_fields,
            "face_match": face_result,
            "liveness": liveness_result,
        })

    except Exception as exc:
        return jsonify({
            "stage": "error", "gate": None, "overall_passed": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }), 500

    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _save_upload(field: str, tmpdir: str, filename: str) -> "str | None":
    f = request.files.get(field)
    if f and f.filename:
        out = str(Path(tmpdir) / filename)
        f.save(out)
        return out

    b64 = request.form.get(f"{field}_b64") or request.form.get(field)
    if b64:
        return _write_b64(b64, tmpdir, filename)

    if request.is_json:
        body = request.get_json(silent=True) or {}
        b64 = body.get(field) or body.get(f"{field}_b64")
        if b64:
            return _write_b64(b64, tmpdir, filename)

    return None


def _write_b64(b64: str, tmpdir: str, filename: str) -> "str | None":
    try:
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        data = base64.b64decode(b64)
        out = str(Path(tmpdir) / filename)
        Path(out).write_bytes(data)
        return out
    except Exception:
        return None


def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "YOUR_PC_IP"


if __name__ == "__main__":
    import os
    ip = _local_ip()
    use_https = os.environ.get("USE_HTTPS", "0") == "1"
    print("=" * 60)
    proto = "https" if use_https else "http"
    print(f"  ID Liveness Verification UI  ({proto.upper()})")
    print(f"  Local  : {proto}://localhost:5000")
    print(f"  Mobile : {proto}://{ip}:5000")
    if use_https:
        print("  NOTE   : Accept the self-signed cert warning in browser")
    else:
        print("  NOTE   : For mobile camera access, run: $env:USE_HTTPS=\"1\"; python liveness_ui.py")
    print("=" * 60)
    ssl_ctx = "adhoc" if use_https else None
    app.run(host="0.0.0.0", port=5000, debug=False,
            threaded=True, ssl_context=ssl_ctx)
 

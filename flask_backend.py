"""
Flask backend for FactSnap-V

This app exposes HTTP endpoints for analyzing audio/video files (and raw text)
using the existing FactSnap-V pipeline, without modifying any other files.
"""

import os
import sys
import tempfile
import time
import traceback
from typing import Optional

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Ensure local imports from this folder work regardless of cwd
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Import existing pipeline components
from main import FactSnapV  # noqa: E402
from config import SUPPORTED_AUDIO_FORMATS, SUPPORTED_VIDEO_FORMATS  # noqa: E402

# Optional: if you later want CORS, install flask-cors and uncomment
# try:
#     from flask_cors import CORS
#     _CORS_AVAILABLE = True
# except Exception:
#     _CORS_AVAILABLE = False


def create_app() -> Flask:
    app = Flask(__name__)

    # Limit uploads to ~500MB by default
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

    # Optional CORS: uncomment if needed and install flask-cors
    try:
        from flask_cors import CORS  # type: ignore
        CORS(app, resources={r"/*": {"origins": "*"}})
    except Exception:
        pass

    # Lazy-initialized singleton for the pipeline with a lock to avoid race conditions
    app.factsnap: Optional[FactSnapV] = None  # type: ignore[attr-defined]

    import threading
    app.init_lock = threading.Lock()  # type: ignore[attr-defined]

    def get_factsnap() -> FactSnapV:
        if app.factsnap is None:
            with app.init_lock:  # type: ignore[attr-defined]
                if app.factsnap is None:
                    # Initialize once and reuse models between requests
                    app.factsnap = FactSnapV()
        return app.factsnap

    @app.get("/health")
    def health():
        try:
            # Touch the instance to ensure it can initialize
            _ = get_factsnap()
            return jsonify({
                "status": "ok",
                "models_loaded": True,
                "supported_audio_formats": SUPPORTED_AUDIO_FORMATS,
                "supported_video_formats": SUPPORTED_VIDEO_FORMATS
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": str(e),
                "trace": traceback.format_exc()
            }), 500

    def _is_allowed_file(filename: str) -> bool:
        if not filename or "." not in filename:
            return False
        ext = os.path.splitext(filename)[1].lower()
        return ext in (SUPPORTED_AUDIO_FORMATS + SUPPORTED_VIDEO_FORMATS)

    @app.post("/analyze-file")
    def analyze_file():
        """
        Analyze an uploaded audio/video file.

        Form-Data:
          - file: the uploaded file (required)
          - extract_claims: optional, "true"/"false" (default: true)
          - export: optional, "true"/"false" to also write outputs to disk (default: false)
        """
        if 'file' not in request.files:
            return jsonify({"error": "Missing 'file' in form-data"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        if not _is_allowed_file(file.filename):
            return jsonify({
                "error": "Unsupported file format",
                "supported": SUPPORTED_AUDIO_FORMATS + SUPPORTED_VIDEO_FORMATS
            }), 400

        extract_claims = request.form.get('extract_claims', 'true').lower() != 'false'
        export_outputs = request.form.get('export', 'false').lower() == 'true'

        tmp_dir = tempfile.mkdtemp(prefix="factsnap_api_")
        tmp_path = os.path.join(tmp_dir, secure_filename(file.filename))

        try:
            file.save(tmp_path)
            fs = get_factsnap()

            t0 = time.time()
            results = fs.analyze_file(tmp_path, extract_claims=extract_claims)
            duration = time.time() - t0

            response = {
                "ok": True,
                "elapsed_sec": round(duration, 3),
                "results": results
            }

            if export_outputs:
                exported = fs.export_results(results)
                response["exported_files"] = exported

            return jsonify(response)

        except Exception as e:
            return jsonify({
                "ok": False,
                "error": str(e),
                "trace": traceback.format_exc()
            }), 500
        finally:
            try:
                # Clean temp input file. Internal components also clean their temps.
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                # Remove tmp dir if empty
                if os.path.isdir(tmp_dir) and not os.listdir(tmp_dir):
                    os.rmdir(tmp_dir)
            except Exception:
                pass

    @app.post("/analyze-text")
    def analyze_text():
        """
        Analyze raw text (bypasses audio). Useful for quick checks.

        JSON Body:
          - text: the input text (required)
          - extract_claims: optional bool (default: true)
        """
        try:
            payload = request.get_json(silent=True) or {}
            text = (payload.get("text") or "").strip()
            extract_claims = bool(payload.get("extract_claims", True))

            if not text:
                return jsonify({"error": "text is required"}), 400

            fs = get_factsnap()

            t0 = time.time()
            # Use existing components for text-only path
            sentences = fs.text_preprocessor.preprocess_transcript(text)
            emotion_results = fs.emotion_detector.analyze_sentences(sentences)
            bias_results = fs.bias_detector.analyze_sentences(sentences)

            fact_results = []
            fact_summary = {}
            if extract_claims:
                fact_results = fs.fact_verifier.verify_sentences_batch(sentences)
                fact_summary = fs.fact_verifier.get_verification_summary(fact_results)

            results = {
                'file_info': {
                    'source': 'text',
                    'analysis_time': time.time() - t0
                },
                'transcript': {
                    'text': text,
                    'sentences': sentences,
                    'sentence_count': len(sentences),
                    'character_count': len(text),
                    'word_count': len(text.split())
                },
                'emotion_analysis': {
                    'results': emotion_results,
                    'summary': fs.emotion_detector.get_emotion_summary(emotion_results)
                },
                'bias_analysis': {
                    'results': bias_results,
                    'summary': fs.bias_detector.get_bias_summary(bias_results)
                },
                'fact_verification': {
                    'results': fact_results,
                    'summary': fact_summary,
                    'claims_extracted': len(fact_results) if fact_results else 0
                }
            }

            return jsonify({"ok": True, "results": results})

        except Exception as e:
            return jsonify({
                "ok": False,
                "error": str(e),
                "trace": traceback.format_exc()
            }), 500

    # WebSocket endpoint for live audio streaming (raw WebSocket via Flask-Sock)
    try:
        from flask_sock import Sock  # type: ignore
        sock = Sock(app)
        print("[WS] Flask-Sock loaded. Enabling WebSocket endpoint at /ws/stream")
        app.config['WS_ENABLED'] = True

        @sock.route('/ws/stream')
        def ws_stream(ws):
            """
            Minimal WebSocket endpoint that accepts audio chunks from the browser.
            Currently acknowledges received bytes. Extend to buffer/process as needed.
            """
            import tempfile
            import shutil
            tmp_dir = tempfile.mkdtemp(prefix="factsnap_ws_")
            total_bytes = 0
            last_ack = time.time()
            try:
                while True:
                    data = ws.receive()
                    if data is None:
                        break
                    if isinstance(data, (bytes, bytearray)):
                        total_bytes += len(data)
                    # Heartbeat/ack every ~5s
                    now = time.time()
                    if now - last_ack > 5:
                        try:
                            ws.send(f"ack:{total_bytes}")
                        except Exception:
                            break
                        last_ack = now
            except Exception:
                pass
            finally:
                try:
                    shutil.rmtree(tmp_dir)
                except Exception:
                    pass
    except Exception as e:
        # If flask-sock/simple-websocket not installed, WS remains unavailable
        app.config['WS_ENABLED'] = False
        print(f"[WS] WebSocket disabled: {e}. Install flask-sock and simple-websocket.")

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FactSnap-V Flask Backend")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")

    args = parser.parse_args()

    app = create_app()
    # threaded allows concurrent requests; good for longer analyses
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)

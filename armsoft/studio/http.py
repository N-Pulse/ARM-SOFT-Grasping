"""HTTP transport for Dataset Studio.

The handler deliberately depends only on the public methods of the supplied
state object. Camera, filesystem, training, and inference logic stay outside
the transport layer.
"""
from __future__ import annotations

import json
from datetime import datetime
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def create_studio_handler(web_root: str | Path, max_request_bytes: int = 1024 * 1024):
    """Build a request handler bound to one static frontend directory."""
    web_root = Path(web_root).resolve()

    class StudioHandler(SimpleHTTPRequestHandler):
        state = None

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(web_root), **kwargs)

        def end_headers(self):
            if not any(header.lower().startswith(b"cache-control:") for header in self._headers_buffer):
                self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            super().end_headers()

        def log_message(self, fmt, *args):
            if not self.path.startswith("/api/frame/"):
                super().log_message(fmt, *args)

        def reply(self, data, code=200, content_type="application/json; charset=utf-8"):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/api/status": return self.reply(self.state.status())
            if url.path == "/api/sessions": return self.reply({"sessions": self.state.sessions()})
            if url.path == "/api/session":
                query = parse_qs(url.query)
                try: return self.reply(self.state.session_detail(query["name"][0]))
                except Exception: return self.send_error(404, "Session not found")
            if url.path == "/api/catalog":
                catalog = self.state.root.parents[1] / "catalog" / "index.json"
                try: return self.reply(json.loads(catalog.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError): return self.reply({"sessions": []})
            if url.path == "/api/storage": return self.reply(self.state.storage_report())
            if url.path == "/api/preprocess/status": return self.reply(self.state.preprocess_status())
            if url.path == "/api/annotation/status": return self.reply(self.state.annotation_status())
            if url.path == "/api/annotation/result":
                query = parse_qs(url.query)
                try: return self.reply(self.state.annotation_result(query["session"][0]))
                except Exception: return self.send_error(404, "Annotation result not found")
            if url.path == "/api/annotation-image":
                query = parse_qs(url.query)
                try: return self.reply(self.state.annotation_image(query["session"][0], query["frame"][0]).read_bytes(), 200, "image/png")
                except Exception: return self.send_error(404)
            if url.path == "/api/review/session":
                query = parse_qs(url.query)
                try: return self.reply(self.state.review_session(query["name"][0]))
                except Exception: return self.send_error(404, "Review session not found")
            if url.path == "/api/review/frame":
                query = parse_qs(url.query)
                try: return self.reply(self.state.review_frame(query["session"][0], query["frame"][0]))
                except Exception: return self.send_error(404, "Review frame not found")
            if url.path == "/api/dataset/inputs": return self.reply(self.state.dataset_inputs())
            if url.path == "/api/dataset/exports": return self.reply(self.state.dataset_exports())
            if url.path == "/api/train/inputs": return self.reply(self.state.training_inputs())
            if url.path == "/api/train/status": return self.reply(self.state.training_status())
            if url.path == "/api/train/runs": return self.reply(self.state.training_runs())
            if url.path == "/api/test/inputs": return self.reply(self.state.test_inputs())
            if url.path == "/api/test/status": return self.reply(self.state.test_status())
            if url.path == "/api/test/frame":
                with self.state.lock: data = self.state.test_image
                return self.reply(data, 200, "image/jpeg") if data else self.send_error(503, "No inference frame")
            if url.path == "/api/preprocess/result":
                query = parse_qs(url.query)
                try: return self.reply(self.state.derived_result(query["session"][0], query["version"][0]))
                except Exception: return self.send_error(404, "Derived result not found")
            if url.path == "/api/derived-image":
                query = parse_qs(url.query)
                try: return self.reply(self.state.derived_image(query["session"][0], query["version"][0], query["kind"][0], query["frame"][0]).read_bytes(), 200, "image/png")
                except Exception: return self.send_error(404)
            if url.path.startswith("/api/frame/"):
                with self.state.lock: data = self.state.images.get(url.path.rsplit('/', 1)[-1])
                return self.reply(data, 200, "image/jpeg") if data else self.send_error(503, "No frame")
            if url.path == "/api/session-image":
                query = parse_qs(url.query)
                try: return self.reply(self.state.image(query["session"][0], query["frame"][0]).read_bytes(), 200, "image/png")
                except Exception: return self.send_error(404)
            if url.path == "/": self.path = "/index.html"
            return super().do_GET()

        def do_POST(self):
            try:
                origin = self.headers.get("Origin")
                if origin and urlparse(origin).netloc != self.headers.get("Host"):
                    return self.reply({"ok": False, "error": "Cross-origin requests are not allowed"}, 403)
                length = int(self.headers.get("Content-Length", 0))
                if length < 0 or length > max_request_bytes:
                    return self.reply({"ok": False, "error": "Request body is too large"}, 413)
                data = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(data, dict): raise ValueError("JSON request body must be an object")
                if self.path == "/api/config":
                    low, high = float(data["min_depth"]), float(data["max_depth"])
                    if not .001 <= low < high <= 10: raise ValueError("Depth range is invalid")
                    with self.state.lock: self.state.config = {"min_depth": low, "max_depth": high}
                elif self.path == "/api/capture/start": self.state.begin(str(data.get("name") or datetime.now().strftime("object_%Y%m%d_%H%M%S")), int(data.get("frames", 100)), int(data.get("every", 3)), str(data.get("notes", "")))
                elif self.path == "/api/capture/stop": self.state.finish()
                elif self.path == "/api/session/delete": return self.reply({"ok": True, "result": self.state.delete_session(str(data.get("name", "")), str(data.get("confirmation", "")))})
                elif self.path == "/api/preprocess/start": return self.reply({"ok": True, "job": self.state.preprocess_start(data)})
                elif self.path == "/api/preprocess/delete": return self.reply({"ok": True, "result": self.state.preprocess_delete(str(data.get("session", "")), str(data.get("version", "")), str(data.get("confirmation", "")))})
                elif self.path == "/api/annotation/start": return self.reply({"ok": True, "job": self.state.annotation_start(data)})
                elif self.path == "/api/annotation/delete": return self.reply({"ok": True, "result": self.state.annotation_delete(str(data.get("session", "")), str(data.get("confirmation", "")))})
                elif self.path == "/api/review/save": return self.reply({"ok": True, "result": self.state.review_save(data)})
                elif self.path == "/api/dataset/validate": return self.reply({"ok": True, "report": self.state.dataset_validate(data)})
                elif self.path == "/api/dataset/export": return self.reply({"ok": True, "result": self.state.dataset_export(data)})
                elif self.path == "/api/dataset/delete": return self.reply({"ok": True, "result": self.state.dataset_delete(str(data.get("version", "")), str(data.get("confirmation", "")))})
                elif self.path == "/api/train/start": return self.reply({"ok": True, "job": self.state.training_start(data)})
                elif self.path == "/api/train/stop": return self.reply({"ok": True, "job": self.state.training_stop()})
                elif self.path == "/api/train/delete": return self.reply({"ok": True, "result": self.state.training_delete(str(data.get("name", "")), str(data.get("confirmation", "")))})
                elif self.path == "/api/test/start": return self.reply({"ok": True, "job": self.state.test_start(data)})
                elif self.path == "/api/test/stop": return self.reply({"ok": True, "job": self.state.test_stop_live()})
                elif self.path == "/api/test/save-failure": return self.reply({"ok": True, "result": self.state.test_save_failure(data)})
                elif self.path == "/api/test/export-pointcloud": return self.reply({"ok": True, "result": self.state.test_export_pointcloud(data)})
                else: return self.send_error(404)
                return self.reply({"ok": True})
            except Exception as error:
                return self.reply({"ok": False, "error": str(error)}, 400)

    StudioHandler.__name__ = "StudioHandler"
    return StudioHandler

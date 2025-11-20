"""EasyOCR client for Talon - connects to standalone server process."""

import base64
import json
import logging
import os
import tempfile
import urllib.error
import urllib.request

from . import _base


def is_server_running(port=8765):
    """Check if EasyOCR server is responsive."""
    try:
        response = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=1)
        return response.status == 200
    except (urllib.error.URLError, ConnectionRefusedError, OSError):
        return False


def start_server_process(port=8765):
    """Check that EasyOCR server is running (user must start manually)."""
    # Server is managed manually by the user
    if not is_server_running(port):
        logging.warning(
            f"EasyOCR server not running on port {port}. "
            "Please start it manually with: uv run _easyocr_server.py --port {port}"
        )
    else:
        logging.info("EasyOCR server is running")


class EasyOcrTalonBackend(_base.OcrBackend):
    """OCR backend using EasyOCR standalone server."""

    def __init__(self, port=8765):
        """Initialize the EasyOCR Talon backend.

        Args:
            port: Port for HTTP server (default: 8765)
        """
        global _server_port
        _server_port = port
        self.port = port
        self.server_url = f"http://localhost:{port}"

        # Start server if not running
        start_server_process(port)

    def run_ocr(self, image) -> _base.OcrResult:
        """Run OCR on a Talon image using the EasyOCR server.

        Args:
            image: Talon image object with .rect attribute

        Returns:
            OcrResult containing detected text and bounding boxes
        """
        temp_path = None
        try:
            # Convert Talon image to PNG file
            # Note: skia.Image.save() requires a file path, not BytesIO
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_path = temp_file.name

            image.save(temp_path)

            # Read PNG file and encode as base64
            with open(temp_path, "rb") as f:
                image_bytes = f.read()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            # Send request
            request_data = json.dumps({"image": image_b64}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.server_url}/ocr",
                data=request_data,
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                response_data = response.read()

            # Parse response
            if not response_data:
                logging.error("Empty response from EasyOCR server")
                return _base.OcrResult(lines=[])

            try:
                detections = json.loads(response_data)
            except json.JSONDecodeError as e:
                logging.error(
                    f"Invalid JSON from server: {response_data[:200]!r} - {e}"
                )
                return _base.OcrResult(lines=[])

            # Convert to OcrResult
            lines = []
            x_offset = image.rect.x
            y_offset = image.rect.y

            for detection in detections:
                word = self._parse_detection(detection, x_offset, y_offset)
                lines.append(_base.OcrLine(words=[word]))

            return _base.OcrResult(lines=lines)

        except Exception as e:
            logging.error(f"EasyOCR server request error: {e}")
            return _base.OcrResult(lines=[])
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def _parse_detection(  # noqa: ARG002
        self, data: dict, x_offset: int, y_offset: int
    ) -> _base.OcrWord:
        """Parse a single detection from EasyOCR JSON output.

        Args:
            data: Dictionary with 'boxes', 'text', and 'confident' keys
            x_offset: X offset from Talon image rect (currently unused)
            y_offset: Y offset from Talon image rect (currently unused)

        Returns:
            OcrWord with text and bounding box information
        """
        boxes = data["boxes"]
        text = data["text"]

        # Calculate bounding box from corner points
        xs = [point[0] for point in boxes]
        ys = [point[1] for point in boxes]

        left = min(xs)
        top = min(ys)
        width = max(xs) - left
        height = max(ys) - top

        return _base.OcrWord(
            text=text,
            left=float(left),
            top=float(top),
            width=float(width),
            height=float(height),
        )

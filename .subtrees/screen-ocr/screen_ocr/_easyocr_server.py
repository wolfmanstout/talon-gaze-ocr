#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "easyocr>=1.7.0",
#   "pillow>=10.0.0",
#   "numpy>=1.24.0",
# ]
#
# [[tool.uv.index]]
# url = "https://download.pytorch.org/whl/cu126"
# ///

"""Standalone EasyOCR HTTP server for Talon integration.

This server runs in a separate process with its own Python environment,
allowing EasyOCR to be used even when it cannot be imported into Talon.
"""

import argparse
import base64
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO

import numpy as np
from PIL import Image


class EasyOCRHandler(BaseHTTPRequestHandler):
    """HTTP request handler for EasyOCR server."""

    # Class-level reader (loaded once per server process)
    reader = None

    @classmethod
    def ensure_reader(cls):
        """Initialize EasyOCR reader if not already loaded."""
        if cls.reader is None:
            logging.info("Loading EasyOCR model...")
            import easyocr

            cls.reader = easyocr.Reader(["en"])
            logging.info("EasyOCR model loaded successfully")

    def log_message(self, format, *args):  # noqa: ARG002
        """Custom logging."""
        logging.info(format % args)

    def do_GET(self):
        """Handle GET requests (health check)."""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle POST requests (OCR)."""
        if self.path != "/ocr":
            self.send_response(404)
            self.end_headers()
            return

        try:
            # Read request
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            request = json.loads(post_data)

            # Decode image
            image_data = base64.b64decode(request["image"])
            image = Image.open(BytesIO(image_data))
            image_array = np.array(image)

            # Ensure reader is loaded
            self.ensure_reader()

            # Run OCR
            assert self.reader is not None
            results = self.reader.readtext(image_array)

            # Format response
            # Convert numpy types to Python native types for JSON serialization
            response = [
                {
                    "boxes": [[int(x), int(y)] for x, y in bbox],
                    "text": text,
                    "confident": float(confidence),
                }
                for bbox, text, confidence in results
            ]

            # Send response
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            logging.error(f"OCR request error: {e}", exc_info=True)
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())


def main():
    """Start the EasyOCR HTTP server."""
    parser = argparse.ArgumentParser(description="EasyOCR HTTP server")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Start server
    server = HTTPServer((args.host, args.port), EasyOCRHandler)
    logging.info(f"EasyOCR server listening on {args.host}:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Server shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()

"""EasyOCR backend for Talon that uses Talon screenshots with EasyOCR CLI processing."""

import json
import logging
import os
import platform
import subprocess
import tempfile

from . import _base


class EasyOcrTalonBackend(_base.OcrBackend):
    """OCR backend using EasyOCR CLI with Talon image objects.

    This backend is designed to work within Talon, receiving Talon image objects
    (which include offset information) and using the EasyOCR command-line tool
    for text recognition.
    """

    def __init__(self, easyocr_command: str = "easyocr"):
        """Initialize the EasyOCR Talon backend.

        Args:
            easyocr_command: Path to the easyocr executable. Defaults to "easyocr"
                           which will use the version available in PATH.
        """
        self.easyocr_command = easyocr_command

    def run_ocr(self, image) -> _base.OcrResult:
        """Run OCR on a Talon image using the EasyOCR CLI.

        Args:
            image: Talon image object with .rect attribute (containing x, y offsets)

        Returns:
            OcrResult containing detected text and bounding boxes
        """
        temp_path = None
        try:
            # Save Talon image (skia.Image) to temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_path = temp_file.name

            # Use skia.Image's save method to write PNG
            image.save(temp_path)

            # Prepare environment for subprocess
            process_env = os.environ.copy()
            if platform.system() == "Windows":
                # Enable UTF-8 mode for Python 3.7+
                process_env["PYTHONUTF8"] = "1"

            # Execute easyocr CLI
            result = subprocess.check_output(
                [
                    self.easyocr_command,
                    "-l",
                    "en",
                    "-f",
                    temp_path,
                    "--detail=1",
                    "--output_format=json",
                ],
                encoding="utf-8",
                stderr=subprocess.PIPE,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0  # type: ignore
                ),
                env=process_env if platform.system() == "Windows" else None,
            )

            # Parse output - one JSON object per line
            lines = []
            x_offset = image.rect.x
            y_offset = image.rect.y

            for line in result.strip().split("\n"):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                    word = self._parse_detection(data, x_offset, y_offset)
                    # Each word gets its own line (matching current _easyocr.py behavior)
                    lines.append(_base.OcrLine(words=[word]))
                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse JSON line: {line!r} - {e}")
                    continue

            return _base.OcrResult(lines=lines)

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else str(e)
            logging.error(f"EasyOCR CLI error: {error_msg}")
            # Return empty result on error
            return _base.OcrResult(lines=[])
        except FileNotFoundError:
            logging.error(
                f"EasyOCR command not found: {self.easyocr_command}. "
                "Please install easyocr or specify the correct path."
            )
            return _base.OcrResult(lines=[])
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def _parse_detection(
        self, data: dict, x_offset: int, y_offset: int
    ) -> _base.OcrWord:
        """Parse a single detection from EasyOCR JSON output.

        Args:
            data: Dictionary with 'boxes', 'text', and 'confident' keys
            x_offset: X offset from Talon image rect (to adjust coordinates)
            y_offset: Y offset from Talon image rect (to adjust coordinates)

        Returns:
            OcrWord with text and bounding box information adjusted for offsets
        """
        # boxes is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        # (top-left, top-right, bottom-right, bottom-left)
        boxes = data["boxes"]
        text = data["text"]

        # Calculate bounding box from the 4 corner points
        xs = [point[0] for point in boxes]
        ys = [point[1] for point in boxes]

        # Coordinates are relative to the cropped image, but we need them
        # relative to the cropped image (no offset adjustment needed since
        # EasyOCR processes the cropped image)
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

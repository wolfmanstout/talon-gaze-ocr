"""Base classes used by backend implementations."""

from dataclasses import dataclass
from typing import List


class OcrBackend:
    """Base class for backend used to perform OCR."""

    def run_ocr(self, image):
        """Return the OcrResult corresponding to the image."""
        raise NotImplementedError()


@dataclass
class OcrWord:
    text: str
    left: float
    top: float
    width: float
    height: float


@dataclass
class OcrLine:
    words: List[OcrWord]


@dataclass
class OcrResult:
    lines: List[OcrLine]

"""Base classes used by backend implementations."""

from dataclasses import dataclass
from typing import List


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


class OcrBackend:
    """Base class for backend used to perform OCR."""

    def run_ocr(self, image) -> OcrResult:
        """Return the OcrResult corresponding to the image."""
        raise NotImplementedError()

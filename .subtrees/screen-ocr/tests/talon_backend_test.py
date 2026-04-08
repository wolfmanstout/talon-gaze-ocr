import importlib
import sys
import types
from typing import Any, cast

import numpy as np

import screen_ocr._screen_ocr as screen_ocr_module


class FakeImage:
    def __init__(
        self,
        array,
        rect=None,
        color_type="BGRA_8888",
        alpha_type="PREMUL",
    ):
        self._array = np.array(array, copy=True)
        self.rect = rect
        self.color_type = color_type
        self.alpha_type = alpha_type
        self.width = self._array.shape[1]
        self.height = self._array.shape[0]

    def __array__(self, dtype=None, copy=None):
        if dtype is None:
            return self._array
        return self._array.astype(dtype)

    @classmethod
    def from_pixels(
        cls,
        data,
        stride,
        width,
        height,
        color_type=None,
        alpha_type=None,
    ):
        row_width = width * 4
        array = np.frombuffer(data, dtype=np.uint8).reshape(height, stride)
        array = array[:, :row_width].reshape(height, width, 4)
        return cls(
            array,
            color_type=color_type or "BGRA_8888",
            alpha_type=alpha_type or "PREMUL",
        )


class FakeOcrModule:
    def __init__(self):
        self.calls = []

    def ocr(self, image):
        self.calls.append(image)
        return []


def load_talon_module(monkeypatch):
    fake_ocr = FakeOcrModule()
    talon_module = types.ModuleType("talon")
    cast(Any, talon_module).skia = types.SimpleNamespace(Image=FakeImage)
    experimental_module = types.ModuleType("talon.experimental")
    cast(Any, experimental_module).ocr = fake_ocr
    cast(Any, talon_module).experimental = experimental_module
    monkeypatch.setitem(sys.modules, "talon", talon_module)
    monkeypatch.setitem(sys.modules, "talon.experimental", experimental_module)
    monkeypatch.delitem(sys.modules, "screen_ocr._talon", raising=False)
    import screen_ocr._talon as talon_module_impl

    talon_module_impl = importlib.reload(talon_module_impl)
    return talon_module_impl, fake_ocr


def test_create_reader_passes_talon_invert_flag(monkeypatch):
    class FakeBackend:
        def __init__(self, invert_dark_images=False):
            self.invert_dark_images = invert_dark_images

    fake_talon_module = types.SimpleNamespace(TalonBackend=FakeBackend)
    monkeypatch.setattr(screen_ocr_module, "_talon", fake_talon_module)

    reader = screen_ocr_module.Reader.create_reader(
        backend="talon",
        invert_dark_images=True,
    )

    assert isinstance(reader._backend, FakeBackend)
    assert reader._backend.invert_dark_images is True


def test_preprocess_leaves_light_image_unchanged(monkeypatch):
    talon_module, _ = load_talon_module(monkeypatch)
    backend = talon_module.TalonBackend(invert_dark_images=True)
    original = np.full((2, 2, 4), 240, dtype=np.uint8)
    original[:, :, 3] = 77
    image = FakeImage(original, rect=(1, 2, 3, 4))

    processed, grayscale = backend._preprocess(image)

    assert processed is image
    expected_grayscale = backend._grayscale(original)
    np.testing.assert_allclose(grayscale, expected_grayscale)


def test_preprocess_inverts_dark_image_and_preserves_metadata(monkeypatch):
    talon_module, _ = load_talon_module(monkeypatch)
    backend = talon_module.TalonBackend(invert_dark_images=True)
    original = np.array(
        [
            [[10, 20, 30, 40], [50, 60, 70, 80]],
            [[90, 100, 110, 120], [30, 40, 50, 60]],
        ],
        dtype=np.uint8,
    )
    image = FakeImage(original, rect=(10, 20, 30, 40))

    processed, grayscale = backend._preprocess(image)
    processed_array = np.array(processed)

    assert processed is not image
    np.testing.assert_array_equal(processed_array[:, :, :3], 255 - original[:, :, :3])
    np.testing.assert_array_equal(processed_array[:, :, 3], original[:, :, 3])
    np.testing.assert_allclose(grayscale, 255 - backend._grayscale(original))
    assert processed.rect == image.rect
    assert processed.color_type == image.color_type
    assert processed.alpha_type == image.alpha_type


def test_run_ocr_uses_preprocessed_image(monkeypatch):
    talon_module, fake_ocr = load_talon_module(monkeypatch)
    backend = talon_module.TalonBackend(invert_dark_images=True)
    original = np.full((2, 2, 4), 20, dtype=np.uint8)
    image = FakeImage(original, rect=(0, 0, 2, 2))

    result = backend.run_ocr(image)

    assert result.lines == []
    assert len(fake_ocr.calls) == 1
    assert fake_ocr.calls[0] is not image
    assert fake_ocr.calls[0].rect == image.rect

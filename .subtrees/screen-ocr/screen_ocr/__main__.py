# Simple script to perform OCR on the current screen contents.

import screen_ocr

# Use radius larger than pixel width of the screen to effectively disable
# cropping and OCR the entire screen.
ocr_reader = screen_ocr.Reader.create_quality_reader(radius=10000)
# Use screen_coordinates to determine where to crop nearby. Won't matter here
# due to large radius.
results = ocr_reader.read_nearby(screen_coordinates=(0, 0))
print(results.as_string())

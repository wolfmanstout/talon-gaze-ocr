# Chromium browsers use smooth scrolling on Windows/Linux which requires a longer
# wait time for scroll detection. Disable smooth scrolling at
# chrome://flags/#smooth-scrolling and add
# "tag(): user.chromium_smooth_scrolling_disabled" to your settings to use the
# default (faster) scroll wait time.
os: windows
os: linux
app: chrome
app: microsoft_edge
app: brave
app: vivaldi
app: opera
not tag: user.chromium_smooth_scrolling_disabled
-
settings():
    user.ocr_scroll_wait_ms = 250

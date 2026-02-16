# Browsers use smooth scrolling on Windows/Linux which requires a longer wait
# time for scroll detection. Disable smooth scrolling and add
# "tag(): user.browser_smooth_scrolling_disabled" to your settings to use the
# default (faster) scroll wait time.
#
# To disable smooth scrolling:
# - Chromium (Chrome, Edge, etc.): chrome://flags/#smooth-scrolling
# - Firefox: about:preferences, search for "smooth scrolling"
os: windows
os: linux
tag: browser
and not tag: user.browser_smooth_scrolling_disabled
-
settings():
    user.ocr_scroll_wait_ms = 250

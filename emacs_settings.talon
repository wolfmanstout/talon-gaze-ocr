# Emacs has discrete scroll amounts, so larger probe values are needed to get
# an accurate scroll ratio. Note: this won't work if the Emacs window is not focused.
app: emacs
-
settings():
    user.ocr_scroll_probe_amount = 200

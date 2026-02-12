# Emacs has discrete scroll amounts, so larger probe values are needed to get
# an accurate scroll ratio.
app: emacs
-
settings():
    user.ocr_scroll_probe_amount = 200

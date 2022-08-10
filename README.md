# talon-gaze-ocr

Talon scripts to enable advanced cursor control using eye tracking and OCR.

All Python package dependencies are present in the `.subtrees` directory so that no pip
installations are needed on Mac. On Windows, the Talon OCR API is not yet available, so you will
need to run `pip install winrt pillow` and set `user.ocr_use_talon_backend = 0`.

Changes can be pushed and pulled to all repositories using `{push,pull}\_all.sh`, provided that
`upstream`, `screen-ocr`, `gaze-ocr`, `rapidfuzz`, and `jarowinkler` are all configured as git
remotes, and git subtree.sh is available. If packages are available through standard pip
installation, these will be preferred (e.g. so that faster binary installations can be used.)

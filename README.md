# talon-gaze-ocr

Talon scripts to enable advanced cursor control using eye tracking and OCR. This is alpha
functionality which uses experimental/unsupported APIs, so it could break at any time.

All Python package dependencies are present in the `.subtrees` directory so that no pip
installations are needed on Mac. On Windows, the Talon OCR API is not yet available, so you will
need to run `C:\Users\<user>\AppData\Roaming\talon\.venv\Scripts\pip.bat install winrt pillow` and
set `user.ocr_use_talon_backend = 0` in your talon settings.

Changes can be pushed and pulled to all repositories using `push_all.sh` and `pull_all.sh`, provided
that `origin`, `screen-ocr`, `gaze-ocr`, `rapidfuzz`, and `jarowinkler` are all configured as git
remotes, and `git subtree` is available. If packages are available through standard pip
installation, these will be preferred (e.g. so that faster binary installations can be used.)

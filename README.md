# talon-gaze-ocr

Talon scripts to enable advanced cursor control using eye tracking and OCR.

All Python package dependencies are present in the `.subtrees` directory so that
no pip installations are needed. Changes can be pushed and pulled to all
repositories using `{push,pull}\_all.sh`, provided that `upstream`,
`screen-ocr`, `gaze-ocr`, `rapidfuzz`, and `jarowinkler` are all configured as
git remotes, and git subtree.sh is available. If packages are available through
standard pip installation, these will be preferred (e.g. so that faster binary
installations can be used.)

# talon-gaze-ocr

[Talon](https://talonvoice.com/) scripts to enable advanced cursor control using
eye tracking and text recognition (OCR). This is alpha functionality which uses
experimental/unsupported APIs, so it could break at any time. See
[my blog post](https://handsfreecoding.org/2022/11/27/gaze-ocr-talon-support-and-10-new-features/)
for an overview.

## Installation

`git clone` this repo into your user directory. Requires
[knausj_talon](https://github.com/knausj85/knausj_talon) to be installed as a
sibling in the same directory.

Required permissions:

- On Mac, Talon requires the ability to read the screen. As per the
  [macOS User Guide](https://support.apple.com/guide/mac-help/control-access-to-screen-recording-on-mac-mchld6aa7d23/mac):
  1. Choose Apple menu > System Settings, then click Privacy & Security in the sidebar. (You may need to scroll down.)
  2. Click Screen Recording.
  3. Turn screen recording on for Talon.

Required Python packages:

- On Mac, all Python package dependencies are present in the `.subtrees` directory
  so that no pip installations are needed.
- On Windows, the Talon OCR API is not yet available, so you will need to run
  `%APPDATA%\talon\.venv\Scripts\pip.bat install winsdk pillow`.
- On all platforms, if packages are available through standard pip installation,
  these will be preferred (e.g. so that faster binary installations can be used.)

## Features:

- Click, select, or position caret adjacent to any text visible onscreen.
- Tracks eye position as you speak to filter matches.
- Offers disambiguation if multiple matches are present.
- Applies fuzzy matching to one or more spoken words.
- Works with or without an eye tracker (just expect slower processing and more
  disambiguation).
- Matches homophones of recognized words (based on CSV in knausj). Also matches
  numbers and punctuation in either their spoken form (e.g. "two" and "period")
  or their symbolic form (e.g. "2" and ".").
- Briefly displays debugging overlay if no matches are present.

## Known issues:

- Only operates on the main screen, as defined by Talon.
- Updates (via git pull) and some settings changes require Talon restart.
- Numbers must be referred to by their individual digits.
- Modifications to punctuation and digit names in knausj not leveraged.
- Depends on OS-provided text recognition (OCR), which is not perfectly accurate.
- Cursor positioning often imprecise around text with underline, strikethrough,
  or squiggly.
- Text caret can interfere with text recognition.
- Command subtitles may cause disambiguation when selecting a range of text.
- Dragon recognition timestamps are slightly off, leading to lower accuracy
  especially during text selection. Works best with Conformer.
- See the [issue tracker](https://github.com/wolfmanstout/talon-gaze-ocr/issues)
  for other bugs that have been discovered.

## Dependencies

The .subtrees directory contains dependency packages needed by talon-gaze-ocr:

- gaze-ocr was cloned from https://github.com/wolfmanstout/gaze-ocr
- screen-ocr was cloned from https://github.com/wolfmanstout/screen-ocr
- rapidfuzz was cloned from https://github.com/maxbachmann/RapidFuzz
- jarowinkler was cloned from https://github.com/maxbachmann/JaroWinkler

To contribute, changes can be pushed and pulled to all repositories using
`push_all.sh` and `pull_all.sh`, provided that `origin`, `screen-ocr`,
`gaze-ocr`, `rapidfuzz`, and `jarowinkler` are all configured as git remotes,
and `git subtree` is available.

## Running without knausj_talon

As noted in the installation instructions,
[knausj_talon](https://github.com/knausj85/knausj_talon) is highly recommended,
but most functionality will still be available in degraded form without it (and
you will see some warning logs). Missing functionality:

- The main `user.timestamped_prose` capture is missing custom vocabulary and
  punctuation support.
- No homophones means no automatic smart handling of homophones (e.g. if "hear"
  is recognized it won't match "here" onscreen).
- "scroll" commands use `user.mouse_scroll_*()`, so they won't work.
- "replace" and "say" commands use `user.prose` to insert text, so they won't
  work.
- `actions.user.dictation_peek` not available means text
  deletion/replacement isn't as smart (i.e. extra space is left over when a word
  is deleted).

#!/bin/sh

git push origin main
git subtree.sh push --prefix=.subtrees/screen-ocr screen-ocr master
git subtree.sh push --prefix=.subtrees/gaze-ocr gaze-ocr master

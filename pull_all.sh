#!/bin/sh

git pull origin main
git subtree.sh pull --prefix=.subtrees/screen-ocr --squash screen-ocr master
git subtree.sh pull --prefix=.subtrees/gaze-ocr --squash gaze-ocr master
git subtree.sh pull --prefix=.subtrees/rapidfuzz --squash rapidfuzz main
git subtree.sh pull --prefix=.subtrees/jarowinkler --squash jarowinkler main

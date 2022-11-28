import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gaze-ocr",
    version="0.4.0",
    author="James Stout",
    author_email="james.wolf.stout@gmail.com",
    description="Library for applying OCR to where the user is looking.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wolfmanstout/gaze-ocr",
    packages=["gaze_ocr"],
    install_requires=[
        "screen-ocr",
    ],
    extras_require={
        "dragonfly": ["dragonfly2", "pythonnet"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)

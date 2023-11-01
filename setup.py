import os

from setuptools import setup, find_packages
from io import open


def load_requirements(file_list=None):
    if file_list is None:
        file_list = ["requirements.txt"]
    if isinstance(file_list, str):
        file_list = [file_list]
    requirements = []
    for file in file_list:
        with open(file, encoding="utf-8-sig") as f:
            requirements.extend(f.readlines())
    return requirements


def readme():
    with open("README.md", encoding="utf-8-sig") as f:
        README = f.read()
    return README

cwd = os.path.dirname(os.path.abspath(__file__))

version_txt = os.path.join(cwd, "version.txt")
with open(version_txt) as f:
    version = f.readline().strip()

setup(
    name="toddleocr",
    packages=find_packages(exclude=("test",)),
    # package_dir={"toddleocr": ""},
    include_package_data=True,
    entry_points={"console_scripts": ["toddleocr = toddleocr.__main__:main"]},
    version=version,
    install_requires=load_requirements(),
    license="Apache License 2.0",
    description="Awesome OCR toolkits based on Torch （8.6M ultra-lightweight pre-trained model, support training and deployment among server, mobile, embedded and IoT devices",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/arry-lee/ToddleOCR",
    download_url="https://github.com/arry-lee/ToddleOCR.git",
    keywords=[
        "ocr textdetection textrecognition toddleocr crnn east star-net rosetta ocrlite db chineseocr chinesetextdetection chinesetextrecognition"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Natural Language :: Chinese (Simplified)",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Utilities",
    ],
)

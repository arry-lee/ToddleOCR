#  Copyright (c) 2023. Arry Lee, <arry_lee@qq.com>
import sys
from pathlib import Path

import torchvision
import toddleocr

SCRIPT_DIR = Path(__file__).parent

def smoke_test_toddleocr():
    print(
        "Is toddleocr usable?"
    )

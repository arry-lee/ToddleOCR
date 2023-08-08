import os
import sys

sys.path.append("D:\dev\github\PaddleOCR")


from ocr.train import Pipeline

import yaml

with open('D:\dev\github\PaddleOCR\ocr\config\det_mv3_east.yml','r',encoding='utf-8') as f:
    config = yaml.safe_load(f)

# print(config)

p = Pipeline(config)
p.train()

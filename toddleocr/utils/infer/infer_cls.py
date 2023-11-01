import numpy as np
import torch
from loguru import logger

from toddleocr.utils.utility import get_image_file_list


@torch.no_grad()
def cls(model=None, infer_img=None):
    transform = model.transforms("infer")
    post_process_class = model.postprocessor
    model = model.model
    model.eval()
    for file in get_image_file_list(infer_img):
        logger.info("infer_img: {}".format(file))
        with open(file, "rb") as f:
            img = f.read()
            data = {"image": img}
        batch = transform(data)
        images = np.expand_dims(batch[0], axis=0)
        images = torch.Tensor(images)
        preds = model(images)
        logger.info(preds)
        post_result = post_process_class(preds)
        for rec_result in post_result:
            logger.info("\t result: {}".format(rec_result))
    logger.info("success!")

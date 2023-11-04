import json
import os

import cv2
import numpy as np
import torch
from loguru import logger
from tools.utility import draw_boxes

from toddleocr.utils.utility import get_image_file_list
from toddleocr.utils.visual import draw_rectangle


@torch.no_grad()
def tab(model, infer_img=None, save_res_path=None):
    transforms = model.transforms("infer")
    post_process_class = model.postprocessor
    model = model.model
    model.eval()
    #
    os.makedirs(save_res_path, exist_ok=True)
    with open(
        os.path.join(save_res_path, "infer.txt"), mode="w", encoding="utf-8"
    ) as f_w:
        for file in get_image_file_list(infer_img):
            logger.info("infer_img: {}".format(file))
            with open(file, "rb") as f:
                img = f.read()
                data = {"image": img}
            batch = transforms(data)
            logger.info("变换后图像")
            logger.info(batch)
            images = np.expand_dims(batch[0], axis=0)
            shape_list = np.expand_dims(batch[-1], axis=0)
            images = torch.Tensor(images)
            logger.info("输入张量")
            logger.info(images)
            preds = model(images)
            logger.info("预测结果")
            logger.info(preds)
            post_result = post_process_class(preds, [shape_list])
            logger.info("后处理结果")
            logger.info(post_result)

            structure_str_list = post_result["structure_batch_list"][0]
            bbox_list = post_result["bbox_batch_list"][0]
            structure_str_list = structure_str_list[0]
            structure_str_list = (
                ["<html>", "<body>", "<table>"]
                + structure_str_list
                + ["</table>", "</body>", "</html>"]
            )
            bbox_list_str = json.dumps(bbox_list.tolist())
            logger.info("result: {}, {}".format(structure_str_list, bbox_list_str))
            f_w.write("result: {}, {}\n".format(structure_str_list, bbox_list_str))
            if len(bbox_list) > 0 and len(bbox_list[0]) == 4:
                img = draw_rectangle(file, bbox_list)
            else:
                img = draw_boxes(cv2.imread(file), bbox_list)
            cv2.imwrite(os.path.join(save_res_path, os.path.basename(file)), img)
            logger.info("save result to {}".format(save_res_path))
        logger.info("success!")

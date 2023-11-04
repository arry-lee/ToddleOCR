import json
import os

import cv2
import numpy as np
import torch
from loguru import logger

from toddleocr.utils.utility import get_image_file_list


def draw_det_res(dt_boxes, img, img_name, save_path):
    if len(dt_boxes) > 0:
        src_im = img
        for box in dt_boxes:
            box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(src_im, [box], True, color=(0, 255, 0), thickness=1)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, os.path.basename(img_name))
        cv2.imwrite(save_path, src_im)
        logger.info("The detected Image saved in {}".format(save_path))


@torch.no_grad()
def det(model, infer_img=None, save_res_path=None):
    postprocessor = model.postprocessor
    transforms = model.transforms("infer")
    model = model.model

    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))
    model.eval()
    with open(save_res_path, "wb") as fout:
        for file in get_image_file_list(infer_img):
            logger.info("infer_img: {}".format(file))
            with open(file, "rb") as f:
                img = f.read()
                data = {"image": img}
            batch = transforms(data)

            images = np.expand_dims(batch[0], axis=0)
            shape_list = np.expand_dims(batch[1], axis=0)
            images = torch.Tensor(images)
            preds = model(images)
            post_result = postprocessor(preds, shape_list)
            src_img = cv2.imread(file)
            dt_boxes_json = []
            if isinstance(post_result, dict):
                det_box_json = {}
                for k in post_result.keys():
                    boxes = post_result[k][0]["points"]
                    dt_boxes_list = []
                    for box in boxes:
                        tmp_json = {"transcription": ""}
                        tmp_json["points"] = np.array(box).tolist()
                        dt_boxes_list.append(tmp_json)
                    det_box_json[k] = dt_boxes_list
                    save_det_path = os.path.dirname(
                        save_res_path
                    ) + "/det_results_{}/".format(k)
                    draw_det_res(boxes, src_img, file, save_det_path)
            else:
                boxes = post_result[0]["points"]
                dt_boxes_json = []
                for box in boxes:
                    tmp_json = {"transcription": ""}
                    tmp_json["points"] = np.array(box).tolist()
                    dt_boxes_json.append(tmp_json)
                save_det_path = os.path.dirname(save_res_path) + "/det_results/"
                draw_det_res(boxes, src_img, file, save_det_path)
            otstr = file + "\t" + json.dumps(dt_boxes_json) + "\n"
            fout.write(otstr.encode())
    logger.info("success!")

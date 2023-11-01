import json
import os

import torch

from toddleocr.utils.utility import get_image_file_list


@torch.no_grad()
def rec(model=None, infer_img=None, save_res_path=None, logger=None):
    post_processor = model.postprocessor
    transforms = model.transforms("infer")
    model = model.model
    model.eval()
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))

    with open(save_res_path, "w") as fout:
        for file in get_image_file_list(infer_img):
            logger.info("infer_img: {}".format(file))
            with open(file, "rb") as f:
                img = f.read()
                data = {"image": img}
            batch = transforms(data)
            # images = np.expand_dims(batch[0], axis=0)
            images = torch.Tensor(batch[0])
            images = torch.unsqueeze(images, 0)
            preds = model(images)
            post_result = post_processor(preds)
            info = None
            if isinstance(post_result, dict):
                rec_info = dict()
                for key in post_result:
                    if len(post_result[key][0]) >= 2:
                        rec_info[key] = {
                            "label": post_result[key][0][0],
                            "score": float(post_result[key][0][1]),
                        }
                info = json.dumps(rec_info, ensure_ascii=False)
            elif isinstance(post_result, list) and isinstance(post_result[0], int):
                info = str(post_result[0])
            elif len(post_result[0]) >= 2:
                info = post_result[0][0] + "\t" + str(post_result[0][1])
            if info is not None:
                logger.info("\t result: {}".format(info))
                fout.write(file + "\t" + info + "\n")
    logger.info("success!")

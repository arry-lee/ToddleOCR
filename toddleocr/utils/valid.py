#  Copyright (c) 2023. Arry Lee, <arry_lee@qq.com>

import platform
import time

import torch
from tqdm import tqdm


def valid(
    model,
    valid_dataloader,
    post_processor,
    evaluator,
    model_type=None,
    extra_input=False,
):
    model.eval()
    with torch.no_grad():
        total_frame = 0.0
        total_time = 0.0
        pbar = tqdm(total=len(valid_dataloader), desc="eval model:", position=0)
        max_iter = (
            len(valid_dataloader) - 1
            if platform.system() == "Windows"
            else len(valid_dataloader)
        )
        sum_images = 0
        for idx, batch in enumerate(valid_dataloader):
            if idx >= max_iter:
                break
            images = batch[0]
            start = time.time()

            if model_type == "table" or extra_input:
                preds = model(images, data=batch[1:])
            elif model_type in ["kie"]:
                preds = model(batch)
            elif model_type in ["can"]:
                preds = model(batch[:3])
            elif model_type in ["sr"]:
                preds = model(batch)
            else:
                preds = model(images)

            batch_numpy = []
            for item in batch:
                if isinstance(item, torch.Tensor):
                    batch_numpy.append(item.numpy())
                else:
                    batch_numpy.append(item)
            # Obtain usable results from post-processing methods
            total_time += time.time() - start
            # Evaluate the results of the current batch
            if model_type in ["table", "kie"]:
                if post_processor:
                    post_result = post_processor(preds, batch_numpy)
                    evaluator(post_result, batch_numpy)
                else:
                    evaluator(preds, batch_numpy)
            elif model_type in ["sr"]:
                evaluator(preds, batch_numpy)
            elif model_type in ["can"]:
                evaluator(preds[0], batch_numpy[2:], epoch_reset=(idx == 0))
            elif model_type in ["cse"]:
                post_result = post_processor(preds)
                evaluator(post_result, batch[1])

            else:
                post_result = post_processor(preds, batch_numpy[1])
                evaluator(post_result, batch_numpy)

            pbar.update(1)
            total_frame += len(images)
            sum_images += 1
        # Get final metric，eg. acc or hmean
        metric = evaluator.get_metric()

    pbar.close()
    model.train()
    metric["fps"] = total_frame / total_time
    return metric

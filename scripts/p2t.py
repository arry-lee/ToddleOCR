import paddle
import torch
from loguru import logger

from toddleocr.models.tab.tab_slanet_pplcnet import Model


def p2t(tensor) -> torch.Tensor:
    return torch.from_numpy(tensor.numpy())


torch.set_printoptions(precision=8)


def convert_params_to_torch(static_model_path, save_path):
    params = paddle.load(static_model_path)
    model = Model().model
    model_state_dict = model.state_dict()

    ends = set()
    for (name, param) in params.items():
        ends.add(name.rsplit('.', 1)[1])
        # if name not in model_state_dict:
        #     logger.info(f"torch模型中找不到{name}")
    logger.info(f"Paddle模型参数{len(params)},后缀{ends}")
    # print(*params.keys)

    ends = set()
    for (name, param) in model_state_dict.items():
        ends.add(name.rsplit('.', 1)[1])
        # if name not in params:
        #     logger.info(f"paddle模型中找不到{name}")
    logger.info(f"Torch模型参数{len(model_state_dict)},后缀{ends}")

    maps = {'running_var': '_variance',
            'running_mean': '_mean'}
    d = {}
    for k, v in model_state_dict.items():
        if k.endswith('num_batches_tracked'):
            d[k] = torch.tensor(0)
            continue

        if (key := k.rsplit('.', 1)[1]) in maps:
            k1 = k.replace(key, maps[key])
        else:
            k1 = k
        if k1 in params:
            if list(params[k1].shape) == list(model_state_dict[k].shape) != [256,
                                                                             256]:  # fixme 如果线性层的weight是方阵，这里就会出bug！！！
                d[k] = torch.from_numpy(params[k1].numpy())
                logger.info(f"{d[k].dtype},{params[k1].dtype}")
            else:
                logger.info(f"需要转置{k}")
                print(params[k1].shape, model_state_dict[k].shape)

                d[k] = torch.from_numpy(params[k1].numpy()).T
                logger.info(f"{d[k].dtype},{params[k1].dtype}")
                # print(params[k1])
                # print(d[k])
        else:
            print(k1)

    # print(*params.keys())
    # k1 = list((k for k in model_state_dict.keys() if not k.endswith('.num_batches_tracked')))
    # k1.sort(key=lambda x: x.replace('running', ''))
    # k2 = sorted([k for k in params.keys()])
    # logger.info(f"Torch模型参数量{len(k1)}")
    # logger.info(f"Paddle模型参数量{len(k2)}")
    #
    # t = set(k1)
    # p = set(k2)
    # logger.info(p.difference(t))
    #
    # d = {k: torch.tensor(0) for k in model_state_dict.keys() if k.endswith('.num_batches_tracked')}
    # for (a, b) in zip(k1, k2):
    #     print(a, b)
    # if params[b].shape == list(model_state_dict[a].shape):
    #     d[a] = torch.from_numpy(params[b].numpy())
    # else:
    #     # logger.info("需要转置",a, b, params[b].shape, list(model_state_dict[a].shape))
    #     d[a] = torch.from_numpy(params[b].numpy()).T

    torch.save(d, save_path)


#
# 调用函数
# static_model_path = 'model/ch_ppstructure_mobile_v2.0_SLANet_train/best_accuracy.pdparams'
# save_path = 'model/ch_ppstructure_mobile_v2.0_SLANet_train/best_accuracy.pt'
# convert_params_to_torch(static_model_path, save_path)
#

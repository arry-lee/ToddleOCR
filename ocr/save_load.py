import errno
import os
import pickle

import torch

__all__ = ['load_model']

from torch import nn

PREFIX = 'torchocr'
def _mkdir_if_not_exist(path, logger):
    """
    mkdir if not exists, ignore the exception when multiprocess mkdir together
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logger.warning(
                    'be happy if some process has already created {}'.format(
                        path))
            else:
                raise OSError('Failed to mkdir {}'.format(path))


def load_model(config, model:nn.Module,logger,optimizer=None, model_type='det'):
    """
    load model from checkpoint or pretrained_model
    """

    global_config = config["Global"]
    checkpoints = global_config.get('checkpoints')
    pretrained_model = global_config.get('pretrained_model')
    best_model_dict = {}
    is_float16 = False

    is_nlp_model = model_type == 'kie' and config["Model"]["algorithm"] not in ["SDMGR"]
    if is_nlp_model:
        # NOTE: for kie model dsitillation, resume training is not supported now
        if config["Model"]["algorithm"] in ["Distillation"]:
            return best_model_dict
        checkpoints = config['Model']['Backbone']['checkpoints'] #模型骨干的checkpoints
        # load kie method metric
        if checkpoints:
            if os.path.exists(os.path.join(checkpoints, 'metric.states')):
                with open(os.path.join(checkpoints, 'metric.states'),
                          'rb') as f:
                    states_dict = pickle.load(
                        f, encoding='latin1')
                best_model_dict = states_dict.get('best_model_dict', {})
                if 'epoch' in states_dict:
                    best_model_dict['start_epoch'] = states_dict['epoch'] + 1
            logger.info("resume from {}".format(checkpoints))

            if optimizer is not None:
                if checkpoints[-1] in ['/', '\\']:
                    checkpoints = checkpoints[:-1]
                if os.path.exists(checkpoints + '.pdopt'):
                    optim_dict = torch.load(checkpoints + '.pdopt')
                    optimizer.load_state_dict(optim_dict)
                else:
                    logger.warning(
                        "{}.pdopt is not exists, params of optimizer is not loaded".
                        format(checkpoints))

        return best_model_dict

    if checkpoints:
        if checkpoints.endswith('.pdparams'):
            checkpoints = checkpoints.replace('.pdparams', '')

        assert os.path.exists(checkpoints + ".pdparams"), "The {}.pdparams does not exists!".format(checkpoints)

        # load params from trained model
        params = torch.load(checkpoints + '.pdparams') # 参数
        state_dict = model.state_dict() # 状态字典
        new_state_dict = {}
        for key, value in state_dict.items():
            if key not in params:
                logger.warning("{} not in loaded params {} !".format(key, params.keys()))
                continue
            pre_value = params[key]
            if pre_value.dtype == torch.float16:
                is_float16 = True
            if pre_value.dtype != value.dtype:
                pre_value = pre_value.astype(value.dtype)
            if list(value.shape) == list(pre_value.shape):
                new_state_dict[key] = pre_value
            else:
                logger.warning(
                    "The shape of model params {} {} not matched with loaded params shape {} !".
                    format(key, value.shape, pre_value.shape))
        model.load_state_dict(new_state_dict)
        
        if is_float16:
            logger.info(
                "The parameter type is float16, which is converted to float32 when loading"
            )
        if optimizer is not None:
            if os.path.exists(checkpoints + '.pdopt'):
                optim_dict = torch.load(checkpoints + '.pdopt')
                optimizer.load_state_dict(optim_dict)
            else:
                logger.warning("{}.pdopt is not exists, params of optimizer is not loaded".format(checkpoints))

        if os.path.exists(checkpoints + '.states'):
            with open(checkpoints + '.states', 'rb') as f:
                states_dict = pickle.load(f, encoding='latin1')
            best_model_dict = states_dict.get('best_model_dict', {})
            if 'epoch' in states_dict:
                best_model_dict['start_epoch'] = states_dict['epoch'] + 1
        logger.info("resume from {}".format(checkpoints))
    elif pretrained_model:
        is_float16 = load_pretrained_params(model, pretrained_model,logger)
    else:
        logger.info('train from scratch')
    best_model_dict['is_float16'] = is_float16
    return best_model_dict


def load_pretrained_params(model, path,logger):
    if path.endswith('.pdparams'):
        path = path.replace('.pdparams', '')
    assert os.path.exists(path + ".pdparams"), "The {}.pdparams does not exists!".format(path)

    params = torch.load(path + '.pdparams')

    state_dict = model.state_dict()

    new_state_dict = {}
    is_float16 = False

    for k in params.keys():

        if k not in state_dict.keys():
            logger.warning("The pretrained params {} not in model".format(k))
        else:
            if params[k].dtype == torch.float16:
                is_float16 = True
            if params[k].dtype != state_dict[k].dtype:
                params[k] = params[k].astype(state_dict[k].dtype)
            if list(state_dict[k].shape) == list(params[k].shape):
                new_state_dict[k] = params[k]
            else:
                logger.warning(
                    "The shape of model params {} {} not matched with loaded params {} {} !".
                    format(k, state_dict[k].shape, k, params[k].shape))

    model.load_state_dict(new_state_dict)
    if is_float16:
        logger.info(
            "The parameter type is float16, which is converted to float32 when loading"
        )
    logger.info("load pretrain successful from {}".format(path))
    return is_float16


def save_model(model,
               optimizer,
               model_path,
               logger,
               config,
               is_best=False,
               prefix='ppocr',
               **kwargs):
    """
    save model to the target path
    """
    _mkdir_if_not_exist(model_path, logger)
    model_prefix = os.path.join(model_path, prefix)
    torch.save(optimizer.state_dict(), model_prefix + '.pdopt')

    is_nlp_model = config['Model']["model_type"] == 'kie' and config["Model"]["algorithm"] not in ["SDMGR"]
    if is_nlp_model:  # for kie system, we follow the save/load rules in NLP
        if config["Global"]['distributed']:
            arch = model._layers
        else:
            arch = model
        if config["Model"]["algorithm"] in ["Distillation"]:
            arch = arch.Student
        arch.backbone.model.save_pretrained(model_prefix)
        metric_prefix = os.path.join(model_prefix, 'metric')
    else:
        torch.save(model.state_dict(), model_prefix + '.pdparams')
        metric_prefix = model_prefix
    # save metric and config
    with open(metric_prefix + '.states', 'wb') as f:
        pickle.dump(kwargs, f, protocol=2)
    if is_best:
        logger.info('save best model is to {}'.format(model_prefix))
    else:
        logger.info("save model in {}".format(model_prefix))

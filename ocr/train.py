# 一个通用的训练脚本
# 角色:
import collections
import copy
import datetime
import errno
import importlib
import logging
import os
import pickle
import platform
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from tqdm import tqdm

import torchvision.transforms
from ocr.save_load import load_model
from ppocr.utils.stats import SmoothedValue

from loguru import logger

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def dynamic_import(module_path):
    module_name, _, class_name = module_path.rpartition('.')
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def build_object(config, section):
    object_config = config[section]
    object_module_path = object_config.pop('class')
    object_class = dynamic_import(object_module_path)
    object_instance = object_class(**object_config)
    return object_instance


class BaseModel(nn.Module):
    """模型的搭建"""

    def __init__(self, config):
        super().__init__()

        self.model_components = nn.ModuleDict()

        in_channels = config.get('in_channels', 3)
        self.model_type = config['model_type']
        self.algorithm = config['algorithm']

        component_order = ['Transform', 'Backbone', 'Neck', 'Head']
        prev_component_out_channels = in_channels

        for component_name in component_order:
            if component_name in config:
                component_config = config[component_name]
                if component_config is None:
                    continue
                component_config['in_channels'] = prev_component_out_channels
                self.model_components[component_name] = self.build_component(component_config)
                prev_component_out_channels = getattr(self.model_components[component_name], "out_channels", None)

        self.return_all_feats = config.get("return_all_feats", False)

    def build_component(self, config):
        component_class_name = config['class']
        component_params = config.copy()
        del component_params['class']
        component_class = dynamic_import(component_class_name)
        # Create an instance of the component class with the provided parameters
        component = component_class(**component_params)
        return component

    def forward(self, x):
        out_dict = {}

        for module_name, module in self.model_components.items():
            x = module(x)
            if isinstance(x, dict):
                out_dict.update(x)
            else:
                out_dict[f"{module_name}_out"] = x

        if self.return_all_feats:
            if self.training:
                return out_dict
            elif isinstance(x, dict):
                return x
            else:
                return {list(out_dict.keys())[-1]: x}
        else:
            return x


class TrainingStats(object):
    def __init__(self, window_size, stats_keys):
        self.window_size = window_size
        self.smoothed_losses_and_metrics = {
            key: SmoothedValue(window_size)
            for key in stats_keys
        }

    def update(self, stats):
        for k, v in stats.items():
            if k not in self.smoothed_losses_and_metrics:
                self.smoothed_losses_and_metrics[k] = SmoothedValue(
                    self.window_size)
            self.smoothed_losses_and_metrics[k].add_value(v)

    def get(self, extras=None):
        stats = collections.OrderedDict()
        if extras:
            for k, v in extras.items():
                stats[k] = v
        for k, v in self.smoothed_losses_and_metrics.items():
            stats[k] = round(v.get_median_value(), 6)

        return stats

    def log(self, extras=None):
        d = self.get(extras)
        strs = []
        for k, v in d.items():
            strs.append('{}: {:x<6f}'.format(k, v))
        strs = ', '.join(strs)
        return strs


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

    is_nlp_model = config['Model']["model_type"] == 'kie' and config[
        "Model"]["algorithm"] not in ["SDMGR"]
    if is_nlp_model is not True:
        torch.save(model.state_dict(), model_prefix + '.pdparams')
        metric_prefix = model_prefix
    else:  # for kie system, we follow the save/load rules in NLP
        if config['Base']['distributed']:
            arch = model._layers
        else:
            arch = model
        if config["Model"]["algorithm"] in ["Distillation"]:
            arch = arch.Student
        arch.backbone.model.save_pretrained(model_prefix)
        metric_prefix = os.path.join(model_prefix, 'metric')
    # save metric and config
    with open(metric_prefix + '.states', 'wb') as f:
        pickle.dump(kwargs, f, protocol=2)
    if is_best:
        logger.info('save best model is to {}'.format(model_prefix))
    else:
        logger.info("save model in {}".format(model_prefix))


def valid(model,
          valid_dataloader,
          post_process_class,
          eval_class,
          model_type=None,
          extra_input=False,
          ):
    model.eval()
    with torch.no_grad():
        total_frame = 0.0
        total_time = 0.0
        pbar = tqdm(
            total=len(valid_dataloader),
            desc='eval model:',
            position=0,
            leave=True)
        max_iter = len(valid_dataloader) - 1 if platform.system(
        ) == "Windows" else len(valid_dataloader)
        sum_images = 0
        for idx, batch in enumerate(valid_dataloader):
            if idx >= max_iter:
                break
            images = batch[0]
            start = time.time()

            if model_type == 'table' or extra_input:
                preds = model(images, data=batch[1:])
            elif model_type in ["kie"]:
                preds = model(batch)
            elif model_type in ['can']:
                preds = model(batch[:3])
            elif model_type in ['sr']:
                preds = model(batch)
                sr_img = preds["sr_img"]
                lr_img = preds["lr_img"]
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
            if model_type in ['table', 'kie']:
                if post_process_class is None:
                    eval_class(preds, batch_numpy)
                else:
                    post_result = post_process_class(preds, batch_numpy)
                    eval_class(post_result, batch_numpy)
            elif model_type in ['sr']:
                eval_class(preds, batch_numpy)
            elif model_type in ['can']:
                eval_class(preds[0], batch_numpy[2:], epoch_reset=(idx == 0))
            else:
                post_result = post_process_class(preds, batch_numpy[1])
                eval_class(post_result, batch_numpy)

            pbar.update(1)
            total_frame += len(images)
            sum_images += 1
        # Get final metric，eg. acc or hmean
        metric = eval_class.get_metric()

    pbar.close()
    model.train()
    metric['fps'] = total_frame / total_time
    return metric


# dist.get_world_size()


class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Pipeline:
    """
    Pipeline for training and validation.
    """

    def __init__(self, config):
        self.config = config
        self.global_config = config.get('Base')
        self.use_gpu = self.global_config.get('use_gpu', False) and torch.cuda.is_available()
        self.device = 'cuda' if self.use_gpu else 'cpu'
        self.use_dist = self.global_config.get('distributed', False)

    def init_distributed(self):
        if self.use_dist:
            # dist.init_process_group(backend='gloo')
            dist.init_process_group(backend='gloo')
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

            logger.info(self.rank)
            logger.info(self.world_size)

    def build_object(self, section):
        object_class, object_config = self.build_class(section)
        if isinstance(object_class, type):
            object_instance = object_class(**object_config)
            return object_instance
        elif callable(object_class):
            return object_class
        raise Exception(f'{object_class} is not callable')

    def build_class(self, section):
        object_config = copy.deepcopy(self.config[section])
        object_module_path = object_config.pop('class')
        object_class = dynamic_import(object_module_path)
        return object_class, object_config

    @property
    def train_dataloader(self):
        return self.build_dataloader('Train')

    @property
    def valid_dataloader(self):
        if self.config.get('Eval'):
            return self.build_dataloader('Eval')
        return None

    def get_dataset(self, mode, seed):
        # torchvision.datasets.DatasetFolder
        dataset_config = copy.deepcopy(self.config[mode]['Dataset'])
        if dataset_config.get('transform'):
            dataset_config['transform'] = build_object(dataset_config, 'transform')
        if dataset_config.get('target_transform'):
            dataset_config['target_transform'] = build_object(dataset_config, 'target_transform')

        transforms_config = dataset_config.pop('transforms', None)
        if transforms_config:
            transforms = []
            for cfg in transforms_config:
                trans_class = dynamic_import(cfg.pop('class'))
                trans_obj = trans_class(**cfg)
                transforms.append(trans_obj)
            transforms = torchvision.transforms.Compose(transforms)
            dataset_config.update(transforms=transforms)
        # torchvision.datasets.MNIST
        class_ = dynamic_import(dataset_config.pop('class'))
        # torchvision.datasets.MNIST
        dataset = class_(**dataset_config)
        return dataset

    def get_batch_sampler(self, mode, dataset, seed):
        loader_config = self.config[mode]['DataLoader']
        batch_size = loader_config['batch_size']
        drop_last = loader_config['drop_last']
        shuffle = loader_config.get('shuffle', False)
        if mode == "Train":
            if dist.is_initialized():
                sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last, seed=seed)
            else:
                sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        else:
            sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=drop_last)
        return batch_sampler

    def build_dataloader(self, mode, seed=None):
        # num_workers = self.config[mode]['loader']['num_workers']
        dataset = self.get_dataset(mode, seed)
        batch_sampler = self.get_batch_sampler(mode, dataset, seed)

        if self.use_gpu:
            num_workers = self.config[mode]['DataLoader']['num_workers']
            pin_memory = self.config[mode]['DataLoader'].get('pin_memory', True)
        else:
            num_workers = 0
            pin_memory = False

        collate_fn = self.config[mode]['DataLoader'].get('collate_fn',None)
        if collate_fn:
            collate_fn = copy.deepcopy(collate_fn)
            collate_fn_class = dynamic_import(collate_fn.pop('class'))
            collate_fn = collate_fn_class(**collate_fn)

        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn)
        return data_loader

    # @cached_property
    # def post_processor_class(self):
    #     self.build_class('PostProcessor')
    #
    #     return module_class

    def update_config(self):
        config = self.config
        post_process_class = self.build_object('PostProcessor')

        if hasattr(post_process_class, 'character'):
            char_num = len(getattr(post_process_class, 'character'))
            if config['Model']["algorithm"] in ["Distillation", ]:  # distillation model
                for key in config['Model']["Models"]:
                    if config['Model']['Models'][key]['Head']['class'] == 'MultiHead':  # for multi head
                        if config['PostProcess']['class'] == 'DistillationSARLabelDecode':
                            char_num = char_num - 2
                        # update SARLoss params
                        assert list(config['Loss']['loss_config_list'][-1].keys())[0] == 'DistillationSARLoss'
                        config['Loss']['loss_config_list'][-1]['DistillationSARLoss']['ignore_index'] = char_num + 1
                        out_channels_list = {}
                        out_channels_list['CTCLabelDecode'] = char_num
                        out_channels_list['SARLabelDecode'] = char_num + 2
                        config['Model']['Models'][key]['Head']['out_channels_list'] = out_channels_list
                    else:
                        config['Model']["Models"][key]["Head"]['out_channels'] = char_num

            elif config['Model']['Head']['class'] == 'MultiHead':  # for multi head
                if config['PostProcess']['class'] == 'SARLabelDecode':
                    char_num = char_num - 2
                # update SARLoss params
                assert list(config['Loss']['loss_config_list'][1].keys())[0] == 'SARLoss'
                if config['Loss']['loss_config_list'][1]['SARLoss'] is None:
                    config['Loss']['loss_config_list'][1]['SARLoss'] = {'ignore_index': char_num + 1}
                else:
                    config['Loss']['loss_config_list'][1]['SARLoss']['ignore_index'] = char_num + 1
                out_channels_list = {}
                out_channels_list['CTCLabelDecode'] = char_num
                out_channels_list['SARLabelDecode'] = char_num + 2
                config['Model']['Head']['out_channels_list'] = out_channels_list
            else:  # base rec model
                config['Model']["Head"]['out_channels'] = char_num

            if config['PostProcess']['class'] == 'SARLabelDecode':  # for SAR model
                config['Loss']['ignore_index'] = char_num - 1

    @property
    def model(self):
        self.update_config()

        _model = BaseModel(self.config['Model'])

        use_sync_bn = self.global_config.get("use_sync_bn", False)
        if use_sync_bn:
            _model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(_model)
            logger.info('convert_sync_batchnorm')

        if self.use_dist:
            _model = DDP(_model)

        # _model.to(device)
        return _model

    # def loss(self):
    @property
    def loss_class(self):
        return self.build_class('Loss')[0]

    def build_optimizer(self, model):
        # example = Optimizer
        optim_config = copy.deepcopy(self.config['Optimizer'])
        optim_name = optim_config.pop('class')

        lr_config = optim_config.pop('lr_scheduler')
        lr_scheduler_name = lr_config.pop('class')
        # torch.optim.Adam
        optimizer = dynamic_import(optim_name)(model.parameters(), **optim_config)

        # epochs = self.global_config['epoch_num']
        # step_each_epoch = len(self.train_dataloader)
        # lr_config.update({'epochs': epochs, 'step_each_epoch': step_each_epoch})
        # torch.optim.lr_scheduler.StepLR
        lr_scheduler = dynamic_import(lr_scheduler_name)(optimizer, **lr_config)
        return optimizer, lr_scheduler

    @property
    def eval_class(self):
        return self.build_class('Metric')[0]

    def is_rank0(self):
        if self.use_dist:
            return dist.get_rank()==0
        return True

    def train(self, log_writer=None):
        """
        config：训练配置，包含了训练过程中所需的各种参数和选项。
        train_dataloader：训练数据集的数据加载器，用于迭代获取训练数据。
        valid_dataloader：验证数据集的数据加载器，用于在训练过程中进行评估。
        device：设备（CPU或GPU）用于训练模型。
        model：待训练的模型。
        loss_class：损失函数的类别，用于计算训练过程中的损失。
        optimizer：优化器，用于更新模型的参数。
        lr_scheduler：学习率调度器，用于调整优化器的学习率。
        post_process_class：后处理的类别，用于处理模型输出并得到最终的预测结果。
        eval_class：评估指标的类别，用于评估模型在验证集上的性能。
        pre_best_model_dict：之前最佳模型的字典，包含了最佳模型的相关信息。
        logger：日志记录器，用于打印训练过程中的日志信息。
        log_writer：日志写入器，将日志信息写入文件。
        """
        self.init_distributed()
        config = self.config
        train_dataloader = self.train_dataloader
        if len(train_dataloader) == 0:
            logger.error(
                "No Images in train dataset, please ensure\n" +
                "\t1. The images num in the train label_file_list should be larger than or equal with batch size.\n"
                +
                "\t2. The annotation file and path in the configuration file are provided normally."
            )
            sys.exit(1)  # 终止进程
        valid_dataloader = self.valid_dataloader

        logger.info('train dataloader has {} iters'.format(len(train_dataloader)))
        if valid_dataloader is not None:
            logger.info('valid dataloader has {} iters'.format(len(valid_dataloader)))

        # device = self.device
        model = self.model
        loss_ = self.build_object('Loss')
        optimizer, lr_scheduler = self.build_optimizer(model)
        post_processor = self.build_object('PostProcessor')
        metric_ = self.build_object('Metric')

        try:
            pre_best_model_dict = load_model(config, model, logger, optimizer, config['Model']["model_type"])
        except AssertionError:
            pre_best_model_dict = dict()
        # 下面开始训练
        cal_metric_during_train = self.global_config.get('cal_metric_during_train', False)
        calc_epoch_interval = self.global_config.get('calc_epoch_interval', 1)
        log_smooth_window = self.global_config['log_smooth_window']
        epoch_num = self.global_config['epoch_num']
        print_batch_step = self.global_config['print_batch_step']
        # profiler_options = config['profiler_options']
        global_step = pre_best_model_dict.get('global_step', 0)

        eval_batch_step = self.global_config['eval_batch_step']
        start_eval_step = 0
        if isinstance(eval_batch_step, list) and len(eval_batch_step) == 2:
            start_eval_step = eval_batch_step[0]
            eval_batch_step = eval_batch_step[1]
            if len(valid_dataloader) == 0:
                logger.info(
                    'No Images in eval dataset, evaluation during training ' \
                    'will be disabled'
                )
                start_eval_step = 1e111
            logger.info(
                "During the training process, after the {}th iteration, " \
                "an evaluation is run every {} iterations".
                format(start_eval_step, eval_batch_step))

        save_epoch_step = self.global_config['save_epoch_step']
        save_model_dir = self.global_config['save_model_dir']

        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)

        main_indicator = metric_.main_indicator  # 主要评估指标
        best_model_dict = {main_indicator: 0}
        best_model_dict.update(pre_best_model_dict)  # 之前最好的指标

        train_stats = TrainingStats(log_smooth_window, ['lr'])

        model.train()

        extra_input_models = [
            "SRN", "NRTR", "SAR", "SEED", "SVTR", "SPIN", "VisionLAN",
            "RobustScanner", "RFL", 'DRRG'
        ]

        extra_input = False
        if self.config['Model']['algorithm'] == 'Distillation':  # 如果算法等于蒸馏
            for key in self.config['Model']["Models"]:
                extra_input = extra_input or self.config['Model']['Models'][key]['algorithm'] in extra_input_models
        else:
            extra_input = self.config['Model']['algorithm'] in extra_input_models

        model_type = self.config['Model'].get('model_type', None)
        algorithm = self.config['Model']['algorithm']

        # 开始批次
        start_epoch = best_model_dict['start_epoch'] if 'start_epoch' in best_model_dict else 1

        total_samples = 0
        train_reader_cost = 0.0
        train_batch_cost = 0.0
        reader_start = time.time()

        eta_meter = AverageMeter()

        # 在Windows操作系统中，存在一个多线程数据加载器的问题。为了解决这个问题，通常会将len(train_dataloader) - 1赋给max_iter。
        # 这是因为在Windows系统中，数据加载器中的最后一个批处理可能会引发一个错误，所以我们要减去1来避免这种情况。
        # 而在其他操作系统中，不需要减去1，直接将len(train_dataloader)赋给max_iter即可。
        # 这是因为其他操作系统上的数据加载器没有这个多线程问题，最后一个批处理不会引发错误。
        max_iter = len(train_dataloader) - 1 if platform.system() == "Windows" else len(train_dataloader)

        # 不用从头开始，从最好的batch开始,需要所有的随机种子要固定
        for epoch in range(start_epoch, epoch_num + 1):
            if hasattr(train_dataloader.dataset, 'need_reset'):  # 数据集需要重置
                train_dataloader = self.build_dataloader('Train', seed=epoch)  # 数据集都有随机种子
                max_iter = len(train_dataloader) - 1 if platform.system() == "Windows" else len(train_dataloader)

            for idx, batch in enumerate(train_dataloader):
                # profiler.add_profiler_step(profiler_options)  # 加性能
                # logger.info('batch',batch)
                train_reader_cost += time.time() - reader_start  # 训练时间
                if idx >= max_iter:
                    break
                lr = lr_scheduler.get_lr()  # 获取学习率

                images = batch[0]

                if model_type == 'table' or extra_input:
                    preds = model(images, data=batch[1:])
                elif model_type in ["kie", 'sr']:
                    preds = model(batch)
                elif algorithm in ['CAN']:
                    preds = model(batch[:3])
                else:
                    preds = model(images)
                loss = loss_(preds, batch)
                avg_loss = loss['loss']
                avg_loss.backward()
                optimizer.step()

                optimizer.zero_grad()

                # 每多少批次计算精度
                if cal_metric_during_train and epoch % calc_epoch_interval == 0:  # only rec and cls need
                    batch = [item.numpy() for item in batch]
                    if model_type in ['kie', 'sr']:
                        metric_(preds, batch)
                    elif model_type in ['table']:
                        post_result = post_processor(preds, batch)
                        metric_(post_result, batch)
                    elif algorithm in ['CAN']:
                        model_type = 'can'
                        metric_(preds[0], batch[2:], epoch_reset=(idx == 0))
                    else:
                        if self.config['Loss']['class'] in ['MultiLoss', 'MultiLoss_v2']:  # for multi head loss
                            post_result = post_processor(
                                preds['ctc'], batch[1])  # for CTC head out
                        elif self.config['Loss']['class'] in ['VLLoss']:
                            post_result = post_processor(preds, batch[1],
                                                         batch[-1])
                        else:
                            post_result = post_processor(preds, batch[1])
                        metric_(post_result, batch)
                    metric = metric_.get_metric()
                    train_stats.update(metric)

                train_batch_time = time.time() - reader_start
                train_batch_cost += train_batch_time
                eta_meter.update(train_batch_time)
                global_step += 1
                # 总样本数
                total_samples += len(images)
                # 不是浮点数而是类
                # if not isinstance(lr_scheduler, float):
                #     lr_scheduler.step()
                lr_scheduler.step()
                # logger and visualdl
                stats = {k: v.detach().numpy().mean() for k, v in loss.items()}  # 每一类的平均损失
                stats['lr'] = lr
                train_stats.update(stats)

                # dist.get_rank()是指在分布式训练中，用于获取当前进程的排名（rank）的函数。它可以在PyTorch的torch.distributed模块中找到。
                # 在分布式训练中，多个进程同时进行模型的训练，每个进程负责处理数据集的一个子集。
                # 每个进程都有一个唯一的排名，用于标识该进程在整个分布式环境中的位置。dist.get_rank()函数的作用就是获取当前进程的排名。
                # 通过使用排名，可以根据需要对分布式训练进行不同的配置和调整，例如在不同排名的进程之间进行通信、同步、梯度聚合等操作
                if log_writer is not None and self.is_rank0():
                    log_writer.log_metrics(metrics=train_stats.get(), prefix="TRAIN", step=global_step)

                if self.is_rank0() and (
                        (global_step > 0 and global_step % print_batch_step == 0) or (
                        idx >= len(train_dataloader) - 1)):
                    logs = train_stats.log()
                    # eta_sec表示预计剩余时间（以秒为单位）
                    eta_sec = ((epoch_num + 1 - epoch) * len(train_dataloader) - idx - 1) * eta_meter.avg

                    eta_sec_format = str(datetime.timedelta(seconds=int(eta_sec)))

                    strs = 'epoch: [{}/{}], global_step: {}, {}, avg_reader_cost: ' \
                           '{:.5f} s, avg_batch_cost: {:.5f} s, avg_samples: {}, ' \
                           'ips: {:.5f} samples/s, eta: {}'.format(
                        epoch, epoch_num, global_step, logs,
                        train_reader_cost / print_batch_step,
                        train_batch_cost / print_batch_step,
                        total_samples / print_batch_step,
                        total_samples / train_batch_cost, eta_sec_format)
                    logger.info(strs)

                    total_samples = 0
                    train_reader_cost = 0.0
                    train_batch_cost = 0.0
                # eval
                # 超过开始评估的步数且固定长度且是主进程
                if global_step > start_eval_step and (
                        global_step - start_eval_step) % eval_batch_step == 0 and self.is_rank0():
                    cur_metric = valid(
                        model,
                        valid_dataloader,
                        post_processor,
                        metric_,
                        model_type,
                        extra_input=extra_input)
                    cur_metric_str = 'cur metric, {}'.format(
                        ', '.join(['{}: {}'.format(k, v) for k, v in cur_metric.items()]))
                    logger.info(cur_metric_str)

                    # logger metric
                    if log_writer is not None:
                        log_writer.log_metrics(metrics=cur_metric, prefix="EVAL", step=global_step)

                    # 如果当期指标优于历史最好
                    if cur_metric[main_indicator] >= best_model_dict[main_indicator]:
                        best_model_dict.update(cur_metric)
                        best_model_dict['best_epoch'] = epoch  # 记下最好的批次并保存模型
                        save_model(
                            model,
                            optimizer,
                            save_model_dir,
                            logger,
                            self.config,
                            is_best=True,
                            prefix='best_accuracy',
                            best_model_dict=best_model_dict,
                            epoch=epoch,
                            global_step=global_step)
                    best_str = 'best metric, {}'.format(
                        ', '.join(['{}: {}'.format(k, v) for k, v in best_model_dict.items()]))
                    logger.info(best_str)
                    # logger best metric
                    if log_writer is not None:
                        log_writer.log_metrics(
                            metrics={"best_{}".format(main_indicator): best_model_dict[main_indicator]},
                            prefix="EVAL",
                            step=global_step)

                        log_writer.log_model(is_best=True,
                                             prefix="best_accuracy",
                                             metadata=best_model_dict)

                reader_start = time.time()  # 重新计时

            # 每个轮次都保存模型
            if self.is_rank0():
                save_model(
                    model,
                    optimizer,
                    save_model_dir,
                    logger,
                    self.config,
                    is_best=False,
                    prefix='latest',
                    best_model_dict=best_model_dict,
                    epoch=epoch,
                    global_step=global_step)

                if log_writer is not None:
                    log_writer.log_model(is_best=False, prefix="latest")
            # 固定间隔保存模型
            if self.is_rank0() and epoch > 0 and epoch % save_epoch_step == 0:
                save_model(
                    model,
                    optimizer,
                    save_model_dir,
                    logger,
                    self.config,
                    is_best=False,
                    prefix='iter_epoch_{}'.format(epoch),
                    best_model_dict=best_model_dict,
                    epoch=epoch,
                    global_step=global_step)
                if log_writer is not None:
                    log_writer.log_model(is_best=False, prefix='iter_epoch_{}'.format(epoch))

        best_str = 'best metric, {}'.format(', '.join(['{}: {}'.format(k, v) for k, v in best_model_dict.items()]))
        logger.info(best_str)
        if self.is_rank0() and log_writer is not None:
            log_writer.close()
        dist.destroy_process_group()
        return

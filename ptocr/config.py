import copy
import json
import math
import warnings

import cv2
import numpy as np

from ptocr.modules.architectures import BaseModel
from ptocr.transforms.rec_img_aug import resize_norm_img_chinese
from tools.infer.utility import get_minarea_rect_crop, get_rotate_crop_image

# 忽略所有警告
warnings.filterwarnings("ignore")

import datetime
import os
import pickle
import platform
import sys
import time
from functools import partial
from typing import Callable, Optional, Tuple, Type

import torch
import torch.distributed as dist
from loguru import logger
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose

from ptocr import hub
from ptocr.optim.lr_scheduler import warmup_scheduler
from ptocr.utils.save_load import _mkdir_if_not_exist, load_pretrained_params
from ptocr.utils.stats import TrainingStats
from ptocr.utils.utility import AverageMeter, get_image_file_list
from tools.train import valid

torch.autograd.set_detect_anomaly(True)

def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes

class _:
    """一个方便写配置文件的辅助类"""

    def __new__(cls, class_=None, /, **kwargs):
        if class_ is None:
            return kwargs
        if isinstance(class_, type):
            return partial(class_, **kwargs)
        # 预热装饰器用于装饰调度器
        if issubclass(class_, LRScheduler) and 'warmup_epoch' in kwargs:
            warmup_epochs = kwargs.pop('warmup_epoch')
            class_ = warmup_scheduler(class_, warmup_epochs)
            return partial(class_, **kwargs)
        # 如果是字符串，则从hub加载
        if isinstance(class_, str):
            class_ = hub(class_)
            return partial(class_, **kwargs)

    def __class_getitem__(cls, item):
        """方便实现Compose"""
        # if isinstance(item, tuple) and all(callable(it) for it in item):
        #     return Compose(item)

        out = [[],[],[]]
        for i in item:
            if isinstance(i,slice):
                ls = [i.start,i.stop,i.step]

                last = None
                for one in ls:
                    if one is not None:
                        last = one
                        break

                for i,one in enumerate(ls):
                    if one is ...:
                        out[i].append(last)
                    elif one:
                        out[i].append(one)
            else:
                for one in out:
                    one.append(i)
        return out

class ConfigModel:
    """基于配置的训练"""

    epoch_num: int  # = 3
    log_window_size: int  # = 20
    log_batch_step: int  # = 2

    save_epoch_step: int  # = 1
    eval_batch_step: Tuple[int, int]  # = (10, 5)

    use_gpu: bool = True
    save_model_dir: Optional[str] = None  # 不指定自动生成"./output/{algorithm}_{Backbone.name}/"
    metric_during_train: bool = False
    pretrained_model: Optional[str] = None  # "./pretrain_models/MobileNetV3_large_x0_5_pretrained"
    checkpoints: Optional[str] = None
    save_infer_dir: Optional[str] = None
    use_visualdl: bool = False
    infer_img: Optional[str] = None  # 训练过程中的测试图
    save_res_path: Optional[str] = None  # 推理结果文件

    model_type: str  # = "det"
    algorithm: str  # = "EAST"

    Backbone: Type[nn.Module]|partial  # = _(MobileNetV3, scale=0.5, model_name="large")
    Neck: Type[nn.Module]|partial  # = _(EASTFPN, model_name="small")
    Head: Type[nn.Module]|partial  # = _(EASTHead, model_name="small")
    loss: nn.Module  # = EASTLoss()
    Optimizer: Type[torch.optim.Optimizer]|partial  # = _(Adam, lr=0.01, betas=(0.9, 0.999))
    LRScheduler: Type[torch.optim.lr_scheduler.LRScheduler] | partial  # = _(ConstantLR, factor=1.0 / 3, total_iters=5, last_epoch=-1)
    postprocessor: Callable  # = EASTPostProcess(score_thresh=0.8, cover_thresh=0.1, nms_thresh=0.2)
    metric: Callable  # = DetMetric(main_indicator="hmean")

    class Train:
        Dataset: Type[VisionDataset]|partial  # = _(FolderDataset,root="E:/00IT/P/uniform/data/bank")
        transforms: Optional[
            Callable
        ] = None  # = _[EASTProcessTrain(image_shape=[512, 512], background_ratio=0, min_crop_side_ratio=0.0, min_text_size=10),KeepKeys(keep_keys=["image", "score_map", "geo_map", "training_mask"]),]
        DATALOADER: dict  # = _(shuffle=False, drop_last=False, batch_size=16, num_workers=4, pin_memory=False)

    class Eval:
        Dataset: Type[VisionDataset]|partial  # = _(FolderDataset,root="E:/00IT/P/uniform/data/banktest",)
        transforms: Optional[
            Callable
        ] = None  # = _[DetResizeForTest(limit_side_len=2400, limit_type=max),NormalizeImage(scale=1.0 / 255.0, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], order="hwc"),ToCHWImage(),KeepKeys(keep_keys=["image", "shape", "polys", "ignore_tags"])]
        DATALOADER: dict  # = _(shuffle=False, drop_last=False, batch_size=1, num_workers=4, pin_memory=False)
    class Infer:
        transforms: Optional[Callable] = None

    rec_image_shape: list = [3,32,320]
    rec_batch_num = 6

    def __init__(self,pretrained=None):
        self.det_box_type = "rect"
        self.use_gpu = torch.cuda.is_available() and self.use_gpu
        self.device = "cuda" if self.use_gpu else "cpu"
        self.use_dist = getattr(self, "distributed", False)
        self.init_distributed()
        self.model = self._build_model()

        self.pretrained = pretrained
        if self.pretrained:
            self.load_pretrained_model(pretrained)

    def _build_model(self):
        _model = BaseModel(in_channels=3, backbone=self.Backbone, neck=self.Neck, head=self.Head)
        use_sync_bn = getattr(self, "use_sync_bn", False)
        if use_sync_bn:
            _model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(_model)
            logger.info("convert_sync_batchnorm")
        if self.use_dist:
            _model = DistributedDataParallel(_model)
        _model.to(self.device)
        return _model

    # def _dataset(self, mode="Train"):
        # return getattr(self, mode).Dataset(transforms=getattr(self, mode).transforms)

    def _sampler(self, mode, dataset, seed, shuffle=None, drop_last=None):
        if mode == "train":
            if dist.is_initialized():
                sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last, seed=seed)
            else:
                sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        else:
            sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        return sampler

    def _dataset(self, mode="Train"):
        train_dict = {k:v for k,v in self.Data.__dict__.items() if not k.startswith('__')}
        if mode=='Train':
            cls = train_dict.pop('dataset')
            return cls(transforms=self._transform(mode),**train_dict)
        valid_dict = copy.deepcopy(train_dict)
        valid_dict.update(self.Data.__annotations__)
        cls = valid_dict.pop('dataset')
        return cls(transforms=self._transform(mode),**valid_dict)

    def _transform(self,mode='Train'):
        train,valid,infer = self.Transforms
        if mode=='Train':
            return Compose(train)
        elif mode == 'Eval':
            return Compose(valid)
        elif mode == 'Infer':
            return Compose(infer)

    def _build_dataloader(self, mode, seed=None):
        if mode == 'Train':
            DATALOADER = {k:v for k,v in self.Data.__dict__.items() if not k.startswith('__')}
        elif mode == 'Eval':
            DATALOADER = {k:v for k,v in self.Data.__dict__.items() if not k.startswith('__')}
            DATALOADER.update(self.Loader.__annotations__)

        shuffle = DATALOADER["shuffle"] if mode == "train" else False
        drop_last = DATALOADER["drop_last"]
        batch_size = DATALOADER["batch_size"]
        if not self.use_gpu:
            num_workers = 0
            pin_memory = False
        else:
            num_workers = DATALOADER["num_workers"]
            pin_memory = DATALOADER.get("pin_memory", True)
        collate_fn = DATALOADER.get("collate_fn", None)

        dataset = self._dataset(mode)
        sampler = self._sampler(mode, dataset, seed, shuffle, drop_last)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
        return data_loader

    def init_distributed(self):
        if self.use_dist:
            # dist.init_process_group(backend='gloo')
            dist.init_process_group(backend="gloo")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

            logger.info(self.rank)
            logger.info(self.world_size)

    def _build_scheduler(self, optimizer, max_epochs, step_each_epoch):
        if self.LRScheduler.func.__name__ == 'CosineAnnealingLR':
            kwargs = {'T_max': step_each_epoch * max_epochs}
        elif self.LRScheduler.func.__name__ == 'CosineAnnealingWarmRestarts':
            kwargs = {'T_0': step_each_epoch * max_epochs}
        elif self.LRScheduler.func.__name__ == 'TwoStepCosineLR':
            kwargs = {'T_max1': step_each_epoch * 200, 'T_max2': step_each_epoch * max_epochs}
        else:
            kwargs = {}
        lr_scheduler = self.LRScheduler(optimizer, **kwargs)
        return lr_scheduler

    def is_rank0(self):
        if self.use_dist:
            return dist.get_rank() == 0
        return True

    def train(self, log_writer=None):
        self.init_distributed()
        train_dataloader = self._build_dataloader("Train")
        if len(train_dataloader) == 0:
            logger.error(
                "No Images in train dataset, please ensure\n"
                + "\t1. The images num in the train label_file_list should be larger than or equal with batch size.\n"
                + "\t2. The annotation file and path in the configuration file are provided normally."
            )
            sys.exit(1)  # 终止进程
        valid_dataloader = self._build_dataloader("Eval")

        logger.info("train dataloader has {} iters".format(len(train_dataloader)))
        if valid_dataloader is not None:
            logger.info("valid dataloader has {} iters".format(len(valid_dataloader)))

        model = self.model
        criterion = self.loss
        optimizer = self.Optimizer(model.parameters())

        # step_each_epoch = len(train_dataloader)
        # max_epochs = self.epoch_num

        lr_scheduler = self._build_scheduler(optimizer, self.epoch_num, len(train_dataloader))
        post_processor = self.postprocessor
        metric_ = self.metric

        try:
            pre_best_model_dict = load_model(model, logger, optimizer, self.checkpoints, self.pretrained_model)
        except AssertionError:
            pre_best_model_dict = {}
        # 下面开始训练
        metric_during_train = getattr(self, "metric_during_train", False)
        calc_epoch_interval = getattr(self, "calc_epoch_interval", 1)
        log_window_size = getattr(self, "log_window_size")
        epoch_num = getattr(self, "epoch_num")
        log_batch_step = getattr(self, "log_batch_step")
        global_step = pre_best_model_dict.get("global_step", 0)

        eval_batch_step = getattr(self, "eval_batch_step")
        start_eval_step = 0
        if isinstance(eval_batch_step, list | tuple) and len(eval_batch_step) == 2:
            start_eval_step, eval_batch_step = eval_batch_step
            if len(valid_dataloader) == 0:
                logger.info("No Images in eval dataset, evaluation during training " "will be disabled")
                start_eval_step = 1e111
            logger.info(
                "During the training process, after the {}th iteration, "
                "an evaluation is run every {} iterations".format(start_eval_step, eval_batch_step)
            )

        save_epoch_step = getattr(self, "save_epoch_step")
        save_model_dir = getattr(self, "save_model_dir")
        os.makedirs(save_model_dir, exist_ok=True)

        main_indicator = metric_.main_indicator  # 主要评估指标
        best_model_dict = {main_indicator: 0}
        best_model_dict.update(pre_best_model_dict)  # 之前最好的指标

        train_stats = TrainingStats(log_window_size, ["lr"])

        model.train()

        extra_input_models = ["SRN", "NRTR", "SAR", "SEED", "SVTR", "SPIN", "VisionLAN", "RobustScanner", "RFL", "DRRG"]

        extra_input = self.algorithm in extra_input_models

        model_type = self.model_type
        algorithm = self.algorithm

        # 开始批次
        start_epoch = best_model_dict.get("start_epoch", 1)

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
            if hasattr(train_dataloader.dataset, "need_reset"):  # 数据集需要重置
                train_dataloader = self._build_dataloader("Train", seed=epoch)  # 数据集都有随机种子
                max_iter = len(train_dataloader) - 1 if platform.system() == "Windows" else len(train_dataloader)

            for idx, batch in enumerate(train_dataloader):
                train_reader_cost += time.time() - reader_start  # 训练数据读取时间
                if idx >= max_iter:
                    break
                lr = lr_scheduler.get_lr()  # 获取学习率

                images = batch[0]

                if model_type == "table" or extra_input:
                    predict = model(images, data=batch[1:])
                elif model_type in ["kie", "sr"]:
                    predict = model(batch)
                elif algorithm in ["CAN"]:
                    predict = model(batch[:3])
                else:
                    predict = model(images)
                # with torch.autograd.set_detect_anomaly(True):
                loss = criterion(predict, batch)  # {'loss':...,'other':...}



                with torch.autograd.detect_anomaly():
                    loss["loss"].backward()

                optimizer.step()
                optimizer.zero_grad()

                # 每多少批次计算精度
                if metric_during_train and epoch % calc_epoch_interval == 0:  # only rec and cls need
                    batch = [item.numpy() for item in batch]
                    if model_type in ["kie", "sr"]:
                        metric_(predict, batch)
                    elif model_type in ["table"]:
                        post_result = post_processor(predict, batch)
                        metric_(post_result, batch)
                    elif algorithm in ["CAN"]:
                        model_type = "can"
                        metric_(predict[0], batch[2:], epoch_reset=(idx == 0))
                    else:
                        if self.loss.__class__.__name__ in ["MultiLoss"]:  # for multi head loss
                            post_result = post_processor(predict["ctc"], batch[1])  # for CTC head out
                        elif self.loss.__class__.__name__ in ["VLLoss"]:
                            post_result = post_processor(predict, batch[1], batch[-1])
                        else:
                            post_result = post_processor(predict, batch[1])  ## todo 后处理是Metric之前的必要步骤，应该合入metric中
                        metric_(post_result, batch)

                    metric = metric_.get_metric()
                    train_stats.update(metric)

                train_batch_time = time.time() - reader_start
                train_batch_cost += train_batch_time
                eta_meter.update(train_batch_time)
                global_step += 1
                # 总样本数
                total_samples += len(images)

                lr_scheduler.step()
                stats = {k: v.detach().numpy().mean() for k, v in loss.items()}  # 每一类的平均损失
                stats["lr"] = lr
                train_stats.update(stats)

                # dist.get_rank()是指在分布式训练中，用于获取当前进程的排名（rank）的函数。它可以在PyTorch的torch.distributed模块中找到。
                # 在分布式训练中，多个进程同时进行模型的训练，每个进程负责处理数据集的一个子集。
                # 每个进程都有一个唯一的排名，用于标识该进程在整个分布式环境中的位置。dist.get_rank()函数的作用就是获取当前进程的排名。
                # 通过使用排名，可以根据需要对分布式训练进行不同的配置和调整，例如在不同排名的进程之间进行通信、同步、梯度聚合等操作
                if log_writer and self.is_rank0():
                    log_writer.log_metrics(metrics=train_stats.get(), prefix="TRAIN", step=global_step)

                if self.is_rank0() and (
                        (global_step > 0 and global_step % log_batch_step == 0) or (idx >= len(train_dataloader) - 1)
                ):
                    logs = train_stats.log()
                    # eta_sec表示预计剩余时间（以秒为单位）
                    eta_sec = ((epoch_num + 1 - epoch) * len(train_dataloader) - idx - 1) * eta_meter.avg

                    eta_sec_format = str(datetime.timedelta(seconds=int(eta_sec)))

                    strs = (
                        "epoch: [{}/{}], global_step: {}, {}, avg_reader_cost: "
                        "{:.5f} s, avg_batch_cost: {:.5f} s, avg_samples: {}, "
                        "ips: {:.5f} samples/s, eta: {}".format(
                            epoch,
                            epoch_num,
                            global_step,
                            logs,
                            train_reader_cost / log_batch_step,
                            train_batch_cost / log_batch_step,
                            total_samples / log_batch_step,
                            total_samples / train_batch_cost,
                            eta_sec_format,
                        )
                    )
                    logger.info(strs)

                    total_samples = 0
                    train_reader_cost = 0.0
                    train_batch_cost = 0.0
                # eval
                # 超过开始评估的步数且固定长度且是主进程
                if (
                        global_step > start_eval_step
                        and (global_step - start_eval_step) % eval_batch_step == 0
                        and self.is_rank0()
                ):
                    cur_metric = valid(
                        model, valid_dataloader, post_processor, metric_, model_type, extra_input=extra_input
                    )
                    cur_metric_str = "cur metric, {}".format(
                        ", ".join(["{}: {}".format(k, v) for k, v in cur_metric.items()])
                    )
                    logger.info(cur_metric_str)

                    # logger metric
                    if log_writer is not None:
                        log_writer.log_metrics(metrics=cur_metric, prefix="EVAL", step=global_step)

                    # 如果当期指标优于历史最好
                    if cur_metric[main_indicator] >= best_model_dict[main_indicator]:
                        best_model_dict.update(cur_metric)
                        best_model_dict["best_epoch"] = epoch  # 记下最好的批次并保存模型
                        save_model(
                            model,
                            optimizer,
                            save_model_dir,
                            logger,
                            is_best=True,
                            prefix="best_accuracy",
                            best_model_dict=best_model_dict,
                            epoch=epoch,
                            global_step=global_step,
                        )
                    best_str = "best metric, {}".format(
                        ", ".join(["{}: {}".format(k, v) for k, v in best_model_dict.items()])
                    )
                    logger.info(best_str)
                    # logger best metric
                    if log_writer is not None:
                        log_writer.log_metrics(
                            metrics={"best_{}".format(main_indicator): best_model_dict[main_indicator]},
                            prefix="EVAL",
                            step=global_step,
                        )

                        log_writer.log_model(is_best=True, prefix="best_accuracy", metadata=best_model_dict)

                reader_start = time.time()  # 重新计时

            # 每个轮次都保存模型
            if self.is_rank0():
                logger.info("Save model checkpoint to {}".format(save_model_dir))
                save_model(
                    model,
                    optimizer,
                    save_model_dir,
                    logger,
                    is_best=False,
                    prefix="latest",
                    best_model_dict=best_model_dict,
                    epoch=epoch,
                    global_step=global_step,
                )

                if log_writer is not None:
                    log_writer.log_model(is_best=False, prefix="latest")
                # 固定间隔保存模型
                if epoch > 0 and epoch % save_epoch_step == 0:
                    save_model(
                        model,
                        optimizer,
                        save_model_dir,
                        logger,
                        is_best=False,
                        prefix="iter_epoch_{}".format(epoch),
                        best_model_dict=best_model_dict,
                        epoch=epoch,
                        global_step=global_step,
                    )
                    if log_writer is not None:
                        log_writer.log_model(is_best=False, prefix="iter_epoch_{}".format(epoch))

        best_str = f"best metric, {', '.join(f'{k}: {v}' for k, v in best_model_dict.items())}"
        logger.info(best_str)
        if log_writer and self.is_rank0():
            log_writer.close()
        if self.use_dist:
            dist.destroy_process_group()
        return

    @torch.no_grad()
    def det_one_image(self,img_or_path):
        self.model.eval()
        if isinstance(img_or_path, str):
            img = cv2.imread(img_or_path)
        else:
            img = img_or_path
        data = {"image": img}
        batch = self._transform('Infer')(data)

        images = np.expand_dims(batch[0], axis=0)
        shape_list = np.expand_dims(batch[1], axis=0)
        images = torch.Tensor(images)
        preds = self.model(images)
        post_result = self.postprocessor(preds, shape_list)
        # parser boxes if post_result is dict
        logger.info("det_result:{}".format(post_result))
        dt_boxes = post_result[0]["points"]
        if self.det_box_type == "poly":
            dt_boxes = self.filter_tag_det_res_only_clip(dt_boxes, img.shape)
        else:
            dt_boxes = self.filter_tag_det_res(dt_boxes, img.shape)
        return dt_boxes,img

    @torch.no_grad()
    def rec_one_image(self,img_or_path):
        self.model.eval()
        if isinstance(img_or_path, str):
            img = cv2.imread(img_or_path)
        else:
            img = img_or_path
        data = {"image": img}
        batch = self._transform('Infer')(data)

        images = np.expand_dims(batch[0], axis=0)
        images = torch.Tensor(images)
        preds = self.model(images)
        post_result = self.postprocessor(preds)
        return post_result

    def load_pretrained_model(self,path):
        state_dict = torch.load(path)  # 参数
        self.model.load_state_dict(state_dict)
        return self.model

    def __call__(self, infer_img):
        return getattr(self, self.model_type)(infer_img)

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))

        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    @torch.no_grad()
    def rec(self, img_list):
        """针对图像列表进行识别"""
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        rec_res = [["", 0.0]] * img_num
        batch_num = self.rec_batch_num # 一批数量

        self.model.eval()

        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []

            imgC, imgH, imgW = self.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH
            # max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio) # 计算最大宽高比
                # logger.info(f"计算最大宽高比{max_wh_ratio}")
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]], max_wh_ratio)# 统一缩放
                # norm_img,_ = resize_norm_img_chinese(norm_img,[3, 32, 320])
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            input_tensor = torch.from_numpy(norm_img_batch)
            # logger.info(input_tensor.shape)
            # self.predictor.run() # how
            # input_tensor = self.Infer.transforms(input_tensor)
            output_tensors = self.model(input_tensor)
            # logger.info(output_tensors)
            # outputs = []
            # for output_tensor in output_tensors:
            #     output = output_tensor.numpy()
            #     outputs.append(output)
            #
            # if len(outputs) != 1:
            #     preds = outputs
            # else:
            #     preds = outputs[0]
            rec_result = self.postprocessor(output_tensors)
            logger.info(rec_result)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]

        return rec_res

    @torch.no_grad()
    def det(self,img,cls=None,rec=None):
        dt_boxes,img = self.det_one_image(img)
        ori_im = img.copy()
        img_crop_list = []
        dt_boxes = sorted_boxes(dt_boxes)
        logger.info(f'有效框:{len(dt_boxes)}')

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.det_box_type == "poly":
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            else:
                img_crop = get_minarea_rect_crop(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if cls:
            img_crop_list, _ = cls(img_crop_list)
        if rec:
            img_crop_list = rec.rec(img_crop_list)
        return img_crop_list

def save_model(model, optimizer, model_path, logger, is_best=False, prefix="ppocr", **kwargs):
    """
    save model to the target path
    """
    _mkdir_if_not_exist(model_path, logger)
    model_prefix = os.path.join(model_path, prefix)
    torch.save(optimizer.state_dict(), model_prefix + ".pdopt")
    torch.save(model.state_dict(), model_prefix + ".pdparams")
    metric_prefix = model_prefix
    # save metric and config
    with open(metric_prefix + ".states", "wb") as f:
        pickle.dump(kwargs, f, protocol=2)
    if is_best:
        logger.info("save best model is to {}".format(model_prefix))
    else:
        logger.info("save model in {}".format(model_prefix))


def load_model(model: nn.Module, logger, optimizer=None, checkpoints=None, pretrained_model=None):
    """
    load model from checkpoint or pretrained_model
    """

    best_model_dict = {}
    is_float16 = False

    if checkpoints:
        if checkpoints.endswith(".pdparams"):
            checkpoints = checkpoints.replace(".pdparams", "")

        assert os.path.exists(checkpoints + ".pdparams"), "The {}.pdparams does not exists!".format(checkpoints)

        # load params from trained model
        params = torch.load(checkpoints + ".pdparams")  # 参数
        state_dict = model.state_dict()  # 状态字典
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
                    "The shape of model params {} {} not matched with loaded params shape {} !".format(
                        key, value.shape, pre_value.shape
                    )
                )
        model.load_state_dict(new_state_dict)

        if is_float16:
            logger.info("The parameter type is float16, which is converted to float32 when loading")
        if optimizer is not None:
            if os.path.exists(checkpoints + ".pdopt"):
                optim_dict = torch.load(checkpoints + ".pdopt")
                optimizer.load_state_dict(optim_dict)
            else:
                logger.warning("{}.pdopt is not exists, params of optimizer is not loaded".format(checkpoints))

        if os.path.exists(checkpoints + ".states"):
            with open(checkpoints + ".states", "rb") as f:
                states_dict = pickle.load(f, encoding="latin1")
            best_model_dict = states_dict.get("best_model_dict", {})
            if "epoch" in states_dict:
                best_model_dict["start_epoch"] = states_dict["epoch"] + 1
        logger.info("resume from {}".format(checkpoints))
    elif pretrained_model:
        is_float16 = load_pretrained_params(model, pretrained_model, logger)
    else:
        logger.info("train from scratch")
    best_model_dict["is_float16"] = is_float16
    return best_model_dict




def draw_det_res(dt_boxes, img, img_name, save_path):
    if len(dt_boxes) > 0:
        import cv2

        src_im = img
        for box in dt_boxes:
            box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, os.path.basename(img_name))
        cv2.imwrite(save_path, src_im)
        logger.info("The detected Image saved in {}".format(save_path))

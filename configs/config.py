import datetime
import os
import pickle
import platform
import sys
import time
from functools import partial
from typing import Callable, List, Optional, Tuple, Type

import torch
import torch.distributed as dist
from loguru import logger
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import ConstantLR
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose

from ocr.datasets.ocr_dataset import FolderDataset
from ppocr import hub
from ppocr.data import SimpleDataSet
from ppocr.data.imaug import (
    DecodeImage,
    DetLabelEncode,
    DetResizeForTest,
    EASTProcessTrain,
    IaaAugment,
    KeepKeys,
    NormalizeImage,
    ToCHWImage,
)
from ppocr.losses import DBLoss, EASTLoss
from ppocr.metrics import DetMetric
from ppocr.modeling.backbones.det_mobilenet_v3 import MobileNetV3
from ppocr.modeling.heads.det_db_head import DBHead
from ppocr.modeling.heads.det_east_head import EASTHead
from ppocr.modeling.necks.db_fpn import DBFPN
from ppocr.modeling.necks.east_fpn import EASTFPN
from ppocr.postprocess import DBPostProcess, EASTPostProcess
from ppocr.utils.save_load import _mkdir_if_not_exist, load_pretrained_params
from ppocr.utils.stats import TrainingStats
from ppocr.utils.utility import AverageMeter
from tools.train import valid


class _:
    """一个方便写配置文件的辅助类"""

    def __new__(cls, class_=None, /, **kwargs):
        if class_ is None:
            return kwargs
        if isinstance(class_, type):
            return partial(class_, **kwargs)
        if isinstance(class_, str):
            return partial(hub(class_), **kwargs)
        else:
            raise NotImplementedError

    def __class_getitem__(cls, item):
        """方便实现Compose"""
        if isinstance(item, tuple) and all(callable(it) for it in item):
            return Compose(item)


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

    Backbone: Type[nn.Module]  # = _(MobileNetV3, scale=0.5, model_name="large")
    Neck: Type[nn.Module]  # = _(EASTFPN, model_name="small")
    Head: Type[nn.Module]  # = _(EASTHead, model_name="small")
    loss: nn.Module  # = EASTLoss()
    Optimizer: Type[torch.optim.Optimizer]  # = _(Adam, lr=0.01, betas=(0.9, 0.999))
    Scheduler: Type[
        torch.optim.lr_scheduler.LRScheduler]  # = _(ConstantLR, factor=1.0 / 3, total_iters=5, last_epoch=-1)
    postprocessor: Callable  # = EASTPostProcess(score_thresh=0.8, cover_thresh=0.1, nms_thresh=0.2)
    metric: Callable  # = DetMetric(main_indicator="hmean")

    class Train:
        Dataset: Type[VisionDataset]  # = _(FolderDataset,root="E:/00IT/P/uniform/data/bank")
        transforms: Optional[
            Callable] = None  # = _[EASTProcessTrain(image_shape=[512, 512], background_ratio=0, min_crop_side_ratio=0.0, min_text_size=10),KeepKeys(keep_keys=["image", "score_map", "geo_map", "training_mask"]),]
        DATALOADER: dict  # = _(shuffle=False, drop_last=False, batch_size=16, num_workers=4, pin_memory=False)

    class Eval:
        Dataset: Type[VisionDataset]  # = _(FolderDataset,root="E:/00IT/P/uniform/data/banktest",)
        transforms: Optional[
            Callable] = None  # = _[DetResizeForTest(limit_side_len=2400, limit_type=max),NormalizeImage(scale=1.0 / 255.0, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], order="hwc"),ToCHWImage(),KeepKeys(keep_keys=["image", "shape", "polys", "ignore_tags"])]
        DATALOADER: dict  # = _(shuffle=False, drop_last=False, batch_size=1, num_workers=4, pin_memory=False)

    def __init__(self):
        self.device = "cuda" if self.use_gpu else "cpu"
        self.use_dist = getattr(self, "distributed", False)
        self.model = self._build_model()

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

    # def _transforms(self, mode="Train"):
    #     transforms = getattr(self, mode).transforms
    #     if transforms:
    #         return Compose(self.Train.transforms)
    #     return None

    def _dataset(self, mode="Train"):
        return getattr(self, mode).Dataset(transforms=getattr(self, mode).transforms)

    def _sampler(self, mode, dataset, seed, shuffle=None, drop_last=None):
        if mode == "train":
            if dist.is_initialized():
                sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last, seed=seed)
            else:
                sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        else:
            sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        return sampler

    def _build_dataloader(self, mode, seed=None):
        DATALOADER = getattr(self, mode).DATALOADER
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

    def train(self, log_writer=None):
        self.init_distributed()
        # config = self.config
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
        loss_ = self.loss
        optimizer = self.Optimizer(model.parameters())
        lr_scheduler = self.Scheduler(optimizer)
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

                loss = loss_(predict, batch)  # {'loss':...,'other':...}
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

    def is_rank0(self):
        if self.use_dist:
            return dist.get_rank() == 0
        return True


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

    # global_config = config["Global"]
    # checkpoints = global_config.get('checkpoints')
    # pretrained_model = global_config.get('pretrained_model')
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


class BaseModel(nn.Module):
    def __init__(self, in_channels, backbone, neck, head, return_all_feats=False):
        super().__init__()
        if backbone:
            self.backbone = backbone(in_channels=in_channels)
            in_channels = self.backbone.out_channels
        if neck:
            self.neck = neck(in_channels=in_channels)
            in_channels = self.neck.out_channels
        if head:
            self.head = head(in_channels=in_channels)
        self.return_all_feats = return_all_feats

    def forward(self, x):
        out_dict = {}
        for module_name, module in self.named_children():
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

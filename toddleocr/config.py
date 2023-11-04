import copy
import warnings

from .postprocess.matcher import TableMatch
from .postprocess.table_master_match import TableMasterMatcher
from .utils.visual import expand, table_view

warnings.filterwarnings("ignore")

import datetime
import os
import pickle
import platform
import time
import types
from functools import partial
from typing import Callable, List, Optional, Tuple, Type

import cv2
import numpy as np
import torch
import torch.distributed as dist
from loguru import logger

from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
)
from torchvision.transforms import Compose

from .modules.architectures import BaseModel
from .optim.lr_scheduler import warmup_scheduler
from .utils.init_args import get_minarea_rect_crop, get_rotate_crop_image
from .utils.stats import TrainingStats
from .utils.utility import (
    AverageMeter,
    filter_tag_det_res,
    filter_tag_det_res_only_clip,
    resize_norm_img,
    sorted_boxes,
)
from .utils.valid import valid

torch.autograd.set_detect_anomaly(True)

PROJECT_DIR = os.path.dirname(__file__)


class _:
    """
    这是一个多功能的辅助类，有以下几个语法功能：
    1. 类似偏函数的偏类：_(DBHead,arg1=0,arg2=1) ==> partial(DBHead, **kwargs)
    2. 字符串动态导入类：_("DBHead",arg1=0,arg2=1) ==> partial(DBHead, **kwargs)
    3. 没有位置参数等效于字典：_(arg1=0,arg2=1) ==> dict(arg1=0,arg2=1)
    4. 预热学习率规划器：
    5. 等效并列列表,用于 Transformers： _[train:eval:infer,train:eval:...],
        三个位置分别表示训练，测试，推理模式下的预处理器，
        特别的：...省略号表示同前一个，
            空的或None表示该位置不需要这个预处理器，例如[DecodeLabel:...:]表示训练和测试需要DecodeLabel，推理不需要
        这种表示方法是为了简化配置，共享处理器减少实例的创建
    """

    def __new__(cls, class_=None, /, **kwargs):
        if class_ is None:
            return kwargs

        if issubclass(class_, LRScheduler) and "warmup_epoch" in kwargs:
            warmup_epochs = kwargs.pop("warmup_epoch")
            class_ = warmup_scheduler(class_, warmup_epochs)
            return partial(class_, **kwargs)

        if isinstance(class_, type | types.FunctionType):
            return partial(class_, **kwargs)

        if isinstance(class_, str):
            from utils.modelhub import Hub

            hub = Hub(os.path.dirname(__file__))  # 这个操作很耗时，尽量不使用字符串形式的导入

            class_ = hub(class_)
            return partial(class_, **kwargs)

    def __class_getitem__(cls, item):
        out = [[], [], []]
        for i in item:
            if isinstance(i, slice):
                ls = [i.start, i.stop, i.step]
                last = None
                for one in ls:
                    if one is not None:
                        last = one
                        break
                for i, one in enumerate(ls):
                    if one is ...:
                        out[i].append(last)
                    elif one:
                        out[i].append(one)
            else:
                for one in out:
                    one.append(i)
        return out


# todo 精简参数
class ConfigModel:
    """
    配置类，摒弃yaml的配置方法，采用纯python语言，类yaml的配置方法，但更灵活
    而且可以使用复杂的引用计算，所配即所得

    Data 和 Loader 子类采用 注解语法区分训练时配置和测试时配置，例如：
        class Loader:
            shuffle:False = True
            drop_last:True = False
            batch_size:1 = 8
            num_workers: 0 = 4
    等号后面表示训练时参数，冒号后面表示测试时参数，没有冒号则相同
    这种表示方法同样是为了简化配置
    """

    model_type: str
    algorithm: str

    epoch_num: int
    meter_epoch_step: int = 0  # 0表示每一代都不评估，正数表示评估间隔
    save_epoch_step: int = 1

    log_window_size: int = 20
    log_batch_step: int = 10
    eval_batch_step: Tuple[int, int] = [0, 1000]

    use_gpu: bool = True
    distributed: bool = False

    model_dir: Optional[str] = None  # 模型存放目录
    pretrained_model: Optional[str] = None  # 预训练模型
    checkpoints: Optional[str] = None  # 训练检查点

    class Data:
        dataset: None
        root: None

    class Loader:
        shuffle = False
        drop_last = False
        batch_size = 1
        num_workers: 0 = 4
        pin_memory = False

    Transforms: List[List[Callable]]
    Backbone: Type[nn.Module] | partial
    Neck: Type[nn.Module] | partial
    Head: Type[nn.Module] | partial
    loss: nn.Module
    Optimizer: Type[Optimizer] | partial
    LRScheduler: Type[LRScheduler] | partial
    postprocessor: Callable
    metric: Callable

    # todo 下面的参数是模型独有的
    rec_image_shape: list = [3, 32, 320]
    cls_image_shape: list = [3, 48, 192]
    rec_batch_num = 6
    cls_batch_num = 6
    det_box_type = "rect"

    def __init__(self, pretrained=None):
        self.use_gpu = self.use_gpu and torch.cuda.is_available()
        self._init_distributed()
        self.model = self._build_model()
        self.pretrained = pretrained or self.pretrained_model
        if self.pretrained:
            self.load_pretrained_model(self.pretrained)
        self.rank = dist.get_rank() if self.distributed else 0
        self.is_rank0 = self.rank == 0

    def _init_distributed(self):
        if self.distributed:
            dist.init_process_group(backend="gloo")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            logger.info(self.rank)
            logger.info(self.world_size)

    def _build_model(self):
        _model = BaseModel(
            in_channels=3,
            backbone=self.Backbone,
            neck=self.Neck,
            head=self.Head,
            # transform=self.Transform
        )
        use_sync_bn = getattr(self, "use_sync_bn", False)
        if use_sync_bn:
            _model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(_model)
            # logger.info("convert_sync_batchnorm")
        if self.distributed:
            _model = DistributedDataParallel(_model)
        _model.to("cuda" if self.use_gpu else "cpu")
        return _model

    def dataset(self, mode="train"):
        train_dict = {
            k: v for (k, v) in self.Data.__dict__.items() if not k.startswith("__")
        }
        if mode == "train":
            cls = train_dict.pop("dataset")
            return cls(transforms=self.transforms(mode), **train_dict)
        valid_dict = copy.deepcopy(train_dict)
        valid_dict.update(self.Data.__annotations__)
        cls = valid_dict.pop("dataset")
        return cls(transforms=self.transforms(mode), **valid_dict)

    def transforms(self, mode="train"):
        if hasattr(self, "Transforms"):
            (train, valid, infer) = self.Transforms
            if mode == "train":
                return Compose(train)
            if mode == "eval":
                return Compose(valid)
            if mode == "infer":
                return Compose(infer)
        return None

    def _build_dataloader(self, mode="train", seed=None):
        if mode == "train":
            dataloader = {
                k: v
                for (k, v) in self.Loader.__dict__.items()
                if not k.startswith("__")
            }
        elif mode == "eval":
            dataloader = {
                k: v
                for (k, v) in self.Loader.__dict__.items()
                if not k.startswith("__")
            }
            dataloader.update(self.Loader.__annotations__)
        shuffle = dataloader["shuffle"] if mode == "train" else False
        drop_last = dataloader["drop_last"]
        batch_size = dataloader["batch_size"]
        if not self.use_gpu:
            num_workers = 1
            pin_memory = False
        else:
            num_workers = dataloader["num_workers"]
            pin_memory = dataloader.get("pin_memory", True)
        collate_fn = dataloader.get("collate_fn", None)
        dataset = self.dataset(mode)
        if mode == "train":
            if dist.is_initialized():
                sampler = DistributedSampler(
                    dataset, shuffle=shuffle, drop_last=drop_last, seed=seed
                )
            else:
                sampler = (
                    RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
                )
        else:
            sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
        return data_loader

    def _build_scheduler(self, optimizer, max_epochs, step_each_epoch):
        if self.LRScheduler.func.__name__ == "CosineAnnealingLR":
            kwargs = {"T_max": step_each_epoch * max_epochs}
        elif self.LRScheduler.func.__name__ == "CosineAnnealingWarmRestarts":
            kwargs = {"T_0": step_each_epoch * max_epochs}
        elif self.LRScheduler.func.__name__ == "TwoStepCosineLR":
            kwargs = {
                "T_max1": step_each_epoch * 200,
                "T_max2": step_each_epoch * max_epochs,
            }
        else:
            kwargs = {}
        return self.LRScheduler(optimizer, **kwargs)

    @torch.no_grad()
    def _det_one_image(self, img_or_path):
        self.model.eval()
        if isinstance(img_or_path, str):
            img = cv2.imread(img_or_path)
        else:
            img = img_or_path
        data = {"image": img}
        batch = self.transforms("infer")(data)
        images = np.expand_dims(batch[0], axis=0)
        shape_list = np.expand_dims(batch[1], axis=0)
        images = torch.Tensor(images)
        preds = self.model(images)
        post_result = self.postprocessor(preds, shape_list)
        # logger.info("det_result:{}".format(post_result))
        dt_boxes = post_result[0]["points"]
        if self.det_box_type == "poly":
            dt_boxes = filter_tag_det_res_only_clip(dt_boxes, img.shape)
        else:
            dt_boxes = filter_tag_det_res(dt_boxes, img.shape)
        return (dt_boxes, img)

    @torch.no_grad()
    def _rec_one_image(self, img_or_path):
        self.model.eval()
        if isinstance(img_or_path, str):
            img = cv2.imread(img_or_path)
        else:
            img = img_or_path
        data = {"image": img}
        batch = self.transforms("infer")(data)
        images = np.expand_dims(batch[0], axis=0)
        images = torch.Tensor(images)
        preds = self.model(images)
        post_result = self.postprocessor(preds)
        return post_result

    @torch.no_grad()
    def cls(self, img_list):
        if not isinstance(img_list, list):
            if isinstance(img_list, str):
                img = cv2.imread(img_list)
            else:
                img = img_list
            img_list = [img]

        img_list = copy.deepcopy(img_list)
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the cls process
        indices = np.argsort(np.array(width_list))
        self.model.eval()
        cls_res = [["", 0.0]] * img_num
        batch_num = self.cls_batch_num
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            starttime = time.time()
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = resize_norm_img(
                    img_list[indices[ino]], max_wh_ratio, self.cls_image_shape
                )
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            input_tensor = torch.from_numpy(norm_img_batch)

            prob_out = self.model(input_tensor)
            cls_result = self.postprocessor(prob_out)
            for rno in range(len(cls_result)):
                label, score = cls_result[rno]
                cls_res[indices[beg_img_no + rno]] = [label, score]
                if "180" in label and score > 0.5:
                    img_list[indices[beg_img_no + rno]] = cv2.rotate(
                        img_list[indices[beg_img_no + rno]], 1
                    )
        return img_list, cls_res

    @torch.no_grad()
    def rec(self, img_list):
        if not isinstance(img_list, list):
            return self._rec_one_image(img_list)

        img_num = len(img_list)
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        indices = np.argsort(np.array(width_list))
        rec_res = [["", 0.0]] * img_num
        batch_num = self.rec_batch_num
        self.model.eval()
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            (imgC, imgH, imgW) = self.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH
            for ino in range(beg_img_no, end_img_no):
                (h, w) = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = resize_norm_img(
                    img_list[indices[ino]], max_wh_ratio, self.rec_image_shape
                )
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            input_tensor = torch.from_numpy(norm_img_batch)
            output_tensors = self.model(input_tensor)
            rec_result = self.postprocessor(output_tensors)
            # logger.info(rec_result)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
        return rec_res

    @torch.no_grad()
    def det(self, img, cls=None, rec=None):
        (dt_boxes, img) = self._det_one_image(img)
        if not rec:
            return {"boxes": dt_boxes}
        ori_im = img.copy()
        img_crop_list = []
        dt_boxes = sorted_boxes(dt_boxes)
        logger.info(f"有效框:{len(dt_boxes)}")
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.det_box_type == "poly":
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            else:
                img_crop = get_minarea_rect_crop(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if cls:
            (img_crop_list, _) = cls(img_crop_list)
        if rec:
            img_crop_list = rec(img_crop_list)
        return {"boxes": dt_boxes, "rec_res": img_crop_list}

    def save(self, model_path, is_best=False, prefix="toddleocr", **kwargs):
        os.makedirs(model_path, exist_ok=True)
        model_prefix = os.path.join(model_path, prefix)
        # torch.save(self.optimizer.state_dict(), model_prefix + ".pto")
        torch.save(self.model.state_dict(), model_prefix + ".pth")
        with open(model_prefix + ".states", "wb") as f:
            pickle.dump(kwargs, f, protocol=2)
        if is_best:
            logger.info("save best model is to {}".format(model_prefix))
        else:
            logger.info("save model in {}".format(model_prefix))

    def load(self):
        best_model_dict = {}
        checkpoints = self.checkpoints
        if self.checkpoints:
            if checkpoints.endswith(".pth"):
                checkpoints = checkpoints.replace(".pth", "")
            params = torch.load(checkpoints + ".pth")
            self.model.load_state_dict(params)
            if os.path.exists(checkpoints + ".pto"):
                optim_dict = torch.load(checkpoints + ".pto")
                self.optimizer.load_state_dict(optim_dict)
            else:
                logger.warning(
                    f"{checkpoints}.pto is not exists, params of optimizer is not loaded"
                )
            if os.path.exists(checkpoints + ".states"):
                with open(checkpoints + ".states", "rb") as f:
                    states_dict = pickle.load(f, encoding="latin1")
                best_model_dict = states_dict.get("best_model_dict", {})
                if "epoch" in states_dict:
                    best_model_dict["start_epoch"] = states_dict["epoch"] + 1
            logger.info("resume from {}".format(checkpoints))
        elif self.pretrained_model:
            is_float16 = self.load_pretrained_model(self.pretrained_model)
        else:
            logger.info("train from scratch")
        return best_model_dict

    def load_pretrained_model(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        return self.model

    def train(self, log_writer=None):
        self._init_distributed()
        train_dataloader = self._build_dataloader("train")
        valid_dataloader = self._build_dataloader("eval")
        logger.info("train dataloader has {} iters".format(len(train_dataloader)))
        logger.info("valid dataloader has {} iters".format(len(valid_dataloader)))
        model = self.model
        criterion = self.loss
        optimizer = self.Optimizer(model.parameters())
        lr_scheduler = self._build_scheduler(
            optimizer, self.epoch_num, len(train_dataloader)
        )
        post_processor = self.postprocessor
        metric_ = self.metric
        pre_best_model_dict = self.load()
        meter_epoch_step = self.meter_epoch_step
        # calc_epoch_interval = self.calc_epoch_interval
        log_window_size = self.log_window_size
        epoch_num = self.epoch_num
        log_batch_step = self.log_batch_step
        global_step = pre_best_model_dict.get("global_step", 0)
        eval_batch_step = self.eval_batch_step
        start_eval_step = 0
        if isinstance(eval_batch_step, list | tuple) and len(eval_batch_step) == 2:
            (start_eval_step, eval_batch_step) = eval_batch_step
            if len(valid_dataloader) == 0:
                logger.info(
                    "No Images in eval dataset, evaluation during training will be disabled"
                )
                start_eval_step = 1e111
            logger.info(
                "During the training process, after the {}th iteration, an evaluation is run every {} iterations".format(
                    start_eval_step, eval_batch_step
                )
            )
        save_epoch_step = self.save_epoch_step
        model_dir = self.model_dir
        os.makedirs(model_dir, exist_ok=True)
        main_indicator = metric_.main_indicator
        best_model_dict = {main_indicator: 0}
        best_model_dict.update(pre_best_model_dict)
        train_stats = TrainingStats(log_window_size, ["lr"])
        model.train()
        extra_input_models = [
            "SRN",
            "NRTR",
            "SAR",
            "SEED",
            "SVTR",
            "SPIN",
            "VisionLAN",
            "RobustScanner",
            "RFL",
            "DRRG",
        ]
        extra_input = self.algorithm in extra_input_models
        model_type = self.model_type
        algorithm = self.algorithm
        start_epoch = best_model_dict.get("start_epoch", 1)
        total_samples = 0
        train_reader_cost = 0.0
        train_batch_cost = 0.0
        reader_start = time.time()
        eta_meter = AverageMeter()
        max_iter = (
            len(train_dataloader) - 1
            if platform.system() == "Windows"
            else len(train_dataloader)
        )
        for epoch in range(start_epoch, epoch_num + 1):
            if hasattr(train_dataloader.dataset, "need_reset"):
                train_dataloader = self._build_dataloader("train", seed=epoch)
                max_iter = (
                    len(train_dataloader) - 1
                    if platform.system() == "Windows"
                    else len(train_dataloader)
                )
            for idx, batch in enumerate(train_dataloader):
                train_reader_cost += time.time() - reader_start
                if idx >= max_iter:
                    break
                lr = lr_scheduler.get_lr()

                images = batch[0]
                # 这里应该是由模型本身决定的，没有必要细分 fixme
                if model_type == "tab" or extra_input:
                    predict = model(images, targets=batch[1:])
                elif model_type in ["kie", "sr"]:
                    predict = model(batch)
                elif algorithm in ["CAN"]:
                    predict = model(batch[:3])
                elif model_type in ["cse"]:
                    predict = model(batch)
                else:
                    predict = model(images)
                loss = criterion(predict, batch)  # 训练时候才使用

                with torch.autograd.detect_anomaly():
                    loss["loss"].backward()
                optimizer.step()
                optimizer.zero_grad()

                # metric 也应当传入所有数据
                if meter_epoch_step > 0 and epoch % meter_epoch_step == 0:
                    batch = [item.numpy() for item in batch]
                    if model_type in ["kie", "sr"]:
                        metric_(predict, batch)
                    elif model_type in ["table"]:
                        post_result = post_processor(predict, batch)
                        metric_(post_result, batch)
                    elif algorithm in ["CAN"]:
                        model_type = "can"
                        metric_(predict[0], batch[2:], epoch_reset=idx == 0)
                    else:
                        if self.loss.__class__.__name__ in ["MultiLoss"]:
                            post_result = post_processor(predict["ctc"], batch[1])
                        elif self.loss.__class__.__name__ in ["VLLoss"]:
                            post_result = post_processor(predict, batch[1], batch[-1])
                        else:
                            post_result = post_processor(predict, batch[1])
                        metric_(post_result, batch)
                    metric = metric_.get_metric()
                    train_stats.update(metric)

                train_batch_time = time.time() - reader_start
                train_batch_cost += train_batch_time
                eta_meter.update(train_batch_time)
                global_step += 1
                total_samples += len(images)
                lr_scheduler.step()
                stats = {k: v.detach().numpy().mean() for (k, v) in loss.items()}
                stats["lr"] = lr
                train_stats.update(stats)
                if log_writer and self.is_rank0:
                    log_writer.log_metrics(
                        metrics=train_stats.get(),
                        prefix="TRAIN",
                        step=global_step,
                    )
                if self.is_rank0 and (
                    global_step > 0
                    and global_step % log_batch_step == 0
                    or idx >= len(train_dataloader) - 1
                ):
                    logs = train_stats.log()
                    eta_sec = (
                        (epoch_num + 1 - epoch) * len(train_dataloader) - idx - 1
                    ) * eta_meter.avg
                    eta_sec_format = str(datetime.timedelta(seconds=int(eta_sec)))
                    strs = "epoch: [{}/{}], global_step: {}, {}, avg_reader_cost: {:.5f} s, avg_batch_cost: {:.5f} s, avg_samples: {}, ips: {:.5f} samples/s, eta: {}".format(
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
                    logger.info(strs)
                    total_samples = 0
                    train_reader_cost = 0.0
                    train_batch_cost = 0.0
                if (
                    global_step > start_eval_step
                    and (global_step - start_eval_step) % eval_batch_step == 0
                    and self.is_rank0
                ):
                    cur_metric = valid(
                        model,
                        valid_dataloader,
                        post_processor,
                        metric_,
                        model_type,
                        extra_input=extra_input,
                    )
                    cur_metric_str = "cur metric, {}".format(
                        ", ".join(
                            ["{}: {}".format(k, v) for (k, v) in cur_metric.items()]
                        )
                    )
                    logger.info(cur_metric_str)
                    if log_writer is not None:
                        log_writer.log_metrics(
                            metrics=cur_metric, prefix="EVAL", step=global_step
                        )
                    if cur_metric[main_indicator] >= best_model_dict[main_indicator]:
                        best_model_dict.update(cur_metric)
                        best_model_dict["best_epoch"] = epoch
                        self.save(
                            model_dir,
                            is_best=True,
                            prefix="best_accuracy",
                            best_model_dict=best_model_dict,
                            epoch=epoch,
                            global_step=global_step,
                        )
                    best_str = "best metric, {}".format(
                        ", ".join(
                            [
                                "{}: {}".format(k, v)
                                for (k, v) in best_model_dict.items()
                            ]
                        )
                    )
                    logger.info(best_str)
                    if log_writer is not None:
                        log_writer.log_metrics(
                            metrics={
                                "best_{}".format(main_indicator): best_model_dict[
                                    main_indicator
                                ]
                            },
                            prefix="EVAL",
                            step=global_step,
                        )
                        log_writer.log_model(
                            is_best=True,
                            prefix="best_accuracy",
                            metadata=best_model_dict,
                        )
                reader_start = time.time()
            if self.is_rank0:
                logger.info("Save model checkpoint to {}".format(model_dir))
                self.save(
                    model_dir,
                    is_best=False,
                    prefix="latest",
                    best_model_dict=best_model_dict,
                    epoch=epoch,
                    global_step=global_step,
                )
                if log_writer is not None:
                    log_writer.log_model(is_best=False, prefix="latest")
                if epoch > 0 and epoch % save_epoch_step == 0:
                    self.save(
                        model_dir,
                        is_best=False,
                        prefix="iter_epoch_{}".format(epoch),
                        best_model_dict=best_model_dict,
                        epoch=epoch,
                        global_step=global_step,
                    )
                    if log_writer is not None:
                        log_writer.log_model(
                            is_best=False, prefix="iter_epoch_{}".format(epoch)
                        )
        best_str = f"best metric, {', '.join((f'{k}: {v}' for (k, v) in best_model_dict.items()))}"
        logger.info(best_str)
        if log_writer and self.is_rank0:
            log_writer.close()
        if self.distributed:
            dist.destroy_process_group()
        return  # 增加一个意外中断，保研数据的功能

    def __call__(self, *args, **kwargs):
        f = getattr(self, self.model_type.lower())
        return f(*args, **kwargs)

    @torch.no_grad()
    def table_one_image(self, img_or_path):
        if isinstance(img_or_path, str):
            img = cv2.imread(img_or_path)
        else:
            img = img_or_path

        self.model.eval()
        data = {"image": img}

        batch = self.transforms("infer")(data)

        if batch[0] is None:
            return None, 0

        images = np.expand_dims(batch[0], axis=0)
        shape_list = np.expand_dims(batch[-1], axis=0)
        images = torch.Tensor(images)
        preds = self.model(images)
        post_result = self.postprocessor(preds, [shape_list])

        structure_str_list = post_result["structure_batch_list"][0]
        bbox_list = post_result["bbox_batch_list"][0]
        structure_str_list = structure_str_list[0]
        structure_str_list = (
            ["<html>", "<body>", "<table>"]
            + structure_str_list
            + ["</table>", "</body>", "</html>"]
        )

        return structure_str_list, bbox_list

    def _ocr(self, img, det, rec):
        h, w = img.shape[:2]
        dt_boxes = det(copy.deepcopy(img))["boxes"]
        dt_boxes = sorted_boxes(dt_boxes)
        r_boxes = []
        for box in dt_boxes:
            x_min = max(0, box[:, 0].min() - 1)
            x_max = min(w, box[:, 0].max() + 1)
            y_min = max(0, box[:, 1].min() - 1)
            y_max = min(h, box[:, 1].max() + 1)
            box = [x_min, y_min, x_max, y_max]
            r_boxes.append(box)
        dt_boxes = np.array(r_boxes)
        # logger.debug("dt_boxes num : {}, elapse : {}".format(len(dt_boxes), det_elapse))
        if dt_boxes is None:
            return None, None

        img_crop_list = []
        for i in range(len(dt_boxes)):
            det_box = dt_boxes[i]
            x0, y0, x1, y1 = expand(2, det_box, img.shape)
            text_rect = img[int(y0) : int(y1), int(x0) : int(x1), :]
            img_crop_list.append(text_rect)
        rec_res = rec(img_crop_list)
        # logger.debug("rec_res num  : {}, elapse : {}".format(len(rec_res), rec_elapse))
        return dt_boxes, rec_res

    def tab(self, img_or_path, det=None, rec=None, view="html", output=None):
        if isinstance(img_or_path, str):
            img = cv2.imread(img_or_path)
        else:
            img = img_or_path
        result = {}
        structure_res = self.table_one_image(copy.deepcopy(img))
        if not rec:
            return structure_res

        result["cell_bbox"] = structure_res[1].tolist()
        dt_boxes, rec_res = self._ocr(copy.deepcopy(img), det, rec)
        result["boxes"] = dt_boxes  # [x.tolist() for x in dt_boxes]
        result["rec_res"] = rec_res

        if self.algorithm in ["TableMaster"]:
            match = TableMasterMatcher()
        else:
            match = TableMatch(filter_ocr_result=True)
        if result["cell_bbox"]:
            pred_html = match(structure_res, dt_boxes, rec_res)
            result["html"] = pred_html
            if view == "html":
                output = output or "."
                table_view(img_or_path, pred_html, output)
        return result

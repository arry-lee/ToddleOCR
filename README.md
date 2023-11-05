<p align="center">
 <img src="docs/toddleocr.png" align="middle" width = "600"/>
<p align="center">
<p align="left">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/arry-lee/ToddleOCR/stargazers"><img src="https://img.shields.io/github/stars/arry-lee/ToddleOCR?color=ccf"></a>
</p>

# ToddleOCR 拓牍

ToddleOCR是一个基于Torch实现的OCR（光学字符识别）项目，它fork自PaddleOCR，并经过作者进行了改进和优化。该项目旨在通过使用Torch框架来学习和理解OCR技术的实现原理。

## 项目名称的含义

ToddleOCR这个名称有三层意思：

1. 结合了Torch和Paddle两个框架的特点，因此命名为Toddle。
2. "Toddle"在英文中意味着蹒跚学步，也代表这个项目是作者在学习过程中的探索和尝试。
3. "Toddle"在中文中音译为"拓牍"，意指从竹片上拓印文字，引申为从图片中提取文字。

## 项目状态

目前，ToddleOCR处于探索阶段，仍在不断地进行改进和优化。欢迎开发者和研究者参与其中，一起探索OCR技术的前沿。~~项目文档目前还是PaddleOCR的文档~~。

## 功能特点

在ToddleOCR中，你可以期望以下功能特点：

- 文字检测：能够在图像中准确地检测出文字区域的位置和边界框。
- 文字识别：能够将检测到的文字区域进行识别，输出对应的文字内容。
- 多语言支持：支持多种语言的文字检测和识别，包括中文、英文等常见语言。
~~- 高性能：经过优化的算法和模型结构，能够在保证准确性的同时提高处理速度。~~

## 快速开始

### 环境要求

在开始之前，请确保你已经安装了以下环境：

* Python 3.x
* Torch 2.x

其他依赖库（具体依赖请参考项目文档）

### 安装

1. 克隆项目代码到本地：

```shell
git clone https://github.com/arry-lee/ToddleOCR.git
```

2. 进入项目目录：

```shell
cd ToddleOCR
```

3. 安装依赖：

```shell
pip install -r requirements.txt
```

4. 下载模型
目前提供的模型可以在[这里](https://github.com/arry-lee/ToddleOCR/releases/weights)手动下载，代码运行时也会自动下载：

- [x] [zh_ocr_cls_v1.rar](https://github.com/arry-lee/ToddleOCR/releases/download/weights/zh_ocr_cls_v1.rar) 文本角度分类模型
- [x] [zh_ocr_det_v3.rar](https://github.com/arry-lee/ToddleOCR/releases/download/weights/zh_ocr_det_v3.rar) 中英文检测模型
- [x] [zh_ocr_rec_v3.rar](https://github.com/arry-lee/ToddleOCR/releases/download/weights/zh_ocr_rec_v3.rar) 中英文识别模型
- [x] [zh_str_tab_m2.rar](https://github.com/arry-lee/ToddleOCR/releases/download/weights/zh_str_tab_m2.rar) 表格识别重建模型

### 使用示例

1. 准备输入图像文件，例如input.jpg。

2. 运行OCR示例脚本：

```shell
python toddleocr input.jpg
```

这将输出检测到的文字区域和对应的识别结果。

## 如何配置新的算法

与PaddleOCR相比，本项目摒弃yaml的配置方法，采用纯python语言，类yaml的配置方法，但更灵活，而且可以使用复杂的引用计算，并且所见即所得，具体的请参考
ptocr/config.py 中的ConfigModel类，继承并重写你的参数。

关于配置的语法，只有两点需要特别注意的，为了方便配置和简化配置量 ,项目内部定义了一个辅助类，提供了一些语法糖，如下：

1. 使用 _ 类，这是一个多功能的辅助类，有以下几个语法功能：
    1. 类似偏函数的偏类：`_(DBHead,arg1=0,arg2=1) ==> partial(DBHead, **kwargs)`
    2. 字符串动态导入类：`_("DBHead",arg1=0,arg2=1) ==> partial(DBHead, **kwargs)`
    3. 没有位置参数等效于字典：`_(arg1=0,arg2=1) ==> dict(arg1=0,arg2=1)`
    4. 预热学习率规划器：
    5. 等效并列列表,用于 Transformers： `_[train:eval:infer,train:eval:...]`,
       切片语法的三个位置分别表示训练，测试，推理模式下的预处理器，
       特别的：...省略号表示同前一个，
       空的或None表示该位置不需要这个预处理器，例如[DecodeLabel:...:]表示训练和测试需要DecodeLabel，推理不需要,这种表示方法是为了
       **简化配置，共享处理器减少实例的创建**

    ```python
    class _:
    
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
                from tools.modelhub import Hub
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
    ```

2. 使用注解语法区分训练参数和测试参数
   ConfigModel 有两个内部类 Data 和 Loader 子类可以采用注解语法区分训练时配置和测试时配置，例如：
    ```python
    class Loader:
        shuffle:False = True
        drop_last:True = False
        batch_size:1 = 8
        num_workers: 0 = 4
   ```
   **等号后面表示训练时参数，冒号后面表示测试时参数**，没有冒号则相同
   这种表示方法同样是为了简化配置

更多的可以参考已经实现的算法，例如`models/det/det_db_rvd.py`

## 贡献

如果你对ToddleOCR感兴趣，并且希望为项目做出贡献，欢迎提交问题、提出建议或者发送Pull Request。我们乐于接受来自社区的贡献，共同推动项目的发展。

## 帮助与支持

如果你在使用过程中遇到任何问题，或者需要进一步的帮助与支持，请参考项目文档或者联系我们的团队。

## 相关链接

- 项目主页：https://github.com/arry-lee/ToddleOCR
- PaddleOCR：https://github.com/paddlepaddle/PaddleOCR
- 文档：https://github.com/arry-lee/ToddleOCR/docs

## 许可证

本项目的发布受<a href="https://github.com/arry-lee/ToddleOCR/blob/torchocr/LICENSE">Apache 2.0 license</a>许可认证。

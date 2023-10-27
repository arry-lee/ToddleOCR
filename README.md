<p align="center">
 <img src="./doc/toddleocr.png" align="middle" width = "600"/>
<p align="center">
<p align="left">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/arry-lee/ToddleOCR/stargazers"><img src="https://img.shields.io/github/stars/arry-lee/ToddleOCR?color=ccf"></a>
</p>

# ToddleOCR 拓牍

ToddleOCR是一个基于Torch实现的OCR（光学字符识别）项目，它源于PaddleOCR，并经过作者进行了改进和优化。该项目旨在通过使用Torch框架来学习和理解OCR技术的实现原理。

## 项目名称的含义

ToddleOCR这个名称有三层意思：

1. 结合了Torch和Paddle两个框架的特点，因此命名为Toddle。
2. "Toddle"在英文中意味着蹒跚学步，也代表这个项目是作者在学习过程中的探索和尝试。
3. "Toddle"在中文中音译为"拓牍"，意指从竹片上拓印文字，引申为从图片中提取文字。

## 项目状态

目前，ToddleOCR处于探索阶段，仍在不断地进行改进和优化。欢迎开发者和研究者参与其中，一起探索OCR技术的前沿。

## 功能特点

在ToddleOCR中，你可以期望以下功能特点：

- 文字检测：能够在图像中准确地检测出文字区域的位置和边界框。
- 文字识别：能够将检测到的文字区域进行识别，输出对应的文字内容。
- 多语言支持：支持多种语言的文字检测和识别，包括中文、英文等常见语言。
- 高性能：经过优化的算法和模型结构，能够在保证准确性的同时提高处理速度。

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

### 使用示例

1. 准备输入图像文件，例如input.jpg。

2. 运行OCR示例脚本：

```shell
python toddleocr.py input.jpg
```

这将输出检测到的文字区域和对应的识别结果。

## 贡献

如果你对ToddleOCR感兴趣，并且希望为项目做出贡献，欢迎提交问题、提出建议或者发送Pull Request。我们乐于接受来自社区的贡献，共同推动项目的发展。

## 帮助与支持

如果你在使用过程中遇到任何问题，或者需要进一步的帮助与支持，请参考项目文档或者联系我们的团队。

## 相关链接

项目主页：https://github.com/arry-lee/ToddleOCR
PaddleOCR：https://github.com/paddlepaddle/PaddleOCR
文档：https://github.com/arry-lee/ToddleOCR/docs

## 许可证

本项目的发布受<a href="https://github.com/arry-lee/ToddleOCR/blob/torchocr/LICENSE">Apache 2.0 license</a>许可认证。

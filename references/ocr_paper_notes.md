# 项目参考文献阅读笔记

## 按时间线排列
* [x] [PP-OCRv3: More Attempts for the Improvement of Ultra Lightweight OCR System](https://arxiv.org/abs/2206.03001v2)
  * [PP-OCRv3：改进超轻量级OCR系统的更多尝试](references/papers/2206.03001.zh.pdf)
    PP-OCRv3在PPOCRv2 的基础上在9个方面升级了文本检测模型和文本 识别模型。对于文本检测器，我们引入了一个具有大感 受野的PAN模块，称为LK-PAN，一个具有残差注意机 制的FPN模块，称为RSE-FPN，以及DML蒸馏策略。对 于文本识别器，我们引入了轻量级文本识别网络SVTRLCNet， 通过注意力进行CTC的引导训练，数据增强策 略TextConAug，通过自监督学习的TextRotNet得到更好 的预训练模型，以及U-DML和UIM来加速模型并提高效果。

    检测训练7w真实6w合成
    文本识别1850w数据，700w真实数据，1150w个合成图像
    主要关注不同背景、旋转、透视变换、噪声、垂直文本 等场景。合成图像的语料库来自真实场景图像。所有 的验证图像也来自真实场景。
    此外，我们收集了800个用于不同真实应用场景的图像，以评估整体OCR系统，包括合同样本、车牌、铭牌、火车票、答题卡、表格、证书、街景图像、名片、数码仪表等。图10显示了测试集的一些图像
* [x] [SVTR: Scene Text Recognition with a Single Visual Model](https://arxiv.org/abs/2205.00159)
* [SVTR：使用单个视觉模型进行场景文本识别](references/papers/2205.00159.zh.pdf)
* [Real-Time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion](https://arxiv.org/abs/2202.10304)
* [基于可微二值化和自适应尺度融合的实时场景文本检测]
* [x] [Benchmarking Chinese Text Recognition:Datasets, Baselines, and an Empirical Study](https://arxiv.org/abs/2112.15093)
* [中文文本识别基准测试：数据集、基线与实证研究](references/papers/2112.15093.zh.pdf)
* [x] [PP-LCNet: A Lightweight CPU Convolutional Neural Network](https://arxiv.org/abs/2109.15099)
* [PP-LCNet：轻量级CPU卷积神经网络](references/papers/2109.15099.zh.pdf)
* [x] [PP-OCRv2: Bag of Tricks for Ultra Lightweight OCR System](https://arxiv.org/abs/2109.03144)
* [PP-OCRv2：超轻量级OCR系统的技巧包](references/papers/2109.03144.zh.pdf)
* [x] VisionLAN [From Two to One: A New Scene Text Recognizer with Visual Language Modeling Network](https://arxiv.org/abs/2108.09661)
* [从二到一：基于视觉语言建模网络的新型场景文本识别器](references/papers/2108.09661.zh.pdf)
* [x] CT [CentripetalText: An Efficient Text Instance Representation for Scene Text Detection](https://arxiv.org/abs/2107.05945)
* [向心文本：用于场景文本检测的高效文本实例表示](references/papers/2107.05945.zh.pdf)
  * CT 说 检测能力和速度大于PAN和DB
* [x] PAN [Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network](https://arxiv.org/abs/1908.05900)
* [基于像素聚合网络的高效、准确的任意形状文本检测](https://readpaper.com/pdf-annotate/note?pdfId=628899607381024768&noteId=2006114988382478592)
* [x] [Vision Transformer for Fast and Efficient Scene Text Recognition](https://arxiv.org/abs/2105.08582)
* [用于快速高效场景文本识别的视觉变换器](references/papers/2105.08582.zh.pdf)
    - [代码](https://github.com/roatienza/deeptext-recognition-benchmark)
    - ViTSTR 一阶段
    - 我们复现了几个强基线模型的结果：CRNN、R2AM、GCRNN、Rosetta、RARE、STAR-Net 和TRBA，
    - 使用RandAugment进行图像反转、弯曲、模糊、噪声、畸变、旋转、 拉伸/压缩、透视和缩小等不同图像变换，可以提高ViTSTR-Tiny的泛化能力1.8％，ViTSTR-Small的泛化能力1.6％
    - ViTSTR是一个简单的单阶段模型架构，强调在准确性、速度和计算要求之间的平衡。通过面向STR的数据增强，ViTSTR可以显著提高准确性，特别是对于不规则数据集。当规模扩大时，ViTSTR保持在准确性、速度和计算要求的前沿。
    - 需要进一步理解Transformer 和 ViT
* [x] [Reciprocal Feature Learning via Explicit and Implicit Tasks in Scene Text Recognition](https://arxiv.org/abs/2105.06229)
* [场景文本识别中通过显式和隐式任务进行倒易特征学习](references/papers/2105.06229.zh.pdf)
    - RF-L 并联了一个字符计数任务，通过一个适配器模块连接两个分支，使得两个分支的特征可以互相学习
    - 是否也可以增加一个笔画计数任务，也没有增加额外的标注成本？
    - [代码](models/rec/rec_rfl_rrfl.pyi)
* [x] FCN [Fourier Contour Embedding for Arbitrary-Shaped Text Detection](https://arxiv.org/abs/2104.10442)
* [用于任意形状文本检测的傅里叶轮廓嵌入](references/papers/2104.10442.zh.pdf)
  - 为任意形状的文本检测设计一个灵活而简单的表示方法具有重要意义

* [x] [PP-OCR: A Practical Ultra Lightweight OCR System](https://arxiv.org/abs/2009.09941)
* [PP-OCR：实用的超轻量级 OCR 系统](references/papers/2009.09941.zh.pdf)
  - 轻量化骨干网络 MobileNetV3
  - 数据增强 BDA TIA
  - 余弦型学习率衰减
  - 特征图分辨率（32x320）
  - 正则化参数 L2
  - 学习率预热
  - 轻量级头部
  - 预训练模型
  - PACT 量化
* [x] SPIN [SPIN: Structure-Preserving Inner Offset Network for Scene Text Recognition](https://arxiv.org/abs/2005.13117)
* [SPIN：用于场景文本识别的结构保持内偏移网络]
* SRN　[Towards Accurate Scene Text Recognition with Semantic Reasoning Networks](https://arxiv.org/abs/2003.12294)
* [基于语义推理网络的场景文本精准识别](https://readpaper.com/pdf-annotate/note?pdfId=628901332967456768&noteId=2008803368415339520)
* DRRG [x] [Deep Relational Reasoning Graph Network for Arbitrary Shape Text Detection](https://arxiv.org/abs/2003.07493)
* [用于任意形状文本检测的深度关系推理图网络](references/papers/2003.07493.zh.pdf)
* DB [x] [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947)
* [具有可微二值化的实时场景文本检测](references/papers/1911.08947.zh.pdf)
* Rosetta [Rosetta: Large Scale System for Text Detection and Recognition in Images](https://arxiv.org/abs/1910.05085)
* [罗塞塔：用于图像文本检测和识别的大型系统]
* SAST [A Single-Shot Arbitrarily-Shaped Text Detector based on Context Attended Multi-Task Learning]
* [基于上下文参与多任务学习的单次任意形状文本检测器](https://arxiv.org/abs/1908.05498)
* [What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis](https://arxiv.org/abs/1904.01906)
* [场景文本识别模型对比有什么问题？数据集和模型分析](https://readpaper.com/pdf-annotate/note?pdfId=628901614598758400&noteId=2007696210588148736)
  - 本文主要介绍了关于场景文本识别（STR）模型比较中存在的数据集和模型不一致的问题，并提出了三个解决方案。首先，分析了训练和评估数据集的不一致性及其对性能差距造成的影响。其次，引入了一个统一的四阶段STR框架，可对以前提出的STR模块进行广泛评估。第三，通过在一致的数据集上评估模块的贡献，分析了准确性、速度和内存需求。本文还提供了大量的实验结果和分析，以及对未来STR研究方向的建议。
* PSE [Shape robust text detection with progressive scale expansion network](https://arxiv.org/abs/1903.12473)
* [通过渐进式缩放扩展网络进行形状稳健的文本检测](https://readpaper.com/pdf-annotate/note?pdfId=628899612883521536&noteId=2008413245649385984)
  - 本文主要介绍了一种基于 kernel 机制和 Progressive Scale Expansion(PSE) 算法的文本检测方法，称为 Progressive Scale Expansion Network(PSENet)。该方法能够准确地检测任意形状的文本实例，并能够将彼此靠近的文本实例分离开来，以便更好地使用基于分割的方法来检测任意形状的文本实例。本文还介绍了该方法的具体实现和在多个数据集上的实验结果，证明了该方法的有效性和优越性。
* SAR [Show, Attend and Read: A Simple and Strong Baseline for Irregular Text Recognition](https://arxiv.org/abs/1811.00751)
* [展示、参加和阅读：不规则文本识别的简单而强大的基线]
* NRTR [NRTR: A No-Recurrence Sequence-to-Sequence Model For Scene Text Recognition](https://arxiv.org/abs/1806.00926)
* NRTR：用于场景文本识别的无重复序列到序列模型
  - 本文主要介绍了一种名为NRTR的无循环序列到序列模型，用于场景文本识别。相比于现有的循环神经网络和卷积神经网络，NRTR采用了自注意力机制，能够在更少的时间和复杂度下进行训练，并取得了非常好的性能表现。文章还介绍了NRTR的具体结构和在各种基准测试中的表现。
* [Deep Mutual Learning](https://arxiv.org/abs/1706.00384)
* 深度互学
* [x] [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155)
* [EAST：高效准确的场景文本检测器](references/papers/1704.03155.zh.pdf)
* RARE [Robust Scene Text Recognition with Automatic Rectification](https://arxiv.org/abs/1603.03915v2)
* [具有自动校正功能的可靠场景文本识别](https://readpaper.com/pdf-annotate/note?pdfId=628901599922888704&noteId=2008656584789306368)
  - 1）在没有几何监督的情况下，学习模型可以自动为人和序列识别网络生成更“可读”的图像； 2）所提出的文本校正方法可以显着提高不规则场景文本的识别精度； 3）与最先进的技术相比，所提出的场景文本识别系统具有竞争力。未来，我们计划通过将 RARE 与场景文本检测方法相结合来解决端到端场景文本阅读问题。
  - STN Spatial Transformer Network
  - SRN Sequential Recognition Network
  - TPS thin-plate-spline
* CRNN [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717)
* 基于图像的序列识别端到端可训练神经网络及其在场景文本识别中的应用


## 说明
将paddleocr 从 paddle 移植到 pytorch

transformers 是与paddle无关的，无需删改

DET 数据集的格式是

"img_path": 图像文件的路径
"label": 标签的json
"image": 图像的二进制数据
"ext_data": 扩展数据，是一个包含图像路径、标签和图像数据的字典列表

然后 label 经过转换后 DetLabelEncoder 增加了

"texts": 文本
"polys": 文本的边界框
"ignore_tags": 文本的忽略标签

自始至终，只有一个 data 字段在处理函数中传递，这个 data 字段是包含一个字典列表的列表，每个字典列表包含一个图像的路径、图像的二进制数据、图像的标签和扩展数据。

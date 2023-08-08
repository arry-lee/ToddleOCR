# GLOBAL

algorithm = "EAST"
cal_metric_during_train = false
checkpoints = null
epoch_num = 10000
eval_batch_step = [4000, 5000]
infer_img = null
log_smooth_window = 20
model_type = "det"
pretrained_model = "./pretrain_models/MobileNetV3_large_x0_5_pretrained"
print_batch_step = 2
save_epoch_step = 1000
save_inference_dir = null
save_model_dir = "./output/east_mv3/"
save_res_path = "./output/det_east/predicts_east.txt"
use_gpu = true
use_visualdl = false

# MODULE

## Backbone

name = "MobileNetV3"
scale = 0.5
model_name = "large"

## Neck

name = "EASTFPN"
model_name = "small"

## Head

name = "EASTHead"
model_name = "small"

## LOSS

name = "EASTLoss"

# OPTIMIZER

name = "Adam"
lr = 0.001
betas = [0.9, 0.999]

## SCHEDULER

name = "StepLR"

# POSTPROCESSOR

name = "EASTPostProcess"
score_thresh = 0.8
cover_thresh = 0.1
nms_thresh = 0.2

# METRIC

name = "DetMetric"
main_indicator = "hmean"

# TRAIN

## Dataset

name = "SimpleDataSet"
data_dir = "./train_data/icdar2015/text_localization/"
label_file_list = ["./train_data/icdar2015/text_localization/train_icdar2015_label.txt"]
ratio_list = [1.0]

### Transforms

#### DecodeImage

img_mode = "BGR"
channel_first = false

#### DetLabelEncode

#### EASTProcessTrain

image_shape = [512, 512]
background_ratio = 0.125
min_crop_side_ratio = 0.1
min_text_size = 10

#### KeepKeys

keep_keys = [
"image",
"score_map",
"geo_map",
"training_mask",
]

## Loader

shuffle = true
drop_last = false
batch_size_per_card = 16
num_workers = 8

# EVAL

## Dataset

name = "SimpleDataSet"
data_dir = "./train_data/icdar2015/text_localization/"
label_file_list = ["./train_data/icdar2015/text_localization/test_icdar2015_label.txt",
"./train_data/icdar2015/text_localization/test_icdar2015_label.txt",
"./train_data/icdar2015/text_localization/test_icdar2015_label.txt",
]

### Transforms

#### DecodeImage

img_mode = "BGR"
channel_first = false

#### DetLabelEncode

#### DetResizeForTest

limit_side_len = 2400
limit_type = "max"

#### NormalizeImage

scale = 0.0039215686
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
order = "hwc"

#### ToCHWImage

#### KeepKeys

keep_keys = ["image", "shape", "polys", "ignore_tags"]

## Loader

shuffle = false
drop_last = false
batch_size_per_card = 1
num_workers = 2

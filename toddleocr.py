import importlib
import os
import sys

__dir__ = os.path.dirname(__file__)

sys.path.append(os.path.join(__dir__, ""))

import cv2
import logging
import numpy as np
from pathlib import Path


def _import_file(module_name, file_path, make_importable=False):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module


ptocr = importlib.import_module("ptocr", "toddleocr")

from loguru import logger
from ptocr.utils.utility import check_and_read, alpha_to_color, binarize_img
from ptocr.utils.network import maybe_download, download_with_progressbar, is_link, confirm_model_dir_url
from tools.utility import draw_ocr, init_args, str2bool, check_gpu

__all__ = [
    "ToddleOCR",
    "draw_ocr",
    "download_with_progressbar",
]

SUPPORT_DET_MODEL = ["DB"]
VERSION = "v1.0.0"
SUPPORT_REC_MODEL = ["CRNN", "SVTR_LCNet"]
BASE_DIR = __dir__
print(BASE_DIR,os.getcwd())
BASE_URL = "https://github.com/arry-lee/ToddleOCR/releases/download/weights/"
DEFAULT_OCR_MODEL_VERSION = "v3"
SUPPORT_OCR_MODEL_VERSION = ["v3"]
# DEFAULT_STRUCTURE_MODEL_VERSION = "PP-StructureV2"
# SUPPORT_STRUCTURE_MODEL_VERSION = ["PP-Structure", "PP-StructureV2"]

MODEL_URLS = {
    "OCR": {
        "v3": {
            "det": {
                "model": "ptocr.models.det.v3.det_db_mv3_rse",
                "ch": {"url": "zh_ocr_det_v3.tar"},
                "en": {"url": "en_ocr_det_v3.tar"},
                "ml": {"url": "ml_ocr_det_v3.tar"},
            },
            "rec": {
                "model": "ptocr.models.rec.v3.rec_svtr_mv1e",
                "ch": {
                    "url": "zh_ocr_rec_v3.tar",
                    "dict": "./ptocr/utils/ptocr_keys_v1.txt",
                },
                "en": {
                    "url": "en_ocr_rec_v3.tar",
                    "dict": "./ptocr/utils/en_dict.txt",
                },
                "korean": {
                    "url": "ko_ocr_rec_v3.tar",
                    "dict": "./ptocr/utils/dict/korean_dict.txt",
                },
                "japan": {
                    "url": "ja_ocr_rec_v3.tar",
                    "dict": "./ptocr/utils/dict/japan_dict.txt",
                },
                "chinese_cht": {
                    "url": "ch_ocr_rec_v3.tar",
                    "dict": "./ptocr/utils/dict/chinese_cht_dict.txt",
                },
                "ta": {
                    "url": "ta_ocr_rec_v3.tar",
                    "dict": "./ptocr/utils/dict/ta_dict.txt",
                },
                "te": {
                    "url": "te_ocr_rec_v3.tar",
                    "dict": "./ptocr/utils/dict/te_dict.txt",
                },
                "ka": {
                    "url": "ka_ocr_rec_v3.tar",
                    "dict": "./ptocr/utils/dict/ka_dict.txt",
                },
                "latin": {
                    "url": "la_ocr_rec_v3.tar",
                    "dict": "./ptocr/utils/dict/latin_dict.txt",
                },
                "arabic": {
                    "url": "ar_ocr_rec_v3.tar",
                    "dict": "./ptocr/utils/dict/arabic_dict.txt",
                },
                "cyrillic": {
                    "url": "ru_ocr_rec_v3.tar",
                    "dict": "./ptocr/utils/dict/cyrillic_dict.txt",
                },
                "devanagari": {
                    "url": "hi_ocr_rec_v3.tar",
                    "dict": "./ptocr/utils/dict/devanagari_dict.txt",
                },
            },
            "cls": {
                "model": "ptocr.models.cls.v2.cls_cls_mv3",
                "ch": {"url": "zh_ocr_cls_v1.tar"}},
        },
        # "v2": {
        #     "det": {"ch": {"url": "zh_ocr_det_v2.tar"}},
        #     "rec": {
        #         "ch": {
        #             "url": "zh_ocr_rec_v2.tar",
        #             "dict": "./ptocr/utils/ptocr_keys_v1.txt",
        #         }
        #     },
        #     "cls": {"ch": {"url": "zh_ocr_cls_v1.tar"}},
        # },
        # "v1": {
        #     "det": {
        #         "ch": {"url": "zh_ocr_det_v1.tar"},
        #         "en": {"url": "en_ocr_det_v1.tar"},
        #         "structure": {"url": "en_tab_det_v1.tar"},
        #     },
        #     "rec": {
        #         "ch": {
        #             "url": "zh_ocr_rec_v1.tar",
        #             "dict": "./ptocr/utils/ptocr_keys_v1.txt",
        #         },
        #         "en": {
        #             "url": "en_ocr_rec_m2.tar",
        #             "dict": "./ptocr/utils/en_dict.txt",
        #         },
        #         "french": {
        #             "url": "fr_ocr_rec_m2.tar",
        #             "dict": "./ptocr/utils/dict/french_dict.txt",
        #         },
        #         "german": {
        #             "url": "de_ocr_rec_m2.tar",
        #             "dict": "./ptocr/utils/dict/german_dict.txt",
        #         },
        #         "korean": {
        #             "url": "ko_ocr_rec_m2.tar",
        #             "dict": "./ptocr/utils/dict/korean_dict.txt",
        #         },
        #         "japan": {
        #             "url": "ja_ocr_rec_m2.tar",
        #             "dict": "./ptocr/utils/dict/japan_dict.txt",
        #         },
        #         "chinese_cht": {
        #             "url": "ch_ocr_rec_m2.tar",
        #             "dict": "./ptocr/utils/dict/chinese_cht_dict.txt",
        #         },
        #         "ta": {
        #             "url": "ta_ocr_rec_m2.tar",
        #             "dict": "./ptocr/utils/dict/ta_dict.txt",
        #         },
        #         "te": {
        #             "url": "te_ocr_rec_m2.tar",
        #             "dict": "./ptocr/utils/dict/te_dict.txt",
        #         },
        #         "ka": {
        #             "url": "ka_ocr_rec_m2.tar",
        #             "dict": "./ptocr/utils/dict/ka_dict.txt",
        #         },
        #         "latin": {
        #             "url": "la_ocr_rec_v1.tar",
        #             "dict": "./ptocr/utils/dict/latin_dict.txt",
        #         },
        #         "arabic": {
        #             "url": "ar_ocr_rec_v1.tar",
        #             "dict": "./ptocr/utils/dict/arabic_dict.txt",
        #         },
        #         "cyrillic": {
        #             "url": "ru_ocr_rec_v1.tar",
        #             "dict": "./ptocr/utils/dict/cyrillic_dict.txt",
        #         },
        #         "devanagari": {
        #             "url": "hi_ocr_rec_v1.tar",
        #             "dict": "./ptocr/utils/dict/devanagari_dict.txt",
        #         },
        #         "structure": {
        #             "url": "en_tab_rec_v1.tar",
        #             "dict": "ptocr/utils/dict/table_dict.txt",
        #         },
        #     },
        #     "cls": {"ch": {"url": "zh_ocr_cls_v1.tar"}},
        # },
    },
    # "STRUCTURE": {
    #     "v1": {
    #         "table": {
    #             "en": {
    #                 "url": "en_tab_str_v1.tar",
    #                 "dict": "ptocr/utils/dict/table_structure_dict.txt",
    #             }
    #         }
    #     },
    #     "v2": {
    #         "table": {
    #             "en": {
    #                 "url": "en_tab_str_m2_slanet.tar",
    #                 "dict": "ptocr/utils/dict/table_structure_dict.txt",
    #             },
    #             "ch": {
    #                 "url": "zh_tab_str_m2_slanet.tar",
    #                 "dict": "ptocr/utils/dict/table_structure_dict_ch.txt",
    #             },
    #         },
    #         "layout": {
    #             "en": {
    #                 "url": "en_lay_det_x1_picodet.tar",
    #                 "dict": "ptocr/utils/dict/layout_dict/layout_publaynet_dict.txt",
    #             },
    #             "ch": {
    #                 "url": "ch_lay_det_x1_picodet.tar",
    #                 "dict": "ptocr/utils/dict/layout_dict/layout_cdla_dict.txt",
    #             },
    #         },
    #     },
    # },
}


def parse_args(main=True):
    import argparse

    parser = init_args()
    parser.add_help = main
    parser.add_argument("--lang", type=str, default="ch")
    parser.add_argument("--det", type=str2bool, default=True)
    parser.add_argument("--rec", type=str2bool, default=True)
    parser.add_argument("--type", type=str, default="ocr")
    parser.add_argument(
        "--ocr_version",
        type=str,
        choices=SUPPORT_OCR_MODEL_VERSION,
        default="v3",
        help="OCR Model version, the current model support list is as follows: "
             "1. v3 Support Chinese and English detection and recognition model, and direction classifier model"
             "2. v2 Support Chinese detection and recognition model. "
             "3. v1 support Chinese detection, recognition and direction classifier and multilingual recognition model.",
    )
    # parser.add_argument(
    #     "--structure_version",
    #     type=str,
    #     choices=SUPPORT_STRUCTURE_MODEL_VERSION,
    #     default="PP-StructureV2",
    #     help="Model version, the current model support list is as follows:"
    #          " 1. PP-Structure Support en table structure model."
    #          " 2. PP-StructureV2 Support ch and en table structure model.",
    # )

    for action in parser._actions:
        if action.dest in ["rec_char_dict_path", "table_char_dict_path", "layout_dict_path"]:
            action.default = None
    if main:
        return parser.parse_args()
    else:
        inference_args_dict = {}
        for action in parser._actions:
            inference_args_dict[action.dest] = action.default
        return argparse.Namespace(**inference_args_dict)


def parse_lang(lang):
    latin_lang = [
        "af",
        "az",
        "bs",
        "cs",
        "cy",
        "da",
        "de",
        "es",
        "et",
        "fr",
        "ga",
        "hr",
        "hu",
        "id",
        "is",
        "it",
        "ku",
        "la",
        "lt",
        "lv",
        "mi",
        "ms",
        "mt",
        "nl",
        "no",
        "oc",
        "pi",
        "pl",
        "pt",
        "ro",
        "rs_latin",
        "sk",
        "sl",
        "sq",
        "sv",
        "sw",
        "tl",
        "tr",
        "uz",
        "vi",
        "french",
        "german",
    ]
    arabic_lang = ["ar", "fa", "ug", "ur"]
    cyrillic_lang = [
        "ru",
        "rs_cyrillic",
        "be",
        "bg",
        "uk",
        "mn",
        "abq",
        "ady",
        "kbd",
        "ava",
        "dar",
        "inh",
        "che",
        "lbe",
        "lez",
        "tab",
    ]
    devanagari_lang = ["hi", "mr", "ne", "bh", "mai", "ang", "bho", "mah", "sck", "new", "gom", "sa", "bgc"]
    if lang in latin_lang:
        lang = "latin"
    elif lang in arabic_lang:
        lang = "arabic"
    elif lang in cyrillic_lang:
        lang = "cyrillic"
    elif lang in devanagari_lang:
        lang = "devanagari"
    assert lang in MODEL_URLS["OCR"][DEFAULT_OCR_MODEL_VERSION]["rec"], "param lang must in {}, but got {}".format(
        MODEL_URLS["OCR"][DEFAULT_OCR_MODEL_VERSION]["rec"].keys(), lang
    )
    if lang == "ch":
        det_lang = "ch"
    elif lang == "structure":
        det_lang = "structure"
    elif lang in ["en", "latin"]:
        det_lang = "en"
    else:
        det_lang = "ml"
    return lang, det_lang


def get_model_config(type, version, model_type, lang):
    if type == "OCR":
        DEFAULT_MODEL_VERSION = DEFAULT_OCR_MODEL_VERSION
    else:
        raise NotImplementedError

    model_urls = MODEL_URLS[type]
    if version not in model_urls:
        version = DEFAULT_MODEL_VERSION
    if model_type not in model_urls[version]:
        if model_type in model_urls[DEFAULT_MODEL_VERSION]:
            version = DEFAULT_MODEL_VERSION
        else:
            logger.error(
                "{} models is not support, we only support {}".format(
                    model_type, model_urls[DEFAULT_MODEL_VERSION].keys()
                )
            )
            sys.exit(-1)

    if lang not in model_urls[version][model_type]:
        if lang in model_urls[DEFAULT_MODEL_VERSION][model_type]:
            version = DEFAULT_MODEL_VERSION
        else:
            logger.error(
                "lang {} is not support, we only support {} for {} models".format(
                    lang, model_urls[DEFAULT_MODEL_VERSION][model_type].keys(), model_type
                )
            )
            sys.exit(-1)
    module = importlib.import_module(model_urls[version][model_type]['model'])
    model = getattr(module, 'Model')

    return model_urls[version][model_type][lang], model


def img_decode(content: bytes):
    np_arr = np.frombuffer(content, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)


def check_img(img):
    if isinstance(img, bytes):
        img = img_decode(img)
    if isinstance(img, str):
        # download net image
        if is_link(img):
            download_with_progressbar(img, "tmp.jpg")
            img = "tmp.jpg"
        image_file = img
        img, flag_gif, flag_pdf = check_and_read(image_file)
        if not flag_gif and not flag_pdf:
            with open(image_file, "rb") as f:
                img = img_decode(f.read())
        if img is None:
            logger.error("error in loading image:{}".format(image_file))
            return None
    if isinstance(img, np.ndarray) and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


class ToddleOCR:
    def __init__(self, **kwargs):
        """
        toddleocr package
        args:
            **kwargs: other params show in toddleocr --help
        """
        params = parse_args(main=False)
        params.__dict__.update(**kwargs)
        # assert params.ocr_version in SUPPORT_OCR_MODEL_VERSION, "ocr_version must in {}, but get {}".format(
        #     SUPPORT_OCR_MODEL_VERSION, params.ocr_version
        # )
        params.use_gpu = check_gpu(params.use_gpu)

        if not params.show_log:
            logger.setLevel(logging.INFO)
        self.use_angle_cls = params.use_angle_cls
        lang, det_lang = parse_lang(params.lang)

        # init model dir
        det_model_config, det_model_cls = get_model_config("OCR", params.ocr_version, "det", det_lang)

        params.det_model_dir, det_url = confirm_model_dir_url(
            params.det_model_dir, os.path.join(BASE_DIR, "weights"), BASE_URL+det_model_config["url"]
        )
        rec_model_config, rec_model_cls = get_model_config("OCR", params.ocr_version, "rec", lang)
        params.rec_model_dir, rec_url = confirm_model_dir_url(
            params.rec_model_dir, os.path.join(BASE_DIR, "weights"), BASE_URL+rec_model_config["url"]
        )
        cls_model_config, cls_model_cls = get_model_config("OCR", params.ocr_version, "cls", "ch")
        params.cls_model_dir, cls_url = confirm_model_dir_url(
            params.cls_model_dir, os.path.join(BASE_DIR, "weights"), BASE_URL+cls_model_config["url"]
        )
        if params.ocr_version == "v3":
            rec_model_cls.rec_image_shape = (3, 48, 320)
        else:
            rec_model_cls.rec_image_shape = (3, 32, 320)
        # download model if using paddle infer
        if not params.use_onnx:
            maybe_download(params.det_model_dir, det_url)
            maybe_download(params.rec_model_dir, rec_url)
            maybe_download(params.cls_model_dir, cls_url)

        if params.det_algorithm not in SUPPORT_DET_MODEL:
            logger.error("det_algorithm must in {}".format(SUPPORT_DET_MODEL))
            sys.exit(0)
        if params.rec_algorithm not in SUPPORT_REC_MODEL:
            logger.error("rec_algorithm must in {}".format(SUPPORT_REC_MODEL))
            sys.exit(0)

        if params.rec_char_dict_path is None:
            params.rec_char_dict_path = str(Path(__file__).parent / rec_model_config["dict"])

        rec_model_cls.character_dict_path = params.rec_char_dict_path

        logger.debug(params)
        self.det_model = det_model_cls(params.det_model_dir + '/inference.pt')
        self.cls_model = cls_model_cls(params.cls_model_dir + '/inference.pt')
        self.rec_model = rec_model_cls(params.rec_model_dir + '/inference.pt')

    def ocr(self, img, det=True, rec=True, cls=True, bin=False, inv=False, alpha_color=(255, 255, 255)):
        """
        OCR with ToddleOCR
        args：
            img: img for OCR, support ndarray, img_path and list or ndarray
            det: use text detection or not. If False, only rec will be exec. Default is True
            rec: use text recognition or not. If False, only det will be exec. Default is True
            cls: use angle classifier or not. Default is True. If True, the text with rotation of 180 degrees can be recognized. If no text is rotated by 180 degrees, use cls=False to get better performance. Text with rotation of 90 or 270 degrees can be recognized even if cls=False.
            bin: binarize image to black and white. Default is False.
            inv: invert image colors. Default is False.
            alpha_color: set RGB color Tuple for transparent parts replacement. Default is pure white.
        """
        assert isinstance(img, (np.ndarray, list, str, bytes))
        if isinstance(img, list) and det == True:
            logger.error("When input a list of images, det must be false")
            exit(0)
        if cls == True and self.use_angle_cls == False:
            logger.warning(
                "Since the angle classifier is not initialized, it will not be used during the forward process"
            )

        if cls:
            cls = self.cls_model
        if rec:
            rec = self.rec_model

        img = check_img(img)
        if not isinstance(img, list):
            imgs = [img]
        else:
            imgs = img
        def preprocess_image(_image):
            _image = alpha_to_color(_image, alpha_color)
            if inv:
                _image = cv2.bitwise_not(_image)
            if bin:
                _image = binarize_img(_image)
            return _image

        ocr_res = []
        for idx, img in enumerate(imgs):
            img = preprocess_image(img)
            dt_boxes = self.det_model(img, cls=cls, rec=rec)
            if not dt_boxes:
                ocr_res.append(None)
                continue
            ocr_res.append(dt_boxes)
        return ocr_res


if __name__ == '__main__':
    t = ToddleOCR(det_model_dir='weights/zh_ocr_det_v3',
        cls_model_dir='weights/zh_ocr_cls_v1',
        rec_model_dir='weights/zh_ocr_rec_v3',
    )
    r = t.ocr("./doc/imgs/11.jpg")
    print(r)

#  Copyright (c) 2023. Arry Lee, <arry_lee@qq.com>
import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent

if __debug__:
    PROJECT_DIR = PACKAGE_DIR.parent
else:
    if os.environ.get("TODDLEOCR_PROJECT_DIR", None):
        PROJECT_DIR = Path(os.environ.get("TODDLEOCR_PROJECT_DIR"))
    else:
        PROJECT_DIR = Path.home() / '.toddleocr'

MODEL_DIR = PROJECT_DIR / 'model'
DATASET_DIR = PROJECT_DIR / 'train_data'
OUTPUT_DIR = PROJECT_DIR / "output"
WEIGHTS_DIR = PROJECT_DIR / "weights"

WEIGHTS_URL = "https://github.com/arry-lee/ToddleOCR/releases/download/weights/"
MODEL_URLS = {
    "OCR": {
        "v3": {
            "det": {
                "model": "models.det.v3.det_db_mv3_rse",
                "ch": {"url": "zh_ocr_det_v3.tar"},
                "en": {"url": "en_ocr_det_v3.tar"},
                "ml": {"url": "ml_ocr_det_v3.tar"},
            },
            "rec": {
                "model": "models.rec.v3.rec_svtr_mv1e",
                "ch": {
                    "url": "zh_ocr_rec_v3.tar",
                    "dict": "utils/dict/chinese_sim_dict.txt",
                },
                "en": {
                    "url": "en_ocr_rec_v3.tar",
                    "dict": "utils/dict/en96_dict.txt",
                },
                "korean": {
                    "url": "ko_ocr_rec_v3.tar",
                    "dict": "utils/dict/korean_dict.txt",
                },
                "japan": {
                    "url": "ja_ocr_rec_v3.tar",
                    "dict": "utils/dict/japan_dict.txt",
                },
                "chinese_cht": {
                    "url": "ch_ocr_rec_v3.tar",
                    "dict": "utils/dict/chinese_cht_dict.txt",
                },
                "ta": {
                    "url": "ta_ocr_rec_v3.tar",
                    "dict": "utils/dict/ta_dict.txt",
                },
                "te": {
                    "url": "te_ocr_rec_v3.tar",
                    "dict": "utils/dict/te_dict.txt",
                },
                "ka": {
                    "url": "ka_ocr_rec_v3.tar",
                    "dict": "utils/dict/ka_dict.txt",
                },
                "latin": {
                    "url": "la_ocr_rec_v3.tar",
                    "dict": "utils/dict/latin_dict.txt",
                },
                "arabic": {
                    "url": "ar_ocr_rec_v3.tar",
                    "dict": "utils/dict/arabic_dict.txt",
                },
                "cyrillic": {
                    "url": "ru_ocr_rec_v3.tar",
                    "dict": "utils/dict/cyrillic_dict.txt",
                },
                "devanagari": {
                    "url": "hi_ocr_rec_v3.tar",
                    "dict": "utils/dict/devanagari_dict.txt",
                },
            },
            "cls": {
                "model": "models.cls.v2.cls_cls_mv3",
                "ch": {"url": "zh_ocr_cls_v1.tar"},
            },
        },
        # "v2": {
        #     "det": {"ch": {"url": "zh_ocr_det_v2.tar"}},
        #     "rec": {
        #         "ch": {
        #             "url": "zh_ocr_rec_v2.tar",
        #             "dict": "utils/dict/chinese_sim_dict.txt",
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
        #             "dict": "utils/dict/chinese_sim_dict.txt",
        #         },
        #         "en": {
        #             "url": "en_ocr_rec_m2.tar",
        #             "dict": "utils/dict/en96_dict.txt",
        #         },
        #         "french": {
        #             "url": "fr_ocr_rec_m2.tar",
        #             "dict": "utils/dict/french_dict.txt",
        #         },
        #         "german": {
        #             "url": "de_ocr_rec_m2.tar",
        #             "dict": "utils/dict/german_dict.txt",
        #         },
        #         "korean": {
        #             "url": "ko_ocr_rec_m2.tar",
        #             "dict": "utils/dict/korean_dict.txt",
        #         },
        #         "japan": {
        #             "url": "ja_ocr_rec_m2.tar",
        #             "dict": "utils/dict/japan_dict.txt",
        #         },
        #         "chinese_cht": {
        #             "url": "ch_ocr_rec_m2.tar",
        #             "dict": "utils/dict/chinese_cht_dict.txt",
        #         },
        #         "ta": {
        #             "url": "ta_ocr_rec_m2.tar",
        #             "dict": "utils/dict/ta_dict.txt",
        #         },
        #         "te": {
        #             "url": "te_ocr_rec_m2.tar",
        #             "dict": "utils/dict/te_dict.txt",
        #         },
        #         "ka": {
        #             "url": "ka_ocr_rec_m2.tar",
        #             "dict": "utils/dict/ka_dict.txt",
        #         },
        #         "latin": {
        #             "url": "la_ocr_rec_v1.tar",
        #             "dict": "utils/dict/latin_dict.txt",
        #         },
        #         "arabic": {
        #             "url": "ar_ocr_rec_v1.tar",
        #             "dict": "utils/dict/arabic_dict.txt",
        #         },
        #         "cyrillic": {
        #             "url": "ru_ocr_rec_v1.tar",
        #             "dict": "utils/dict/cyrillic_dict.txt",
        #         },
        #         "devanagari": {
        #             "url": "hi_ocr_rec_v1.tar",
        #             "dict": "utils/dict/devanagari_dict.txt",
        #         },
        #         "structure": {
        #             "url": "en_tab_rec_v1.tar",
        #             "dict": "utils/dict/table_dict.txt",
        #         },
        #     },
        #     "cls": {"ch": {"url": "zh_ocr_cls_v1.tar"}},
        # },
    },
    "STRUCTURE": {
        "v1": {
            "table": {
                "en": {
                    "url": "en_tab_str_v1.tar",
                    "dict": "utils/dict/table_structure_dict.txt",
                }
            }
        },
        "v2": {
            "table": {
                "model": "models.tab.tab_slanet_pplcnet",
                "en": {
                    "url": "en_str_tab_m2.tar",
                    "dict": "utils/dict/table_structure_dict.txt",
                },
                "ch": {
                    "url": "zh_str_tab_m2.tar",
                    "dict": "utils/dict/table_structure_dict_ch.txt",
                },
            },
            # "layout": {
            #     "en": {
            #         "url": "en_lay_det_x1_picodet.tar",
            #         "dict": "utils/dict/layout_dict/layout_publaynet_dict.txt",
            #     },
            #     "ch": {
            #         "url": "ch_lay_det_x1_picodet.tar",
            #         "dict": "utils/dict/layout_dict/layout_cdla_dict.txt",
            #     },
            # },
        },
    },
}

import numpy
import torch
from loguru import logger
def p2t(tensor)->torch.Tensor:
    if isinstance(tensor,numpy.ndarray):
        return torch.from_numpy(tensor)
    return torch.from_numpy(tensor.numpy())

M = {
    "OCR": {
        "PP-OCRv3": {
            "det": {
                "ch": {"url": "zh_ocr_det_v3.tar"},
                "en": {"url": "en_ocr_det_v3.tar"},
                "ml": {"url": "ml_ocr_det_v3.tar"},
            },
            "rec": {
                "ch": {
                    "url": "zh_ocr_rec_v3.tar",
                    "dict_path": "./ptocr/utils/ppocr_keys_v1.txt",
                },
                "en": {
                    "url": "en_ocr_rec_v3.tar",
                    "dict_path": "./ppocr/utils/en_dict.txt",
                },
                "korean": {
                    "url": "ko_ocr_rec_v3.tar",
                    "dict_path": "./ppocr/utils/dict/korean_dict.txt",
                },
                "japan": {
                    "url": "ja_ocr_rec_v3.tar",
                    "dict_path": "./ppocr/utils/dict/japan_dict.txt",
                },
                "chinese_cht": {
                    "url": "ch_ocr_rec_v3.tar",
                    "dict_path": "./ppocr/utils/dict/chinese_cht_dict.txt",
                },
                "ta": {
                    "url": "ta_ocr_rec_v3.tar",
                    "dict_path": "./ppocr/utils/dict/ta_dict.txt",
                },
                "te": {
                    "url": "te_ocr_rec_v3.tar",
                    "dict_path": "./ppocr/utils/dict/te_dict.txt",
                },
                "ka": {
                    "url": "ka_ocr_rec_v3.tar",
                    "dict_path": "./ppocr/utils/dict/ka_dict.txt",
                },
                "latin": {
                    "url": "la_ocr_rec_v3.tar",
                    "dict_path": "./ppocr/utils/dict/latin_dict.txt",
                },
                "arabic": {
                    "url": "ar_ocr_rec_v3.tar",
                    "dict_path": "./ppocr/utils/dict/arabic_dict.txt",
                },
                "cyrillic": {
                    "url": "ru_ocr_rec_v3.tar",
                    "dict_path": "./ppocr/utils/dict/cyrillic_dict.txt",
                },
                "devanagari": {
                    "url": "hi_ocr_rec_v3.tar",
                    "dict_path": "./ppocr/utils/dict/devanagari_dict.txt",
                },
            },
            "cls": {"ch": {"url": "zh_ocr_cls_v1.tar"}},
        },
        "PP-OCRv2": {
            "det": {"ch": {"url": "zh_ocr_det_v2.tar"}},
            "rec": {
                "ch": {
                    "url": "zh_ocr_rec_v2.tar",
                    "dict_path": "./ptocr/utils/ppocr_keys_v1.txt",
                }
            },
            "cls": {"ch": {"url": "zh_ocr_cls_v1.tar"}},
        },
        "PP-OCR": {
            "det": {
                "ch": {"url": "zh_ocr_det_v1.tar"},
                "en": {"url": "en_ocr_det_v1.tar"},
                "structure": {"url": "en_tab_det_v1.tar"},
            },
            "rec": {
                "ch": {
                    "url": "zh_ocr_rec_v1.tar",
                    "dict_path": "./ptocr/utils/ppocr_keys_v1.txt",
                },
                "en": {
                    "url": "en_ocr_rec_m2.tar",
                    "dict_path": "./ppocr/utils/en_dict.txt",
                },
                "french": {
                    "url": "fr_ocr_rec_m2.tar",
                    "dict_path": "./ppocr/utils/dict/french_dict.txt",
                },
                "german": {
                    "url": "de_ocr_rec_m2.tar",
                    "dict_path": "./ppocr/utils/dict/german_dict.txt",
                },
                "korean": {
                    "url": "ko_ocr_rec_m2.tar",
                    "dict_path": "./ppocr/utils/dict/korean_dict.txt",
                },
                "japan": {
                    "url": "ja_ocr_rec_m2.tar",
                    "dict_path": "./ppocr/utils/dict/japan_dict.txt",
                },
                "chinese_cht": {
                    "url": "ch_ocr_rec_m2.tar",
                    "dict_path": "./ppocr/utils/dict/chinese_cht_dict.txt",
                },
                "ta": {
                    "url": "ta_ocr_rec_m2.tar",
                    "dict_path": "./ppocr/utils/dict/ta_dict.txt",
                },
                "te": {
                    "url": "te_ocr_rec_m2.tar",
                    "dict_path": "./ppocr/utils/dict/te_dict.txt",
                },
                "ka": {
                    "url": "ka_ocr_rec_m2.tar",
                    "dict_path": "./ppocr/utils/dict/ka_dict.txt",
                },
                "latin": {
                    "url": "la_ocr_rec_v1.tar",
                    "dict_path": "./ppocr/utils/dict/latin_dict.txt",
                },
                "arabic": {
                    "url": "ar_ocr_rec_v1.tar",
                    "dict_path": "./ppocr/utils/dict/arabic_dict.txt",
                },
                "cyrillic": {
                    "url": "ru_ocr_rec_v1.tar",
                    "dict_path": "./ppocr/utils/dict/cyrillic_dict.txt",
                },
                "devanagari": {
                    "url": "hi_ocr_rec_v1.tar",
                    "dict_path": "./ppocr/utils/dict/devanagari_dict.txt",
                },
                "structure": {
                    "url": "en_tab_rec_v1.tar",
                    "dict_path": "ppocr/utils/dict/table_dict.txt",
                },
            },
            "cls": {"ch": {"url": "zh_ocr_cls_v1.tar"}},
        },
    },
    "STRUCTURE": {
        "PP-Structure": {
            "table": {
                "en": {
                    "url": "en_tab_str_v1.tar",
                    "dict_path": "ppocr/utils/dict/table_structure_dict.txt",
                }
            }
        },
        "PP-StructureV2": {
            "table": {
                "en": {
                    "url": "en_tab_str_m2_slanet.tar",
                    "dict_path": "ppocr/utils/dict/table_structure_dict.txt",
                },
                "ch": {
                    "url": "zh_tab_str_m2_slanet.tar",
                    "dict_path": "ppocr/utils/dict/table_structure_dict_ch.txt",
                },
            },
            "layout": {
                "en": {
                    "url": "en_lay_det_x1_picodet.tar",
                    "dict_path": "ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt",
                },
                "ch": {
                    "url": "ch_lay_det_x1_picodet.tar",
                    "dict_path": "ppocr/utils/dict/layout_dict/layout_cdla_dict.txt",
                },
            },
        },
    },
}

import paddle
zh = paddle.load(r'D:\dev\.model\toddleocr\zh_ocr_det_v3\inference')
en = paddle.load(r'D:\dev\.model\toddleocr\en_ocr_det_v3\inference')
ml = paddle.load(r'D:\dev\.model\toddleocr\ml_ocr_det_v3\inference')

assert list(zh.keys())==list(en.keys())==list(ml.keys())

from ptocr.models.det.v3.det_db_mv3_rse import Model
# from ptocr.models.det.det_db_rvd import Model
md = Model().model.state_dict()
ignore = ['num_batches_tracked']


def parse_prefix(md):
    # global md_prefix, md_suffix, k, p, s
    md_prefix = set()
    md_suffix = set()
    for k in md:
        p, s = k.rsplit('.', 1)
        md_prefix.add(p)
        md_suffix.add(s)

    return md_prefix,md_suffix

md_prefix,md_suffix = parse_prefix(md)
print(md_prefix,md_suffix)

zh_prefix,zh_suffix = parse_prefix(zh)
print(zh_prefix,zh_suffix)

diff = zh_prefix.difference(md_prefix)
diff2 = md_prefix.difference(zh_prefix)

print('ZH:',diff)
print('MD:',diff2)
if diff2:
    logger.info(f'推理模型中不存在这些参数，请注意{diff2}')

diff_name={'running_mean':'_mean','running_var':'_variance'}
transpose='weight'
linear_prefix = []
for k in md:
    p, s = k.rsplit('.', 1)
    if s in diff_name:
        if (key:=p+'.'+diff_name[s]) in zh:
            md[k] = p2t(zh[key])
    elif s ==transpose and p in linear_prefix:
        if k in zh:
            md[k] = p2t(zh[k]).T
    else:
        if k in zh:
            md[k] = p2t(zh[k])

torch.save(md,r'D:\dev\.model\toddleocr\zh_ocr_det_v3\inference.pt')



# md_keys= [k for k in md if k.rsplit('.',1)[1] not in ignore]
#
# print(len(md_keys),len(zh))
#
# for k,v in zh.items():
#     print(k,v.shape)
#
# for k,v in md.items():
#     print(k,v.shape)

# def transmodel():
#     import torch
#     import paddle
#
#     def p2t(tensor) -> torch.Tensor:
#         return torch.from_numpy(tensor.numpy())
#     m = Model()
#     x = m.model.state_dict()
#     t_keys = [k for k in x.keys()]
#     print('模型参数', len(t_keys))
#     paddle_params = paddle.load('D:\\dev\\.model\\paddocr\\ch_ppocr_server_v2.0_det_train\\best_accuracy.pdparams')
#     print('文件参数', len(paddle_params))
#     keys = list(paddle_params.keys())
#     keys.sort()
#     print(t_keys)
#     print(keys)
#     print(len(t_keys), len(keys))
#     maps = {'bn.running_mean': '_batch_norm._mean', 'bn.running_var': '_batch_norm._variance', 'bn.bias': '_batch_norm.bias', 'bn.weight': '_batch_norm.weight', 'conv.weight': '_conv.weight'}
#     tset = set()
#     for k in t_keys:
#         (l, m, r) = k.rsplit('.', 2)
#         if m + '.' + r in maps:
#             k = l + '.' + maps[m + '.' + r]
#         tset.add(k)
#     pset = set(keys)
#     u = pset - tset
#     print(u)
#     print(len(u))
#     d = {}
#     transmap = {'_variance': 'running_var', '_mean': 'running_mean'}
#     d = dict()
#     for (k, v) in paddle_params.items():
#         (p, n) = k.rsplit('.', 1)
#         if n in transmap:
#             o = p + '.' + transmap[n]
#             d[o] = p2t(v)
#             if n == '_variance':
#                 d[p + '.num_batches_tracked'] = torch.tensor(200)
#         else:
#             d[k] = p2t(v)
#     torch.save(d, 'D:\\dev\\.model\\paddocr\\ch_ppocr_server_v2.0_det_train\\best_accuracy.pth')

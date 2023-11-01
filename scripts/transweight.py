"""
https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar
https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar
https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar
https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar
https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar
https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/arabic_PP-OCRv3_rec_infer.tar
https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/chinese_cht_PP-OCRv3_rec_infer.tar
https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_infer.tar
https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/devanagari_PP-OCRv3_rec_infer.tar
https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_infer.tar
https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ka_PP-OCRv3_rec_infer.tar
https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_infer.tar
https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_infer.tar
https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ta_PP-OCRv3_rec_infer.tar
https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/te_PP-OCRv3_rec_infer.tar
https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar
https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar
https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/arabic_ppocr_mobile_v2.0_rec_infer.tar
https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/chinese_cht_mobile_v2.0_rec_infer.tar
https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/cyrillic_ppocr_mobile_v2.0_rec_infer.tar
https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/devanagari_ppocr_mobile_v2.0_rec_infer.tar
https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_infer.tar
https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_ppocr_mobile_v2.0_det_infer.tar
https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/french_mobile_v2.0_rec_infer.tar
https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/german_mobile_v2.0_rec_infer.tar
https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/japan_mobile_v2.0_rec_infer.tar
https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ka_mobile_v2.0_rec_infer.tar
https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/korean_mobile_v2.0_rec_infer.tar
https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/latin_ppocr_mobile_v2.0_rec_infer.tar
https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ta_mobile_v2.0_rec_infer.tar
https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/te_mobile_v2.0_rec_infer.tar
https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_det_infer.tar
https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_rec_infer.tar
https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar
https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_cdla_infer.tar
https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar
https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/ch_ppstructure_mobile_v2.0_SLANet_infer.tar
https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar
"""

MODEL_URLS = {
    "OCR": {
        "PP-OCRv3": {
            "det": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar",
                },
                "en": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
                },
                "ml": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar"
                },
            },
            "rec": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar",
                    "dict": "./toddleocr/utils/dict/chinese_sim_dict.txt",
                },
                "en": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar",
                    "dict": "./ppocr/utils/dict/en96_dict.txt",
                },
                "korean": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_infer.tar",
                    "dict": "./ppocr/utils/dict/korean_dict.txt",
                },
                "japan": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_infer.tar",
                    "dict": "./ppocr/utils/dict/japan_dict.txt",
                },
                "chinese_cht": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/chinese_cht_PP-OCRv3_rec_infer.tar",
                    "dict": "./ppocr/utils/dict/chinese_cht_dict.txt",
                },
                "ta": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ta_PP-OCRv3_rec_infer.tar",
                    "dict": "./ppocr/utils/dict/ta_dict.txt",
                },
                "te": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/te_PP-OCRv3_rec_infer.tar",
                    "dict": "./ppocr/utils/dict/te_dict.txt",
                },
                "ka": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ka_PP-OCRv3_rec_infer.tar",
                    "dict": "./ppocr/utils/dict/ka_dict.txt",
                },
                "latin": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_infer.tar",
                    "dict": "./ppocr/utils/dict/latin_dict.txt",
                },
                "arabic": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/arabic_PP-OCRv3_rec_infer.tar",
                    "dict": "./ppocr/utils/dict/arabic_dict.txt",
                },
                "cyrillic": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_infer.tar",
                    "dict": "./ppocr/utils/dict/cyrillic_dict.txt",
                },
                "devanagari": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/devanagari_PP-OCRv3_rec_infer.tar",
                    "dict": "./ppocr/utils/dict/devanagari_dict.txt",
                },
            },
            "cls": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
                }
            },
        },
        "PP-OCRv2": {
            "det": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar",
                },
            },
            "rec": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar",
                    "dict": "./toddleocr/utils/dict/chinese_sim_dict.txt",
                }
            },
            "cls": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
                }
            },
        },
        "PP-OCR": {
            "det": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar",
                },
                "en": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_ppocr_mobile_v2.0_det_infer.tar",
                },
                "structure": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_det_infer.tar"
                },
            },
            "rec": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar",
                    "dict": "./toddleocr/utils/dict/chinese_sim_dict.txt",
                },
                "en": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_infer.tar",
                    "dict": "./ppocr/utils/dict/en96_dict.txt",
                },
                "french": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/french_mobile_v2.0_rec_infer.tar",
                    "dict": "./ppocr/utils/dict/french_dict.txt",
                },
                "german": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/german_mobile_v2.0_rec_infer.tar",
                    "dict": "./ppocr/utils/dict/german_dict.txt",
                },
                "korean": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/korean_mobile_v2.0_rec_infer.tar",
                    "dict": "./ppocr/utils/dict/korean_dict.txt",
                },
                "japan": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/japan_mobile_v2.0_rec_infer.tar",
                    "dict": "./ppocr/utils/dict/japan_dict.txt",
                },
                "chinese_cht": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/chinese_cht_mobile_v2.0_rec_infer.tar",
                    "dict": "./ppocr/utils/dict/chinese_cht_dict.txt",
                },
                "ta": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ta_mobile_v2.0_rec_infer.tar",
                    "dict": "./ppocr/utils/dict/ta_dict.txt",
                },
                "te": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/te_mobile_v2.0_rec_infer.tar",
                    "dict": "./ppocr/utils/dict/te_dict.txt",
                },
                "ka": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ka_mobile_v2.0_rec_infer.tar",
                    "dict": "./ppocr/utils/dict/ka_dict.txt",
                },
                "latin": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/latin_ppocr_mobile_v2.0_rec_infer.tar",
                    "dict": "./ppocr/utils/dict/latin_dict.txt",
                },
                "arabic": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/arabic_ppocr_mobile_v2.0_rec_infer.tar",
                    "dict": "./ppocr/utils/dict/arabic_dict.txt",
                },
                "cyrillic": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/cyrillic_ppocr_mobile_v2.0_rec_infer.tar",
                    "dict": "./ppocr/utils/dict/cyrillic_dict.txt",
                },
                "devanagari": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/devanagari_ppocr_mobile_v2.0_rec_infer.tar",
                    "dict": "./ppocr/utils/dict/devanagari_dict.txt",
                },
                "structure": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_rec_infer.tar",
                    "dict": "ppocr/utils/dict/table_dict.txt",
                },
            },
            "cls": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
                }
            },
        },
    },
    "STRUCTURE": {
        "PP-Structure": {
            "table": {
                "en": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar",
                    "dict": "ppocr/utils/dict/table_structure_dict.txt",
                }
            }
        },
        "PP-StructureV2": {
            "table": {
                "en": {
                    "url": "https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar",
                    "dict": "ppocr/utils/dict/table_structure_dict.txt",
                },
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/ch_ppstructure_mobile_v2.0_SLANet_infer.tar",
                    "dict": "ppocr/utils/dict/table_structure_dict_ch.txt",
                },
            },
            "layout": {
                "en": {
                    "url": "https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar",
                    "dict": "ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt",
                },
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_cdla_infer.tar",
                    "dict": "ppocr/utils/dict/layout_dict/layout_cdla_dict.txt",
                },
            },
        },
    },
}

# 指定需要重命名的文件夹路径
# folder_path = r'D:\dev\.model\toddleocr'
#
# # 遍历文件夹内的所有文件
# for filename in os.listdir(folder_path):
#     # 获取文件名和扩展名
#     name, ext = os.path.splitext(filename)
#     # 将文件名转换为小写
#     new_name = name.lower()
#     # 生成新的文件名（包括扩展名）
#     new_filename = new_name + ext
#     # 构建新的文件路径
#     old_path = os.path.join(folder_path, filename)
#     new_path = os.path.join(folder_path, new_filename)
#     # 重命名文件
#     print(new_filename)
# os.rename(old_path, new_path)
# print(f'Renamed file "{filename}" to "{new_filename}"')
keys = """
arabic_pp-ocrv3_rec_infer.tar 
arabic_ppocr_mobile_v2.0_rec_infer.tar 
chinese_cht_mobile_v2.0_rec_infer.tar 
chinese_cht_pp-ocrv3_rec_infer.tar 
ch_pp-ocrv2_det_infer.tar 
ch_pp-ocrv2_rec_infer.tar 
ch_pp-ocrv3_det_infer.tar 
ch_pp-ocrv3_rec_infer.tar 
ch_ppocr_mobile_v2.0_cls_infer.tar 
ch_ppocr_mobile_v2.0_det_infer.tar 
ch_ppocr_mobile_v2.0_rec_infer.tar 
ch_ppstructure_mobile_v2.0_slanet_infer.tar 
cyrillic_pp-ocrv3_rec_infer.tar 
cyrillic_ppocr_mobile_v2.0_rec_infer.tar 
devanagari_pp-ocrv3_rec_infer.tar 
devanagari_ppocr_mobile_v2.0_rec_infer.tar 
en_number_mobile_v2.0_rec_infer.tar 
en_pp-ocrv3_det_infer.tar 
en_pp-ocrv3_rec_infer.tar 
en_ppocr_mobile_v2.0_det_infer.tar 
en_ppocr_mobile_v2.0_table_det_infer.tar 
en_ppocr_mobile_v2.0_table_rec_infer.tar 
en_ppocr_mobile_v2.0_table_structure_infer.tar 
en_ppstructure_mobile_v2.0_slanet_infer.tar 
french_mobile_v2.0_rec_infer.tar 
german_mobile_v2.0_rec_infer.tar 
japan_mobile_v2.0_rec_infer.tar 
japan_pp-ocrv3_rec_infer.tar 
ka_mobile_v2.0_rec_infer.tar 
ka_pp-ocrv3_rec_infer.tar 
korean_mobile_v2.0_rec_infer.tar 
korean_pp-ocrv3_rec_infer.tar 
latin_pp-ocrv3_rec_infer.tar 
latin_ppocr_mobile_v2.0_rec_infer.tar 
multilingual_pp-ocrv3_det_infer.tar 
picodet_lcnet_x1_0_fgd_layout_cdla_infer.tar 
picodet_lcnet_x1_0_fgd_layout_infer.tar 
ta_mobile_v2.0_rec_infer.tar 
ta_pp-ocrv3_rec_infer.tar 
te_mobile_v2.0_rec_infer.tar 
te_pp-ocrv3_rec_infer.tar 
"""

vals = """
ar_ocr_rec_v3.tar
ar_ocr_rec_v1.tar
ch_ocr_rec_m2.tar
ch_ocr_rec_v3.tar
zh_ocr_det_v2.tar
zh_ocr_rec_v2.tar
zh_ocr_det_v3.tar
zh_ocr_rec_v3.tar
zh_ocr_cls_v1.tar
zh_ocr_det_v1.tar
zh_ocr_rec_v1.tar
zh_tab_str_m2_slanet.tar
ru_ocr_rec_v3.tar
ru_ocr_rec_v1.tar
hi_ocr_rec_v3.tar
hi_ocr_rec_v1.tar
en_ocr_rec_m2.tar
en_ocr_det_v3.tar
en_ocr_rec_v3.tar
en_ocr_det_v1.tar
en_tab_det_v1.tar
en_tab_rec_v1.tar
en_tab_str_v1.tar
en_tab_str_m2_slanet.tar
fr_ocr_rec_m2.tar
de_ocr_rec_m2.tar
ja_ocr_rec_m2.tar
ja_ocr_rec_v3.tar
ka_ocr_rec_m2.tar
ka_ocr_rec_v3.tar
ko_ocr_rec_m2.tar
ko_ocr_rec_v3.tar
la_ocr_rec_v3.tar
la_ocr_rec_v1.tar
ml_ocr_det_v3.tar
ch_lay_det_x1_picodet.tar
en_lay_det_x1_picodet.tar                
ta_ocr_rec_m2.tar
ta_ocr_rec_v3.tar
te_ocr_rec_m2.tar
te_ocr_rec_v3.tar
"""

# d = {}
# for k,v in zip(keys.splitlines(),vals.splitlines()):
#     d[k.strip()]=v.strip()
#
# folder_path = r'D:\dev\.model\toddleocr'


# # 遍历文件夹内的所有文件
# for filename in os.listdir(folder_path):
#     # 获取文件名和扩展名
#     name, ext = os.path.splitext(filename)
#     # 将文件名转换为小写
#     new_name = name.lower()
#     # 生成新的文件名（包括扩展名）
#     new_filename = d[new_name+'.tar']
#     # 构建新的文件路径
#     old_path = os.path.join(folder_path, filename)
#     new_path = os.path.join(folder_path, new_filename)
#     # 重命名文件
#     print(new_filename)
#     os.rename(old_path, new_path)
#     print(f'Renamed file "{filename}" to "{new_filename}"')
# def f(x):
#     for k,v in x.items():
#         if k=='url':
#             x[k] = d[x[k].lower().rsplit('/',1)[1]]
#         elif isinstance(v,dict):
#             f(v)
#
# f(MODEL_URLS)
#
# print(MODEL_URLS)
# M = {'OCR': {'PP-OCRv3': {'det': {'ch': {'url': 'zh_ocr_det_v3.tar'}, 'en': {'url': 'en_ocr_det_v3.tar'}, 'ml': {'url': 'ml_ocr_det_v3.tar'}}, 'rec': {'ch': {'url': 'zh_ocr_rec_v3.tar', 'dict_path': './toddleocr/utils/dict/chinese_sim_dict.txt'}, 'en': {'url': 'en_ocr_rec_v3.tar', 'dict_path': './ppocr/utils/dict/en96_dict.txt'}, 'korean': {'url': 'ko_ocr_rec_v3.tar', 'dict_path': './ppocr/utils/dict/korean_dict.txt'}, 'japan': {'url': 'ja_ocr_rec_v3.tar', 'dict_path': './ppocr/utils/dict/japan_dict.txt'}, 'chinese_cht': {'url': 'ch_ocr_rec_v3.tar', 'dict_path': './ppocr/utils/dict/chinese_cht_dict.txt'}, 'ta': {'url': 'ta_ocr_rec_v3.tar', 'dict_path': './ppocr/utils/dict/ta_dict.txt'}, 'te': {'url': 'te_ocr_rec_v3.tar', 'dict_path': './ppocr/utils/dict/te_dict.txt'}, 'ka': {'url': 'ka_ocr_rec_v3.tar', 'dict_path': './ppocr/utils/dict/ka_dict.txt'}, 'latin': {'url': 'la_ocr_rec_v3.tar', 'dict_path': './ppocr/utils/dict/latin_dict.txt'}, 'arabic': {'url': 'ar_ocr_rec_v3.tar', 'dict_path': './ppocr/utils/dict/arabic_dict.txt'}, 'cyrillic': {'url': 'ru_ocr_rec_v3.tar', 'dict_path': './ppocr/utils/dict/cyrillic_dict.txt'}, 'devanagari': {'url': 'hi_ocr_rec_v3.tar', 'dict_path': './ppocr/utils/dict/devanagari_dict.txt'}}, 'cls': {'ch': {'url': 'zh_ocr_cls_v1.tar'}}}, 'PP-OCRv2': {'det': {'ch': {'url': 'zh_ocr_det_v2.tar'}}, 'rec': {'ch': {'url': 'zh_ocr_rec_v2.tar', 'dict_path': './toddleocr/utils/dict/chinese_sim_dict.txt'}}, 'cls': {'ch': {'url': 'zh_ocr_cls_v1.tar'}}}, 'PP-OCR': {'det': {'ch': {'url': 'zh_ocr_det_v1.tar'}, 'en': {'url': 'en_ocr_det_v1.tar'}, 'structure': {'url': 'en_tab_det_v1.tar'}}, 'rec': {'ch': {'url': 'zh_ocr_rec_v1.tar', 'dict_path': './toddleocr/utils/dict/chinese_sim_dict.txt'}, 'en': {'url': 'en_ocr_rec_m2.tar', 'dict_path': './ppocr/utils/dict/en96_dict.txt'}, 'french': {'url': 'fr_ocr_rec_m2.tar', 'dict_path': './ppocr/utils/dict/french_dict.txt'}, 'german': {'url': 'de_ocr_rec_m2.tar', 'dict_path': './ppocr/utils/dict/german_dict.txt'}, 'korean': {'url': 'ko_ocr_rec_m2.tar', 'dict_path': './ppocr/utils/dict/korean_dict.txt'}, 'japan': {'url': 'ja_ocr_rec_m2.tar', 'dict_path': './ppocr/utils/dict/japan_dict.txt'}, 'chinese_cht': {'url': 'ch_ocr_rec_m2.tar', 'dict_path': './ppocr/utils/dict/chinese_cht_dict.txt'}, 'ta': {'url': 'ta_ocr_rec_m2.tar', 'dict_path': './ppocr/utils/dict/ta_dict.txt'}, 'te': {'url': 'te_ocr_rec_m2.tar', 'dict_path': './ppocr/utils/dict/te_dict.txt'}, 'ka': {'url': 'ka_ocr_rec_m2.tar', 'dict_path': './ppocr/utils/dict/ka_dict.txt'}, 'latin': {'url': 'la_ocr_rec_v1.tar', 'dict_path': './ppocr/utils/dict/latin_dict.txt'}, 'arabic': {'url': 'ar_ocr_rec_v1.tar', 'dict_path': './ppocr/utils/dict/arabic_dict.txt'}, 'cyrillic': {'url': 'ru_ocr_rec_v1.tar', 'dict_path': './ppocr/utils/dict/cyrillic_dict.txt'}, 'devanagari': {'url': 'hi_ocr_rec_v1.tar', 'dict_path': './ppocr/utils/dict/devanagari_dict.txt'}, 'structure': {'url': 'en_tab_rec_v1.tar', 'dict_path': 'ppocr/utils/dict/table_dict.txt'}}, 'cls': {'ch': {'url': 'zh_ocr_cls_v1.tar'}}}}, 'STRUCTURE': {'PP-Structure': {'table': {'en': {'url': 'en_tab_str_v1.tar', 'dict_path': 'ppocr/utils/dict/table_structure_dict.txt'}}}, 'PP-StructureV2': {'table': {'en': {'url': 'en_tab_str_m2_slanet.tar', 'dict_path': 'ppocr/utils/dict/table_structure_dict.txt'}, 'ch': {'url': 'zh_tab_str_m2_slanet.tar', 'dict_path': 'ppocr/utils/dict/table_structure_dict_ch.txt'}}, 'layout': {'en': {'url': 'en_lay_det_x1_picodet.tar', 'dict_path': 'ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt'}, 'ch': {'url': 'ch_lay_det_x1_picodet.tar', 'dict_path': 'ppocr/utils/dict/layout_dict/layout_cdla_dict.txt'}}}}}
import numpy
import torch
from loguru import logger


def p2t(tensor) -> torch.Tensor:
    if isinstance(tensor, numpy.ndarray):
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
                    "dict": "./toddleocr/utils/dict/chinese_sim_dict.txt",
                },
                "en": {
                    "url": "en_ocr_rec_v3.tar",
                    "dict": "./ppocr/utils/dict/en96_dict.txt",
                },
                "korean": {
                    "url": "ko_ocr_rec_v3.tar",
                    "dict": "./ppocr/utils/dict/korean_dict.txt",
                },
                "japan": {
                    "url": "ja_ocr_rec_v3.tar",
                    "dict": "./ppocr/utils/dict/japan_dict.txt",
                },
                "chinese_cht": {
                    "url": "ch_ocr_rec_v3.tar",
                    "dict": "./ppocr/utils/dict/chinese_cht_dict.txt",
                },
                "ta": {
                    "url": "ta_ocr_rec_v3.tar",
                    "dict": "./ppocr/utils/dict/ta_dict.txt",
                },
                "te": {
                    "url": "te_ocr_rec_v3.tar",
                    "dict": "./ppocr/utils/dict/te_dict.txt",
                },
                "ka": {
                    "url": "ka_ocr_rec_v3.tar",
                    "dict": "./ppocr/utils/dict/ka_dict.txt",
                },
                "latin": {
                    "url": "la_ocr_rec_v3.tar",
                    "dict": "./ppocr/utils/dict/latin_dict.txt",
                },
                "arabic": {
                    "url": "ar_ocr_rec_v3.tar",
                    "dict": "./ppocr/utils/dict/arabic_dict.txt",
                },
                "cyrillic": {
                    "url": "ru_ocr_rec_v3.tar",
                    "dict": "./ppocr/utils/dict/cyrillic_dict.txt",
                },
                "devanagari": {
                    "url": "hi_ocr_rec_v3.tar",
                    "dict": "./ppocr/utils/dict/devanagari_dict.txt",
                },
            },
            "cls": {"ch": {"url": "zh_ocr_cls_v1.tar"}},
        },
        "PP-OCRv2": {
            "det": {"ch": {"url": "zh_ocr_det_v2.tar"}},
            "rec": {
                "ch": {
                    "url": "zh_ocr_rec_v2.tar",
                    "dict": "./toddleocr/utils/dict/chinese_sim_dict.txt",
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
                    "dict": "./toddleocr/utils/dict/chinese_sim_dict.txt",
                },
                "en": {
                    "url": "en_ocr_rec_m2.tar",
                    "dict": "./ppocr/utils/dict/en96_dict.txt",
                },
                "french": {
                    "url": "fr_ocr_rec_m2.tar",
                    "dict": "./ppocr/utils/dict/french_dict.txt",
                },
                "german": {
                    "url": "de_ocr_rec_m2.tar",
                    "dict": "./ppocr/utils/dict/german_dict.txt",
                },
                "korean": {
                    "url": "ko_ocr_rec_m2.tar",
                    "dict": "./ppocr/utils/dict/korean_dict.txt",
                },
                "japan": {
                    "url": "ja_ocr_rec_m2.tar",
                    "dict": "./ppocr/utils/dict/japan_dict.txt",
                },
                "chinese_cht": {
                    "url": "ch_ocr_rec_m2.tar",
                    "dict": "./ppocr/utils/dict/chinese_cht_dict.txt",
                },
                "ta": {
                    "url": "ta_ocr_rec_m2.tar",
                    "dict": "./ppocr/utils/dict/ta_dict.txt",
                },
                "te": {
                    "url": "te_ocr_rec_m2.tar",
                    "dict": "./ppocr/utils/dict/te_dict.txt",
                },
                "ka": {
                    "url": "ka_ocr_rec_m2.tar",
                    "dict": "./ppocr/utils/dict/ka_dict.txt",
                },
                "latin": {
                    "url": "la_ocr_rec_v1.tar",
                    "dict": "./ppocr/utils/dict/latin_dict.txt",
                },
                "arabic": {
                    "url": "ar_ocr_rec_v1.tar",
                    "dict": "./ppocr/utils/dict/arabic_dict.txt",
                },
                "cyrillic": {
                    "url": "ru_ocr_rec_v1.tar",
                    "dict": "./ppocr/utils/dict/cyrillic_dict.txt",
                },
                "devanagari": {
                    "url": "hi_ocr_rec_v1.tar",
                    "dict": "./ppocr/utils/dict/devanagari_dict.txt",
                },
                "structure": {
                    "url": "en_tab_rec_v1.tar",
                    "dict": "ppocr/utils/dict/table_dict.txt",
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
                    "dict": "ppocr/utils/dict/table_structure_dict.txt",
                }
            }
        },
        "PP-StructureV2": {
            "table": {
                "en": {
                    "url": "en_tab_str_m2_slanet.tar",
                    "dict": "ppocr/utils/dict/table_structure_dict.txt",
                },
                "ch": {
                    "url": "zh_tab_str_m2_slanet.tar",
                    "dict": "ppocr/utils/dict/table_structure_dict_ch.txt",
                },
            },
            "layout": {
                "en": {
                    "url": "en_lay_det_x1_picodet.tar",
                    "dict": "ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt",
                },
                "ch": {
                    "url": "ch_lay_det_x1_picodet.tar",
                    "dict": "ppocr/utils/dict/layout_dict/layout_cdla_dict.txt",
                },
            },
        },
    },
}

import paddle


# from toddleocr.models.det.det_db_rvd import Model


def transmodel(model_cls,
        pdmodel,
        linear_prefix=(),
):
    """det model"""
    # global transpose
    md = model_cls().model.state_dict()
    zh = paddle.load(pdmodel)
    logger.info(zh.keys())

    def parse_prefix(md):
        # global md_prefix, md_suffix, k, p, s
        md_prefix = set()
        md_suffix = set()
        for k in md:
            p, s = k.rsplit('.', 1)
            md_prefix.add(p)
            md_suffix.add(s)

        return md_prefix, md_suffix

    md_prefix, md_suffix = parse_prefix(md)
    print(md_prefix, md_suffix)
    zh_prefix, zh_suffix = parse_prefix(zh)
    print(zh_prefix, zh_suffix)
    diff = zh_prefix.difference(md_prefix)
    diff2 = md_prefix.difference(zh_prefix)
    print('ZH:', diff)
    print('MD:', diff2)
    if diff2:
        logger.info(f'推理模型中不存在这些参数，请注意{diff2}')

    replace = {'running_mean': '_mean',
               'running_var': '_variance',
               'dw_conv.bn': '_depthwise_conv._batch_norm',
               'pw_conv.bn': '_pointwise_conv._batch_norm',
               'dw_conv.conv': "_depthwise_conv._conv",
               'pw_conv.conv': '_pointwise_conv._conv',
               'backbone.conv1.conv': 'backbone.conv1._conv',
               'backbone.conv1.bn': 'backbone.conv1._batch_norm',
               'head.ctc_encoder.encoder.conv1.bn': 'head.ctc_encoder.encoder.conv1.norm',
               'head.ctc_encoder.encoder.conv2.bn': 'head.ctc_encoder.encoder.conv2.norm',
               'head.ctc_encoder.encoder.conv3.bn': 'head.ctc_encoder.encoder.conv3.norm',
               'head.ctc_encoder.encoder.conv4.bn': 'head.ctc_encoder.encoder.conv4.norm',
               'head.ctc_encoder.encoder.conv1x1.bn': 'head.ctc_encoder.encoder.conv1x1.norm',
               '.se': '._se'
               }
    transpose = 'weight'
    # linear_prefix = []
    for k in md:
        p, s = k.rsplit('.', 1)
        if k in zh:
            md[k] = p2t(zh[k])
        elif key := indict(k, replace):
            if key in zh:
                md[k] = p2t(zh[key])
            else:
                logger.info(f'not found {key} in zh')

        if s == transpose and startswith(p, linear_prefix):
            md[k] = md[k].T

    torch.save(md, fr'{pdmodel}.pt')  # pickle_protocol=-1


def startswith(p, ls):
    for s in ls:
        if p.startswith(s):
            return True
    return False


def indict(p, dc):
    for k in dc:
        if k in p:
            p = p.replace(k, dc[k])
    return p




ls = ["ar_ocr_rec_v3",
      "ch_ocr_rec_v3",
      "zh_ocr_rec_v3",
      "ru_ocr_rec_v3",
      "hi_ocr_rec_v3",
      "en_ocr_rec_v3",
      "ja_ocr_rec_v3",
      "ka_ocr_rec_v3",
      "ko_ocr_rec_v3",
      "la_ocr_rec_v3",
      "ta_ocr_rec_v3",
      "te_ocr_rec_v3", ]

# M = {'OCR': {'PP-OCRv3': {'det': {'ch': {'url': 'zh_ocr_det_v3.tar'}, 'en': {'url': 'en_ocr_det_v3.tar'}, 'ml': {'url': 'ml_ocr_det_v3.tar'}}, 'rec': {'ch': {'url': 'zh_ocr_rec_v3.tar', 'dict_path': './toddleocr/utils/dict/chinese_sim_dict.txt'}, 'en': {'url': 'en_ocr_rec_v3.tar', 'dict_path': './ppocr/utils/dict/en96_dict.txt'}, 'korean': {'url': 'ko_ocr_rec_v3.tar', 'dict_path': './ppocr/utils/dict/korean_dict.txt'}, 'japan': {'url': 'ja_ocr_rec_v3.tar', 'dict_path': './ppocr/utils/dict/japan_dict.txt'}, 'chinese_cht': {'url': 'ch_ocr_rec_v3.tar', 'dict_path': './ppocr/utils/dict/chinese_cht_dict.txt'}, 'ta': {'url': 'ta_ocr_rec_v3.tar', 'dict_path': './ppocr/utils/dict/ta_dict.txt'}, 'te': {'url': 'te_ocr_rec_v3.tar', 'dict_path': './ppocr/utils/dict/te_dict.txt'}, 'ka': {'url': 'ka_ocr_rec_v3.tar', 'dict_path': './ppocr/utils/dict/ka_dict.txt'}, 'latin': {'url': 'la_ocr_rec_v3.tar', 'dict_path': './ppocr/utils/dict/latin_dict.txt'}, 'arabic': {'url': 'ar_ocr_rec_v3.tar', 'dict_path': './ppocr/utils/dict/arabic_dict.txt'}, 'cyrillic': {'url': 'ru_ocr_rec_v3.tar', 'dict_path': './ppocr/utils/dict/cyrillic_dict.txt'}, 'devanagari': {'url': 'hi_ocr_rec_v3.tar
# for i in ls:
from toddleocr.models.rec.v3.rec_svtr_mv1e import Model

transmodel(Model,
    rf'D:\dev\.model\toddleocr\te_ocr_rec_v3\inference',
    linear_prefix=("head.ctc_encoder.encoder.svtr_block", "head.ctc_head.fc"))

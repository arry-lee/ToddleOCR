#  Copyright (c) 2023. Arry Lee, <arry_lee@qq.com>

from engine import ToddleOCR
from utils.visual import draw_ocr_box_txt


def main():
    import sys

    t = ToddleOCR(
        det_model_dir="weights/zh_ocr_det_v3",
        cls_model_dir="weights/zh_ocr_cls_v1",
        rec_model_dir="weights/zh_ocr_rec_v3",
        tab_model_dir="weights/zh_str_tab_m2",
    )
    img = sys.argv[1]
    r = t.ocr(img, tab=False)[0]
    print(r)
    from PIL import Image

    im = Image.open(img)
    # boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[1])), (int(i[2]), int(i[3])), (int(i[0]), int(i[3]))] for i in
    #          r['boxes']]
    boxes = r["boxes"]
    print(boxes)
    res = draw_ocr_box_txt(im, boxes, [t[0] for t in r["rec_res"]])
    res.show()


if __name__ == "__main__":
    main()

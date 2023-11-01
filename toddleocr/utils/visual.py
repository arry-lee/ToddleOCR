import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_ser_results(
    image, ocr_results, font_path="doc/fonts/simfang.ttf", font_size=14
):
    np.random.seed(2021)
    color = (
        np.random.permutation(range(255)),
        np.random.permutation(range(255)),
        np.random.permutation(range(255)),
    )
    color_map = {
        idx: (color[0][idx], color[1][idx], color[2][idx]) for idx in range(1, 255)
    }
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, str) and os.path.isfile(image):
        image = Image.open(image).convert("RGB")
    img_new = image.copy()
    draw = ImageDraw.Draw(img_new)

    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    for ocr_info in ocr_results:
        if ocr_info["pred_id"] not in color_map:
            continue
        color = color_map[ocr_info["pred_id"]]
        text = "{}: {}".format(ocr_info["pred"], ocr_info["transcription"])

        if "bbox" in ocr_info:
            # draw with ocr engine
            bbox = ocr_info["bbox"]
        else:
            # draw with ocr groundtruth
            bbox = trans_poly_to_bbox(ocr_info["points"])
        draw_box_txt(bbox, text, draw, font, font_size, color)

    img_new = Image.blend(image, img_new, 0.7)
    return np.array(img_new)


def draw_box_txt(bbox, text, draw, font, font_size, color):
    # draw ocr results outline
    bbox = ((bbox[0], bbox[1]), (bbox[2], bbox[3]))
    draw.rectangle(bbox, fill=color)

    # draw ocr results
    tw = font.getsize(text)[0]
    th = font.getsize(text)[1]
    start_y = max(0, bbox[0][1] - th)
    draw.rectangle(
        [(bbox[0][0] + 1, start_y), (bbox[0][0] + tw + 1, start_y + th)],
        fill=(0, 0, 255),
    )
    draw.text((bbox[0][0] + 1, start_y), text, fill=(255, 255, 255), font=font)


def trans_poly_to_bbox(poly):
    x1 = np.min([p[0] for p in poly])
    x2 = np.max([p[0] for p in poly])
    y1 = np.min([p[1] for p in poly])
    y2 = np.max([p[1] for p in poly])
    return [x1, y1, x2, y2]


def draw_re_results(image, result, font_path="doc/fonts/simfang.ttf", font_size=18):
    np.random.seed(0)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, str) and os.path.isfile(image):
        image = Image.open(image).convert("RGB")
    img_new = image.copy()
    draw = ImageDraw.Draw(img_new)

    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    color_head = (0, 0, 255)
    color_tail = (255, 0, 0)
    color_line = (0, 255, 0)

    for ocr_info_head, ocr_info_tail in result:
        draw_box_txt(
            ocr_info_head["bbox"],
            ocr_info_head["transcription"],
            draw,
            font,
            font_size,
            color_head,
        )
        draw_box_txt(
            ocr_info_tail["bbox"],
            ocr_info_tail["transcription"],
            draw,
            font,
            font_size,
            color_tail,
        )

        center_head = (
            (ocr_info_head["bbox"][0] + ocr_info_head["bbox"][2]) // 2,
            (ocr_info_head["bbox"][1] + ocr_info_head["bbox"][3]) // 2,
        )
        center_tail = (
            (ocr_info_tail["bbox"][0] + ocr_info_tail["bbox"][2]) // 2,
            (ocr_info_tail["bbox"][1] + ocr_info_tail["bbox"][3]) // 2,
        )

        draw.line([center_head, center_tail], fill=color_line, width=5)

    img_new = Image.blend(image, img_new, 0.5)
    return np.array(img_new)


def draw_rectangle(img_path, boxes):
    boxes = np.array(boxes)
    img = cv2.imread(img_path)
    img_show = img.copy()
    for box in boxes.astype(int):
        x1, y1, x2, y2 = box
        cv2.rectangle(img_show, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return img_show


def table_view(image_file, pred_html, output):
    f_html = open(os.path.join(output, "show.html"), mode="w", encoding="utf-8")
    f_html.write("<html>\n<body>\n")
    f_html.write('<table border="1">\n')
    f_html.write(
        '<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />'
    )
    f_html.write("<tr>\n")
    f_html.write("<td>img name\n")
    f_html.write("<td>ori image</td>")
    f_html.write("<td>table html</td>")
    f_html.write("<td>cell box</td>")
    f_html.write("</tr>\n")
    f_html.write("<tr>\n")
    f_html.write(f"<td> {os.path.basename(image_file)} <br/>\n")
    f_html.write(f'<td><img src="{image_file}" width=640></td>\n')
    f_html.write(
        '<td><table  border="1">'
        + pred_html.replace("<html><body><table>", "").replace(
            "</table></body></html>", ""
        )
        + "</table></td>\n"
    )
    f_html.write(f'<td><img src="{os.path.basename(image_file)}" width=640></td>\n')
    f_html.write("</tr>\n")
    f_html.write("</table>\n")
    f_html.close()
    import webbrowser

    webbrowser.open(os.path.join(output, "show.html"))


def expand(pix, det_box, shape):
    x0, y0, x1, y1 = det_box
    h, w, c = shape
    tmp_x0 = x0 - pix
    tmp_x1 = x1 + pix
    tmp_y0 = y0 - pix
    tmp_y1 = y1 + pix
    x0_ = tmp_x0 if tmp_x0 >= 0 else 0
    x1_ = tmp_x1 if tmp_x1 <= w else w
    y0_ = tmp_y0 if tmp_y0 >= 0 else 0
    y1_ = tmp_y1 if tmp_y1 <= h else h
    return x0_, y0_, x1_, y1_

import math
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT = "simfang.ttf"


def draw_ser_results(image, ocr_results, font_path=DEFAULT_FONT, font_size=14):
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


def draw_re_results(image, result, font_path=DEFAULT_FONT, font_size=18):
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


def draw_e2e_res(dt_boxes, strs, img_path):
    src_im = cv2.imread(img_path)
    for box, str in zip(dt_boxes, strs):
        box = box.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
        cv2.putText(
            src_im,
            str,
            org=(int(box[0, 0, 0]), int(box[0, 0, 1])),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=0.7,
            color=(0, 255, 0),
            thickness=1,
        )
    return src_im


def draw_text_det_res(dt_boxes, img):
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(img, [box], True, color=(255, 255, 0), thickness=2)
    return img


def resize_img(img, input_size=600):
    """
    resize img and limit the longest side of the image to input_size
    """
    img = np.array(img)
    im_shape = img.shape
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(input_size) / float(im_size_max)
    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
    return img


def draw_ocr(
    image,
    boxes,
    txts=None,
    scores=None,
    drop_score=0.5,
    font_path=DEFAULT_FONT,
):
    """
    Visualize the results of OCR detection and recognition
    args:
        image(Image|array): RGB image
        boxes(list): boxes with shape(N, 4, 2)
        txts(list): the texts
        scores(list): txxs corresponding scores
        drop_score(float): only scores greater than drop_threshold will be visualized
        font_path: the path of font which is used to draw text
    return(array):
        the visualized img
    """
    if scores is None:
        scores = [1] * len(boxes)
    box_num = len(boxes)
    for i in range(box_num):
        if scores is not None and (scores[i] < drop_score or math.isnan(scores[i])):
            continue
        box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    if txts is not None:
        img = np.array(resize_img(image, input_size=600))
        txt_img = text_visual(
            txts,
            scores,
            img_h=img.shape[0],
            img_w=600,
            threshold=drop_score,
            font_path=font_path,
        )
        img = np.concatenate([np.array(img), np.array(txt_img)], axis=1)
        return img
    return image


def draw_ocr_box_txt(
    image,
    boxes,
    txts=None,
    scores=None,
    drop_score=0.5,
    font_path=DEFAULT_FONT,
):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = np.ones((h, w, 3), dtype=np.uint8) * 255
    random.seed(0)

    draw_left = ImageDraw.Draw(img_left)
    if txts is None or len(txts) != len(boxes):
        txts = [None] * len(boxes)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        draw_left.polygon(box, fill=color)
        img_right_text = draw_box_txt_fine((w, h), box, txt, font_path)
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_right_text, [pts], True, color, 1)
        img_right = cv2.bitwise_and(img_right, img_right_text)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new("RGB", (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(Image.fromarray(img_right), (w, 0, w * 2, h))
    return img_show


def draw_box_txt_fine(img_size, box, txt, font_path=DEFAULT_FONT):
    box_height = int(
        math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)
    )
    box_width = int(
        math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
    )

    if box_height > 2 * box_width and box_height > 30:
        img_text = Image.new("RGB", (box_height, box_width), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            font = create_font(txt, (box_height, box_width), font_path)
            draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)
        img_text = img_text.transpose(Image.ROTATE_270)
    else:
        img_text = Image.new("RGB", (box_width, box_height), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            font = create_font(txt, (box_width, box_height), font_path)
            draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)

    pts1 = np.float32(
        [[0, 0], [box_width, 0], [box_width, box_height], [0, box_height]]
    )
    pts2 = np.array(box, dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)

    img_text = np.array(img_text, dtype=np.uint8)
    img_right_text = cv2.warpPerspective(
        img_text,
        M,
        img_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return img_right_text


def create_font(txt, sz, font_path=DEFAULT_FONT):
    font_size = int(sz[1] * 0.99)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    length = font.getlength(txt)
    if length > sz[0]:
        font_size = int(font_size * sz[0] / length)
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    return font


def str_count(s):
    """
    Count the number of Chinese characters,
    a single English character and a single number
    equal to half the length of Chinese characters.
    args:
        s(string): the input of string
    return(int):
        the number of Chinese characters
    """
    import string

    count_zh = count_pu = 0
    s_len = len(s)
    en_dg_count = 0
    for c in s:
        if c in string.ascii_letters or c.isdigit() or c.isspace():
            en_dg_count += 1
        elif c.isalpha():
            count_zh += 1
        else:
            count_pu += 1
    return s_len - math.ceil(en_dg_count / 2)


def text_visual(
    texts,
    scores,
    img_h=400,
    img_w=600,
    threshold=0.0,
    font_path=DEFAULT_FONT,
):
    """
    create new blank img and draw txt on it
    args:
        texts(list): the text will be draw
        scores(list|None): corresponding score of each txt
        img_h(int): the height of blank img
        img_w(int): the width of blank img
        font_path: the path of font which is used to draw text
    return(array):
    """
    if scores is not None:
        assert len(texts) == len(
            scores
        ), "The number of txts and corresponding scores must match"

    def create_blank_img():
        blank_img = np.ones(shape=[img_h, img_w], dtype=np.int8) * 255
        blank_img[:, img_w - 1 :] = 0
        blank_img = Image.fromarray(blank_img).convert("RGB")
        draw_txt = ImageDraw.Draw(blank_img)
        return blank_img, draw_txt

    blank_img, draw_txt = create_blank_img()

    font_size = 20
    txt_color = (0, 0, 0)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    gap = font_size + 5
    txt_img_list = []
    count, index = 1, 0
    for idx, txt in enumerate(texts):
        index += 1
        if scores[idx] < threshold or math.isnan(scores[idx]):
            index -= 1
            continue
        first_line = True
        while str_count(txt) >= img_w // font_size - 4:
            tmp = txt
            txt = tmp[: img_w // font_size - 4]
            if first_line:
                new_txt = str(index) + ": " + txt
                first_line = False
            else:
                new_txt = "    " + txt
            draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
            txt = tmp[img_w // font_size - 4 :]
            if count >= img_h // gap - 1:
                txt_img_list.append(np.array(blank_img))
                blank_img, draw_txt = create_blank_img()
                count = 0
            count += 1
        if first_line:
            new_txt = str(index) + ": " + txt + "   " + "%.3f" % (scores[idx])
        else:
            new_txt = "  " + txt + "  " + "%.3f" % (scores[idx])
        draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
        # whether add new blank img or not
        if count >= img_h // gap - 1 and idx + 1 < len(texts):
            txt_img_list.append(np.array(blank_img))
            blank_img, draw_txt = create_blank_img()
            count = 0
        count += 1
    txt_img_list.append(np.array(blank_img))
    if len(txt_img_list) == 1:
        blank_img = np.array(txt_img_list[0])
    else:
        blank_img = np.concatenate(txt_img_list, axis=1)
    return np.array(blank_img)


def draw_boxes(image, boxes, scores=None, drop_score=0.5):
    if scores is None:
        scores = [1] * len(boxes)
    for box, score in zip(boxes, scores):
        if score < drop_score:
            continue
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    return image


def resize_image(im, max_side_len=512):
    """
    resize image to a size multiple of max_stride which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    """
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    if resize_h > resize_w:
        ratio = float(max_side_len) / resize_h
    else:
        ratio = float(max_side_len) / resize_w

    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    max_stride = 128
    resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
    resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
    im = cv2.resize(im, (int(resize_w), int(resize_h)))
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def resize_image_min(im, max_side_len=512):
    """ """
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    if resize_h < resize_w:
        ratio = float(max_side_len) / resize_h
    else:
        ratio = float(max_side_len) / resize_w

    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    max_stride = 128
    resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
    resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
    im = cv2.resize(im, (int(resize_w), int(resize_h)))
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)
    return im, (ratio_h, ratio_w)


def resize_image_for_totaltext(im, max_side_len=512):
    """ """
    h, w, _ = im.shape

    resize_w = w
    resize_h = h
    ratio = 1.25
    if h * ratio > max_side_len:
        ratio = float(max_side_len) / resize_h

    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    max_stride = 128
    resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
    resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
    im = cv2.resize(im, (int(resize_w), int(resize_h)))
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)
    return im, (ratio_h, ratio_w)


def point_pair2poly(point_pair_list):
    """
    Transfer vertical point_pairs into poly point in clockwise.
    """
    pair_length_list = []
    for point_pair in point_pair_list:
        pair_length = np.linalg.norm(point_pair[0] - point_pair[1])
        pair_length_list.append(pair_length)
    pair_length_list = np.array(pair_length_list)
    pair_info = (
        pair_length_list.max(),
        pair_length_list.min(),
        pair_length_list.mean(),
    )

    point_num = len(point_pair_list) * 2
    point_list = [0] * point_num
    for idx, point_pair in enumerate(point_pair_list):
        point_list[idx] = point_pair[0]
        point_list[point_num - 1 - idx] = point_pair[1]
    return np.array(point_list).reshape(-1, 2), pair_info


def shrink_quad_along_width(quad, begin_width_ratio=0.0, end_width_ratio=1.0):
    """
    Generate shrink_quad_along_width.
    """
    ratio_pair = np.array([[begin_width_ratio], [end_width_ratio]], dtype=np.float32)
    p0_1 = quad[0] + (quad[1] - quad[0]) * ratio_pair
    p3_2 = quad[3] + (quad[2] - quad[3]) * ratio_pair
    return np.array([p0_1[0], p0_1[1], p3_2[1], p3_2[0]])


def expand_poly_along_width(poly, shrink_ratio_of_width=0.3):
    """
    expand poly along width.
    """
    point_num = poly.shape[0]
    left_quad = np.array([poly[0], poly[1], poly[-2], poly[-1]], dtype=np.float32)
    left_ratio = (
        -shrink_ratio_of_width
        * np.linalg.norm(left_quad[0] - left_quad[3])
        / (np.linalg.norm(left_quad[0] - left_quad[1]) + 1e-6)
    )
    left_quad_expand = shrink_quad_along_width(left_quad, left_ratio, 1.0)
    right_quad = np.array(
        [
            poly[point_num // 2 - 2],
            poly[point_num // 2 - 1],
            poly[point_num // 2],
            poly[point_num // 2 + 1],
        ],
        dtype=np.float32,
    )
    right_ratio = 1.0 + shrink_ratio_of_width * np.linalg.norm(
        right_quad[0] - right_quad[3]
    ) / (np.linalg.norm(right_quad[0] - right_quad[1]) + 1e-6)
    right_quad_expand = shrink_quad_along_width(right_quad, 0.0, right_ratio)
    poly[0] = left_quad_expand[0]
    poly[-1] = left_quad_expand[-1]
    poly[point_num // 2 - 1] = right_quad_expand[1]
    poly[point_num // 2] = right_quad_expand[2]
    return poly


def norm2(x, axis=None):
    if axis:
        return np.sqrt(np.sum(x**2, axis=axis))
    return np.sqrt(np.sum(x**2))


def cos(p1, p2):
    return (p1 * p2).sum() / (norm2(p1) * norm2(p2))

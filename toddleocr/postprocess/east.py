import cv2
import numpy as np
import torch

from toddleocr.ops.locality_aware_nms import nms_locality

__all__ = ["EASTPostProcess"]


class EASTPostProcess:
    """
    The post process for EAST.
    """

    def __init__(self, score_thresh=0.8, cover_thresh=0.1, nms_thresh=0.2, **kwargs):
        self.score_thresh = score_thresh  # 分数阈值，用于过滤文本框的得分
        self.cover_thresh = cover_thresh  # 覆盖度阈值，用于过滤文本框的覆盖度
        self.nms_thresh = nms_thresh  # 非极大值抑制的阈值，用于合并文本框

    def restore_rectangle_quad(self, origin, geometry):
        """
        Restore rectangle from quadrangle.
        """
        # 将原始坐标按顺序复制4份，使其变为8维坐标 (n, 8)
        origin_concat = np.concatenate((origin, origin, origin, origin), axis=1)
        # 根据几何信息和原始坐标恢复文本框的四边形坐标
        pred_quads = origin_concat - geometry
        pred_quads = pred_quads.reshape((-1, 4, 2))  # 将形状重塑为 (n, 4, 2)
        return pred_quads

    def detect(
        self, score_map, geo_map, score_thresh=0.8, cover_thresh=0.1, nms_thresh=0.2
    ):
        """
        restore text boxes from score map and geo map
        """
        # 获取分数图和几何图的尺寸信息
        score_map = score_map[0]
        geo_map = np.swapaxes(geo_map, 1, 0)
        geo_map = np.swapaxes(geo_map, 1, 2)

        # 根据分数阈值过滤文本框
        xy_text = np.argwhere(score_map > score_thresh)
        if len(xy_text) == 0:
            return []

        # 根据y轴排序文本框
        xy_text = xy_text[np.argsort(xy_text[:, 0])]

        # 恢复四边形提议
        text_box_restored = self.restore_rectangle_quad(
            xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :]
        )

        # 构造包含文本框信息的数组
        boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = text_box_restored.reshape((-1, 8))
        boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]

        try:
            import lanms

            # 使用lanms库进行非极大值抑制，合并文本框
            boxes = lanms.merge_quadrangle_n9(boxes, nms_thresh)
        except:
            print(
                "you should install lanms by pip3 install lanms-nova to speed up nms_locality"
            )
            boxes = nms_locality(boxes.astype(np.float64), nms_thresh)

        if boxes.shape[0] == 0:
            return []

        # 根据平均分数图过滤一些低分文本框，与原论文不同
        for i, box in enumerate(boxes):
            mask = np.zeros_like(score_map, dtype=np.uint8)
            cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
            boxes[i, 8] = cv2.mean(score_map, mask)[0]

        # 根据覆盖度阈值过滤文本框
        boxes = boxes[boxes[:, 8] > cover_thresh]
        return boxes

    def sort_poly(self, p):
        """
        Sort polygons.
        """
        min_axis = np.argmin(np.sum(p, axis=1))
        p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
        if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
            return p
        else:
            return p[[0, 3, 2, 1]]

    def __call__(self, outs_dict, shape_list):
        score_list = outs_dict["f_score"]
        geo_list = outs_dict["f_geo"]
        if isinstance(score_list, torch.Tensor):
            score_list = score_list.numpy()
            geo_list = geo_list.numpy()
        img_num = len(shape_list)
        dt_boxes_list = []
        for ino in range(img_num):
            score = score_list[ino]
            geo = geo_list[ino]

            # 检测文本框
            boxes = self.detect(
                score_map=score,
                geo_map=geo,
                score_thresh=self.score_thresh,
                cover_thresh=self.cover_thresh,
                nms_thresh=self.nms_thresh,
            )
            boxes_norm = []
            if len(boxes) > 0:
                h, w = score.shape[1:]
                src_h, src_w, ratio_h, ratio_w = shape_list[ino]
                boxes = boxes[:, :8].reshape((-1, 4, 2))
                boxes[:, :, 0] /= ratio_w
                boxes[:, :, 1] /= ratio_h

                # 对文本框进行排序并添加到结果列表中
                for i_box, box in enumerate(boxes):
                    box = self.sort_poly(box.astype(np.int32))
                    if (
                        np.linalg.norm(box[0] - box[1]) < 5
                        or np.linalg.norm(box[3] - box[0]) < 5
                    ):
                        continue
                    boxes_norm.append(box)

            dt_boxes_list.append({"points": np.array(boxes_norm)})
        return dt_boxes_list

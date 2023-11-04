from ..utils.utility import get_dict
from ._det_eval import combine_results, get_score_A, get_score_B

__all__ = ["E2EMetric"]


class E2EMetric:
    def __init__(
        self,
        mode,
        gt_mat_dir,
        character_dict_path,
        main_indicator="f_score_e2e",
        **kwargs
    ):
        self.mode = mode
        self.gt_mat_dir = gt_mat_dir
        self.label_list = get_dict(character_dict_path)
        self.max_index = len(self.label_list)
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, batch, **kwargs):
        if self.mode == "A":
            gt_polyons_batch = batch[2]
            temp_gt_strs_batch = batch[3][0]
            ignore_tags_batch = batch[4]
            gt_strs_batch = []
            for temp_list in temp_gt_strs_batch:
                t = ""
                for index in temp_list:
                    if index < self.max_index:
                        t += self.label_list[index]
                gt_strs_batch.append(t)
            for (pred, gt_polyons, gt_strs, ignore_tags) in zip(
                [preds], gt_polyons_batch, [gt_strs_batch], ignore_tags_batch
            ):
                gt_info_list = [
                    {"points": gt_polyon, "text": gt_str, "ignore": ignore_tag}
                    for (gt_polyon, gt_str, ignore_tag) in zip(
                        gt_polyons, gt_strs, ignore_tags
                    )
                ]
                e2e_info_list = [
                    {"points": det_polyon, "texts": pred_str}
                    for (det_polyon, pred_str) in zip(pred["points"], pred["texts"])
                ]
                result = get_score_A(gt_info_list, e2e_info_list)
                self.results.append(result)
        else:
            img_id = batch[5][0]
            e2e_info_list = [
                {"points": det_polyon, "texts": pred_str}
                for (det_polyon, pred_str) in zip(preds["points"], preds["texts"])
            ]
            result = get_score_B(self.gt_mat_dir, img_id, e2e_info_list)
            self.results.append(result)

    def get_metric(self):
        metrics = combine_results(self.results)
        self.reset()
        return metrics

    def reset(self):
        self.results = []

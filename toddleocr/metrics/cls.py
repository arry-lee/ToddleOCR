__all__ = ["ClsMetric"]


class ClsMetric:
    def __init__(self, main_indicator="acc", **kwargs):
        self.main_indicator = main_indicator
        self.eps = 1e-05
        self.reset()

    def __call__(self, pred_label, *args, **kwargs):
        (preds, labels) = pred_label
        correct_num = 0
        all_num = 0
        for ((pred, pred_conf), (target, _)) in zip(preds, labels):
            if pred == target:
                correct_num += 1
            all_num += 1
        self.correct_num += correct_num
        self.all_num += all_num
        return {"acc": correct_num / (all_num + self.eps)}

    def get_metric(self):
        acc = self.correct_num / (self.all_num + self.eps)
        self.reset()
        return {"acc": acc}

    def reset(self):
        self.correct_num = 0
        self.all_num = 0

__all__ = ["VQASerTokenMetric"]


class VQASerTokenMetric:
    def __init__(self, main_indicator="hmean", **kwargs):
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, batch, **kwargs):
        preds, labels = preds
        self.pred_list.extend(preds)
        self.gt_list.extend(labels)

    def get_metric(self):
        from seqeval.metrics import f1_score, precision_score, recall_score

        metrics = {
            "precision": precision_score(self.gt_list, self.pred_list),
            "recall": recall_score(self.gt_list, self.pred_list),
            "hmean": f1_score(self.gt_list, self.pred_list),
        }
        self.reset()
        return metrics

    def reset(self):
        self.pred_list = []
        self.gt_list = []

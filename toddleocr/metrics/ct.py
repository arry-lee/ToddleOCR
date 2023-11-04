from ._det_eval import combine_results, get_score_C

__all__ = ["CTMetric"]


class CTMetric:
    def __init__(self, main_indicator, delimiter="\t", **kwargs):
        self.delimiter = delimiter
        self.main_indicator = main_indicator
        self.reset()

    def reset(self):
        self.results = []

    def __call__(self, preds, batch, **kwargs):
        assert len(preds) == 1, "CentripetalText test now only support batch_size=1."
        label = batch[2]
        text = batch[3]
        pred = preds[0]["points"]
        result = get_score_C(label, text, pred)
        self.results.append(result)

    def get_metric(self):
        metrics = combine_results(self.results, rec_flag=False)
        self.reset()
        return metrics

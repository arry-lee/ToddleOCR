import importlib

__all__ = ["DistillationMetric"]


class DistillationMetric:
    def __init__(self, key=None, base_metric_name=None, main_indicator=None, **kwargs):
        self.main_indicator = main_indicator
        self.key = key
        self.main_indicator = main_indicator
        self.base_metric_name = base_metric_name
        self.kwargs = kwargs
        self.metrics = None

    def _init_metrics(self, preds):
        self.metrics = dict()
        mod = importlib.import_module(__name__)
        for key in preds:
            self.metrics[key] = getattr(mod, self.base_metric_name)(
                main_indicator=self.main_indicator, **self.kwargs
            )
            self.metrics[key].reset()

    def __call__(self, preds, batch, **kwargs):
        assert isinstance(preds, dict)
        if self.metrics is None:
            self._init_metrics(preds)
        for key in preds:
            self.metrics[key].__call__(preds[key], batch, **kwargs)

    def get_metric(self):
        output = dict()
        for key in self.metrics:
            metric = self.metrics[key].get_metric()
            if key == self.key:
                output.update(metric)
            else:
                for sub_key in metric:
                    output["{}_{}".format(key, sub_key)] = metric[sub_key]
        return output

    def reset(self):
        for key in self.metrics:
            self.metrics[key].reset()

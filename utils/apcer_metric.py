import numpy as np

from utils.metric import Metric


class APCERMetric(Metric):

    def _compute_metric(self):
        basic_metrics = self._compute_basic_metrics()

        APCER = basic_metrics["false_positives"] / (basic_metrics["true_positives"] + basic_metrics["false_positives"])
        return APCER

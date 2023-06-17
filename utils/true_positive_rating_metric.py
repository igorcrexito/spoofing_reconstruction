import numpy as np

from utils.metric import Metric


class TPRMetric(Metric):

    def _compute_metric(self):
        basic_metrics = self._compute_basic_metrics()

        TPR = basic_metrics["true_positives"] / (basic_metrics["true_positives"] + basic_metrics["false_negatives"])
        return TPR

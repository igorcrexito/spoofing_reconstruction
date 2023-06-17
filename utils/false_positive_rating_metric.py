import numpy as np

from utils.metric import Metric


class FPRMetric(Metric):

    def _compute_metric(self):
        basic_metrics = self._compute_basic_metrics()

        FPR = basic_metrics["false_positives"] / (basic_metrics["false_positives"] + basic_metrics["true_negatives"])
        return FPR

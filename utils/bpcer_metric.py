import numpy as np

from utils.metric import Metric


class BPCERMetric(Metric):

    def _compute_metric(self):
        basic_metrics = self._compute_basic_metrics()

        BPCER = basic_metrics["false_negatives"] / (basic_metrics["true_positives"] + basic_metrics["false_negatives"])
        return BPCER

import numpy as np

from utils.metric import Metric


class FRRMetric(Metric):

    def _compute_metric(self):
        basic_metrics = self._compute_basic_metrics()

        FRR = basic_metrics["false_negatives"] / (basic_metrics['true_negatives'] + basic_metrics['false_negatives'])
        return FRR

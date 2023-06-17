import numpy as np

from utils.metric import Metric


class NPCERMetric(Metric):

    def _compute_metric(self):
        basic_metrics = self._compute_basic_metrics()

        NPCER = basic_metrics["false_negatives"] / (basic_metrics["false_negatives"] + basic_metrics["true_positives"])
        return NPCER

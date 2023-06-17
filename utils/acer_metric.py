import numpy as np

from utils.metric import Metric


class ACERMetric(Metric):

    def _compute_metric(self):
        basic_metrics = self._compute_basic_metrics()

        APCER = basic_metrics["false_positives"] / (basic_metrics["true_positives"] + basic_metrics["false_positives"])
        NPCER = basic_metrics["false_negatives"] / (basic_metrics["false_negatives"] + basic_metrics["true_positives"])

        ACER = (APCER + NPCER)/2
        return ACER

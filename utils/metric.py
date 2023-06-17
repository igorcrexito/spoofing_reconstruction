import numpy as np


class Metric:

    def __init__(self, metric_name: str, input_data: np.ndarray, label_data: np.ndarray):
        self.metric_name = metric_name
        self.input_data = input_data
        self.label_data = label_data
        self.metric_value = self._compute_metric()

    def _compute_basic_metrics(self):
        ## computing TP, FP, TN, FN
        TP, FP, TN, FN = 0, 0, 0, 0

        for index, data in enumerate(self.input_data):
            label = self.label_data[index]

            if label == 1:
                if data == 1:
                    TP += 1
                else:
                    FN += 1
            else:
                if data == 1:
                    FP += 1
                else:
                    TN += 1

        return {"true_positives": TP, "false_positives": FP, "true_negatives": TN, "false_negatives": FN}

    def _compute_metric(self):
        pass

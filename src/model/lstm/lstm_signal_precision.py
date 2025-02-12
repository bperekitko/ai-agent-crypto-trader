from typing import List

import tensorflow as tf
from keras import Metric

AVERAGE_SELECTED_CLASS_PRECISION_METRIC_NAME = "average_selected_classes_precision"

class AverageSelectedClassPrecision(Metric):
    def __init__(self, class_ids: List[int], **kwargs):
        super().__init__(name=AVERAGE_SELECTED_CLASS_PRECISION_METRIC_NAME, **kwargs)
        self.class_ids = class_ids

        self.true_positives = [self.add_weight(name=f"tp_{i}", initializer="zeros") for i in range(len(class_ids))]
        self.predicted_positives = [self.add_weight(name=f"pp_{i}", initializer="zeros") for i in range(len(class_ids))]

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_labels = tf.argmax(y_true, axis=-1)  #  returns index of max value - true class value
        y_pred_labels = tf.argmax(y_pred, axis=-1)  # same as above but for predicted values

        for i, class_id in enumerate(self.class_ids):
            y_true_class = tf.cast(tf.equal(y_true_labels, class_id), tf.float32) # checks if true class is the class_id
            y_pred_class = tf.cast(tf.equal(y_pred_labels, class_id), tf.float32) # checks if predicted class is the class_id

            tp = tf.reduce_sum(y_true_class * y_pred_class)  # Calculating TP - y_true_class * y_pred_class = 1 when both are equal to 1.0
            pp = tf.reduce_sum(y_pred_class)  # Calculating PP (TP + FP) - all predictions of class_id

            self.true_positives[i].assign_add(tp)
            self.predicted_positives[i].assign_add(pp)

    def result(self):
        precision_per_class = [tf.math.divide_no_nan(tp, pp) for tp, pp in zip(self.true_positives, self.predicted_positives)]
        return tf.reduce_mean(precision_per_class)

    def reset_state(self):
        for i in range(len(self.class_ids)):
            self.true_positives[i].assign(0)
            self.predicted_positives[i].assign(0)

import numpy as np


class BinaryLabelEncoder(object):

    def __init__(self):
        self._labels_dict = dict()
        self._inverse_labels_dict = dict()

    def fit(self, neg, pos):
        self._labels_dict[pos] = 1
        self._labels_dict[neg] = 0
        self._inverse_labels_dict = {v: k for k, v in self._labels_dict.items()}

    def transform(self, class_list):
        result = []
        for item in class_list:
            item = item.strip('"')
            if item in self._labels_dict:
                result.append(self._labels_dict[item])
            else:
                result.append(int(item))
        return np.asarray(result)

    def inverse_transform(self, class_list):
        result = []
        for item in class_list:
            result.append(self._inverse_labels_dict[item])
        return result


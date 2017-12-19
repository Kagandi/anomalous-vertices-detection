import turicreate as tc
from itertools import izip
import numpy as np

from anomalous_vertices_detection.learners import AbstractLearner
from anomalous_vertices_detection.utils.dataset import DataSetFactory, DataSet
from anomalous_vertices_detection.utils.exceptions import *


class GlLearner(AbstractLearner):
    def __init__(self, classifier=None, labels=None):
        super(GlLearner, self).__init__(classifier)
        if len(labels) == 2:
            self._labels = labels
        else:
            raise NonBinaryLabels("Must be only two labels, negative and positive")
        self._base_classifier = classifier

    def convert_data_to_format(self, features, labels=None, feature_id_col_name=None, metadata_cols=None):
        return DataSetFactory().convert_data_to_graphlab_format(features, labels, feature_id_col_name, metadata_cols,
                                                                self._labels)

    def set_boosted_trees_classifier(self):
        return GlLearner(tc.boosted_trees_classifier.create, self._labels)

    def set_svm_classifier(self):
        return GlLearner(tc.svm_classifier.create, self._labels)

    def set_neuralnet_classifier(self):
        return GlLearner(tc.neuralnet_classifier.create, self._labels)

    def set_randomforest_classifier(self):
        return GlLearner(tc.random_forest_classifier.create, self._labels)

    def train_classifier(self, dataset, **kwargs):
        self._classifier = self._classifier(dataset.features, target=dataset.labels)
        print(self.get_evaluation(dataset))
        return self

    def get_prediction(self, prediction_data):
        return self._classifier.predict(prediction_data.features)

    def get_evaluation(self, data):
        return self._classifier.evaluate(data.features)

    def get_prediction_probabilities(self, prediction_data):
        # print self.get_evaluation(prediction_data)
        return self._classifier.predict(prediction_data.features, output_type='probability')

    def split_kfold(self, data, labels=None, n_folds=10):
        pos_data = data[data["label"] == 0]
        neg_data = data[data["label"] == 1]
        pos_split = np.arange(pos_data.num_rows())
        neg_split = np.arange(neg_data.num_rows())
        np.random.shuffle(pos_split)
        np.random.shuffle(neg_split)
        pos_split = np.array_split(pos_split, n_folds)
        neg_split = np.array_split(neg_split, n_folds)
        neg_split_copy = np.copy(neg_split)
        pos_split_copy = np.copy(pos_split)
        for i, item in enumerate(izip(neg_split, pos_split)):
            neg_item, pos_item = item
            pos_split_copy = np.delete(pos_split_copy, i, 0)
            neg_split_copy = np.delete(neg_split_copy, i, 0)
            yield np.hstack(pos_split_copy + neg_split_copy), neg_item + pos_item
            pos_split_copy = np.insert(pos_split_copy, i, pos_item, axis=0)
            neg_split_copy = np.insert(neg_split_copy, i, neg_item, axis=0)

    def split_k(self, data, n_folds=10):
        x = np.arange(data.num_rows())
        np.random.shuffle(x)
        split = np.array_split(x, n_folds)
        split_copy = np.copy(split)
        for i, item in enumerate(split):
            temp = item
            split_copy = np.delete(split_copy, i, 0)
            yield np.hstack(split_copy), item
            split_copy = np.insert(split_copy, i, item, axis=0)

    def classify_by_links_probability(self, probas, features_ids, labels=None, threshold=0.78):
        """
        @todo add metadata usgae
        Parameters
        ----------
        probas
        features_ids
        labels
        threshold

        Returns
        -------

        """
        train_df = tc.SFrame()
        train_df["probas"] = probas
        train_df["src_id"] = features_ids
        train_df["link_label"] = train_df["probas"].apply(lambda avg: 1 if avg >= threshold else 0)
        train_df = train_df.groupby("src_id", operations={"prob": tc.aggregate.MEAN('probas'),
                                                          "median_prob": tc.aggregate.QUANTILE('probas', 0.5),
                                                          "mean_link_label": tc.aggregate.MEAN('link_label'),
                                                          "sum_link_label": tc.aggregate.SUM('link_label'),
                                                          "STDV_probability": tc.aggregate.STDV('probas'),
                                                          "STDV_predicted_label": tc.aggregate.STDV('link_label'),
                                                          "var_probability": tc.aggregate.VAR('probas'),
                                                          "var_predicted_label": tc.aggregate.VAR('link_label'),
                                                          "count": tc.aggregate.COUNT('link_label')})
        # .agg({0:['mean', 'count'], 1:'mean', "link_label":['mean', 'sum']})
        train_df["predicted"] = train_df["prob"].apply(
            lambda avg: labels["pos"] if avg >= threshold else labels["neg"])
        train_df["median_prob"] = train_df["median_prob"].apply(lambda median_prob: median_prob[0])
        return train_df
        # return train_df.

    def merge_with_labels(self, classified, labels_path, merge_col_name="id", default_label='real'):
        node_labels = tc.SFrame.read_csv(labels_path, column_type_hints={"id": str})
        merged_data = classified.join(node_labels, on={'src_id': merge_col_name}, how='left')
        merged_data = merged_data.fillna("label", default_label)
        merged_data = merged_data.rename({"label": "actual"})
        merged_data['actual'] = merged_data['actual'] == self._labels['pos']
        return merged_data

    def validate_prediction_by_links(self, result):
        targets, predictions, probas = result["actual"], result["predicted"], \
                                       result["prob"]
        precision = tc.evaluation.precision(targets, predictions)
        accuracy = tc.evaluation.accuracy(targets, predictions)
        recall = tc.evaluation.recall(targets, predictions)
        auc = tc.evaluation.auc(targets, predictions)
        return {"recall": recall,
                "precision": precision,
                "accuracy": accuracy,
                "auc": auc
                }

    # def cross_validate(self, dataset, n_folds=10):
    #     folds = self.split_kfold(dataset.features, n_folds)
    #     params = {'target': dataset.labels}
    #     job = tc.cross_validation.cross_val_score(folds,
    #                                               self._base_classifier,
    #                                               params)
    #     res = job.get_results()["summary"]
    #     return {"validation_accuracy": res["validation_accuracy"].mean(),
    #             "training_accuracy": res["training_accuracy"].mean()}

    def get_classification_metrics(self, l_test, prediction, probas):
        # fpr, tpr, thresholds = roc_curve(l_test, prediction)
        false_positive = float(
            len(np.where(l_test - prediction == -1)[0]))  # 0 (truth) - 1 (prediction) == -1 which is a false positive
        true_negative = float(
            len(np.where(l_test + prediction == 0)[0]))  # 0 (truth) - 0 (prediction) == 0 which is a true positive
        return {"fpr": false_positive / (true_negative + false_positive),
                "tnr": true_negative / (true_negative + false_positive)
                }

    def cross_validate(self, dataset, n_folds=10):
        roc_auc, recall, precision, accuracy, fpr, tpr, tnr = [], [], [], [], [], [], []
        dataset.features['id'] = range(dataset.features.num_rows())
        for train_index, test_index in self.split_kfold(dataset.features, dataset.labels, n_folds):
            f_train, f_test = DataSet(dataset.features.filter_by(train_index, 'id'), 'label'), DataSet(
                dataset.features.filter_by(test_index, 'id'), 'label')
            # l_train, l_test = dataset.labels.filter_by(train_index, 'id'), dataset.labels.filter_by(test_index, 'id')

            # l_train, l_test = dataset.labels[train_index], dataset.labels[test_index]
            # prediction = self.train_classifier(DataSet(f_train, l_train)).predict(f_test)
            cross_val = self.set_randomforest_classifier()
            prediction = cross_val.train_classifier(f_train).get_prediction(f_test)
            probas = cross_val.get_prediction_probabilities(f_test)
            metrics = cross_val.get_classification_metrics(f_test.features['label'], prediction, probas)
            fpr.append(metrics["fpr"])
            tnr.append(metrics["tnr"])
        # return {"auc": np.mean(roc_auc)}
        # print classification_report(l_test, prediction)
        # print "Predicted: 0   1"
        # print confusion_matrix(l_test, prediction)
        return {"fpr": np.mean(fpr), "tnr": np.mean(tnr)}

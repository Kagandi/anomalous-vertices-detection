import numpy as np
import pandas as pd
from sklearn import svm, tree, ensemble, feature_extraction, preprocessing
from sklearn.metrics import recall_score, precision_score, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from anomalous_vertices_detection.configs.config import *
from anomalous_vertices_detection.learners import AbstractLearner
from anomalous_vertices_detection.utils.dataset import DataSetFactory, DataSet
from anomalous_vertices_detection.utils.exceptions import *


def dict_to_array(my_dict):
    vec = feature_extraction.DictVectorizer()
    return vec.fit_transform(my_dict).toarray()


def encode_labels(labels):
    le = preprocessing.LabelEncoder()
    return le.fit_transform(labels)


class SkLearner(AbstractLearner):
    def __init__(self, classifier=None, labels=None):
        super(SkLearner, self).__init__(classifier)
        if labels is not None and isinstance(labels, dict):
            if len(labels) == 2:
                self.fit_labels([labels['neg'], labels['pos']])
            else:
                raise NonBinaryLabels("Must be only two labels, negative and positive")
        else:
            self._label_encoder = labels

    def merge_with_labels(self, classified, labels_path, merge_col_name="id", default_label=0):
        node_labels = pd.read_csv(labels_path, dtype={"id": str})
        labels = node_labels.pop("label").values
        # labels = self.transform_labels(labels)
        node_labels["actual"] = labels
        merged_data = pd.merge(classified, node_labels, left_on='src_id', right_on=merge_col_name, how='left')
        merged_data["actual"].fillna(default_label, inplace=True)
        merged_data["actual"] = self.transform_labels(merged_data["actual"])
        return merged_data

    @staticmethod
    def fit_labels(labels):
        return label_encoder.fit(*labels)

    @staticmethod
    def transform_labels(labels):
        return label_encoder.transform(labels)

    @staticmethod
    def inverse_transform_labels(labels):
        return label_encoder.inverse_transform(labels)

    def convert_data_to_format(self, features, labels=None, feature_id_col_name=None, metadata_cols=None):
        return DataSetFactory().convert_data_to_sklearn_format(features, labels, feature_id_col_name, metadata_cols)

    def set_decision_tree_classifier(self, tree_number=100):
        return SkLearner(tree.DecisionTreeClassifier())

    def set_svm_classifier(self):
        return SkLearner(svm.SVC(), self._label_encoder)

    def set_randomforest_classifier(self):
        return SkLearner(ensemble.RandomForestClassifier(n_estimators=100, criterion="entropy"))

    def set_adaboost_classifier(self):
        return SkLearner(ensemble.AdaBoostClassifier())

    def set_bagging_classifier(self):
        return SkLearner(ensemble.BaggingClassifier(tree.DecisionTreeClassifier()))

    def train_classifier(self, dataset):
        self._classifier = self._classifier.fit(dataset.features, dataset.labels)
        return self

    def get_prediction(self, prediction_data):
        if isinstance(prediction_data, DataSet):
            return self._classifier.predict(prediction_data.features)
        else:
            return self._classifier.predict(prediction_data)

    def get_prediction_probabilities(self, prediction_data):
        if isinstance(prediction_data, DataSet):
            return self._classifier.predict_proba(prediction_data.features)
        else:
            return self._classifier.predict_proba(prediction_data)

    def split_kfold(self, features, labels=None, n_folds=10):
        skf = StratifiedKFold(n_folds)
        for train_index, test_index in skf.split(features, labels):
            yield train_index, test_index
            # return cross_validation.StratifiedKFold(labels, n_folds)

    def get_classification_metrics(self, l_test, prediction, probas):
        # fpr, tpr, thresholds = roc_curve(l_test, prediction)
        false_positive = float(
            len(np.where(l_test - prediction == -1)[0]))  # 0 (truth) - 1 (prediction) == -1 which is a false positive
        true_negative = float(
            len(np.where(l_test + prediction == 0)[0]))  # 0 (truth) - 0 (prediction) == 0 which is a true positive
        return {"auc": roc_auc_score(l_test, probas),
                "recall": recall_score(l_test, prediction),
                "precision": precision_score(l_test, prediction),
                "accuracy": accuracy_score(l_test, prediction),
                "fpr": false_positive / (true_negative + false_positive),
                "tnr": true_negative / (true_negative + false_positive)
                }

    def cross_validate(self, dataset, n_folds=10):
        roc_auc, recall, precision, accuracy, fpr, tpr, tnr = [], [], [], [], [], [], []
        for train_index, test_index in self.split_kfold(dataset.features, dataset.labels, n_folds):
            f_train, f_test = dataset.features[train_index], dataset.features[test_index]
            l_train, l_test = dataset.labels[train_index], dataset.labels[test_index]
            # prediction = self.train_classifier(DataSet(f_train, l_train)).predict(f_test)
            prediction = self.train_classifier(DataSet(f_train, l_train)).get_prediction(f_test)
            probas = self.train_classifier(DataSet(f_train, l_train)).get_prediction_probabilities(f_test)[:, 1]
            metrics = self.get_classification_metrics(l_test, prediction, probas)
            roc_auc.append(metrics["auc"])
            recall.append(metrics["recall"])  # TPR
            precision.append(metrics["precision"])
            accuracy.append(metrics["accuracy"])
            fpr.append(metrics["fpr"])
            tnr.append(metrics["tnr"])
        # return {"auc": np.mean(roc_auc)}
        # print classification_report(l_test, prediction)
        # print "Predicted: 0   1"
        # print confusion_matrix(l_test, prediction)
        return {"auc": np.mean(roc_auc), "recall": np.mean(recall), "precision": np.mean(precision),
                "accuracy": np.mean(accuracy), "fpr": np.mean(fpr), "tnr": np.mean(tnr)}

    def get_evaluation(self, data):
        prediction = self.get_prediction(data)
        probas = self.get_prediction_probabilities(data)[:, 1]
        data.merge_dataset_with_predictions(prediction)
        return self.get_classification_metrics(data.labels, prediction, probas)

    def validate_prediction_by_links(self, prediction):
        roc_auc, recall, precision, accuracy, fpr, tpr = [], [], [], [], [], []

        try:
            metrics = self.get_classification_metrics(prediction["predicted_label"].values, prediction["actual"].values,
                                                      prediction["pos probability"].values)
            roc_auc.append(metrics["auc"])
            recall.append(metrics["recall"])  # TPR
            precision.append(metrics["precision"])
            accuracy.append(metrics["accuracy"])
            fpr.append(metrics["fpr"])
        except ValueError:
            print "Error"
        return {"auc": np.mean(roc_auc), "recall": np.mean(recall), "precision": np.mean(precision),
                "accuracy": np.mean(accuracy), "fpr": np.mean(fpr)}

    def classify_by_links_probability(self, probas, features_ids, labels=None, threshold=0.5):
        if not labels:
            labels = {"neg": 0, "pos": 1}
        train_df = pd.DataFrame(probas)
        train_df["src_id"] = pd.DataFrame(features_ids)
        train_df["link_label"] = train_df[0].apply(lambda avg: 1 if avg <= threshold else 0)
        train_df = train_df.groupby("src_id", as_index=False).agg(
            {0: ['mean', 'count'], 1: 'mean', "link_label": ['mean', 'sum']})
        train_df.columns = ['src_id', "neg probability", 'edge number', "pos probability", 'mean_link_label',
                            'sum_link_label']
        train_df["predicted_label"] = train_df["pos probability"].apply(
            lambda avg: labels["pos"] if avg >= threshold else labels["neg"])
        return train_df

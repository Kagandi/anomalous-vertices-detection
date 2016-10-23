from sklearn import svm, tree, ensemble, feature_extraction, cross_validation, preprocessing
import numpy as np
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, accuracy_score, roc_auc_score
import pandas as pd
from GraphML.configs.config import *
from GraphML.learners import AbstractLearner
from GraphML.utils.dataset import DataSetFactory, DataSet
from GraphML.utils.exceptions import *

def dict_to_array(my_dict):
    vec = feature_extraction.DictVectorizer()
    return vec.fit_transform(my_dict).toarray()


def encode_labels(labels):
    le = preprocessing.LabelEncoder()
    return le.fit_transform(labels)
    # le.fit(labels)
    # return le.transform(labels)


# false_positive = float(len(np.where(l_test - prediction == -1)[0])) # 0 (truth) - 1 (prediction) == -1 which is a false positive
# false_negative = float(len(np.where(l_test - prediction == 1)[0])) # 1 (truth) - 0 (prediction) == 1 which is a false negative
# true_negative = float(len(np.where(l_test + prediction == 0)[0])) # 0 (truth) - 0 (prediction) == 0 which is a false positive
# true_positive = float(len(np.where(l_test + prediction == 2)[0]))# 1 (truth) - 1 (prediction) == 0 which is a false positive
# print true_positive/(true_positive+false_negative) #tpr
# print false_negative/(true_positive+false_negative) #fnr
# print false_positive/(true_negative+false_positive) #fpr
# print true_negative/(true_negative+false_positive) #tnr


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

    def convert_data_to_format(self, features, labels=None, feature_id_col_name=None, metadata_cols=[]):
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
        return self._classifier

    def get_prediction(self, prediction_data):
        return self._classifier.predict(prediction_data.features)

    def get_prediction_probabilities(self, prediction_data):
        return self._classifier.predict_proba(prediction_data.features)

    def split_kfold(self, labels, n_folds=10):
        # StratifiedKFold(self._labels, n_folds)
        return cross_validation.StratifiedKFold(labels, n_folds)

    def get_classification_metrics(self, l_test, prediction):
        fpr, tpr, thresholds = roc_curve(l_test, prediction)
        # false_positive = float(len(np.where(l_test - prediction == -1)[0])) # 0 (truth) - 1 (prediction) == -1 which is a false positive
        # false_negative = float(len(np.where(l_test - prediction == 1)[0])) # 1 (truth) - 0 (prediction) == 1 which is a false negative
        # true_negative = float(len(np.where(l_test + prediction == 0)[0])) # 0 (truth) - 0 (prediction) == 0 which is a false positive
        # true_positive = float(len(np.where(l_test + prediction == 2)[0]))# 1 (truth) - 1 (prediction) == 0 which is a false positive
        # print true_positive/(true_positive+false_negative) #tpr
        # print false_negative/(true_positive+false_negative) #fnr
        # print false_positive/(true_negative+false_positive) #fpr
        # print true_negative/(true_negative+false_positive) #tnr
        false_positive = float(
            len(np.where(l_test - prediction == -1)[0]))  # 0 (truth) - 1 (prediction) == -1 which is a false positive
        true_negative = float(
            len(np.where(l_test + prediction == 0)[0]))  # 0 (truth) - 0 (prediction) == 0 which is a true positive
        return {"auc": roc_auc_score(l_test, prediction)
                # "recall": recall_score(l_test, prediction),
                # "precision": precision_score(l_test, prediction),
                # "accuracy": accuracy_score(l_test, prediction),
                # "fpr": false_positive / (true_negative + false_positive),
                # "tnr": true_negative / (true_negative + false_positive)
                }

    def cross_validate(self, dataset, n_folds=10):
        roc_auc, recall, precision, accuracy, fpr, tpr, tnr = [], [], [], [], [], [], []
        for train_index, test_index in self.split_kfold(dataset.labels, n_folds):
            f_train, f_test = dataset.features[train_index], dataset.features[test_index]
            l_train, l_test = dataset.labels[train_index], dataset.labels[test_index]
            # prediction = self.train_classifier(DataSet(f_train, l_train)).predict(f_test)
            prediction = self.train_classifier(DataSet(f_train, l_train)).predict_proba(f_test)[:, 1]
            metrics = self.get_classification_metrics(l_test, prediction)
            roc_auc.append(metrics["auc"])
            # recall.append(metrics["recall"])  # TPR
            # precision.append(metrics["precision"])
            # accuracy.append(metrics["accuracy"])
            # fpr.append(metrics["fpr"])
            # tnr.append(metrics["tnr"])
        return {"auc": np.mean(roc_auc)}
        # return {"auc": np.mean(roc_auc), "recall": np.mean(recall), "precision": np.mean(precision),
        #         "accuracy": np.mean(accuracy), "fpr": np.mean(fpr), "tnr": np.mean(tnr)}

    def get_evaluation(self, data):
        prediction = self.get_prediction_probabilities(data)
        data.merge_dataset_with_predictions(prediction).to_csv(results_path + "res.csv")
        return self.get_classification_metrics(data.labels, prediction[:, 1])

    def validate_prediction_by_links(self, train, test, nodes_label_path):
        roc_auc, recall, precision, accuracy, fpr, tpr = [], [], [], [], [], []

        # f_train, f_test = features[train_index], features[test_index]
        # l_train, l_test = labels[train_index], labels[test_index]
        # print l_train
        prediction = self.train_classifier(train).predict_proba(test)
        # prediction.to_csv("res2.csv")
        prediction = self.classify_by_links_probability(prediction, test.features_ids, threshold=0.5, metadata=test.metadata)
        prediction = self.merge_with_labels(prediction, nodes_label_path, default_label="Real")
        # prediction.to_csv("res3.csv")
        try:
            metrics = self.get_classification_metrics(prediction["label_y"].values, prediction["label_x"].values)
            roc_auc.append(metrics["auc"])
            recall.append(metrics["recall"])  # TPR
            precision.append(metrics["precision"])
            accuracy.append(metrics["accuracy"])
            fpr.append(metrics["fpr"])
        except ValueError:
            print "Error"
            # print recall
        return {"auc": np.mean(roc_auc), "recall": np.mean(recall), "precision": np.mean(precision),
                "accuracy": np.mean(accuracy), "fpr": np.mean(fpr)}

    def classify_by_links_probability(self, probas, features_ids, labels={"neg": 0, "pos": 1}, threshold=0.5, metadata=[]):
        train_df = pd.DataFrame(probas)
        train_df["src_id"] = pd.DataFrame(features_ids)
        if isinstance(metadata, pd.DataFrame):
            train_df["dst_id"] = metadata["dst"]
            # train_df.to_csv(results_path + "res.csv")
            train_df.pop("dst_id")
        train_df["link_label"] = train_df[0].apply(lambda avg: 1 if avg <= threshold else 0)
        train_df = train_df.groupby("src_id", as_index=False).agg(
            {0: ['mean', 'count'], 1: 'mean', "link_label": ['mean', 'sum']})
        train_df.columns = ['src_id', "neg probability", 'edge number', "pos probability", 'mean_link_label', 'sum_link_label']
        train_df["predicted_label"] = train_df["pos probability"].apply(lambda avg: labels["pos"] if avg >= threshold else labels["neg"])
        return train_df
        # return train_df.loc[train_df[0] <= threshold], train_df.loc[train_df[0] <= threshold]

import graphlab as gl

from anomalous_vertices_detection.learners import AbstractLearner
from anomalous_vertices_detection.utils.dataset import DataSetFactory
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
        return GlLearner(gl.boosted_trees_classifier.create, self._labels)

    def set_svm_classifier(self):
        return GlLearner(gl.svm_classifier.create, self._labels)

    def set_neuralnet_classifier(self):
        return GlLearner(gl.neuralnet_classifier.create, self._labels)

    def set_randomforest_classifier(self):
        return GlLearner(gl.random_forest_classifier.create, self._labels)

    def train_classifier(self, dataset, **kwargs):
        self._classifier = self._classifier(dataset.features, target=dataset.labels)
        print self.get_evaluation(dataset)
        return self

    def get_prediction(self, prediction_data):
        return self._classifier.predict(prediction_data.features)

    def get_evaluation(self, data):
        return self._classifier.evaluate(data.features)

    def get_prediction_probabilities(self, prediction_data):
        # print self.get_evaluation(prediction_data)
        return self._classifier.predict(prediction_data.features, output_type='probability')

    def split_kfold(self, data, labels=None, n_folds=10):
        return gl.cross_validation.KFold(data, n_folds)

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
        train_df = gl.SFrame()
        train_df["probas"] = probas
        train_df["src_id"] = features_ids
        train_df["link_label"] = train_df["probas"].apply(lambda avg: 1 if avg >= threshold else 0)
        train_df = train_df.groupby("src_id", operations={"prob": gl.aggregate.MEAN('probas'),
                                                          "median_prob": gl.aggregate.QUANTILE('probas', 0.5),
                                                          "mean_link_label": gl.aggregate.MEAN('link_label'),
                                                          "sum_link_label": gl.aggregate.SUM('link_label'),
                                                          "STDV_probability": gl.aggregate.STDV('probas'),
                                                          "STDV_predicted_label": gl.aggregate.STDV('link_label'),
                                                          "var_probability": gl.aggregate.VAR('probas'),
                                                          "var_predicted_label": gl.aggregate.VAR('link_label'),
                                                          "count": gl.aggregate.COUNT('link_label')})
        # .agg({0:['mean', 'count'], 1:'mean', "link_label":['mean', 'sum']})
        train_df["predicted"] = train_df["prob"].apply(
            lambda avg: labels["pos"] if avg >= threshold else labels["neg"])
        train_df["median_prob"] = train_df["median_prob"].apply(lambda median_prob: median_prob[0])
        return train_df
        # return train_df.

    def merge_with_labels(self, classified, labels_path, merge_col_name="id", default_label='real'):
        node_labels = gl.SFrame.read_csv(labels_path, column_type_hints={"id": str})
        merged_data = classified.join(node_labels, on={'src_id': merge_col_name}, how='left')
        merged_data = merged_data.fillna("label", default_label)
        merged_data = merged_data.rename({"label": "actual"})
        merged_data['actual'] = merged_data['actual'] == self._labels['pos']
        return merged_data

    def validate_prediction_by_links(self, result):
        targets, predictions, probas = result["actual"], result["predicted"], \
                                       result["prob"]
        precision = gl.evaluation.precision(targets, predictions)
        accuracy = gl.evaluation.accuracy(targets, predictions)
        recall = gl.evaluation.recall(targets, predictions)
        auc = gl.evaluation.auc(targets, predictions)
        return {"recall": recall,
                "precision": precision,
                "accuracy": accuracy,
                "auc": auc
                }

    def cross_validate(self, dataset, n_folds=10):
        folds = self.split_kfold(dataset.features, n_folds)
        params = {'target': dataset.labels}
        job = gl.cross_validation.cross_val_score(folds,
                                                  self._base_classifier,
                                                  params)
        res = job.get_results()["summary"]
        return {"validation_accuracy": res["validation_accuracy"].mean(),
                "training_accuracy": res["training_accuracy"].mean()}

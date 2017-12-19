from anomalous_vertices_detection.utils.dataset import DataSet
from anomalous_vertices_detection.utils.exceptions import ValueNotSet


class MlController(object):
    def __init__(self, learner):
        self._learner = learner
        self._training = DataSet()
        self._test = DataSet()

    def load_training_set(self, features, labels=None, feature_id_col_name=None, metadata_col_names=None):
        """Loads the training set extracted features.

        Parameters
        ----------
        features : string
            path to the csv file that contains the features.
        labels : string, (default=None)
            The name of the label field in the csv  file.
        feature_id_col_name : string, (default=None)
            The name of the id field in the csv  file.
        metadata_col_names :  list[string], (default=None)
            List contains the name of the fields that should not be loaded.
        """
        self._training = self._learner.convert_data_to_format(features, labels, feature_id_col_name, metadata_col_names)

    def load_test_set(self, features, labels=None, feature_id_col_name=None, metadata_col_names=None):
        """Loads the test set extracted features.

        Parameters
        ----------
        features : string
            path to the csv file that contains the features.
        labels : string, (default=None)
            The name of the label field in the csv  file.
        feature_id_col_name : string, (default=None)
            The name of the id field in the csv  file.
        metadata_col_names :  list[string], (default=None)
            List contains the name of the fields that should not be loaded.
        """
        self._test = self._learner.convert_data_to_format(features, labels, feature_id_col_name, metadata_col_names)

    def evaluate(self, features, labels=None, feature_id_col_name=None, metadata_col_names=None):
        """Run evaluation function on the test set.
         If there is no test set ValueNotSet
         exception will be thrown.


        Parameters
        ----------
        features : string
            path to the csv file that contains the features.
        labels : string, (default=None)
            The name of the label field in the csv  file.
        feature_id_col_name : string, (default=None)
            The name of the id field in the csv  file.
        metadata_col_names :  list[string], (default=None)
            List contains the name of the fields that should not be loaded.


        Returns
        -------
        Frame
            Contains the classified results.
        """
        if features:
            if not isinstance(features, DataSet):
                features = self._learner.convert_data_to_format(features, labels, feature_id_col_name,
                                                                metadata_col_names)
            return self._learner.get_evaluation(features)
        else:
            raise ValueNotSet("Test set was not defined.")

    def train_classifier(self):
        """Train the classifier.
        """
        self._learner = self._learner.train_classifier(self._training)
        return self

    def predict(self, features, labels=None, feature_id_col_name=None, metadata_col_names=None):
        """Return a binary classification on the test set.

        Returns
        -------
            The predicted label for each line  in the test set.
        """
        if not isinstance(features, DataSet):
            features = self._learner.convert_data_to_format(features, labels, feature_id_col_name, metadata_col_names)
        return self._learner.get_prediction(features)

    def predict_class_probabilities(self, features, labels=None, feature_id_col_name=None, metadata_col_names=None):
        """Return the probability of every item in the test set to be the positive class.

        Returns
        -------
        Frame
            DataFrame or Sframe containing the probabilities of the test being positive class.
        """
        if not isinstance(features, DataSet):
            features = self._learner.convert_data_to_format(features, labels, feature_id_col_name, metadata_col_names)
        return self._learner.get_prediction_probabilities(features)

    def k_fold_validation(self, k=10):
        """Return k-fold validation results.

        Parameters
        ----------
        k : int
            Number of folds.

        Returns
        -------
        Dict
            Dict contains the AUC
        """
        return self._learner.cross_validate(self._training, k)

    def validate_prediction_by_links(self, result):
        """ Return validation values for anomaly detection
        by link predication.

        Parameters
        ----------
        result : Frame
            The output of classify_by_links_probability can be DataFrame, SFrame.

        Returns
        -------
        Dict
            Dictionary of different metrics such as auc, fpr etc
        """
        return self._learner.validate_prediction_by_links(result)

    def classify_by_links_probability(self, features, labels=None, feature_id_col_name=None, metadata_col_names=None,
                                      labels_map=None):
        """Return the metrics for anomaly detection
        by link predication.

        Parameters
        ----------
        features
        labels
        feature_id_col_name
        metadata_col_names
        labels_map : dict, (defual=={"neg": 0, "pos": 1})
            Dictionary containing map of the labels.

        Returns
        -------
        Frame
            DataFrame or SFrame containing the aggregated results.
        """
        if not labels_map:
            labels_map = {"neg": 0, "pos": 1}
        if not isinstance(features, DataSet):
            features = self._learner.convert_data_to_format(features, labels, feature_id_col_name, metadata_col_names)
        probas = self.predict_class_probabilities(features)
        avg_prob = self._learner.classify_by_links_probability(probas, features.features_ids, labels_map)
        try:
            avg_prob = avg_prob.sort_values("mean_link_label")
        except AttributeError:
            avg_prob = avg_prob.sort("mean_link_label")
        return avg_prob

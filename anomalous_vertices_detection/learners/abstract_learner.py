import abc
import numpy as np
from sklearn.model_selection import StratifiedKFold
from anomalous_vertices_detection.utils.dataset import DataSetFactory, DataSet


class AbstractLearner(object):
    """Interface for learning class
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, classifier):
        """

        Parameters
        ----------
        classifier : function
        """
        self._classifier = classifier

    def set_decision_tree_classifier(self):
        """Set the classifier that should be used as decision tree.

        Returns
        -------
        AbstractLearner
            Learner class object with classifier set as decision tree.
        """
        pass

    def set_svm_classifier(self):
        """Set the classifier that should be used as SVM.

        Returns
        -------
        AbstractLearner
            Learner class object with classifier set as decision tree.
        """
        pass

    @abc.abstractmethod
    def train_classifier(self, dataset):
        """ Train the classifier on supplied data.

        Parameters
        ----------
        dataset : DataSet

        Returns
        -------
        Object
            Classifier object
        """
        pass

    @abc.abstractmethod
    def get_prediction(self, prediction_data):
        """Return classification of the provided data.

        Parameters
        ----------
        prediction_data : DataSet

        Returns
        -------
        Frame
            DataFrame or Sframe containing the classification of the test set.
        """
        pass

    @abc.abstractmethod
    def get_prediction_probabilities(self, prediction_data):
        """Return the probability of every item in DataSet object
        Parameters
        ----------
        prediction_data : DataSet

        Returns
        -------
        Frame
            DataFrame or Sframe containing the probabilities of the test being positive class.
        """
        pass

    def split_kfold(self, features, labels=None, n_folds=10):
        """Return cross validation results dictionary

        Parameters
        ----------
        labels
        data :
        n_folds : int (default=10)
            Number of folds

        Returns
        -------
        Dict
            Dict contains the AUC
        """
        skf = StratifiedKFold(n_folds)
        for train_index, test_index in skf.split(features, labels):
            yield train_index, test_index

    def get_evaluation(self, data):
        """Run evaluation function on provided data.

        Parameters
        ----------
        data : DataSet

        Returns
        -------
        out : dict
            Dictionary of evaluation results where the key is the name of the
            evaluation metric (e.g. `accuracy`) and the value is the evaluation
            score.
        """
        pass

    def cross_validate(self, dataset, n_folds=10):
        pass
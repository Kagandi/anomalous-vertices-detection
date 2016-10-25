import os

from anomalous_vertices_detection.samplers.graph_sampler import GraphSampler
from pandas import DataFrame
from configs.config import *
from configs.predefined_features_sets import *
from feature_controller import FeatureController
from ml_controller import MlController
from utils import utils
from graphlab import SFrame


class GraphLearningController:
    def __init__(self, cls, labels, config):
        """ Initialize a class the combines ml and graphs.

        Parameters
        ----------
        cls : AbstractLearner
            An object of a class the implements AbstractLearner
        labels : dict
            Map of binary labels for instance: {"neg": "Negative", "pos": "Positive"}
        config : GraphConfig
            config object that contains all the necessary information about the graph
        """
        self._labels = labels
        self._ml = MlController(cls.set_randomforest_classifier())
        self._config = config

    @staticmethod
    def extract_features_for_set(graph, dataset, output_path, feature_dict, max_items_num=None):
        """ Extracts features for given set and writes it to file.

            Parameters
            ----------
            graph : AbstractGraph

            dataset : list, iterable
               List that contains the name of the objects(vertex/edge) that
               their features should be extracted.
            output_path : string
               Path including file name for writing the extracted features.
            feature_dict : dict, optional

        """
        features = FeatureController(graph)
        print "Graph loaded"
        features.extract_features(dataset, feature_dict, max_items_num=max_items_num)
        print "Features were written to: " + output_path

    def create_training_test_sets(self, my_graph, test_path, train_path, test_size, training_size,
                                  feature_dict, labels_path=None):
        """
        Creates and extracts features for training and test set.

        Parameters
        ----------
        my_graph : AbstractGraph
            A graph object that implements the AbstractGraph interface
        test_path : string
            A path to where the test set should be saved
        train_path : string
            A path to where the training set should be saved
        test_size : int
            The size of the test set that should be generated
        training_size : int
            The size of the training set that should be generated
        labels_path : string, (default=None)
            The path to where the labels should be saved.
        """
        if not (utils.is_valid_path(train_path) and utils.is_valid_path(test_path)):
            gs = GraphSampler(my_graph, self._config.vertex_min_edge_number, self._config.vertex_max_edge_number)
            if labels_path:
                my_graph.write_nodes_labels(labels_path)
            training_set, test_set = gs.split_training_test_set(training_size, test_size)

            self.extract_features_for_set(my_graph, test_set, test_path, feature_dict,
                                          test_size["neg"] + test_size["pos"])
            self.extract_features_for_set(my_graph, training_set, train_path, feature_dict,
                                          training_size["neg"] + training_size["pos"])
        else:
            print "Existing files were loaded."

    def classify_data(self, my_graph, test_path, train_path, test_size,
                      training_size, id_col_name="src", labels_path=None, feature_dict=fast_link_features):
        """Execute the link classifier

        Parameters
        ----------
        my_graph : AbstractGraph
            A graph object that implements the AbstractGraph interface
        test_path : string
            A path to where the test set should be saved
        train_path : string
            A path to where the training set should be saved
        test_size : int
            The size of the test set that should be generated
        training_size : int
            The size of the training set that should be generated
        id_col_name : string
            The column name of the vertices id
        labels_path : string, (default=None)
            The path to where the labels should be saved.
        feature_dict: dict
        """
        print "Setting training and test sets"
        if my_graph.is_directed:
            meta_data_cols = ["dst", "src", "out_degree_v", "in_degree_v", "out_degree_u", "in_degree_u"]
            # meta_data_cols = ["dst", "src"]
        else:
            meta_data_cols = ["dst", "src", "number_of_friends_u", "number_of_friends_v"]

        # meta_data_cols = ["dst"]
        self.create_training_test_sets(my_graph, test_path, train_path, test_size=test_size,
                                       training_size=training_size,
                                       labels_path=labels_path, feature_dict=feature_dict)  # Training the classifier
        self._ml.load_training_set(train_path, "edge_label", id_col_name, meta_data_cols)
        print("Training 10-fold validation: {}".format(self._ml.k_fold_validation()))
        self._ml.load_test_set(test_path, "edge_label", id_col_name, meta_data_cols)
        self._ml.train_classifier()
        # Testing the classifier
        print("Test evaluation: {}".format(self._ml.evaluate_test()))

    def classify_by_links(self, my_graph, test_path, train_path, results_output_path, real_labels_path, test_size,
                          train_size):
        """Execute the vertex anomaly detection process

        Parameters
        ----------
        my_graph : AbstractGraph
            A graph object that implements the AbstractGraph interface
        test_path : string
            A path to where the test set should be saved
        training_size : int
            The size of the training set that should be generated
        results_output_path : string
            The path to where the classification results should be saved
        real_labels_path : string
            The path to which the labels should be saved
        train_path : string
            A path to where the training set should be saved
        test_size : int
            The size of the test set that should be generated
        """
        self.classify_data(my_graph, test_path, train_path, test_size, train_size, labels_path=real_labels_path)
        classified = self._ml.classify_by_links_probability()
        # Output
        classified = self._ml._learner.merge_with_labels(classified, real_labels_path)
        if isinstance(classified, SFrame):
            classified.save(results_output_path)
        if isinstance(classified, DataFrame):
            classified.to_csv(results_output_path)

        print self._ml.validate_prediction_by_links(classified)


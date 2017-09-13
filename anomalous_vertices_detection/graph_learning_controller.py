from graphlab import SFrame
from pandas import DataFrame
from anomalous_vertices_detection.samplers.graph_sampler import GraphSampler
from configs.predefined_features_sets import *
from feature_controller import FeatureController
from ml_controller import MlController
from utils import utils
from configs.config import TEMP_DIR
import os


class GraphLearningController:
    def __init__(self, cls, config):
        """ Initialize a class the combines ml and graphs.

        Parameters
        ----------
        cls : AbstractLearner
            An object of a class the implements AbstractLearner
        config : GraphConfig
            config object that contains all the necessary information about the graph
        """
        self._ml = MlController(cls.set_randomforest_classifier())
        self._config = config
        self._test_path = os.path.join(TEMP_DIR, config.name + "_test.csv")
        self._train_path = os.path.join(TEMP_DIR, config.name + "_trsin.csv")
        self._labels_path = os.path.join(TEMP_DIR, config.name + "_labels.csv")

    @staticmethod
    def extract_features_for_set(graph, dataset, output_path, feature_dict, max_items_num=None):
        """ Extracts features for given set and writes it to file.

            Parameters
            ----------
            max_items_num
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
        features.extract_features(dataset, feature_dict, output_path, max_items_num=max_items_num)
        print "Features were written to: " + output_path

    def create_training_test_sets(self, my_graph, test_size, training_size,
                                  feature_dict):
        """
        Creates and extracts features for training and test set.

        Parameters
        ----------
        feature_dict
        my_graph : AbstractGraph
            A graph object that implements the AbstractGraph interface
        test_size : int
            The size of the test set that should be generated
        training_size : int
            The size of the training set that should be generated
        """
        if not (utils.is_valid_path(self._train_path) and utils.is_valid_path(self._test_path)):
            gs = GraphSampler(my_graph, self._config.vertex_min_edge_number, self._config.vertex_max_edge_number)
            if self._labels_path:
                my_graph.write_nodes_labels(self._labels_path)
            training_set, test_set = gs.split_training_test_set(training_size, test_size)

            self.extract_features_for_set(my_graph, test_set, self._test_path, feature_dict[my_graph.is_directed],
                                          test_size["neg"] + test_size["pos"])
            self.extract_features_for_set(my_graph, training_set, self._train_path, feature_dict[my_graph.is_directed],
                                          training_size["neg"] + training_size["pos"])
        else:
            print "Existing files were loaded."

    def evaluate_classifier(self, my_graph, test_size=0,
                            training_size=0, id_col_name="src", feature_dict=fast_link_features,
                            meta_data_cols=None):
        """Execute the link classifier

        Parameters
        ----------
        meta_data_cols
        my_graph : AbstractGraph
            A graph object that implements the AbstractGraph interface
        test_size : int
            The size of the test set that should be generated
        training_size : int
            The size of the training set that should be generated
        id_col_name : string
            The column name of the vertices id
        feature_dict: dict
        """
        print "Setting training and test sets"
        if not meta_data_cols:
            if my_graph.is_directed:
                meta_data_cols = ["dst", "src"]

        self.create_training_test_sets(my_graph, test_size=test_size,
                                       training_size=training_size,
                                       feature_dict=feature_dict)  # Training the classifier
        self._ml.load_training_set(self._train_path, "edge_label", id_col_name, meta_data_cols)
        # self._ml.load_test_set(test_path, "edge_label", id_col_name, meta_data_cols)
        self._ml = self._ml.train_classifier()
        print("Training 10-fold validation: {}".format(self._ml.k_fold_validation()))
        # Testing the classifier
        print(
            "Test evaluation: {}".format(self._ml.evaluate(self._test_path, "edge_label", id_col_name, meta_data_cols)))

    def classify_by_links(self, my_graph, results_output_path, test_size,
                          train_size, meta_data_cols=None, id_col_name="src", temp_folder=TEMP_DIR):
        """Execute the vertex anomaly detection process

        Parameters
        ----------
        temp_folder
        train_size
        meta_data_cols
        id_col_name
        my_graph : AbstractGraph
            A graph object that implements the AbstractGraph interface
        results_output_path : string
            The path to where the classification results should be saved
        test_size : int
            The size of the test set that should be generated
        """

        self.evaluate_classifier(my_graph, test_size, train_size, meta_data_cols=meta_data_cols)
        classified = self._ml.classify_by_links_probability(self._test_path, "edge_label", id_col_name, meta_data_cols)
        # Output
        classified = self._ml._learner.merge_with_labels(classified, self._labels_path)
        if isinstance(classified, SFrame):
            classified.save(results_output_path)
        if isinstance(classified, DataFrame):
            classified.to_csv(results_output_path)

        return self._ml.validate_prediction_by_links(classified)

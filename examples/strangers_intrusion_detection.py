"""
Implementation of an algorithm for the detection of spammers and fake profiles in social networks.
The algorithm, which is based solely on the topology of the social network, detects users who
randomly connect to others by detecting anomalies in that network's topology.

References
----------
Fire, Michael, Gilad Katz, and Yuval Elovici. "Strangers intrusion detection
detecting spammers and fake profiles in social networks based on topology
anomalies." Human Journal 1, no. 1 (2012): 26-39.
"""
from anomalous_vertices_detection.configs.graph_config import GraphConfig
from anomalous_vertices_detection.graph_learning_controller import *
from anomalous_vertices_detection.graphs.graph_factory import GraphFactory
from anomalous_vertices_detection.learners.gllearner import GlLearner

labels = {"neg": "Real", "pos": "Fake"}
#
twitter_path = "C:\\Users\\user\\Documents\\Datasets\\Twitter\\twitter_clean2.csv"
twitter_labels = "C:\\Users\\user\\Documents\\Datasets\\Twitter\\fake_users.txt"
dataset_config = GraphConfig("twitter", twitter_path, True, labels_path=twitter_labels, type="simulation",
                             vertex_min_edge_number=3, vertex_max_edge_number=50000)
glc = GraphLearningController(GlLearner(labels=labels), dataset_config)
output_foldr = "../output/"
test_path, training_path, result_path, labels_output_path = output_foldr + dataset_config.name + "_test.csv", \
                                                            output_foldr + dataset_config.name + "_train.csv", \
                                                            output_foldr + dataset_config.name + "_res.csv", \
                                                            output_foldr + dataset_config.name + "_labels.csv"

my_graph = GraphFactory().make_graph_with_fake_profiles(dataset_config.data_path,
                                                        is_directed=dataset_config.is_directed,
                                                        labels_path=dataset_config.labels_path,
                                                        pos_label=labels["pos"],
                                                        fake_users_number=1000,
                                                        neg_label=labels["neg"], max_num_of_edges=1000000)
print("Graph was loaded")
sampler = GraphSampler(my_graph, 3, 10000)
glc.extract_features_for_set(my_graph, sampler.generate_sample_for_labeled_vertices(100, 100), training_path,
                             stranger_intrusion_features[my_graph.is_directed])
glc.extract_features_for_set(my_graph, sampler.generate_sample_for_test_labeled_vertices(900, 100), test_path,
                             stranger_intrusion_features[my_graph.is_directed])
meta_data_cols = ["src", "vertex_label"]
glc._ml.load_training_set(training_path, "label", "src", meta_data_cols)
glc._ml = glc._ml.train_classifier()
print("Training 10-fold validation: {}".format(glc._ml.k_fold_validation()))
# Testing the classifier
print("Test evaluation: {}".format(glc._ml.evaluate(test_path, "label", "src", meta_data_cols)))


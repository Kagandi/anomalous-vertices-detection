from anomalous_vertices_detection.configs.graph_config import GraphConfig
from anomalous_vertices_detection.graph_learning_controller import *
from anomalous_vertices_detection.learners.sklearner import SkLearner
from anomalous_vertices_detection.datasets.academia import load_data

labels = {"neg": "Real", "pos": "Fake"}


academia_graph, academia_config = load_data(labels_map=labels)

glc = GraphLearningController(SkLearner(labels=labels), academia_config)

output_folder = "../output/"
test_path, training_path, result_path, labels_output_path = output_folder + academia_config.name + "_test.csv", \
                                                            output_folder + academia_config.name + "_train.csv", \
                                                            output_folder + academia_config.name + "_res.csv", \
                                                            output_folder + academia_config.name + "_labels.csv"


if academia_graph.is_directed:
    meta_data_cols = ["dst", "src", "out_degree_v", "in_degree_v", "out_degree_u", "in_degree_u"]
    # meta_data_cols = ["dst", "src"]
else:
    meta_data_cols = ["dst", "src", "number_of_friends_u", "number_of_friends_v"]

glc.classify_by_links(academia_graph, test_path, training_path, result_path,
                      labels_output_path, test_size={"neg": 1000, "pos": 100},
                      train_size={"neg": 20000, "pos": 20000},meta_data_cols=meta_data_cols)

from anomalous_vertices_detection.graph_learning_controller import *
from anomalous_vertices_detection.learners.sklearner import SkLearner
from anomalous_vertices_detection.datasets.academia import load_data
import os

labels = {"neg": "Real", "pos": "Fake"}

academia_graph, academia_config = load_data(labels_map=labels, simulate_fake_vertices=True)

glc = GraphLearningController(SkLearner(labels=labels), academia_config)

output_folder = "../output/"
result_path = os.path.join(output_folder, academia_config.name + "_res.csv")

if academia_graph.is_directed:
    meta_data_cols = ["dst", "src", "out_degree_v", "in_degree_v", "out_degree_u", "in_degree_u"]
else:
    meta_data_cols = ["dst", "src", "number_of_friends_u", "number_of_friends_v"]

glc.classify_by_links(academia_graph, result_path, test_size={"neg": 1000, "pos": 100},
                      train_size={"neg": 20000, "pos": 20000}, meta_data_cols=meta_data_cols)

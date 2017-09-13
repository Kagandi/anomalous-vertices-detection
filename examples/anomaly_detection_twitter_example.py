from anomalous_vertices_detection.graph_learning_controller import *
from anomalous_vertices_detection.learners.sklearner import SkLearner
from anomalous_vertices_detection.datasets.twitter import load_data

labels = {"neg": "Real", "pos": "Fake"}

twiiter_graph, twitter_config = load_data(labels_map=labels)

glc = GraphLearningController(SkLearner(labels=labels), twitter_config)
output_folder = "../output/"

result_path = os.path.join(output_folder, twitter_config.name + "_res.csv")

if twiiter_graph.is_directed:
    meta_data_cols = ["dst", "src", "out_degree_v", "in_degree_v", "out_degree_u", "in_degree_u"]
else:
    meta_data_cols = ["dst", "src", "number_of_friends_u", "number_of_friends_v"]

glc.classify_by_links(twiiter_graph, result_path, test_size={"neg": 5000, "pos": 1000},
                      train_size={"neg": 5000, "pos": 5000}, meta_data_cols=meta_data_cols)

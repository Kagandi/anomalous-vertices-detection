from anomalous_vertices_detection.configs.graph_config import GraphConfig
from anomalous_vertices_detection.graph_learning_controller import *
from anomalous_vertices_detection.graphs.graph_factory import GraphFactory
from anomalous_vertices_detection.learners.sklearner import SkLearner

labels = {"neg": "Real", "pos": "Fake"}

dataset_config = GraphConfig("academia", "..\\data\\academia.csv.gz", True, graph_type="simulation",
                             vertex_min_edge_number=3, vertex_max_edge_number=50000)
glc = GraphLearningController(SkLearner(labels=labels), dataset_config)
output_folder = "../output/"
result_path = os.path.join(output_folder, dataset_config.name + "_res.csv")

my_graph = GraphFactory().make_graph_with_fake_profiles(dataset_config.data_path,
                                                        is_directed=dataset_config.is_directed,
                                                        pos_label=labels["pos"],
                                                        neg_label=labels["neg"], max_num_of_edges=100000)

if my_graph.is_directed:
    meta_data_cols = ["dst", "src", "out_degree_v", "in_degree_v", "out_degree_u", "in_degree_u"]
else:
    meta_data_cols = ["dst", "src", "number_of_friends_u", "number_of_friends_v"]

glc.classify_by_links(my_graph, result_path, test_size={"neg": 1000, "pos": 100},
                      train_size={"neg": 20000, "pos": 20000}, meta_data_cols=meta_data_cols)

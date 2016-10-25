from anomalous_vertices_detection.configs.graph_config import GraphConfig
from anomalous_vertices_detection.graph_learning_controller import *
from anomalous_vertices_detection.graphs.graph_factory import GraphFactory

labels = {"neg": "Real", "pos": "Fake"}

dataset_config = GraphConfig("academia", "..\\data\\academia.bz2", True, type="simulation",
                             vertex_min_edge_number=3, vertex_max_edge_number=50000)
edges_output_path = "../output/" + dataset_config.name + "_edges.csv"
vetices_output_path = "../output/" + dataset_config.name + "_vertices.csv"

my_graph = GraphFactory().make_graph_with_fake_profiles(dataset_config.data_path,
                                                        is_directed=dataset_config.is_directed,
                                                        pos_label=labels["pos"], neg_label=labels["neg"],
                                                        max_num_of_edges=100000)
features = FeatureController(my_graph)
# Edge feature extraction
features.extract_features(my_graph.edges[:1000], fast_link_features, edges_output_path)
# Vertex feature extraction
features.extract_features(my_graph.vertices[:1000], vetices_output_path, fast_vertex_features)

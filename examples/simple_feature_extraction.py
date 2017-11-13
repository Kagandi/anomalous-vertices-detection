from anomalous_vertices_detection.graph_learning_controller import *
from anomalous_vertices_detection.datasets.academia import load_data
from itertools import islice
labels = {"neg": "Real", "pos": "Fake"}

my_graph, dataset_config = load_data(labels_map=labels, limit=50000)
print(my_graph.number_of_vertices)
print(len(my_graph.edges))
edges_output_path = "../output/" + dataset_config.name + "_edges.csv"
vetices_output_path = "../output/" + dataset_config.name + "_vertices.csv"
print len(my_graph.vertices)
print my_graph.vertices[10]

features = FeatureController(my_graph)
# Edge feature extraction
features.extract_features_to_file(my_graph.edges[:1000], fast_link_features[my_graph.is_directed], edges_output_path)
# Vertex feature extraction
features.extract_features_to_file(my_graph.vertices[:1000], fast_vertex_features[my_graph.is_directed], vetices_output_path)

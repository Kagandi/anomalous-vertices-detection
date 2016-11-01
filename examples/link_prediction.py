from anomalous_vertices_detection.configs.graph_config import GraphConfig
from anomalous_vertices_detection.graph_learning_controller import *
from anomalous_vertices_detection.graphs.graph_factory import GraphFactory
from anomalous_vertices_detection.learners.gllearner import GlLearner

labels = {"neg": "Real", "pos": "Fake"}
#
dataset_config = GraphConfig("academia", "..\\data\\academia.csv.gz", True, type="simulation",
                             vertex_min_edge_number=3, vertex_max_edge_number=50000)
glc = GraphLearningController(GlLearner(labels=labels), labels, dataset_config)
output_foldr = "../output/"
test_path, training_path, result_path, labels_output_path = output_foldr + dataset_config.name + "_test.csv", \
                                                            output_foldr + dataset_config.name + "_train.csv", \
                                                            output_foldr + dataset_config.name + "_res.csv", \
                                                            output_foldr + dataset_config.name + "_labels.csv"

my_graph = GraphFactory().make_graph_with_fake_profiles(dataset_config.data_path,
                                                        is_directed=dataset_config.is_directed,
                                                        pos_label=labels["pos"],
                                                        neg_label=labels["neg"], max_num_of_edges=1000000)

sampler = GraphSampler(my_graph, 3, 10000)
glc.extract_features_for_set(my_graph, sampler.generate_sample_for_unlabeled_links(10000, 10000), training_path,
                             fast_link_features[my_graph.is_directed])
glc.extract_features_for_set(my_graph, sampler.generate_sample_for_labeled_links(10000, 10000), test_path,
                             fast_link_features[my_graph.is_directed])
print glc.evaluate_classifier(my_graph, test_path, training_path)

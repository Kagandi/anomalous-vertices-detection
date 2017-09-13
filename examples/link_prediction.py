from anomalous_vertices_detection.datasets.academia import load_data
from anomalous_vertices_detection.graph_learning_controller import *
from anomalous_vertices_detection.learners.gllearner import GlLearner

labels = {"neg": "Real", "pos": "Fake"}

my_graph, dataset_config = load_data(labels_map=labels, simulate_fake_vertices=True)

glc = GraphLearningController(GlLearner(labels=labels), dataset_config)
output_folder = "../output/"
test_path, training_path = os.path.join(output_folder, dataset_config.name + "_test.csv"),\
                                        os.path.join(output_folder, dataset_config.name + "_train.csv")

sampler = GraphSampler(my_graph, 3, 10000)
glc.extract_features_for_set(my_graph, sampler.generate_sample_for_unlabeled_links(10000, 10000), training_path,
                             fast_link_features[my_graph.is_directed])
glc.extract_features_for_set(my_graph, sampler.generate_sample_for_labeled_links(10000, 10000), test_path,
                             fast_link_features[my_graph.is_directed])
glc.evaluate_classifier(my_graph, test_path, training_path)

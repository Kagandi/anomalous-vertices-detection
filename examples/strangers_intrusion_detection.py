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

from graphlab import aggregate

from anomalous_vertices_detection.configs.graph_config import GraphConfig
from anomalous_vertices_detection.graph_learning_controller import *
from anomalous_vertices_detection.graphs.graph_factory import GraphFactory
from anomalous_vertices_detection.learners.gllearner import GlLearner


def aggreagate_res(data_folder, res_path):
    results_frame = SFrame()
    for f in os.listdir(data_folder):
        temp_sf = SFrame.read_csv(data_folder + "\\" + f,
                                  column_type_hints={"prob": float})
        results_frame = results_frame.append(temp_sf)

    results_frame = results_frame.groupby("src_id", operations={"prob": aggregate.MEAN('prob'), "actual": aggregate.SELECT_ONE('actual')})

    # res["actual"] = res["actual"].apply(lambda x: "P" if x == 1 else "N")
    results_frame.save(res_path, 'csv')


labels = {"neg": "Real", "pos": "Fake"}
#
output_folder = "../output/twitter/"
twitter_path = "C:\\Users\\user\\Documents\\Datasets\\Twitter\\twitter_clean2.csv"
twitter_labels = "C:\\Users\\user\\Documents\\Datasets\\Twitter\\fake_users.txt"
dataset_config = GraphConfig("twitter", twitter_path, True, labels_path=twitter_labels, graph_type="simulation",
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
                                                        fake_users_number=5000,
                                                        neg_label=labels["neg"], max_num_of_edges=1000000)
print("Graph was loaded")
for i in range(10):
    sampler = GraphSampler(my_graph, 3, 10000)
    features = FeatureController(my_graph)
    features.extract_features_to_file(sampler.generate_sample_for_labeled_vertices(1000, 100),
                                      stranger_intrusion_features[my_graph.is_directed], training_path)
    features.extract_features_to_file(sampler.generate_sample_for_test_labeled_vertices(900, 100),
                                      stranger_intrusion_features[my_graph.is_directed], test_path)
    meta_data_cols = ["src", "vertex_label"]
    ml = MlController(GlLearner(labels=labels).set_randomforest_classifier())
    ml.load_training_set(training_path,
                         "label", "src", meta_data_cols)

    ml = ml.train_classifier()
    features = ml._learner.convert_data_to_format(test_path, "label", "src", meta_data_cols)
    res = SFrame()
    res["src_id"] = features.features_ids
    res["actual"] = features.features["label"]
    res["prob"] = ml.predict_class_probabilities(features)
    result_path = output_folder + dataset_config.name + "_" + str(i) + "res.csv"
    res.save(result_path, "csv")

# Testing the classifier
# print("Test evaluation: {}".format(glc._ml.evaluate(test_path, "label", "src", meta_data_cols)))
aggreagate_res(output_folder, "twitter_res.csv")

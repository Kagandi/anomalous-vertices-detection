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


def aggreagate_res(data_folder, res_path):
    res = SFrame()
    for f in os.listdir(data_folder):
        temp_sf = SFrame.read_csv(data_folder + "\\" + f,
                                  column_type_hints={"prob": float})
        res = res.append(temp_sf)

    res = res.groupby("src_id", operations={"prob": aggregate.MEAN('prob')})

    # res["actual"] = res["actual"].apply(lambda x: "P" if x == 1 else "N")
    res.save(res_path, 'csv')


labels = {"neg": "Real", "pos": "Fake"}
#
kids_path = "C:\Users\user\Documents\Datasets\German Friend\\Heidler_et_al_2013_Relationship_patterns_in_the_19th_century.csv"
dataset_config = GraphConfig("kids", kids_path, True, graph_type="simulation",
                             vertex_min_edge_number=1, vertex_max_edge_number=50000)

output_folder = "../output/kids/"
test_path, training_path, labels_output_path = output_folder + dataset_config.name + "_test.csv", \
                                               output_folder + dataset_config.name + "_train.csv", \
                                               output_folder + dataset_config.name + "_labels.csv"

# print("Graph was loaded")
# for i in range(100):
#     my_graph = GraphFactory().make_graph_with_fake_profiles(dataset_config.data_path,
#                                                             is_directed=dataset_config.is_directed,
#                                                             labels_path=dataset_config.labels_path,
#                                                             pos_label=labels["pos"],
#                                                             fake_users_number=20,
#                                                             neg_label=labels["neg"], max_num_of_edges=1000000)
#
#     result_path = output_folder + dataset_config.name + "_" + str(i) + "res.csv"
#     sampler = GraphSampler(my_graph, 1, 10000)
#     features = FeatureController(my_graph)
#     features.extract_features(sampler.generate_sample_for_labeled_vertices(20, 20),
#                               stranger_intrusion_features[my_graph.is_directed], training_path, max_items_num=40)
#     features.extract_features(sampler.generate_sample_for_test_labeled_vertices(40, 0),
#                               stranger_intrusion_features[my_graph.is_directed], test_path, max_items_num=40)
#
#     meta_data_cols = ["src", "vertex_label"]
#     ml = MlController(GlLearner(labels=labels).set_randomforest_classifier())
#     ml.load_training_set(training_path,
#                          "label", "src", meta_data_cols)
#
#     ml = ml.train_classifier()
#     features = ml._learner.convert_data_to_format(test_path, "label", "src", meta_data_cols)
#     res = SFrame()
#     res["src_id"] = features.features_ids
#     res["prob"] = ml.predict_class_probabilities(features)
#     res.save(result_path, "csv")

aggreagate_res(output_folder, "res_kids.csv")

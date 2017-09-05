from anomalous_vertices_detection.configs.graph_config import GraphConfig
from anomalous_vertices_detection.graph_learning_controller import *
from anomalous_vertices_detection.graphs.graph_factory import GraphFactory
from anomalous_vertices_detection.learners.sklearner import SkLearner
import graphlab as gl

labels = {"neg": "Real", "pos": "Fake"}

dataset_config = GraphConfig("instacart", "C:\Users\user\\Dropbox\\instacart\\product_graph.csv", False,
                             graph_type="simulation",
                             vertex_min_edge_number=0, vertex_max_edge_number=50000)
glc = GraphLearningController(SkLearner(labels=labels), dataset_config)
output_folder = "../output/"
test_path, training_path, result_path, labels_output_path = output_folder + dataset_config.name + "_test.csv", \
                                                            output_folder + dataset_config.name + "_train.csv", \
                                                            output_folder + dataset_config.name + "_res.csv", \
                                                            output_folder + dataset_config.name + "_labels.csv"

my_graph = GraphFactory().make_graph_with_fake_profiles(dataset_config.data_path,
                                                        is_directed=dataset_config.is_directed,
                                                        pos_label=labels["pos"],
                                                        neg_label=labels["neg"], max_num_of_edges=1000000)

if my_graph.is_directed:
    meta_data_cols = ["dst", "src", "out_degree_v", "in_degree_v", "out_degree_u", "in_degree_u"]
    # meta_data_cols = ["dst", "src"]
else:
    meta_data_cols = ["dst", "src", "number_of_friends_u", "number_of_friends_v"]

glc.classify_by_links(my_graph, test_path, training_path, result_path,
                      labels_output_path, test_size={"neg": 5000, "pos": 1000},
                      train_size={"neg": 5000, "pos": 5000}, meta_data_cols=meta_data_cols)
#
# results = gl.SFrame.read_csv(result_path)
# ratting = gl.SFrame.read_csv("C:\\Users\\user\\Dropbox\\Datasets\\imdb\\id-name.csv")
# merged_data = results.join(ratting, on={'src_id': 'id'}, how='left')
# merged_data = merged_data.fillna("ratting", 0)
# merged_data.save("res.csv")

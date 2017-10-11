from anomalous_vertices_detection.graph_learning_controller import *
from anomalous_vertices_detection.learners.sklearner import SkLearner
from anomalous_vertices_detection.datasets.twitter import load_data
import pandas as pd


def aggreagate_res(data_folder, res_path):
    results_frame = DataFrame()
    for f in os.listdir(data_folder):
        temp_df = pd.read_csv(data_folder + "\\" + f)
        results_frame = results_frame.append(temp_df)

    results_frame = results_frame.groupby("src_id").mean()

    results_frame.to_csv(res_path)


labels = {"neg": "Real", "pos": "Fake"}

twiiter_graph, twitter_config = load_data(labels_map=labels)

output_folder = "../output/twitter"

if twiiter_graph.is_directed:
    meta_data_cols = ["dst", "src", "out_degree_v", "in_degree_v", "out_degree_u", "in_degree_u"]
else:
    meta_data_cols = ["dst", "src", "number_of_friends_u", "number_of_friends_v"]

for i in range(10):
    twitter_config._name = "twitter_" + str(i)
    print(twitter_config.name)
    glc = GraphLearningController(SkLearner(labels=labels), twitter_config)
    result_path = os.path.join(output_folder, twitter_config.name  + "_res.csv")
    glc.classify_by_links(twiiter_graph, result_path, test_size={"neg": 10000, "pos": 1000},
                          train_size={"neg": 20000, "pos": 20000}, meta_data_cols=meta_data_cols)

aggreagate_res(output_folder, "res.csv")

from anomalous_vertices_detection.utils.utils import download_file
import os
from anomalous_vertices_detection.configs.config import TWITTER_URL, DATA_DIR, TWITTER_LABELS_URL
from anomalous_vertices_detection.configs.graph_config import GraphConfig
from anomalous_vertices_detection.graphs.graph_factory import GraphFactory


def load_data(dataset_file_name="twitter.csv.gz", labels_file_name="twitter_fake_ids.csv", labels_map=None, limit=5000000):
    data_path = os.path.join(DATA_DIR, dataset_file_name)
    if not os.path.exists(data_path):
        data_path = download_file(TWITTER_URL, dataset_file_name)

    labels_path = os.path.join(DATA_DIR, labels_file_name)
    if not os.path.exists(labels_path):
        labels_path = download_file(TWITTER_LABELS_URL, labels_file_name)

    twitter_config = GraphConfig("twitter", data_path,
                                 is_directed=True, labels_path=labels_path,
                                 graph_type="regular", vertex_min_edge_number=10, vertex_max_edge_number=50000)
    return GraphFactory().factory(twitter_config, labels=labels_map, limit=limit), twitter_config

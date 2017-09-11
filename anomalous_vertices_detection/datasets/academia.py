from anomalous_vertices_detection.utils.utils import download_file
import os
from anomalous_vertices_detection.configs.config import ACADEMIA_URL, DATA_DIR
from anomalous_vertices_detection.configs.graph_config import GraphConfig
from anomalous_vertices_detection.graphs.graph_factory import GraphFactory


def load_data(dataset_file_name="academia.csv.gz", labels_map=None):
    data_path = os.path.join(DATA_DIR, dataset_file_name)
    if not os.path.exists(data_path):
        data_path = download_file(ACADEMIA_URL, dataset_file_name)

    academia_config = GraphConfig("academia_config", data_path,
                                  is_directed=True,
                                  graph_type="simulation", vertex_min_edge_number=10, vertex_max_edge_number=50000)
    return GraphFactory().factory(academia_config, labels=labels_map), academia_config

from anomalous_vertices_detection.utils.utils import download_file
import os
from anomalous_vertices_detection.configs.config import KIDS_URL, DATA_DIR, KIDS_LABELS_URL
from anomalous_vertices_detection.configs.graph_config import GraphConfig
from anomalous_vertices_detection.graphs.graph_factory import GraphFactory


def load_data(dataset_file_name="Relationship_patterns_in_the_19th_century.csv",
              labels_file_name="Relationship_patterns_labels.csv", labels_map=None):
    data_path = os.path.join(DATA_DIR, dataset_file_name)
    if not os.path.exists(data_path):
        data_path = download_file(KIDS_URL, dataset_file_name)

    labels_path = os.path.join(DATA_DIR, labels_file_name)
    if not os.path.exists(labels_path):
        labels_path = download_file(KIDS_LABELS_URL, labels_file_name)

    kids_config = GraphConfig("kids_config", data_path,
                                 is_directed=True, labels_path=labels_path,
                                 graph_type="regular", vertex_min_edge_number=3, vertex_max_edge_number=50000)
    return GraphFactory().factory(kids_config, labels=labels_map), kids_config

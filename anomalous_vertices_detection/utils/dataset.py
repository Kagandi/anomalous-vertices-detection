import numpy as np
import pandas as pd

try:
    import graphlab as gl
except ImportError:
    gl = None

from anomalous_vertices_detection.configs.config import *
from anomalous_vertices_detection.utils import utils


class DataSet(object):
    def __init__(self, features=None, labels=None, features_ids=None, metadata=None, container_type="DataFrame"):
        if features is None:
            features = []
        if features_ids is None:
            features_ids = []
        if metadata is None:
            metadata = []
        if labels is None:
            labels = []

        self._features_ids = features_ids
        self._labels = labels
        self._features = features
        self._metadata = metadata
        self._container_type = container_type

    def get_complement(self, b):
        if self.container_type == "DataFrame":
            complement_ids = ~pd.Series(self.features_ids).isin(b.features_ids)
            features_ids = pd.Series(self._features_ids)[complement_ids]
            labels = pd.Series(self._labels)[complement_ids]
            features = pd.DataFrame(self._features)[complement_ids]
            return DataSet(features.values, labels.values, features_ids.values, self.metadata, self.container_type)
        return []

    @property
    def features(self):
        return self._features

    @property
    def features_ids(self):
        return self._features_ids

    @property
    def labels(self):
        return self._labels

    @property
    def metadata(self):
        return self._metadata

    @property
    def container_type(self):
        return self._container_type

    def __len__(self):
        return len(self._features_ids)

    def merge_dataset_with_predictions(self, predictions):
        res = pd.DataFrame(predictions)
        res["src"] = self._features_ids
        res["real labels"] = self._labels
        res["dst"] = self.metadata["dst"]
        return res


class DataSetFactory(object):
    def convert_data_to_sklearn_format(self, features, labels=None, feature_id_col_name=None, metadata_cols=None):
        if not metadata_cols:
            metadata_cols = []
        features_id = None
        metadata = pd.DataFrame()
        if utils.is_valid_path(features):
            features = pd.read_csv(features, dtype={feature_id_col_name: str})
        elif isinstance(features[0], dict):
            features = pd.DataFrame(features)
        if labels is not None:
            if isinstance(labels, str):
                labels = features.pop(labels).values
            labels = label_encoder.transform(labels)
        if feature_id_col_name is not None:
            features_id = features.pop(feature_id_col_name).values
        for metadata_col in metadata_cols:
            if metadata_col in features:
                metadata[metadata_col] = features[metadata_col]
                features = features.drop(metadata_col, axis=1)
        if isinstance(features, pd.DataFrame):
            features = features.values
        if isinstance(features[0], list) or isinstance(features, np.ndarray):
            return DataSet(features, labels, features_id, metadata)

    def convert_data_to_graphlab_format(self, features, labels=None, feature_id_col_name=None, metadata_cols=None,
                                        labels_map=None):
        features_id = None
        if not metadata_cols:
            metadata_cols = []
        if utils.is_valid_path(features):
            features = gl.SFrame.read_csv(features, column_type_hints={feature_id_col_name: str, "dst": str})
        elif isinstance(features[0], dict):
            features = pd.DataFrame(features)
        if feature_id_col_name:
            features_id = features[feature_id_col_name]
        temp_metadata_cols = []
        for col in metadata_cols:
            if col in features.column_names():
                temp_metadata_cols.append(col)
        features = features.remove_columns(temp_metadata_cols)
        if 'label' not in features.column_names():
            features.rename({labels: 'label'})
            if labels_map:
                features["label"] = features["label"] == labels_map["pos"]
        return DataSet(features, 'label', features_id)

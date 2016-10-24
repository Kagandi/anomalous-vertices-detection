from configs.config import *
from feature_extractor import FeatureExtractor
from utils.utils import dict_writer, delete_file_content


class FeatureController(object):
    """
    This class control the extraction of features from FeatureController.
    """

    def __init__(self, graph_obj, enabled_features={}):
        self._enabled_features = enabled_features
        self.init_enabled_features()
        self._graph = graph_obj
        self._fe = FeatureExtractor(self._graph)
        # self._fe.load_centrality_features()
        self._temp_result = {}
        self._labels = []

    def init_enabled_features(self):
        if "same_community" in self._enabled_features or "number_of_neighbors_communities" in self._enabled_features:
            self._enabled_features["disjoint_communities"] = 1
        if "hubs" in self._enabled_features or "authorities" in self._enabled_features:
            self._enabled_features["hits"] = 1

    def is_enabled(self, feature_name):
        if len(self._enabled_features) == 0:
            return True
        elif feature_name in self._enabled_features:
            return True
        return False

    def set_label(self, v, u=None):
        """Set the edge/vertex label attribute in temp_result.
        If v and u are provided it sets the label of the edge
        between v and u. If only v are provided then it sets
        the label of the vertex v.

        Parameters
        ----------
        v : string
            Vertex
        u : string, (default=None)
            Vertex
        """
        if u is None:
            self._temp_result["vertex_label"] = self._fe.get_node_label(v)
        else:
            self._temp_result["edge_label"] = self._fe.get_edge_label(v, u)

    def set_edge_weight(self, u, v):
        if self._graph.has_weight:
            self._temp_result["edge_weight"] = self._graph.get_edge_weight(u, v)

    def set_feature(self, feature_name, feature_func, *args):
        """ Execute and a feature extraction function and set its
         value in temp_result dictionary.

        Parameters
        ----------
        feature_name : string
            The name that should be given to the extracted feature.
        feature_func : string
            The function that should be executed to extract the feature.
        args : tuple
            The feature extraction function arguments.
        """
        self._temp_result[feature_name] = getattr(self._fe, feature_func)(*args)

    def init_entry(self, v, u=None):
        """ Initializing entry for an edge feature extraction.
         Set the edge source destination and label.

        Parameters
        ----------
        v : string
            Vertex
        u : string
            Vertex
        """
        if u:
            self._temp_result = {"src": v, "dst": u}
        else:
            self._temp_result = {"src": v}
        self.set_label(v, u)
        # self.set_edge_weight(vertex1, vertex2)

    def extract_all_features(self, features_list, vertex1, vertex2=None):
        """Extract all features from predefined feature list for given vertices or edges.
        If two vertices are given then it will extract vertex and edge features.
        If only one vertex the just vertex features will be extracted.

        Parameters
        ----------
        features_list : dict
            Dictionary that contains the feature that should be extracted and with which function.
        vertex1 : string
            Vertex
        vertex2 : string, (default=None)
            Vertex
        """
        self.extract_single_entry(features_list, "vertex_v", vertex1)
        if vertex2:
            self.extract_single_entry(features_list, "link", vertex1, vertex2)
            self.extract_single_entry(features_list, "vertex_u", vertex2)

    def extract_single_entry(self, features_list, features_type, *args):
        """Extract all features for a single vertex/edge.

        Parameters
        ----------
        features_list : dict
            Dict that contains the features that should be extracted by type.
        features_type : string
            The feature type that should be extracted
        args : tuple
            The feature extraction function arguments.
        """
        if features_type in features_list:
            for name, feature in features_list[features_type].iteritems():
                self.set_feature(name, feature, *args)

    def features_generator(self, features_dict, item_iter):
        """ Generator that yield the extracted feature from the edge_iter.

        Parameters
        ----------
        features_dict : dict
            A dictionary that contains the features that should be extracted.
        item_iter : iterator
            Iterator over a list of vertecies/edges that features should be extracted for them.
        """
        for count, item in enumerate(item_iter):
            if type(item) is str: item = (item,)
            self.init_entry(*item[:2])
            if len(item) > 1 and self._graph.has_edge(item[0], item[1]):
                self.extract_features_for_existing_edge(features_dict, item)
            else:
                self.extract_all_features(features_dict, *(item[:2]))
            yield self._temp_result

    def extract_features_for_existing_edge(self, features_dict, item):
        """

        Parameters
        ----------
        features_dict : dict
            A dictionary that contains the features that should be extracted.
        item : tuple
            A tuple contains an edge (v,u).
        """
        curr_edge = (item[0], item[1], self._graph.edge(*item))
        self._graph.remove_edge(item[0], item[1])
        self.extract_all_features(features_dict, item[0], item[1])
        self._graph.add_edge(*curr_edge)
        self._graph.get_neighbors.reset()

    @staticmethod
    def save_progress(array):
        """ Save the progress od the feature extraction.

        Parameters
        ----------
        array : list[dict]
        """
        dict_writer(array, temp_path, "a+")

    def extract_features(self, data_iter, features_dict, max_items_num=10000):
        """ Extract features from the graph and saves them to file.

        Parameters
        ----------
        data_iter : iterator
            An iterable objects that contains the vertices/edges that their features should be extracted
        max_items_num : int, (default=1000)
            Maximal number of features that should be extracted, used to show progress information.
        """
        delete_file_content(temp_path)
        self.run_feature_extractor(data_iter, max_items_num, features_dict[self._graph.is_directed])

    def run_feature_extractor(self, item_iter, max_items_num, features_list):
        """Extract edge features from the graph and saves them to file.

        Parameters
        ----------
        item_iter : iterator
            An iterable objects that contains the vertices/edges that their features should be extracted
        max_items_num : int, (default=1000)
            Maximal number of features that should be extracted, used to show progress information.
        """
        features_array = []
        for count, entry in enumerate(self.features_generator(features_list, item_iter)):
            if count % (max_items_num / 10) == 0:
                print "%d%% (%d out of %d features were extracted)." % \
                      (100 * count / max_items_num, count, max_items_num)
            features_array.append(entry)
            if len(features_array) == save_progress_interval:
                self.save_progress(features_array)
                features_array = []
        if features_array:
            self.save_progress(features_array)

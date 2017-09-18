from anomalous_vertices_detection.configs.config import *
import anomalous_vertices_detection.utils.utils as utils
import collections
import types


class AbstractGraph(object):

    def __init__(self, weight_field=None):
        self._graph = None
        self._labels_dict = collections.defaultdict(lambda: "Real")
        self._labels_map = {"pos": 1, "neg": 0}
        self._weight_field = weight_field

    """Basic Operations"""

    def add_edge(self, vertex1, vertex2, weight=None):
        pass

    def remove_edge(self, vertex1, vertex2):
        pass

    @property
    def has_weight(self):
        return self._weight_field is not None

    def map_labels(self, positive=None, negative=None):
        if positive is not None:
            self._labels_map["pos"] = positive
        if negative is not None:
            self._labels_map["neg"] = negative

    @property
    def has_labels(self):
        return bool(self._labels_dict)

    @property
    def positive_label(self):
        return self._labels_map["pos"]

    @property
    def negative_label(self):
        return self._labels_map["neg"]

    def get_label_by_type(self, label_type):
        return self._labels_map[label_type]

    # def get_vertex(self, vertex):
    #     return vertex
    # @profile
    def load_graph(self, graph_path, direction=1, start_line=1, limit=graph_max_edge_number, blacklist=set(),
                   delimiter=','):
        """Load a graph.
        The graph should be in csv format

        Parameters
        ----------
        graph_path : string
            The path to the graph csv file.
        direction : int, optional
            The direction of the edge in the data, in other words
            if direction is 1 then 1,2 represents an edge 1->2 if 0 then 2->1,
        start_line : int, optional
            If set, this first line number that that should be inserted into the graph.
        limit : int, optional
            Maximal number of edges the should be loaded.
        blacklist : set, optional
            List of nodes the their edges should not be loaded.
        delimiter : string, optional
            Te delimiter used for parsing csv files.
        Returns
        -------

        """
        if isinstance(graph_path, str) or isinstance(graph_path, unicode):
            if graph_path.lower().endswith(".bz2"):
                f = utils.read_bz2(graph_path)
            elif graph_path.lower().endswith(".gz"):
                f = utils.read_gzip(graph_path)
            else:
                f = utils.read_file(graph_path)
        else:
            f = graph_path
        if isinstance(f, types.GeneratorType):
            for i, edge in enumerate(f):
                edge_attr = {}
                if i >= start_line:
                    if i == limit + start_line:
                        break
                    edge = utils.extract_items_from_line(edge, delimiter)[0:3]
                    if direction is 0:
                        edge = reversed(edge)
                    if len(edge) >= 2 and edge[0] not in blacklist and edge[1] not in blacklist:
                        edge = list(edge)
                        if edge[0] == edge[1]:
                            continue
                        edge_attr['edge_label'] = self.generate_edge_label(edge[0], edge[1])
                        if len(edge) == 3:
                            try:
                                edge_attr[self._weight_field] = int(edge[2])
                            except ValueError:
                                pass
                        edge = edge[:2]
                        edge.append(edge_attr)
                        self.add_edge(*edge)

    def load_labels(self, labels_path):
        labels_list = utils.read_file_by_lines(labels_path)
        for label in labels_list[1:]:
            label = utils.extract_items_from_line(label, ",")
            self._labels_dict[label[0].strip()] = label[1].strip()

    def get_node_label(self, vertex):
        return self._labels_dict[vertex]

    def generate_edge_label(self, vertex1, vertex2):
        if self.get_node_label(vertex1) == self.positive_label or self.get_node_label(vertex2) == self.positive_label:
            return self.positive_label
        return self.negative_label

    def get_edge_label(self, vertex1, vertex2):
        pass

    def get_vertex_name(self, vertex):
        return vertex

    @property
    def vertices(self):
        pass

    @property
    def edges(self):
        pass

    def get_vertex_edges(self, vertex):
        pass

    @property
    def number_of_vertices(self):
        pass

    @property
    def is_directed(self):
        pass

    @property
    def graph_degree(self):
        pass

    def has_edge(self, vertex1, vertex2):
        pass

    """Degrees """

    def get_vertex_degree(self, vertex):
        if self.is_directed:
            return self.get_vertex_out_degree(vertex) + self.get_vertex_in_degree(vertex)
        else:
            return self.get_vertex_out_degree(vertex)

    def get_vertex_in_degree(self, vertex):
        pass

    def get_vertex_out_degree(self, vertex):
        pass

    def get_vertex_bi_degree(self, vertex):
        if self.is_directed:
            return float(len(self.get_bi_neighbors(vertex)))
        return None

    def get_bi_neighbors(self, vertex):
        if self.is_directed:
            return utils.intersect(self.get_neighbors(vertex), self.get_followers(vertex))
        return []

    def average_neighbor_degree(self):
        pass

    def degree_centrality(self):
        pass

    def in_degree_centrality(self):
        pass

    def out_degree_centrality(self):
        pass

    """Neighborhoods """

    def get_neighborhoods_union(self, vertices):
        return utils.union(*[self.get_neighbors(vertex) for vertex in vertices])

    def get_neighborhoods_plus_union(self, vertices):
        return utils.union(*[self.get_neighbors_plus(vertex) for vertex in vertices])

    """Subgraphs """

    def get_subgraph(self, vertices):
        pass

    def neighbors_iter(self, vertex):
        pass

    def get_inner_subgraph(self, vertex1, vertex2):
        inner_subgraph = self.__class__(self.is_directed)
        for v_friend in self.neighbors_iter(vertex1):
            for u_friend in self.neighbors_iter(vertex2):
                if self.has_edge(u_friend, v_friend):
                    inner_subgraph.add_edge(u_friend, v_friend)
                if self.is_directed and self.has_edge(v_friend, u_friend):
                    inner_subgraph.add_edge(v_friend, u_friend)
        return inner_subgraph

    def get_neighborhoods_subgraph(self, vertices):
        vertices = utils.to_iterable(vertices)
        return self.get_subgraph(list(self.get_neighborhoods_union(vertices)))

    def get_neighborhoods_subgraph_edges(self, vertices):
        subgraph = self.get_neighborhoods_subgraph(vertices)
        return list(subgraph.edges)

    def get_neighborhoods_subgraph_plus(self, vertices):
        vertices = utils.to_iterable(vertices)
        return self.get_subgraph(list(self.get_neighborhoods_plus_union(vertices)))

    def get_neighborhoods_subgraph_edges_plus(self, vertices):
        subgraph = self.get_neighborhoods_subgraph_plus(vertices)
        return list(subgraph.edges)

    def get_neighbors(self, vertex):
        pass

    def get_neighbors_plus(self, vertex):
        return list(utils.union([vertex], self.get_neighbors(vertex)))

    """Scc/Wcc"""

    def get_scc_number(self, vertices):
        pass

    def get_scc_number_plus(self, vertices):
        pass

    def get_wcc_number(self, vertices):
        pass

    def get_inner_subgraph_scc_number(self, vertex1, vertex2):
        pass

    def get_inner_subgraph_wcc_number(self, vertex1, vertex2):
        pass

    """Common Friends"""

    @utils.memoize2
    def get_common_neighbors(self, vertex1, vertex2):
        vertex1_neighbors = self.get_neighbors(vertex1)
        vertex2_neighbors = self.get_neighbors(vertex2)
        return utils.intersect(vertex1_neighbors, vertex2_neighbors)

    def get_in_common_neighbors(self, vertex1, vertex2):
        return self.get_common_neighbors(vertex1, vertex2)

    def get_out_common_neighbors(self, vertex1, vertex2):
        vertex1_in_neighbors = self.get_neighbors(vertex1)
        vertex2_in_neighbors = self.get_neighbors(vertex2)
        return utils.intersect(vertex1_in_neighbors, vertex2_in_neighbors)

    def get_bi_common_neighbors(self, vertex1, vertex2):
        vertex1_bi_neighbors = self.get_bi_neighbors(vertex1)
        vertex2_bi_neighbors = self.get_bi_neighbors(vertex2)
        return utils.intersect(vertex1_bi_neighbors, vertex2_bi_neighbors)

    def get_transitive_friends(self, vertex1, vertex2):
        vertex1_out_neighbors = self.get_followers(vertex1)
        vertex2_in_neighbors = self.get_neighbors(vertex2)
        return utils.intersect(vertex1_out_neighbors, vertex2_in_neighbors)

    """Algorithms"""

    def nodes_number_of_cliques(self):
        pass

    # @property
    def pagerank(self):
        pass

    # @property
    def hits(self):
        pass

    # @property
    def eigenvector(self):
        pass

    # @property
    def load_centrality(self):
        pass

    # @property
    def communicability_centrality(self):
        pass

    def get_jaccard_coefficie_nt(self, vertex1, vertex2):
        pass

    def betweenness_centrality(self):
        pass

    def closeness(self):
        pass

    def get_shortest_path_length(self, vertex1, vertex2):
        pass

    @property
    def clusters(self):
        pass

    def disjoint_communities(self):
        pass

    @property
    def connected_components(self):
        pass

    def write_nodes_labels(self, output_path):
        return utils.write_hash_table(self._labels_dict, output_path, ("id", "label"))

    def get_followers(self, vertex):
        pass

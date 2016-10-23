import math

import numpy as np

from GraphML.utils.utils import memoize2


class FeatureExtractor(object):
    """
    Class for feature extraction from a graph.
    This class only implements the features.
    """

    def __init__(self, graph):
        """

        Parameters
        ----------
        graph : Graph
            One of the class that implements AbstractGraph API
        """
        self._graph = graph
        self.init_centrality_features()

    def init_centrality_features(self):
        """
        Initialize all centrality features
        """
        self._degree_centrality = None
        self._out_degree_centrality = None
        self._in_degree_centrality = None
        self._eigenvector = None
        self._pagerank = None
        self._hits = None
        self._closeness = None
        self._disjoint_communities = None
        self._load_centrality = None
        self._nodes_number_of_cliques = None
        self._average_neighbor_degree = None
        self._communicability_centrality = None
        self._betweenness = None

    # def get_output_path(self, feature_name):
    #     """
    #
    #     Parameters
    #     ----------
    #     feature_name :
    #
    #     Returns
    #     -------
    #
    #     """
    #     return "Output\\" + self._graph_name + feature_name + ".pkl"

    def get_node_label(self, vertex):
        """Return the vertex label.

        Parameters
        ----------
        vertex : string
            The vertex that its label should be returend.

        Returns
        -------
            Return the label if exist otherwise return the nehative label value
        """
        node_label = self._graph.get_node_label(vertex)
        if node_label is not None:
            return self._graph.get_node_label(vertex)
        else:
            return self._graph.negative_label

    def get_edge_label(self, vertex1, vertex2):
        """Return the edge label.

        Parameters
        ----------
        vertex1 : The source vertex
        vertex2 : The destantion vertex

        Returns
        -------
        The edge label

        """
        if self._graph.has_edge(vertex1, vertex2):
            return self._graph.get_edge_label(vertex1, vertex2)
        else:
            return self._graph.positive_label

    # def is_enabled(self, feature_name):
    #     if len(self._enabled_features) == 0:
    #         return True
    #     elif feature_name in self._enabled_features:
    #         return True
    #     return False

    # def get_vertices_by_label(self, label_values):
    #     """
    #
    #     Parameters
    #     ----------
    #     label_values :
    #
    #     Returns
    #     -------
    #
    #     """
    #     return self._graph.get_vertices_by_label(label_values)

    def get_graph(self):
        return self._graph

    def load_feature(self, feature, feature_func):
        if feature is None:
            feature = feature_func()

    # def load_centrality_features(self):
    #     return
    #     self._degree_centrality = self.load_feature("degree_centrality", self._graph.degree_centrality)
    #     self._out_degree_centrality = self.load_feature("out_degree_centrality", self._graph.out_degree_centrality)
    #     self._in_degree_centrality = self.load_feature("degree_centrality", self._graph.in_degree_centrality)
    #     self._pagerank = self.load_feature("pagerank", self._graph.pagerank)
    #     self._disjoint_communities = self.load_feature("disjoint_communities", self._graph.disjoint_communities)
    #     self._hits = self.load_feature("hits", self._graph.hits)
    #     self._closeness = self.load_feature("closeness", self._graph.closeness)
    #     self._load_centrality = self.load_feature("load_centrality", self._graph.load_centrality)
    #     self._nodes_number_of_cliques = self.load_feature("nodes_number_of_cliques", self._graph.nodes_number_of_cliques)
    #     self._average_neighbor_degree = self.load_feature("average_neighbor_degree",
    #     self._graph.average_neighbor_degree)
    #     self._communicability_centrality = self.load_feature("communicability", self._graph.communicability_centrality)
    #     self._betweenness = self.load_feature("betweenness", self._graph.betweenness_centrality)
    #     self._eigenvector = self.load_feature("eigenvector", self._graph.eigenvector)

    @property
    def nodes(self):
        return self._graph.vertices

    @property
    def edges_iter(self):
        return self._graph.edges_iter

    @property
    def number_of_nodes(self):
        """Return the of the vertices in the graph.

        Returns
        -------
        int: The number of vertices in the graph.
        """
        return self._graph.number_of_vertices

    # def get_vertex(self, vertex):
    #     return self._graph.get_vertex(vertex)

    def get_edge_weight(self, u, v):
        """ Return the edge weight.

        Parameters
        ----------
        v: string
            The source vertex.
        u: string
            The destination vertex.

        Returns
        -------
        int: The edge weight value
        """
        return self._graph.get_edge_weight(u, v)

    @memoize2
    def get_weight_features(self, v):
        """ Return weight based features.
        All the four weight features based on the same calculations
        to save calculation time we use this function to memoize the features.

        Parameters
        ----------
        v : string
            The vertex for witch the features is calculated.

        Returns
        -------
        tuple
            Return total_edge_weight, average_edge_weight, stdv_edge_weight, max_edge_weight
        """
        total_edge_weight = []
        for u in self._graph.neighbors_iter(v):
            total_edge_weight.append(self.get_edge_weight(v, u))
        weight_sum = sum(total_edge_weight)
        if len(total_edge_weight):
            return weight_sum, weight_sum / len(total_edge_weight), np.std(total_edge_weight), max(total_edge_weight)
        return 0, 0, 0, 0

    def get_total_edge_weight(self, v):
        """Return the total edge weight of v edges

        Parameters
        ----------
        v : string
            The vertex for witch the features is calculated.

        Returns
        -------
        int
            Total edge weight
        """
        return self.get_weight_features(v)[0]

    def get_average_edge_weight(self, v):
        """Return the average edge weight of v edges

        Parameters
        ----------
        v : string
            The vertex for witch the features is calculated.

        Returns
        -------
        int
            Average edge weight
        """
        return self.get_weight_features(v)[1]

    def get_stdv_edge_weight(self, v):
        """Return the stdv edge weight of v edges

        Parameters
        ----------
        v : string
            The vertex for witch the features is calculated.

        Returns
        -------
        int
            Stdv edge weight
        """
        return self.get_weight_features(v)[2]

    def get_max_edge_weight(self, v):
        """Return the total edge weight of v edges

        Parameters
        ----------
        v : string
            The vertex for witch the features is calculated.

        Returns
        -------
        int
            Total edge weight
        """
        return self.get_weight_features(v)[3]

    def get_level_two_common_friends(self, v, u):
        """Return average common friends number between two vertices.

        Parameters
        ----------
        v : string
            Vertex
        u : string
            Vertex

        Returns
        -------
        int
            Average common friends number between v an u friends.
        """
        total_common_friends = []
        for z in self._graph.neighbors_iter(u):
            total_common_friends.append(self.get_common_friends(v, z))
        return sum(total_common_friends) / float(len(total_common_friends))

    def get_level_two_jaccards_coefficient(self, v, u):
        """Return  average jaccards coefficient between two vertices.

        Parameters
        ----------
        v : string
            Vertex
        u : string
            Vertex

        Returns
        -------
        int
            Average jaccards coefficient between v an u friends.
        """
        total_jaccards_coefficient = []
        for z in self._graph.neighbors_iter(u):
            total_jaccards_coefficient.append(self.get_jaccards_coefficient(v, z))
        return sum(total_jaccards_coefficient) / float(len(total_jaccards_coefficient))

    def get_total_friends(self, v, u):
        """Return the number of distinct friends between two vertices v and u.
        Let v,u \in V ;  then Total-Friends of v and u will be defined as the
        number of vertices in the union of v friends with u friends.

        Parameters
        ----------
        v : string
            Vertex
        u : string
            Vertex

        Returns
        -------
        int
            The union of the number of friends of two vertices

        """
        return float(len(self._graph.get_neighborhoods_union([v, u])))

    def get_number_of_friends(self, v):
        """ Return the number of friends of a vertex.

        Parameters
        ----------
        v : string
            vertex
        """
        return len(self._graph.get_neighbors(v))

    def get_sum_of_friends(self, u, v):
        """Return the number of sum of friends between two vertices v and u.

        Parameters
        ----------
        v : string
            Vertex
        u : string
            Vertex

        Returns
        -------
        int
            The sum of the number of friends of two vertices

        """
        return self.get_number_of_friends(u) + self.get_number_of_friends(v)

    def get_common_friends(self, v, u):
        """ Return the number of common friends between two vertices v and u.

        Parameters
        ----------
        v : string
            Vertex
        u : string
            Vertex
        """
        return float(len(self._graph.get_common_neighbors(v, u)))

    def get_in_common_friends(self, v, u):
        """ Return the number of inbound common friends between two vertices v and u.

        Parameters
        ----------
        v : string
            Vertex
        u : string
            Vertex
        """
        return len(self._graph.get_in_common_neighbors(v, u))

    def get_out_common_friends(self, v, u):
        """ Return the number of outbound common friends between two vertices v and u.

        Parameters
        ----------
        v : string
            Vertex
        u : string
            Vertex
        """
        return len(self._graph.get_out_common_neighbors(v, u))

    def get_bi_common_friends(self, v, u):
        """ Return the number of bi-directional common friends between two vertices v and u.

        Parameters
        ----------
        v : string
            Vertex
        u : string
            Vertex
        """
        return len(self._graph.get_bi_common_neighbors(v, u))

    def get_jaccards_coefficient(self, v, u):
        """ This is one of the most known link prediction features. It measures
        similarity between two groups of items. Jaccard's Coefficient is
        defined as the ratio between CommonFriends and TotalFriends.
        If total friends is 0 then the function will return 0.

        Parameters
        ----------
        v : string
            Vertex
        u : string
            Vertex

        Returns
        -------
        int
             Return Jaccard's Coefficient of two vertices

        """
        total_friends = self.get_total_friends(v, u)
        if total_friends != 0:
            return float(self.get_common_friends(v, u) / total_friends)
        else:
            return 0

    def is_in_distance_on_n_hops(self, source, target, n):
        return n == len(self._graph.get_shortest_path_length_with_limit(source, target, n))

    def get_number_of_transitive_friends(self, v, u):
        """  For vertices v and u in a directed graph G calculates
         the number of transitive friends of u ,v and v,u.

        Parameters
        ----------
        v : string
            Vertex
        u : string
            Vertex

        Returns
        -------
        int
             Return number of transitive friends of u ,v and v,u.

        """
        return len(self._graph.get_transitive_friends(v, u))

    def get_in_degree_centrality(self, node):
        self.load_feature(self._in_degree_centrality, self._graph.in_degree_centrality)
        if node in self._in_degree_centrality.keys():
            return self._in_degree_centrality[node]
        else:
            return 0

    def get_out_degree_centrality(self, node):
        self.load_feature(self._out_degree_centrality, self._graph.out_degree_centrality)
        if node in self._out_degree_centrality.keys():
            return self._out_degree_centrality[node]
        else:
            return 0

    def get_nodes_number_of_cliques(self, node):
        self.load_feature(self._nodes_number_of_cliques, self._graph.nodes_number_of_cliques)
        if node in self._nodes_number_of_cliques:
            return self._nodes_number_of_cliques[node]
        else:
            return 0

    def get_degree_centrality(self, node):
        self.load_feature(self._degree_centrality, self._graph.degree_centrality)
        if node in self._degree_centrality.keys():
            return self._degree_centrality[node]
        else:
            return 0

    def get_pagerank(self, node):
        self.load_feature(self._pagerank, self._graph.pagerank)
        if node in self._pagerank[0].keys():
            return self._pagerank[0][node]
        else:
            return 0

    def get_hubs(self, node):
        self.load_feature(self._hits, self._graph.hits)
        if node in self._hits[0].keys():
            return self._hits[0][node]
        else:
            return 0

    def get_authorities(self, node):
        self.load_feature(self._hits, self._graph.hits)
        if node in self._hits[1].keys():
            return self._hits[1][node]
        else:
            return 0

    def get_average_neighbor_degree(self, node):
        self.load_feature(self._average_neighbor_degree, self._graph.average_neighbor_degree)
        if node in self._average_neighbor_degree.keys():
            return self._average_neighbor_degree[node]
        else:
            return 0

    def get_eigenvector(self, node):
        self.load_feature(self._eigenvector, self._graph.eigenvector)
        if node in self._eigenvector.keys():
            return self._eigenvector[node]
        else:
            return 0

    def get_betweenness(self, node):
        self.load_feature(self._betweenness, self._graph.betweenness)
        if node in self._betweenness.keys():
            return self._betweenness[node]
        else:
            return 0

    def get_load_centrality(self, node):
        self.load_feature(self._load_centrality, self._graph.load_centrality)
        if node in self._load_centrality.keys():
            return self._load_centrality[node]
        else:
            return 0

    def get_communicability_centrality(self, node):
        self.load_feature(self._communicability_centrality, self._graph.communicability_centrality)
        if node in self._communicability_centrality.keys():
            return self._communicability_centrality[node]
        else:
            return 0

    def get_closeness(self, node):
        self.load_feature(self._closeness, self._graph.closeness)
        if node in self._closeness.keys():
            return self._closeness[node]
        else:
            return 0

    def get_preferential_attachment_score(self, v, u):
        """ This is a well-known features. It is based on the idea that in
        social networks the rich get richer. The Preferential Attachment
        defined as the multiplication of the number of friends of two
        vertices v and u.

        Parameters
        ----------
        v : string
            Vertex
        u : string
            Vertex

        Returns
        -------
        int
             Return the multiplication of the number of friends of two vertices v and u.
        """
        return self.get_number_of_friends(v) * float(self.get_number_of_friends(u))

    def get_knn_in_weight(self, v):
        return 1 / math.sqrt(1 + float(self.get_in_degree(v)))

    def get_knn_out_weight(self, v):
        return 1 / math.sqrt(1 + float(self.get_out_degree(v)))

    def get_knn_weight1(self, v, u):
        return self.get_knn_in_weight(v) + self.get_knn_in_weight(u)

    def get_knn_weight2(self, v, u):
        return self.get_knn_in_weight(v) + self.get_knn_out_weight(u)

    def get_knn_weight3(self, v, u):
        return self.get_knn_out_weight(v) + self.get_knn_in_weight(u)

    def get_knn_weight4(self, v, u):
        return self.get_knn_out_weight(v) + self.get_knn_out_weight(u)

    def get_knn_weight5(self, v, u):
        return self.get_knn_in_weight(v) * self.get_knn_in_weight(u)

    def get_knn_weight6(self, v, u):
        return self.get_knn_in_weight(v) * self.get_knn_out_weight(u)

    def get_knn_weight7(self, v, u):
        return self.get_knn_out_weight(v) * self.get_knn_in_weight(u)

    def get_knn_weight8(self, v, u):
        return self.get_knn_out_weight(v) * self.get_knn_out_weight(u)

    def get_cosine(self, v, u):
        return self.get_common_friends(v, u) / self.get_preferential_attachment_score(v, u)

    def get_inner_subgraph_link_number(self, v, u):
        return len(self._graph.get_inner_subgraph(v, u).edges)

    def is_linked(self, v, u):
        """ Return true if two edges v and u linked otherwise false.

        Parameters
        ----------
        v : string
            Vertex
        u : string
            Vertex
        """
        if v == u or self._graph.has_edge(u, v) or self._graph.has_edge(v, u):
            return 1
        else:
            return 0

    def get_friend_measure(self, v, u):
        """When looking at two vertices in a social network, we can assume that the
        more connections their neighborhoods have with each other, the higher the
        chances are that the two vertices are connected. We accept the logic of
        this statement and define the Friends measure as the number of
        connections between u and v neighborhoods. The formal
        definitions of Friends measure is: Let be G = <V, E>
        and u,v \in V.

        Parameters
        ----------
        v : string
            Vertex
        u : string
            Vertex

        Returns
        -------
        int
             Return the friend measure of two vertices v and u.
        """
        friend_measure = 0
        for v_friend in self._graph.neighbors_iter(v):
            for u_friend in self._graph.neighbors_iter(u):
                # friend_measure = Parallel(n_jobs=4)(delayed(self.is_linked)(v_friend, u_friend) for u_friend in self._graph.get_neighbors(u))
                # friend_measure = sum(friend_measure)
                if self.is_linked(v_friend, u_friend):
                    friend_measure += 1
        return friend_measure

    def get_alt_friend_measure(self, v, u):
        """Alternative implementation of friend measure.

        Parameters
        ----------
        v : string
            Vertex
        u : string
            Vertex

        Returns
        -------
        int
             Return the friend measure of two vertices v and u.
        """
        v_neighbors, u_neighbors = set(), set()
        for v_friend in self._graph.neighbors_iter(v):
            v_neighbors.update(self._graph.get_neighbors(v_friend))
        for u_friend in self._graph.neighbors_iter(u):
            u_neighbors.update(self._graph.get_neighbors(u_friend))
        return len(u_neighbors & v_neighbors)

    def get_secondary_neighbors_log(self, v):
        """ Return the som of logarithm of the secondary neighbor count.

        Parameters
        ----------
        v : string
            Vertex

        References
        ----------
        Al Hasan, Mohammad, et al.
        Link prediction using supervised learning (2006).
        """
        secondary_friends_number = 0
        for v_friend in self._graph.neighbors_iter(v):
            secondary_friends_number += self.get_number_of_friends(v_friend)
        if secondary_friends_number != 0:
            secondary_friends_number = math.log10(secondary_friends_number)
        return secondary_friends_number

    def get_adamic_adar_index(self, v, u):
        """Adamic adar is a similarity measure for undirected graphs which measures how strongly two
        vertices are related. Higher scores will be given to edges that have rare connections.

        Parameters
        ----------
        v : string
            Vertex
        u : string
            Vertex

        """
        u, v, p = self._graph.get_adamic_adar_index(v, u)
        return p

    def get_resource_allocation_index(self, v, u):
        u, v, p = self._graph.get_resource_allocation_index(v, u)
        return p

    def is_same_community(self, v, u):
        return self._disjoint_communities[v] == self._disjoint_communities[u]

    # def is_same_community(self, v, u):
    # disjoint_communities = self._disjoint_communities
    #     for com in set(disjoint_communities.values()):
    #         list_nodes = [nodes for nodes in disjoint_communities.keys()
    #                       if disjoint_communities[nodes] == com]
    #         if u in list_nodes and v in list_nodes:
    #             return 1
    #     return 0

    def get_number_of_neighbors_communities(self, v):
        communities = []
        for u in self._graph.neighbors_iter(v):
            communities.append(self._disjoint_communities[u])
        return len(set(communities))

    def get_bi_degree_density(self, v):
        if self._graph.is_directed:
            return self._graph.get_vertex_bi_degree(v) / self._graph.get_vertex_degree(v)
        else:
            return None

    def get_communication_reciprocity(self, v):
        if self._graph.is_directed:
            try:
                return self._graph.get_vertex_bi_degree(v) / self._graph.get_vertex_out_degree(v)
            except ZeroDivisionError:
                return 0
        else:
            return None

    def get_out_degree_density(self, v):
        if self._graph.is_directed:
            return self._graph.get_vertex_out_degree(v) / self._graph.get_vertex_degree(v)
        else:
            return None

    def get_in_degree_density(self, v):
        if self._graph.is_directed:
            return self._graph.get_vertex_in_degree(v) / self._graph.get_vertex_degree(v)
        else:
            return None

    def get_bi_degree(self, v):
        return self._graph.get_vertex_bi_degree(v)

    def get_in_degree(self, v):
        in_degree = self._graph.get_vertex_in_degree(v)
        if in_degree is None:
            return 0
        return in_degree

    def get_out_degree(self, v):
        out_degree = self._graph.get_vertex_out_degree(v)
        if out_degree is None:
            return 0
        return out_degree

    def get_vertex_degree(self, v):
        return self._graph.get_vertex_degree(v)

    def get_subgraph_node_link_number(self, v):
        return len(self._graph.get_neighborhoods_subgraph_edges(v))

    def get_subgraph_node_link_number_plus(self, v):
        return len(self._graph.get_neighborhoods_subgraph_edges_plus(v))

    def get_subgraph_link_number(self, v, u):
        return len(self._graph.get_neighborhoods_subgraph_edges([v, u]))

    def get_subgraph_link_number_plus(self, v, u):
        return len(self._graph.get_neighborhoods_subgraph_edges_plus([v, u]))

    def get_density_neighborhood_subgraph(self, v):
        try:
            return self._graph.get_vertex_degree(v) / self.get_subgraph_node_link_number(v)
        except ZeroDivisionError:
            return 0

    def get_density_neighborhood_subgraph_plus(self, v):
        try:
            return self._graph.get_vertex_degree(v) / self.get_subgraph_node_link_number_plus(v)
        except ZeroDivisionError:
            return 0

    def get_average_scc(self, v):
        try:
            return self._graph.get_vertex_degree(v) / self._graph.get_scc_number(v)
        except ZeroDivisionError:
            return 0

    def get_average_scc_plus(self, v):
        try:
            return self._graph.get_vertex_degree(v) / self._graph.get_scc_number_plus(v)
        except ZeroDivisionError:
            return 0

    def get_average_wcc(self, v):
        try:
            return self._graph.get_vertex_degree(v) / self._graph.get_wcc_number(v)
        except ZeroDivisionError:
            return 0

    def get_shortest_path_length(self, v, u):
        return self._graph.get_shortest_path_length(v, u)

    def is_opposite_direction_friends(self, v, u):
        if self._graph.is_directed:
            return self._graph.has_edge(v, u) and self._graph.has_edge(u, v)
        else:
            return self._graph.has_edge(v, u)

    def get_scc_number(self, node1, node2):
        return self._graph.get_scc_number([node1, node2])

    def get_scc_number_plus(self, node1, node2):
        return self._graph.get_scc_number_plus([node1, node2])

    def get_wcc_number(self, node1, node2):
        return self._graph.get_wcc_number([node1, node2])

    def get_inner_subgraph_scc_number(self, node1, node2):
        return self._graph.get_inner_subgraph_scc_number(node1, node2)

    def get_inner_subgraph_wcc_number(self, node1, node2):
        return self._graph.get_inner_subgraph_wcc_number(node1, node2)

    def get_label(self, v, default_label='neg'):
        """ Return the label of a vertex.

        Parameters
        ----------
        v : string
            Vertex
        default_label : string, (default=neg)
            The value to return when there is no lable.
        """
        label = self.get_node_label(v)
        if label is None:
            return self._graph.get_label_by_type(default_label)
        else:
            return label

    def get_vertex(self, v):
        return v
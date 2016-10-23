from GraphML.utils import utils
import string_gtgraph as gt
from GraphML.graphs import AbstractGraph


class GtGraph(AbstractGraph):
    __slots__ = []
    def __init__(self, is_directed=False):
        super(GtGraph, self).__init__()
        self._graph = gt.StringGtGraph(directed=is_directed)
        self._graph.vertex_properties["vertex-name"] = self._graph.new_vertex_property("string")
        self._graph.vertex_properties["label"] = self._graph.new_vertex_property("string")

    # def add_vertex(self, vertex_name):
    # if not vertex_name in self._vertices:
    #         vertex = self._graph.add_vertex()
    #         self._vertices[vertex_name] = vertex
    #     else:
    #         vertex = self._vertices[vertex_name]
    #     return vertex

    def __getitem__(self, key):
        return self._graph[key]

    def draw_graph(self, output_name="graph"):
        gt.graph_draw(self._graph, output=output_name + ".pdf")

    ##
    # Todo weights
    ##
    def add_edge(self, vertex1, vertex2, edge_atrr):
        vertex1, vertex2 = str(vertex1).strip(), str(vertex2).strip()
        if not self.has_edge(vertex1, vertex2):
            v1 = self.add_vertex(vertex1)
            v2 = self.add_vertex(vertex2)
            self._graph.add_edge(v1, v2)

    def get_vertices_by_label(self, label_values):
        vertices_by_labels = {}
        for vertex in self.vertices:
            label = self.get_label(vertex)
            if label in label_values:
                if label not in vertices_by_labels:
                    vertices_by_labels[label] = []
                vertices_by_labels[label].append(vertex)
        return vertices_by_labels

    def get_label(self, vertex):
        return self._graph.vp["label"][vertex]
        # return self._labels_dict[self.get_vertex_name(vertex)]

    def add_vertex(self, vertex):
        v = self._graph.add_vertex(vertex)
        self._graph.vp["vertex-name"][v] = vertex
        if vertex in self._labels_dict:
            self._graph.vp["label"][v] = self._labels_dict[vertex]
        return v

    def create_edge(self, vertex1, vertex2):
        v1 = self.add_vertex(vertex1)
        v2 = self.add_vertex(vertex2)
        return int(v1), int(v2)

    def add_edge_list(self, edges):
        edge_list = []
        count = 0
        for edge in edges:
            count += 1
            if count % 100000 == 0:
                print count
            # if count % 10000 == 0:
            #     break
            edge = utils.extract_items_from_line(edge, ",")
            vertex1, vertex2 = self.create_edge(edge[0].strip(), edge[1].strip())
            if not self.has_edge(vertex1, vertex2) and (vertex1, vertex2) not in edge_list:
                edge_list.append((vertex1, vertex2))
        self._graph.add_edge_list(edge_list)

    def has_edge(self, vertex1, vertex2):
        return not self._graph.edge(vertex1, vertex2) is None

    @property
    def edges(self):
        return list(self._graph.edges())

    def get_vertex_name(self, vertex):
        return self._graph.vp["vertex-name"][vertex]

        # Todo vertices list
        # def load_graph(self, graph_data, direction=1):
        #     json_object = utils.is_json(graph_data)
        #     if not json_object is False:
        #         graph_data = json_object
        #     else:
        #         graph_data = utils.read_file_by_lines(graph_data)
        #     self.add_edge_list(graph_data)
        #     del self._labels_dict
        # for edge in graph_data:
        #     count += 1
        #     if count % 100000 == 0:
        #         print count
        #     if json_object is False:
        #         edge = utils.extract_items_from_line(edge, ",")
        #     if direction:
        #         self.add_edge(edge[0].strip(), edge[1].strip())
        #     else:
        #         self.add_edge(edge[1].strip(), edge[1].strip())

    @property
    def vertices(self):
        return list(self._graph.vertices())
        # return self._vertices.keys()

    @property
    def number_of_vertices(self):
        return self._graph.num_vertices()

    @property
    def is_directed(self):
        return self._graph.is_directed()

    # @property
    def pagerank(self):
        return gt.pagerank(self._graph)

    # @property
    def hits(self):
        egg, authority, hub = gt.hits(self._graph, max_iter=100, epsilon=1.0e-8)
        return authority, hub

    # @property
    def eigenvector(self):
        max_egg, egg = gt.eigenvector(self._graph)
        return egg

    # @property
    def betweenness_centrality(self):
        vp, ep = gt.betweenness(self._graph)
        return vp

    def closeness(self):
        return gt.closeness(self._graph, norm=True, harmonic=True)

    def get_followers(self, vertex):
        return list(self[vertex].in_neighbours())

    def get_neighbors(self, vertex):
        return list(self[vertex].out_neighbours())

    def get_vertex_in_degree(self, vertex):
        return float(self[vertex].in_degree())

    def get_neighbors_plus(self, vertex):
        return list(utils.union([self[vertex]], self.get_neighbors(vertex)))

    def get_vertex_out_degree(self, vertex):
        return float(self[vertex].out_degree())

    def get_shortest_path_length(self, vertex1, vertex2):
        return len(gt.shortest_path(self._graph, self[vertex1], self[vertex2])[1])

    def get_subgraph(self, vertices):
        if self.is_directed:
            subgraph = self.__class__(True)
        else:
            subgraph = self.__class__(False)
        for vertex1 in vertices:
            for vertex2 in vertices:
                if self.has_edge(vertex1, vertex2):
                    subgraph.add_edge(vertex1, vertex2)
        return subgraph

    def get_subgraph_edges(self, subgraph):
        return list(subgraph.edges_iter)

    def get_vertex_all_edges(self, vertex):
        return vertex.all_edges()

    @property
    def get_strongly_connected_components_number(self):
        comp, hist = gt.label_components(self._graph)
        return len(set(comp.a.copy()))

    def get_scc_number(self, vertices):
        neighborhood_subgraph = self.get_neighborhoods_subgraph(vertices)
        scc = neighborhood_subgraph.get_strongly_connected_components_number
        return scc

    def get_scc_number_plus(self, vertices):
        neighborhood_subgraph = self.get_neighborhoods_subgraph_plus(vertices)
        scc = neighborhood_subgraph.get_strongly_connected_components_number
        return scc

    def get_inner_subgraph_scc_number(self, vertex1, vertex2):
        neighborhood_subgraph = self.get_inner_subgraph(vertex1, vertex2)
        scc = neighborhood_subgraph.get_strongly_connected_components_number
        return scc

    # _________________________________________________________________
    # Todo test
    @property
    def get_weakly_connected_components_number(self):
        self._graph.set_directed(False)
        comp, hist = gt.label_components(self._graph)
        # print hist.copy().size == len(set(comp.a.copy()))
        return hist.copy().size

    # Todo test
    @property
    def clusters(self):
        return gt.local_clustering(self._graph)
        # print(gt.vertex_average(g, clust))

    # Todo ask micky
    def disjoint_communities(self):
        return []

    def get_wcc_number(self, vertices):
        neighborhood_subgraph = self.get_neighborhoods_subgraph(vertices)
        return neighborhood_subgraph.get_weakly_connected_components_number

    def get_inner_subgraph_wcc_number(self, vertex1, vertex2):
        neighborhood_subgraph = self.get_inner_subgraph(vertex1, vertex2)
        return neighborhood_subgraph.get_weakly_connected_components_number

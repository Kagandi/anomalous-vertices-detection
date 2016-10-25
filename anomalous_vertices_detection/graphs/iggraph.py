try:
    import igraph
except ImportError:
    pass

from anomalous_vertices_detection.configs.config import *
from anomalous_vertices_detection.graphs import AbstractGraph
from anomalous_vertices_detection.utils.utils import *


class IGraph(AbstractGraph):
    __slots__ = ['_is_directed']

    def __init__(self, is_directed=False, weight_field=None, graph_obj=[]):
        super(IGraph, self).__init__(weight_field)
        if graph_obj:
            self._graph = graph_obj
        else:
            self._graph = igraph.Graph(directed=is_directed)
            self._is_directed = is_directed

    def save_graph(self):
        pass

    def load_saved_graph(self):
        pass

    def get_node_label(self, vertex):
        return self._labels_dict[self.get_vertex_name(vertex)]

    def add_edge(self, vertex1, vertex2, edge_atrr={}):
        try:
            vertex1, vertex2 = vertex1.strip(), vertex2.strip()
        except:
            pass
        if not self.has_edge(vertex1, vertex2):
            vertex1 = self.add_vertex(vertex1)
            vertex2 = self.add_vertex(vertex2)
            self._graph.add_edge(vertex1, vertex2)
            edge_id = self._graph.get_eid(vertex1, vertex2)
            self._graph.es[edge_id][self._weight_field] = edge_atrr[self._weight_field]
            self._graph.es[edge_id]["edge_label"] = edge_atrr["edge_label"]

    def add_vertex(self, vertex):
        if not self.has_vertex(vertex):
            self._graph.add_vertex(vertex)
        return self._graph.vs.find(vertex).index

    def load_graph(self, graph_path, direction=1, start_line=1, limit=graph_max_edge_number, blacklist=set(),
                   delimiter=','):
        graph_path = read_file(graph_path)
        vetices_list = set()
        edges_list, label_list, weight_list = [], [], []
        for i, edge in graph_path:
            edge_attr = {}
            if i >= start_line:
                if i == limit + start_line:
                    break
                edge = extract_items_from_line(edge, delimiter)[0:3]
                if direction is 0:
                    edge = reversed(edge)
                if len(edge) >= 2 and edge[0] not in blacklist and edge[1] not in blacklist:
                    vetices_list.add(edge[0])
                    vetices_list.add(edge[1])
                    label_list.append(self.generate_edge_label(edge[0], edge[1]))
                    if len(edge) == 3:
                        weight_list.append(int(edge[2]))
                        edge = edge[:2]
                    edges_list.append(edge)
        self.load_igraph(list(vetices_list), edges_list, weight_list, label_list)

    def load_igraph(self, vertices_list, edges_list, weight_list, label_list):
        self._graph.add_vertices(len(vertices_list))
        self._graph.vs["name"] = vertices_list
        v_dict = {vertices_list[i]: i for i in range(len(vertices_list))}
        edges_list = [(v_dict[e[0]], v_dict[e[1]]) for e in edges_list]
        self._graph.add_edges(edges_list)
        if weight_list:
            self._graph.es[self._weight_field] = weight_list
        if edges_list:
            self._graph.es["edge_label"] = label_list

    def delete_graph(self):
        del self._graph
        self._graph = igraph.Graph(self._is_directed)

    def remove_edge(self, v, u, *args):
        self._graph.delete_edges(self._graph.get_eid(v, u))

    # return edges ids insted of edges
    def get_vertex_edges(self, vertex, mode="out"):
        edges_ind = []
        if mode == "out":
            edges_ind = self._graph.incident(vertex, mode="out")
        if mode == "in":
            edges_ind = self._graph.incident(vertex, mode="in")
        if mode == "all":
            edges_ind = self._graph.incident(vertex, mode="in") + self._graph.incident(vertex, mode="out")
        return [self._graph.es.find(x).tuple for x in edges_ind]

    def get_edge_label(self, vertex1, vertex2):
        if self.has_edge(vertex1, vertex2):
            # self._graph.get_eid(vertex1,vertex2)
            if "edge_label" in self._graph.es.attributes():
                return self._graph.es['edge_label'][self._graph.get_eid(vertex1, vertex2)]
        return self.negative_label

    def get_edge_weight(self, u, v):
        return self._graph.es[self._weight_field][self._graph.get_eid(u, v)]

    def edge(self, u, v):
        return self._graph.es[self._graph.get_eid(u, v)].attributes()

    @property
    def vertices(self):
        return self._graph.vs.indices

    @property
    def vertices_iter(self):
        return self.vertices()

    @property
    def edges(self):
        return [self._graph.es.find(x).tuple for x in self._graph.es.indices]

    @property
    def edges_iter(self):
        for x in self._graph.es.indices:
            yield self._graph.es.find(x).tuple

    @property
    def number_of_vertices(self):
        return self._graph.vcount()

    @property
    def is_directed(self):
        return self._is_directed

    # todo upgrade
    def pagerank(self):
        return self._graph.pagerank(weights=self._weight_field)

    def eigenvector(self):
        return self._graph.evcent(directed=self.is_directed(), weights=self._weight_field)

    def closeness(self):
        return self._graph.closeness

    def get_shortest_path_length(self, vertex1, vertex2):
        try:
            self._graph.bfs()
            return self._graph.shortest_paths(source=vertex1, target=vertex2)
        except:
            return 0

    def get_vertex_name(self, vertex):
        if self.has_vertex(vertex):
            return self._graph.vs[vertex]["name"]
        else:
            return vertex

    def get_vertex_in_degree(self, vertex):
        return float(self._graph.indegree(vertex))

    def get_vertex_out_degree(self, vertex):
        return float(self._graph.outdegree(vertex))

    def has_edge(self, node1, node2):
        if not self.has_vertex(node1):
            return False
        if not self.has_vertex(node2):
            return False
        return self._graph.are_connected(node1, node2)

    def has_vertex(self, vertex):
        try:
            self._graph.vs.find(vertex)
            return True
        except ValueError:
            return False

    @memoize
    def get_neighbors(self, node):
        return self._graph.successors(node)

    def neighbors_iter(self, node):
        for node in self.get_neighbors(node):
            yield node

    def get_followers(self, node):
        if self.is_directed:
            return self._graph.predecessors(node)
        else:
            return self.get_neighbors(node)

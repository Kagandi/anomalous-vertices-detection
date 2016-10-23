from graphlab import (Edge, SArray, SFrame, SGraph, aggregate,
                      connected_components, pagerank,
                      shortest_path, canvas,  aggregate as agg)

from GraphML.configs.config import *
from GraphML.graphs import AbstractGraph
from GraphML.utils import utils


class GlGraph(AbstractGraph):

    __slots__ = ['_is_directed']

    def __init__(self, is_directed=False, weight_field="", graph_obj=SGraph()):
        super(GlGraph, self).__init__(weight_field)
        self._graph = graph_obj
        self._is_directed = is_directed

    def set_sgraph(self, sgraph):
        self._graph = sgraph

    def load_graph(self, graph_path, direction=1, start_line=0, limit=graph_max_edge_number, header=False):
        json_object = utils.is_json(graph_path)
        if json_object is not False:
            # print json_object
            graph_path = SFrame(SArray(json_object).unpack())
            graph_path.rename({'X.0': 'X1', 'X.1': 'X2', 'X.2': 'Weight'})
        else:
            # load_sgraph()
            graph_path = SFrame.read_csv(graph_path, header=False, column_type_hints={
                'X1': str, 'X2': str}, nrows=graph_max_edge_number, skiprows=start_line)
            if self._weight_field != "":
                graph_path.rename({'X3': 'Weight'})
        # print graph_data
        self._graph = self._graph.add_edges(
            graph_path, src_field='X1', dst_field='X2')
        if not self.is_directed:
            self.to_undirected()

    def draw_graph(self):
        self._graph.show(vlabel='id', arrows=True)
        raw_input()

    def add_edge(self, vertex1, vertex2, edge_atrr={}):
        self._graph = self._graph.add_edges(Edge(vertex1, vertex2, attr=edge_atrr))

    @property
    def is_directed(self):
        return self._is_directed

    @property
    def vertices(self):
        return self._graph.get_vertices()['__id']

    @property
    def edges(self):
        edges = self._graph.get_edges()
        if self.is_directed:
            return edges
        else:
            # self._graph._edges['edata'] = edges['__src_id'] + edges['__dst_id']
            edges['edata'] = edges.apply(
                lambda x: x['__src_id'] + x['__dst_id'] if x['__src_id'] > x['__dst_id'] else x['__dst_id'] + x[
                    '__src_id'])
            edges = edges.groupby(['edata'], {'__src_id': aggregate.SELECT_ONE('__src_id'),
                                              '__dst_id': aggregate.SELECT_ONE('__dst_id')})
            del edges['edata']
            return edges

    @property
    def number_of_vertices(self):
        return len(self.vertices)

    def has_edge(self, vertex1, vertex2):
        return len(self._graph.get_edges(vertex1, vertex2)) == 1

    def to_undirected(self):
        # Possible to do apply and afterwards unique
        for edge in self.edges:
            if not self.has_edge(edge["__dst_id"], edge["__src_id"]):
                self.add_edge(edge["__dst_id"], edge["__src_id"])

    # following
    def get_neighbors(self, node):
        return self._graph.get_edges(node, [None])["__dst_id"]

    def neighbors_iter(self, node):
        return self.get_neighbors(node)

    def get_followers(self, node):
        return self._graph.get_edges([None], node)["__src_id"]

    def get_vertex_in_degree(self, vertex):
        return float(len(self.get_followers(vertex)))

    def get_vertex_out_degree(self, vertex):
        return float(len(self.get_neighbors(vertex)))

    # @property
    def pagerank(self):
        return pagerank.create(self._graph)['pagerank']

    def get_shortest_path(self, vertex):
        shortest_paths = shortest_path.create(
            self._graph, vertex, weight_field=self._weight_field)['distance']["distance"]
        return [x for x in shortest_paths if x != 1e30]

    def get_shortest_path_length(self, vertex1, vertex2):
        shortest_paths = shortest_path.create(self._graph, vertex1, weight_field=self._weight_field)['distance']
        path_distance = (
            item for item in shortest_paths if item["__id"] == vertex2).next()['distance']
        # print path_distance
        return 0 if path_distance == 1e30 else path_distance
    #
    # def get_subgraph(self, nodes):
    #     # print nodes
    #     try:
    #         return GlGraph(self._is_directed, self._weight_field, self._graph.get_neighborhood(nodes, 0))
    #     except RuntimeError:
    #         return GlGraph(self._is_directed, self._weight_field)

    def get_subgraph(self, ids, radius=1, full_subgraph=True):
            verts = ids

            ## find the vertices within radius (and the path edges)
            for i in range(radius):
                edges_out = self._graph.get_edges(src_ids=verts)
                # edges_in = self._graph.get_edges(dst_ids=verts)

                verts = list(edges_out['__src_id']) + list(edges_out['__dst_id'])
                verts = list(set(verts))

            ## make a new graph to return and add the vertices
            g = SGraph()
            g = g.add_vertices(self._graph.get_vertices(verts), vid_field='__id')

            ## add the requested edge set
            if full_subgraph is True:
                df_induced = self._graph.get_edges(src_ids=verts)
                # induced_edge_in = self._graph.get_edges(dst_ids=verts)
                # df_induced = induced_edge_out.append(induced_edge_in)
                df_induced = df_induced.groupby(df_induced.column_names(), {})

                verts_sa = SArray(list(ids))
                edges = df_induced.filter_by(verts_sa, "__src_id")
                edges.append(df_induced.filter_by(verts_sa, "__dst_id"))

            g = g.add_edges(edges, src_field='__src_id', dst_field='__dst_id')
            return g


    def get_number_weakly_connected_components(self, g):
        cc = connected_components.create(g)
        return len(cc['component_size'])

    def get_node_wcc_number(self, node):
        neighborhood_subgraph = self.get_neighborhoods_subgraph(node)
        return self.get_number_weakly_connected_components(neighborhood_subgraph._graph)

    def get_wcc_number(self, vertices):
        neighborhood_subgraph = self.get_neighborhoods_subgraph(vertices)
        return self.get_number_weakly_connected_components(neighborhood_subgraph._graph)

    def get_inner_subgraph_wcc_number(self, node1, node2):
        inner_subgraph = self.get_inner_subgraph(node1, node2)
        return self.get_number_weakly_connected_components(inner_subgraph._graph)

    # closeness_centrality
    def closeness(self, u=None, normalized=True):
        """
        if distance is not None:
            # use Dijkstra's algorithm with specified attribute as edge weight
            path_length = functools.partial(nx.single_source_dijkstra_path_length,
                                            weight=distance)
        else:
            path_length = self.get_shotest_paths
            :param normalized:
            :param u:
        """
        if u is None:
            nodes = self._graph.get_vertices()
        else:
            nodes = [u]
        closeness_centrality = {}
        for n in nodes["__id"]:
            sp = self.get_shortest_path(n)
            totsp = sum(sp)
            if totsp > 0.0 and self.number_of_vertices > 1:
                closeness_centrality[n] = (len(sp) - 1.0) / totsp
                # normalize to number of nodes-1 in connected part
                if normalized:
                    s = (len(sp) - 1.0) / (self.number_of_vertices - 1)
                    closeness_centrality[n] *= s
            else:
                closeness_centrality[n] = 0.0
        if u is not None:
            return closeness_centrality[u]
        else:
            return closeness_centrality

    # Todo

    def get_node_scc_number(self, node):
        # neighborhood_subgraph = self.get_node_neighborhood_subgraph(node)
        # return nx.number_strongly_connected_components(neighborhood_subgraph)
        return 0

        # Todo

    def get_node_scc_number_plus(self, node):
        # neighborhood_subgraph = self.get_node_neighborhood_subgraph_plus(node)
        # return nx.number_strongly_connected_components(neighborhood_subgraph)
        return 0

        # Todo

    def get_scc_number(self, vertices):
        # neighborhood_subgraph = self.get_neighborhoods_subgraph(node1, node2)
        # return nx.number_strongly_connected_components(neighborhood_subgraph)
        return 0

        # Todo

    def get_scc_number_plus(self, vertices):
        # neighborhood_subgraph = self.get_neighborhoods_subgraph_plus(node1, node2)
        # return nx.number_strongly_connected_components(neighborhood_subgraph)
        return 0

        # Todo

    def get_inner_subgraph_scc_number(self, node1, node2):
        # neighborhood_subgraph = self.get_inner_subgraph(node1, node2)
        # return nx.number_strongly_connected_components(neighborhood_subgraph)
        return 0

    # Todo
    @property
    def clusters(self):
        # return nx.clustering(self._graph)
        return []

    # Todo
    def disjoint_communities(self):
        # if self.is_directed:
        # partition = community.best_partition(self._graph.to_undirected())
        # else:
        # partition = community.best_partition(self._graph)
        # return partition
        return []

    # Todo
    def hits(self):
        return {}

    # Todo
    def eigenvector(self):
        return {}

    # Todo
    def load_centrality(self):
        # return nx.load_centrality(self._graph)
        return []

    # Todo
    def communicability_centrality(self):
        # if self.is_directed:
        #     return nx.communicability_centrality(self._graph.to_undirected())
        # else:
        #     return nx.communicability_centrality(self._graph)
        return []

    # Todo
    def betweenness_centrality(self):
        # return nx.betweenness_centrality(self._graph)
        return []

    def save_graph(self, graph_name):
        self._graph.save(graph_name + ".csv", format='csv')
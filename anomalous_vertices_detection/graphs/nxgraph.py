from itertools import product

import community
import networkx as nx
import numpy as np

from anomalous_vertices_detection.graphs import AbstractGraph
from anomalous_vertices_detection.utils.graphlab_utils import load_nxgraph_from_sgraph, save_nx_as_sgraph
from anomalous_vertices_detection.utils.utils import *


class NxGraph(AbstractGraph):
    __slots__ = []

    def __init__(self, is_directed=False, weight_field=None, graph_obj=None):
        """ Initialize a graph

        Parameters
        ----------
        is_directed: bool, optional (default=False)
            True if the graph direted otherwise False.
        weight_field: string, optional (default=None)
            The name of the weight attribute if exist.
        graph_obj: Graph (NetworkX), optional (default=[])
            A NetworkX graph. If graph_obj=None (default) an empty
            NetworkX graph is created.
        Examples
        --------
        >>> G = NxGraph()
        >>> G = NxGraph(True, "weight")
        >>> e = [(1,2),(2,3),(3,4)]
        >>> nx_graph = nx.Graph(e) #NetworkX Graph
        >>> G = NxGraph(graph_obj=nx_graph)
        """
        super(NxGraph, self).__init__(weight_field)
        if graph_obj:
            self._graph = graph_obj
        else:
            if is_directed is False:
                self._graph = nx.Graph()
            else:
                self._graph = nx.DiGraph()

    def save_graph(self, save_path, type="pickle"):
        """ Saves the graph into a file
        
        Parameters
        ----------
        save_path: string,
            The path where the file shpuld be saved
        type: string, optional (default=pickle)
            The format in which the graph should be saved

        Examples
        --------
        >>> g.save_graph("Facebook.", "graphml")
        """ 
        save_path += "_"
        if type == "pickle":
            nx.write_gpickle(self._graph,  save_path)
        elif type == "sgraph":
            self.save_as_sgraph(save_path)
        elif type == "graphml":
            nx.write_graphml(self._graph, save_path)
        else:
            msg = "The file type %s is unknown" % (type)
            raise TypeError(msg)

    def save_as_sgraph(self, graph_path):
        """  Saves the graph as sgraph

        Parameters
        ----------
        graph_path: string
            The graph path.
        
        Examples
        --------
        >>>  g.save_as_sgraph("graph.sgraph")
        """
        edges = nx.to_edgelist(self._graph)
        res = {'source': [], 'dest': [], "attr": []}
        for edge in edges:
            res['source'].append(edge[0])
            res['dest'].append(edge[1])
            res['attr'].append(edge[2])
        save_nx_as_sgraph(res, graph_path)

    def load_saved_graph(self, graph_path, type="pickle"):
        """ Load a saved graph

        Parameters
        ----------
        graph_path: string,
            The path of the graph that is should be loaded
        type: string, optional (default=pickle)
            The format of the loaded graph

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>  g.load_saved_graph("graph")
        """

        if type == "sgraph":
            return self.load_saved_sgraph(graph_path)
        if type == "pickle":
            return self.load_saved_pickle(graph_path)
        if type == "graphml":
            return self.load_graphml(graph_path)

    @classmethod
    def load_saved_pickle(cls, graph_path):
        """Loads a graph saved as pickle

        Parameters
        ----------
        graph_path: The path of the graph that should be loaded

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>> g.load_saved_pickle("graph.bz2")
        """
        return cls(graph_obj=nx.read_gpickle(graph_path))

    @classmethod
    def load_saved_sgraph(cls, graph_path):
        """Loads a graph saved as sgraph

        Parameters
        ----------
        graph_path: The path of the graph that should be loaded

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>> g.load_saved_sgraph("graph.sgraph")
        """
        return cls(graph_obj=load_nxgraph_from_sgraph(graph_path))

    @classmethod
    def load_gaphml(cls, graph_path):
        """Loads a graph saved as gaphml

        Parameters
        ----------
        graph_path: The path of the graph that should be loaded

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>> g.load_gaphml("graph.gaphml")
        """
        return cls(graph_obj=nx.read_graphml(graph_path))

    def add_edge(self, vertex1, vertex2, edge_atrr={}):
        """ Adds a new edge to the graph
        Parameters
        ----------
        vertex1: string
            The name of the source vertex
        vertex2: string
            The name of the destination vertex
        edge_atrr: dict
            The attributes and the values of the edge

        Examples
        --------
        >>> g.add_edge("a","b")
        >>> g.add_edge("c","b",{"weight": 5})
        """
        vertex1, vertex2 = str(vertex1).strip(), str(vertex2).strip()
        self._graph.add_edge(vertex1, vertex2, edge_atrr)

    def remove_edge(self, v, u):
        """Remove the edge between u and v.

        Parameters
        ----------
        v: string
            The source vertex.
        u: string
            The destination vertex.


        Examples
        --------
        >>> g.remove_edge("1","2")
        """
        self._graph.remove_edge(v, u)

    def get_vertex_edges(self, vertex, mode="out"):
        """ Return a list of edges.

        Parameters
        ----------
        vertex: string
            The node that its edge should be returned
        mode: string, optional (default=out)
            Equals to "out" to return outbound edges.
            Equals to "in" to return inbound edges.
            Equals to "all" to return all edges.

        Returns
        -------
        edge_list: list
            Edges that are adjacent to vertex.

        Examples
        --------
        >>> g.get_vertex_edges("1", "out")
        """
        if mode == "out":
            return self._graph.edges(vertex)
        if mode == "in":
            return self._graph.in_edges(vertex)
        if mode == "all":
            return self._graph.edges(vertex) + self._graph.in_edges(vertex)

    def get_edge_label(self, vertex1, vertex2):
        """ Return the label of the edge.

        Parameters
        ----------
        vertex1: string
            The source vertex.
        vertex2: string
            The destination vertex.

        Returns
        -------
        string: The edge label.

        Examples
        --------
        >>> g.get_edge_label("1", "2")
        """
        if self.has_edge(vertex1, vertex2):
            if "edge_label" in self._graph[vertex1][vertex2]:
                return self._graph[vertex1][vertex2]['edge_label']
        return self.negative_label

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

        Examples
        --------
        >>> g.get_edge_weight("1", "2")
        """
        return self.edge(u, v)[self._weight_field]

    def edge(self, u, v):
        """ Return the edge
        Parameters
        ----------
        v: string
            The source vertex.
        u: string
            The destination vertex.

        Returns
        -------
        edge: The edge

        Examples
        --------
        >>> g.edge("1", "2")
        """
        return self._graph[u][v]

    @property
    def vertices(self):
        """ Return all the vertices in the graph.

        Returns
        -------
        list: List of all the vertices.

        Examples
        --------
        >>> g.vertices
        """
        return self._graph.nodes()

    @property
    def vertices_iter(self):
        """Return an iterator over the vertices.

        Returns
        -------
        iterator: An iterator over all the vertices.

        Examples
        --------
        >>> g.vertices_iter
        """
        return self._graph.nodes_iter()

    @property
    def edges(self):
        """Return all the edges in the graph.

        Returns
        -------
        list: list of all edges

        Examples
        --------
        >>> g.edges
        """
        return self._graph.edges()

    @property
    def edges_iter(self):
        """Return an iterator over the edges.

        Returns
        -------
        iterator: An iterator over all the edges.

        Examples
        --------
        >>> g.edges_iter
        """
        return self._graph.edges_iter()

    @property
    def number_of_vertices(self):
        """Return the of the vertices in the graph.

        Returns
        -------
        int: The number of vertices in the graph.

        Examples
        --------
        >>> g.number_of_vertices()
        """
        return self._graph.number_of_nodes()

    @property
    def is_directed(self):
        """ Return True if graph is directed, False otherwise.

        Examples
        --------
        >>> g.is_directed
        """
        return self._graph.is_directed()

    def simrank(self, r=0.9, max_iter=100, eps=1e-4):
        """
        Source: http://stackoverflow.com/questions/9767773/calculating-simrank-using-networkx

        Parameters
        ----------
        r: relative importance factor.
        max_iter: maximum number of iterations.
        eps: convergence threshold.

        Returns
        -------
        numpy: A matrix (numpy array) containing the simrank scores of the nodes

        Examples
        --------
        >>>
        """

        nodes = self.vertices
        nodes_i = {k: v for (k, v) in [(nodes[i], i) for i in range(0, len(nodes))]}

        sim_prev = np.zeros(len(nodes))
        sim = np.identity(len(nodes))

        for i in range(max_iter):
            if np.allclose(sim, sim_prev, atol=eps):
                break
            sim_prev = np.copy(sim)
            for u, v in product(nodes, nodes):
                if u is v:
                    continue
                u_ns, v_ns = self.get_neighbors(u), self.get_neighbors(v)
                s_uv = sum([sim_prev[nodes_i[u_n]][nodes_i[v_n]] for u_n, v_n in product(u_ns, v_ns)])
                sim[nodes_i[u]][nodes_i[v]] = (r * s_uv) / (len(u_ns) * len(v_ns))

        return sim

    # @property
    def pagerank(self):
        """Return the PageRank of the nodes in the graph.

        Returns
        -------
        pagerank : dictionary
            Dictionary of nodes with PageRank as value

        Examples
        --------
        >>> g.pagerank()
         """
        return nx.pagerank(self._graph, weight=self._weight_field)

    def katz(self):
        """Compute the Katz centrality for the nodes of the graph G.

        Returns
        -------
        nodes : dictionary
        Dictionary of nodes with Katz centrality as the value.
        """
        return nx.katz_centrality(self._graph, weight=self._weight_field)

    # @property
    def hits(self, max_iter=100, normalized=True):
        """Return HITS hubs and authorities values for nodes.
        max_iter : interger, optional
          Maximum number of iterations in power method.

        normalized : bool (default=True)

        Normalize results by the sum of all of the values.
        Returns
        -------
        (hubs,authorities) : two-tuple of dictionaries
            Two dictionaries keyed by node containing the hub and authority
            values.

        Examples
        --------
        >>>
        """
        return nx.hits(self._graph, max_iter=max_iter, normalized=normalized)

    # @property
    def eigenvector(self):
        """ Compute the eigenvector centrality for the graph G.

        Returns
        -------
        nodes : dictionary
            Dictionary of nodes with eigenvector centrality as the value.

        Examples
        --------
        >>>
        """
        return nx.eigenvector_centrality(self._graph, weight=self._weight_field)

    # @property
    def load_centrality(self):
        """Compute load centrality for nodes.

        Returns
        -------
        nodes : dictionary
            Dictionary of nodes with centrality as the value.

        Examples
        --------
        >>>
        """
        return nx.load_centrality(self._graph, weight=self._weight_field)

    # @property
    def communicability_centrality(self):
        """Return communicability centrality for each node in G.

        If is the graph is directed it will be converted to undirected.

        Returns
        -------
        nodes: dictionary
            Dictionary of nodes with communicability centrality as the value.

        Examples
        --------
        >>>
        """
        if self.is_directed:
            return nx.communicability_centrality(self._graph.to_undirected())
        else:
            return nx.communicability_centrality(self._graph)

    # @property
    def betweenness_centrality(self):
        """
        Parameters
        ----------

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        return nx.betweenness_centrality(self._graph, weight=self._weight_field)

    def closeness(self):
        """
        Parameters
        ----------

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        return nx.closeness_centrality(self._graph)

    def get_vertex_degree(self, vertex):
        """
        Parameters
        ----------
        vertex:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        return float(self._graph.degree(vertex))

    def get_shortest_path_length(self, vertex1, vertex2):
        """
        Parameters
        ----------
        vertex1:
        vertex2:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        try:
            return nx.shortest_path_length(self._graph, source=vertex1, target=vertex2, weight=self._weight_field)
        except nx.NetworkXNoPath:
            return 0

    def get_shortest_path_length_with_limit(self, vertex1, vertex2, cutoff=None):
        """
        Parameters
        ----------
        vertex1:
        vertex2:
        cutoff:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        try:
            return nx.single_source_dijkstra(self._graph, source=vertex1, target=vertex2, cutoff=cutoff,
                                             weight=self._weight_field)
        except nx.NetworkXNoPath:
            return 0

    def get_vertex_in_degree(self, vertex):
        """
        Parameters
        ----------
        vertex:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        if self.is_directed:
            return float(self._graph.in_degree(vertex))
        return None

    def get_vertex_out_degree(self, vertex):
        """
        Parameters
        ----------
        vertex:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        if self.is_directed:
            return float(self._graph.out_degree(vertex))
        return float(self._graph.degree(vertex))

    def get_subgraph(self, vertices):
        """
        Parameters
        ----------
        vertices:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        return NxGraph(self.is_directed, self._weight_field, self._graph.subgraph(vertices))

    def has_edge(self, node1, node2):
        """
        Parameters
        ----------
        node1:
        node2:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        return self._graph.has_edge(node1, node2)

    @property
    def connected_components(self):
        """
        Parameters
        ----------

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        return nx.connected_components(self._graph)

    @memoize
    def get_neighbors(self, node):
        """
        Parameters
        ----------
        node:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        return self._graph.neighbors(node)

    def neighbors_iter(self, node):
        """
        Parameters
        ----------
        node:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        return self._graph.neighbors_iter(node)

    def get_followers(self, node):
        """
        Parameters
        ----------
        node:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        if self.is_directed:
            return self._graph.predecessors(node)
        else:
            return self.get_neighbors(node)

    def get_clustering_coefficient(self, node):
        """
        Parameters
        ----------
        node:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        return nx.clustering(self._graph, node, weight=self._weight_field)

    def disjoint_communities(self):
        """
        Parameters
        ----------

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        if self.is_directed:
            partition = community.best_partition(self._graph.to_undirected())
        else:
            partition = community.best_partition(self._graph)
        return partition

    def average_neighbor_degree(self):
        """
        Parameters
        ----------

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        return nx.average_neighbor_degree(self._graph)

    def degree_centrality(self):
        """
        Parameters
        ----------

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        return nx.degree_centrality(self._graph)

    def in_degree_centrality(self):
        """
        Parameters
        ----------

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        return nx.in_degree_centrality(self._graph)

    def out_degree_centrality(self):
        """
        Parameters
        ----------

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        return nx.out_degree_centrality(self._graph)

    def get_scc_number(self, vertices):
        """
        Parameters
        ----------
        vertices:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        neighborhood_subgraph = self.get_neighborhoods_subgraph(vertices)
        return nx.number_strongly_connected_components(neighborhood_subgraph._graph)

    def get_scc_number_plus(self, vertices):
        """
        Parameters
        ----------
        vertices:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        neighborhood_subgraph = self.get_neighborhoods_subgraph_plus(vertices)
        return nx.number_strongly_connected_components(neighborhood_subgraph._graph)

    def get_wcc_number(self, vertices):
        """
        Parameters
        ----------
        vertices:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        neighborhood_subgraph = self.get_neighborhoods_subgraph(vertices)
        return nx.number_weakly_connected_components(neighborhood_subgraph._graph)

    def get_inner_subgraph_scc_number(self, node1, node2):
        """
        Parameters
        ----------
        node1:
        node2:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        inner_subgraph = self.get_inner_subgraph(node1, node2)
        return nx.number_strongly_connected_components(inner_subgraph._graph)

    def get_inner_subgraph_wcc_number(self, node1, node2):
        """
        Parameters
        ----------
        node1:
        node2:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        inner_subgraph = self.get_inner_subgraph(node1, node2)
        return nx.number_weakly_connected_components(inner_subgraph._graph)

    def nodes_number_of_cliques(self):
        """
        Parameters
        ----------

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        if self.is_directed:
            return nx.number_of_cliques(self._graph.to_undirected())
        else:
            return nx.number_of_cliques(self._graph)

    def get_adamic_adar_index(self, node1, node2):
        """
        Parameters
        ----------
        node1:
        node2:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        try:
            return nx.adamic_adar_index(self._graph, [(node1, node2)]).next()
        except:
            return 0, 0, 0

    def get_resource_allocation_index(self, node1, node2):
        """
        Parameters
        ----------
        node1:
        node2:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        return nx.resource_allocation_index(self._graph, [(node1, node2)]).next()

    def write_graph(self, output_path):
        """
        Parameters
        ----------
        output_path:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        nx.write_edgelist(self._graph, output_path, delimiter=',', data=False)
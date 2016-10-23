import random

from networkx.generators import barabasi_albert_graph
from networkx.utils import create_degree_sequence, powerlaw_sequence
import GraphML.utils.utils as utils
from GraphML.graphs.glgraph import GlGraph
from GraphML.graphs.gtgraph import GtGraph
from GraphML.graphs.iggraph import IGraph
from GraphML.graphs.nxgraph import NxGraph
from GraphML.samplers.graph_sampler import GraphSampler

def get_graph(package="Networkx"):
    packages = dict(Networkx=NxGraph,
                    GraphLab=GlGraph,
                    IGraph=IGraph,
                    GraphTool=GtGraph)

    return packages[package]


class GraphFactory(object):
    def factory(self, type):
        if type =="ba":
            return self.make_barabasi_albert_graph
        if type =="simulation":
            return self.make_graph_with_fake_profiles
        if type =="regular":
            return self.make_graph

    def make_barabasi_albert_graph(self, node_number, edge_number, package="Networkx"):
        ba_graph = barabasi_albert_graph(node_number, edge_number)
        # write_adjlist(ba_graph,"ba_test.csv")
        return get_graph(package)(ba_graph.is_directed(), graph_obj=ba_graph)

    def make_ba_graph_with_fake_profiles(self, node_number, edge_number, fake_users_number, max_neighbors,
                                         pos_label=None, neg_label=None, package="Networkx"):
        graph = self.make_barabasi_albert_graph(node_number, edge_number)
        graph.write_graph("tmp_graph.csv")
        is_directed = graph.is_directed
        return self.make_graph_with_fake_profiles("tmp_graph.csv", fake_users_number, max_neighbors, is_directed,
                                                  package="Networkx", pos_label=pos_label, neg_label=neg_label)

    def make_graph(self, graph_path, is_directed=False, labels_path=False, package="Networkx", pos_label=None,
                   neg_label=None, start_line=0, max_num_of_edges=10000000, weight_field=None, blacklist_path=False,
                   delimiter=','):
        """
            Loads graph into specified package.
            Parameters
            ----------
            graph_path : string

            is_directed : boolean, optional (default=False)
               Hold true if the graph is directed otherwise false.

            labels_path : string or None, optional (default=False)
               The path of the node labels file.

            package : string(Networkx, GraphLab or GraphTool), optional (default="Networkx")
               The name of the package to should be used to load the graph.

            pos_label : string or None, optional (default=None)
               The positive label.

            neg_label : string or None, optional (default=None)
               The negative label.

            start_line : integer, optional (default=0)
               The number of the first line in the file to be read.

            max_num_of_edges : integer, optional (default=10000000)
               The maximal number of edges that should be loaded.

            weight_field : string

            Returns
            -------
            g : AbstractGraph
                A graph object with the randomly generated nodes.

        """
        graph = get_graph(package)(is_directed, weight_field)
        if labels_path:
            print("Loading labels...")
            graph.load_labels(labels_path)
        if blacklist_path:
            print("Loading black list...")
            blacklist = utils.read_set_from_file(blacklist_path)
        else:
            blacklist = []
        graph.map_labels(positive=pos_label, negative=neg_label)
        print("Loading graph...")
        graph.load_graph(graph_path, start_line=start_line, limit=max_num_of_edges, blacklist=blacklist,
                         delimiter=delimiter)
        print("Data loaded.")
        return graph

    def load_graph(self, is_directed=False, labels_path=False, package="Networkx", pos_label=None,
                   neg_label=None, weight_field=None, blacklist_path=False, delimiter=',', graph_name=""):
        """
            Loads graph into specified package.
            Parameters
            ----------
            graph_path : string

            is_directed : boolean, optional (default=False)
               Hold true if the graph is directed otherwise false.

            labels_path : string or None, optional (default=False)
               The path of the node labels file.

            package : string(Networkx, GraphLab or GraphTool), optional (default="Networkx")
               The name of the package to should be used to load the graph.

            pos_label : string or None, optional (default=None)
               The positive label.

            neg_label : string or None, optional (default=None)
               The negative label.

            start_line : integer, optional (default=0)
               The number of the first line in the file to be read.

            max_num_of_edges : integer, optional (default=10000000)
               The maximal number of edges that should be loaded.

            weight_field : string

            Returns
            -------
            g : AbstractGraph
                A graph object with the randomly generated nodes.

        """
        graph = get_graph(package)(is_directed, weight_field)
        if labels_path and utils.is_valid_path(labels_path):
            print("Loading labels...")
            graph.load_labels(labels_path)
        if blacklist_path:
            print("Loading black list...")
            blacklist = utils.read_set_from_file(blacklist_path)
        else:
            blacklist = []
        graph.map_labels(positive=pos_label, negative=neg_label)
        print("Loading graph...")
        graph = graph.load_saved_graph(graph_name)
        print("Data loaded.")
        return graph

    # noinspection PyUnresolvedReferences
    def create_random_vertex(self, graph, neighbors_number, graph_vertices, node_name, vertex_label):
        """
            Create a new node and links it randomly to other nodes in the graph
            Parameters
            ----------
            graph : AbstractGraph

            node_name : string

            graph_vertices : list

            neighbors_number : integer

            vertex_label : string
               The label of the generated nodes.


        """
        random_vertices = random.sample(graph_vertices, neighbors_number)
        graph._labels_dict[node_name] = vertex_label
        for rand_vertex in random_vertices:
            if len(graph.get_vertex_edges(rand_vertex, "out")) > 1:
                graph.add_edge(node_name, rand_vertex, {"edge_label": vertex_label})

    def add_random_vertices(self, graph, random_vertices_number, max_neighbors, vertex_label):
        """ Return a graph with simulated random vertices

            Parameters
            ----------
            graph : AbstractGraph

            random_vertices_number : integer
               Hold the number of nodes that should be generated.

            stdv_neighbors : integer
               The stdv of neighbours of the generated node.

            max_neighbors : integer
               The average number of the number of neighbours of the generated node.

            vertex_label : string
               The label of the generated nodes.

            Returns
            -------
            g : AbstractGraph
                A graph object with the randomly generated nodes.
        """
        print("Generating " + str(random_vertices_number) + " vertices.")
        graph_vertices = graph.vertices
        for i,followers_neighbors_number in enumerate(GraphSampler.sample_vertecies_by_degree_distribution(graph,random_vertices_number)):
            self.create_random_vertex(graph, int(followers_neighbors_number), graph_vertices, "Fake" + str(i), vertex_label)
        print(str(random_vertices_number) + " fake users generated.")
        return graph

    def make_graph_with_fake_profiles(self, graph_path, fake_users_number=None, edge_number=290,
                                      is_directed=True, labels_path=False, package="Networkx", pos_label=None,
                                      neg_label=None, start_line=0, max_num_of_edges=10000000, weight_field=None,
                                      delimiter=','):
        graph = self.make_graph(graph_path, is_directed, labels_path, package, pos_label, neg_label, start_line,
                                max_num_of_edges, weight_field, delimiter=delimiter)
        if not fake_users_number:
            fake_users_number = int(0.1*graph.number_of_vertices)
        if max_num_of_edges > 2:
            graph = self.add_random_vertices(graph, fake_users_number, edge_number, pos_label)
        return graph
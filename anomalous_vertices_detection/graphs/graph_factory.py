import random

from networkx.generators import barabasi_albert_graph

import anomalous_vertices_detection.utils.utils as utils
from anomalous_vertices_detection.graphs.glgraph import GlGraph
from anomalous_vertices_detection.graphs.gtgraph import GtGraph
from anomalous_vertices_detection.graphs.iggraph import IGraph
from anomalous_vertices_detection.graphs.nxgraph import NxGraph
from anomalous_vertices_detection.samplers.graph_sampler import GraphSampler

from tqdm import tqdm

def get_graph(package="Networkx"):
    packages = dict(Networkx=NxGraph,
                    GraphLab=GlGraph,
                    IGraph=IGraph,
                    GraphTool=GtGraph)

    return packages[package]


class GraphFactory(object):
    def factory(self, graph_config, labels=None, fake_users_number=None, limit=5000000,
                package="Networkx"):
        if not labels:
            labels = {"neg": "Real", "pos": "Fake"}
        if graph_config.type == "ba":
            return self.make_barabasi_albert_graph(graph_config.node_number, graph_config.edge_number, package=package)
        if graph_config.type == "simulation":
            return self.make_graph_with_fake_profiles(graph_config.data_path, fake_users_number,
                                                      graph_config.is_directed, graph_config.labels_path,
                                                      max_num_of_edges=limit, start_line=graph_config.first_line,
                                                      package=package, pos_label=labels["pos"],
                                                      neg_label=labels["neg"], delimiter=graph_config.delimiter)
        if graph_config.type == "regular":
            return self.make_graph(graph_config.data_path, graph_config.is_directed, graph_config.labels_path,
                                   max_num_of_edges=limit, start_line=graph_config.first_line,
                                   package=package, pos_label=labels["pos"],
                                   neg_label=labels["neg"], delimiter=graph_config.delimiter)

    def make_barabasi_albert_graph(self, node_number, edge_number, package="Networkx"):
        ba_graph = barabasi_albert_graph(node_number, edge_number)
        return get_graph(package)(ba_graph.is_directed(), graph_obj=ba_graph)

    def make_ba_graph_with_fake_profiles(self, node_number, edge_number, fake_users_number,
                                         pos_label=None, neg_label=None, package="Networkx"):
        graph = self.make_barabasi_albert_graph(node_number, edge_number)
        graph.write_graph("tmp_graph.csv")
        is_directed = graph.is_directed
        return self.make_graph_with_fake_profiles("tmp_graph.csv", fake_users_number, is_directed,
                                                  package=package, pos_label=pos_label, neg_label=neg_label)

    def make_graph(self, graph_path, is_directed=False, labels_path=None, package="Networkx", pos_label=None,
                   neg_label=None, start_line=0, max_num_of_edges=10000, weight_field=None, blacklist_path=False,
                   delimiter=','):
        """
            Loads graph into specified package.
            Parameters
            ----------
            blacklist_path
            delimiter
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

    def load_saved_graph(self, graph_path, is_directed=False, labels_path=False, package="Networkx", pos_label=None,
                         neg_label=None, weight_field=None):
        """
            Load graph that was save by the library into specified package.
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
        graph.map_labels(positive=pos_label, negative=neg_label)
        print("Loading graph...")
        graph = graph.load_saved_graph(graph_path)
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
        graph.add_node(node_name, {"type": "sim"})
        for rand_vertex in random_vertices:
            if len(graph.get_vertex_edges(rand_vertex, "out")) >= 0:
                graph.add_edge(node_name, rand_vertex, {"edge_label": vertex_label})

    def add_random_vertices(self, graph, random_vertices_number, vertex_label):
        """ Return a graph with simulated random vertices

            Parameters
            ----------
            graph : AbstractGraph

            random_vertices_number : integer
               Hold the number of nodes that should be generated.

            vertex_label : string
               The label of the generated nodes.

            Returns
            -------
            g : AbstractGraph
                A graph object with the randomly generated nodes.
        """
        print("Generating " + str(random_vertices_number) + " vertices.")
        graph_vertices = list(graph.vertices)
        for i, followers_neighbors_number in tqdm(enumerate(
                GraphSampler.sample_vertices_by_degree_distribution(graph, random_vertices_number)),
                total=random_vertices_number, unit=" simulated vertices"):
            self.create_random_vertex(graph, int(followers_neighbors_number), graph_vertices, "Fake" + str(i),
                                      vertex_label)
        print(str(random_vertices_number) + " fake users generated.")
        return graph

    def make_graph_with_fake_profiles(self, graph_path, fake_users_number=None,
                                      is_directed=True, labels_path=None, package="Networkx", pos_label=None,
                                      neg_label=None, start_line=0, max_num_of_edges=10000000, weight_field=None,
                                      delimiter=','):
        graph = self.make_graph(graph_path, is_directed, labels_path, package, pos_label, neg_label, start_line,
                                max_num_of_edges, weight_field, delimiter=delimiter)
        if not fake_users_number:
            fake_users_number = int(0.1 * graph.number_of_vertices)
        if max_num_of_edges > 2:
            graph = self.add_random_vertices(graph, fake_users_number, pos_label)
        return graph

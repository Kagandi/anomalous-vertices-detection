from anomalous_vertices_detection.graphs.graph_factory import GraphFactory
from anomalous_vertices_detection.graphs import NxGraph

import unittest


class SimulationTest(unittest.TestCase):
    def setUp(self):
        self._graph = NxGraph(False)
        self._graph.load_graph("input/test1.txt", start_line=0)
        self._digraph = NxGraph(True)
        self._digraph.load_graph("input/test1.txt", start_line=0)
        self._factory = GraphFactory()
        self._count = 0

    def test_added_random_vertex(self):
        self._factory.create_random_vertex(self._graph, 2, self._graph.vertices, "Sim", "Sim")
        self._factory.create_random_vertex(self._digraph, 2, self._digraph.vertices, "Sim", "Sim")
        self.assertEqual(self._graph.number_of_vertices, 7)
        self.assertEqual(self._digraph.number_of_vertices, 7)
        self.assertEqual(len(self._graph.edges), 11)
        self.assertEqual(len(self._digraph.edges), 12)

    def test_added_random_vertices(self):
        graph = self._factory.add_random_vertices(self._graph, 3, "Sim")
        digraph = self._factory.add_random_vertices(self._digraph, 3, "Sim")
        self.assertEqual(graph.number_of_vertices, 9)
        self.assertEqual(digraph.number_of_vertices, 9)

    def test_added_correct_labels(self):
        graph = self._factory.add_random_vertices(self._graph, 2, "Sim")
        sim_label_counter = 0
        real_label_counter = 0
        for v in graph.vertices:
            if graph.get_node_label(v) == "Sim":
                sim_label_counter += 1
            else:
                real_label_counter += 1
        self.assertEqual(sim_label_counter, 2)
        self.assertEqual(real_label_counter, 6)

        sim_label_counter = 0
        real_label_counter = 0
        digraph = self._factory.add_random_vertices(self._digraph, 2, "Sim")
        for v in digraph.vertices:
            if digraph.get_node_label(v) == "Sim":
                sim_label_counter += 1
            else:
                real_label_counter += 1
        self.assertEqual(sim_label_counter, 2)
        self.assertEqual(real_label_counter, 6)


if __name__ == '__main__':
    unittest.main()

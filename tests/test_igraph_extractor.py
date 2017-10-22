from anomalous_vertices_detection.feature_extractor import FeatureExtractor
from anomalous_vertices_detection.graphs import IGraph
import os
import unittest


class ExtractorIgraphTest(unittest.TestCase):
    def setUp(self):
        tests_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        tests_input = os.path.join(tests_path, "input/test1.txt")
        self._graph = IGraph(False)
        self._graph.load_graph(tests_input, start_line=0)
        self._digraph = IGraph(True)
        self._digraph.load_graph(tests_input, start_line=0)
        self._fe = FeatureExtractor(self._graph)
        self._dife = FeatureExtractor(self._digraph)

    def test_total_friends(self):
        self.assertEqual(self._fe.get_total_friends("1", "4"), 4)
        self.assertEqual(self._dife.get_total_friends("1", "4"), 4)

    def test_common_friends(self):
        self.assertEqual(self._fe.get_common_friends("1", "4"), 2)
        self.assertEqual(self._dife.get_common_friends("1", "4"), 2)

    def test_jaccards_coefficient(self):
        self.assertEqual(self._fe.get_jaccards_coefficient("1", "4"), 0.5)
        self.assertEqual(self._dife.get_jaccards_coefficient("1", "4"), 0.5)

    def test_preferential_attachment_score(self):
        self.assertEqual(self._fe.get_preferential_attachment_score("1", "4"), 9)
        self.assertEqual(self._dife.get_preferential_attachment_score("1", "4"), 9)

    def test_inner_subgraph_link_number(self):
        self.assertEqual(self._fe.get_inner_subgraph_link_number("1", "4"), 3)
        self.assertEqual(self._dife.get_inner_subgraph_link_number("1", "4"), 3)

    def test_friend_measure(self):
        self.assertEqual(self._fe.get_friend_measure("1", "4"), 5)
        self.assertEqual(self._dife.get_friend_measure("1", "4"), 5)

    # def test_is_same_community(self):
    #     self.assertEqual(self._fe.is_same_community("1", "4"), 0)
    #     self.assertEqual(self._dife.is_same_community("1", "4"), 0)

    def test_in_degree_density(self):
        self.assertEqual(self._dife.get_in_degree_density("1"), 0.25)

    def test_out_degree_density(self):
        self.assertEqual(self._dife.get_out_degree_density("1"), 0.75)

    def test_bi_degree_density(self):
        self.assertEqual(self._dife.get_bi_degree_density("1"), 0.25)
        self.assertEqual(self._dife.get_bi_degree_density("4"), 0)

    def test_subgraph_node_link_number(self):
        self.assertEqual(self._fe.get_subgraph_node_link_number("1"), 2)
        self.assertEqual(self._dife.get_subgraph_node_link_number("1"), 2)

    def test_subgraph_node_link_number_plus(self):
        self.assertEqual(self._fe.get_subgraph_node_link_number_plus("1"), 5)
        self.assertEqual(self._dife.get_subgraph_node_link_number_plus("1"), 6)

    def test_subgraph_link_number(self):
        self.assertEqual(self._fe.get_subgraph_link_number("1", "4"), 3)
        self.assertEqual(self._dife.get_subgraph_link_number("1", "4"), 3)

    def test_subgraph_link_number_plus(self):
        self.assertEqual(self._fe.get_subgraph_link_number_plus("1", "4"), 9)
        self.assertEqual(self._dife.get_subgraph_link_number_plus("1", "4"), 10)

    def test_density_neighborhood_subgraph(self):
        self.assertEqual(self._fe.get_density_neighborhood_subgraph("1"), 1.5)
        self.assertEqual(self._dife.get_density_neighborhood_subgraph("1"), 2)

    def test_density_neighborhood_subgraph_plus(self):
        self.assertEqual(self._fe.get_density_neighborhood_subgraph_plus("1"), 0.6)
        self.assertEqual(self._dife.get_density_neighborhood_subgraph_plus("1"), float(2) / float(3))

    def test_shortest_path_length(self):
        self.assertEqual(self._fe.get_shortest_path_length("1", "4"), 2)
        self.assertEqual(self._dife.get_shortest_path_length("1", "4"), 0)

    def test_is_opposite_direction_friends(self):
        self.assertEqual(self._dife.is_opposite_direction_friends("1", "4"), 0)
        self.assertEqual(self._dife.is_opposite_direction_friends("1", "3"), 1)
        #
        # def test_average_scc(self):
        #     self.assertEqual(self._dife.get_average_scc("1"), float(4) / float(3))

        # def test_average_scc_plus(self):
        #     self.assertEqual(self._dife.get_average_scc_plus("1"), 2)

        # def test_average_wcc(self):
        #     self.assertEqual(self._dife.get_average_wcc("1"), 4)
        #
        # def test_scc_number(self):
        #     self.assertEqual(self._dife.get_scc_number("1", "4"), 4)
        #
        # def test_scc_number_plus(self):
        #     self.assertEqual(self._dife.get_scc_number_plus("1", "4"), 4)

        # def test_wcc_number(self):
        #     self.assertEqual(self._dife.get_wcc_number("1", "4"), 1)
        #
        # def test_inner_subgraph_scc_number(self):
        #     self.assertEqual(self._dife.get_inner_subgraph_scc_number("1", "4"), 4)
        #
        # def test_inner_subgraph_wcc_number(self):
        #     self.assertEqual(self._dife.get_inner_subgraph_wcc_number("1", "4"), 1)


if __name__ == '__main__':
    unittest.main()

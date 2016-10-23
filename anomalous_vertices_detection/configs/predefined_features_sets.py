link06 = {
    "link": {"sum_of_friends": "get_sum_of_friends",
             "shortest_distance": "get_shortest_path_length"
             },
    "vertex_v": {"clustering_coefficient_v": "get_clustering_coefficient"},
    "vertex_u": {"clustering_coefficient_u": "get_clustering_coefficient"}

}
fast_link_features = {
    True: {
        "link": {"jaccards_coefficient": "get_jaccards_coefficient",
                 "preferential_attachment_score": "get_preferential_attachment_score",
                 # "friend_measure": "get_friend_measure",
                 "number_of_transitive_friends": "get_number_of_transitive_friends",
                 "is_opposite_direction_friends": "is_opposite_direction_friends",
                 "in_common_friends": "get_in_common_friends",
                 "out_common_friends": "get_out_common_friends",
                 "total_friends": "get_total_friends",
                 "knn_weight1": "get_knn_weight1",
                 "knn_weight2": "get_knn_weight2",
                 "knn_weight3": "get_knn_weight3",
                 "knn_weight4": "get_knn_weight4",
                 "knn_weight5": "get_knn_weight5",
                 "knn_weight6": "get_knn_weight6",
                 "knn_weight7": "get_knn_weight7",
                 "knn_weight8": "get_knn_weight8",
                 # "cosine": "get_cosine",
                 "bi_common_friends": "get_bi_common_friends"
                 }
        ,
        "vertex_v": {"out_degree_v": "get_out_degree",
                     "in_degree_v": "get_in_degree"},
        "vertex_u": {"out_degree_u": "get_out_degree",
                     "in_degree_u": "get_in_degree",
                     # "secondary_neighbors_log": "get_secondary_neighbors_log"
                     # "total_edge_weight": "get_total_edge_weight",
                     # "average_edge_weight": "get_average_edge_weight"
                     # "u_label": "get_label"
                     # "secondary_neighbors_log": "get_secondary_neighbors_log"
                     }
    },
    False: {
        "link": {  # "common_friends2": "get_level_two_common_friends",
            "jaccards_coefficient": "get_jaccards_coefficient",
            "common_friends": "get_common_friends",
            "preferential_attachment_score": "get_preferential_attachment_score",
            # "friend_measure": "get_friend_measure",
            # "alt_friend_measure": "get_alt_friend_measure"
            "total_friends": "get_total_friends",
            "sum_of_friends": "get_sum_of_friends",
            "adamic_adar_index": "get_adamic_adar_index",
            "knn_weight4": "get_knn_weight4",
            "knn_weight8": "get_knn_weight8"

            # "resource_allocation_index": "get_resource_allocation_index"
        }  # ,
        # "vertex_v": {"number_of_friends_v": "get_number_of_friends"#,
        #              # "total_edge_weight_v": "get_total_edge_weight",
        #              # "average_edge_weight_v": "get_average_edge_weight"
        #              },
        # "vertex_u": {"number_of_friends_u": "get_number_of_friends"#,
        #              # "total_edge_weight_u": "get_total_edge_weight",
        #              # "average_edge_weight_u": "get_average_edge_weight"
        #              # "secondary_neighbors_log": "get_secondary_neighbors_log"
        #              }
    }
}

fast_vertex_features = {
    True: {"vertex_v":
               {"src": "get_vertex",
                "label": "get_label",
                "subgraph_node_link_number": "get_subgraph_node_link_number",
                "subgraph_node_link_number_plus": "get_subgraph_node_link_number_plus",
                "density_neighborhood_subgraph": "get_density_neighborhood_subgraph",
                "density_neighborhood_subgraph_plus": "get_density_neighborhood_subgraph_plus",
                "average_scc": "get_average_scc",
                "average_scc_plus": "get_average_scc_plus",
                "average_wcc": "get_average_wcc",
                # "degree_centrality": "get_degree_centrality",
                "out_degree": "get_out_degree",
                # "out_degree_centrality": "get_out_degree_centrality",
                "in_degree": "get_in_degree",
                "bi_degree": "get_bi_degree",
                # "pagerank": "get_pagerank",
                "bi_degree_density": "get_bi_degree_density",
                "in_degree_density": "get_in_degree_density",
                "out_degree_density": "get_out_degree_density",
                # "number_of_neighbors_communities": "get_number_of_neighbors_communities",
                # "communication_reciprocity": "get_communication_reciprocity"
                }
           }
}

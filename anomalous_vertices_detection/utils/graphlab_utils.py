import json

try:
    import graphlab as gl
except ImportError:
    gl = None


def load_nxgraph_from_sgraph(graph_path):
    sg = gl.load_sgraph(graph_path)
    import networkx as nx
    g = nx.Graph()

    # Put the nodes and edges from the SGraph into a NetworkX graph
    g.add_nodes_from(list(sg.vertices['__id']))
    g.add_edges_from([(e['__src_id'], e['__dst_id'], e['attr']) for e in sg.edges])
    return g


def save_nx_as_sgraph(df, output_path):
    sf = gl.SFrame(data=df)
    sg = gl.SGraph().add_edges(sf, src_field="source", dst_field="dest")
    sg.save(output_path)


def sgraph_to_csv(sgraph_path, output_path):
    sg = gl.load_sgraph(sgraph_path)
    sg.save(output_path, 'csv')


def sframe_to_csv(sframe_path, output_path):
    sf = gl.SFrame(sframe_path, format='array')
    sf.save(output_path, 'csv')


def sarray_to_csv(sarray_path, output_path):
    sf = gl.SArray(sarray_path)
    sf.save(output_path, 'csv')


def json_to_csv(json_path, output_path):
    with open(json_path) as json_data:
        sarray_to_csv(json.load(json_data), output_path)

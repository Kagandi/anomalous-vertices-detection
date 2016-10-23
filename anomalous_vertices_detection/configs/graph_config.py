class GraphConfig(object):
    def __init__(self, set_name, dataset_path, is_directed=True, labels_path=False,
                 type="regular",max_neighbors_number=False, stdv_neighbors=False,
                 vertex_min_edge_number=1, vertex_max_edge_number=50000, delimiter=',', first_line=2):
        self._dataset_path = dataset_path
        self._labels_path = labels_path
        self._is_directed = is_directed
        self._name = set_name
        self._max_neighbors_number = max_neighbors_number
        self._stdv_neighbors = stdv_neighbors
        self._vertex_min_edge_number = vertex_min_edge_number
        self._vertex_max_edge_number = vertex_max_edge_number
        self._delimiter = delimiter
        self._first_line = first_line
        self._type = type

    @property
    def delimiter(self):
        return self._delimiter

    @property
    def max_neighbors_number(self):
        return self._max_neighbors_number

    @property
    def type(self):
        return self._type

    @property
    def stdv_neighbors(self):
        return self._stdv_neighbors

    @property
    def vertex_min_edge_number(self):
        return self._vertex_min_edge_number

    @property
    def vertex_max_edge_number(self):
        return self._vertex_max_edge_number

    @property
    def name(self):
        return self._name

    @property
    def data_path(self):
        return self._dataset_path

    @property
    def labels_path(self):
        return self._labels_path

    @property
    def is_directed(self):
        return self._is_directed


class GraphSimConfig(GraphConfig):

    def __init__(self, node_number, edge_number, *args, **kwargs):
        super(GraphSimConfig, self).__init__(*args, **kwargs)
        self._edge_number = edge_number
        self._node_number = node_number

    @property
    def node_number(self):
        return self._node_number

    @property
    def edge_number(self):
        return self._edge_number
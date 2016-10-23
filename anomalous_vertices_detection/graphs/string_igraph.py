from iggraph import *


class StringIGraph(Graph):
    def __init__(self, directed=True):
        super(StringIGraph, self).__init__()
        self._vertices = dict()

    def __getitem__(self, key):
        if isinstance(key, basestring):
            if key in self._vertices:
                return self._vertices[key]
            else:
                return None
        else:
            return super(StringIGraph, self).vertex(key)

    def __setitem__(self, key, val):
        self._vertices[key] = val

    def __delitem__(self, key):
        self._vertices.__delitem__(key)

    def clear(self):
        self._vertices.clear()

    def add_vertex(self, vertex_name):
        if vertex_name not in self._vertices:
            vertex = super(StringIGraph, self).add_vertex()
            self._vertices[vertex_name] = vertex
        else:
            vertex = self._vertices[vertex_name]
        return vertex

    def vertex(self, vertex_id):
        if isinstance(vertex_id, basestring):
            return self._vertices[vertex_id]
        else:
            return super(StringIGraph, self).vs[vertex_id]

    def edge(self, s, t, all_edges=False):
        if isinstance(s, basestring):
            s = self[s]
        if isinstance(t, basestring):
            t = self[t]
        if s is not None and t is not None:
            return super(StringIGraph, self).edge(s, t, all_edges=False)
        return None

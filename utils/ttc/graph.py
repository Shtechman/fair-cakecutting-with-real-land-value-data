class Vertex(object):
    """**
        * A util class for representing Graph's Vertex.
        *
        * @author Itay Shtechman
        * @since 2019-10
        """

    def __init__(self, graph, vertex_id):
        self.graph = graph

        self.vertex_id = vertex_id

        self.outgoing_edges = set()

        self.incoming_edges = set()

    def __hash__(self):
        return hash(self.vertex_id)

    def __repr__(self):
        return "Vertex(%s)" % (repr(self.vertex_id),)

    def __str__(self):
        return repr(self)

    def out_degree(self):
        return len(self.outgoing_edges)

    def any_next(self):
        return self.graph[list(self.outgoing_edges)[0][1]]


class Graph(object):
    """**
        * A util class for representing a Direct Graph.
        *
        * @author Itay Shtechman
        * @since 2019-10
        """

    def __init__(self, vertex_ids):

        self.vertices = dict((name, Vertex(self, name)) for name in vertex_ids)

        self.edges = set()

    def __getitem__(self, key):

        return self.vertices[key]

    def any_vertex(self):

        for k, v in self.vertices.items():
            return v

    def add_edge(self, source, target):

        self.edges.add((source, target))

        self[source].outgoing_edges.add((source, target))

        self[target].incoming_edges.add((source, target))

    def add_edges(self, edges):

        for e in edges:
            self.add_edge(*e)

    def delete(self, vertex):

        if type(vertex) is Vertex:
            vertex = vertex.vertex_id

        involved_edges = (
            self[vertex].outgoing_edges | self[vertex].incoming_edges
        )

        for (u, v) in involved_edges:
            self[v].incoming_edges.remove((u, v))

            self[u].outgoing_edges.remove((u, v))

            self.edges.remove((u, v))

        del self.vertices[vertex]

    def __repr__(self):

        return repr(set(self.vertices.keys())) + ", " + repr(self.edges)

    def __str__(self):

        return repr(self)

from utils.graph import *

"""**
    * A util script implementing TTC logic.
    *
    * @author Itay Shtechman
    * @since 2019-10
    """


# getAgents: graph, vertex -> set(vertex)

# get the set of agents on a cycle starting at the given vertex


def get_agents(cycle, agents):
    # a cycle in G is represented by any vertex of the cycle

    # out_degree guarantee means we don't care which vertex it is

    # make sure starting vertex is a house

    if cycle.vertex_id in agents:
        cycle = cycle.any_next()

    starting_piece = cycle

    current_vertex = starting_piece.any_next()

    the_agents = set()

    while current_vertex not in the_agents:
        the_agents.add(current_vertex)

        current_vertex = current_vertex.any_next()

        current_vertex = current_vertex.any_next()

    return the_agents


# anyCycle: graph -> vertex

# find any vertex involved in a cycle


def any_cycle(graph):
    visited = set()

    v = graph.any_vertex()

    while v not in visited:
        visited.add(v)

        v = v.any_next()

    return v


# find a core matching of agents to pieces

# agents and pieces are unique identifiers for the agents and pieces involved

# agent_preferences is a dictionary with keys being agents and values being

# lists that are permutations of the list of all pieces.

# initiail_ownerships is a dict {pieces:agents}


def top_trading_cycles(agents, pieces, agent_preferences, initial_ownership):
    """ Ensure agent ids and pieces ids are unique"""
    agents = {"a_" + a for a in agents}
    pieces = {"h_" + h for h in pieces}
    initial_ownership = {
        "h_" + h: "a_" + initial_ownership[h] for h in initial_ownership
    }
    agent_preferences = {
        "a_" + a: ["h_" + h for h in agent_preferences[a]]
        for a in agent_preferences
    }

    """ Form the initial graph """
    agents = set(agents)

    vertex_set = set(agents) | set(pieces)

    graph = Graph(vertex_set)

    # maps agent to an index of the list agentPreferences[agent]

    current_preference_index = dict((a, 0) for a in agents)

    preferred_piece = lambda a: agent_preferences[a][
        current_preference_index[a]
    ]

    for a in agents:
        graph.add_edge(a, preferred_piece(a))

    for h in pieces:
        graph.add_edge(h, initial_ownership[h])

    # iteratively remove top trading cycles

    allocation = dict()

    while len(graph.vertices) > 0:

        cycle = any_cycle(graph)

        cycle_agents = get_agents(cycle, agents)

        # assign agents in the cycle their piece

        for a in cycle_agents:
            h = a.any_next().vertex_id

            allocation[a.vertex_id.replace("a_", "")] = h.replace("h_", "")

            graph.delete(a)

            graph.delete(h)

        for a in agents:

            if a in graph.vertices and graph[a].out_degree() == 0:

                while preferred_piece(a) not in graph.vertices:
                    current_preference_index[a] += 1

                graph.add_edge(a, preferred_piece(a))

    return allocation

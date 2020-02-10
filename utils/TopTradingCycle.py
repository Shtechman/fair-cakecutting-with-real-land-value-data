from utils.Graph import *

"""**
    * A util script implementing TTC logic.
    *
    * @author Itay Shtechman
    * @since 2019-10
    """

# getAgents: graph, vertex -> set(vertex)

# get the set of agents on a cycle starting at the given vertex

def getAgents(G, cycle, agents):
    # a cycle in G is represented by any vertex of the cycle

    # outdegree guarantee means we don't care which vertex it is

    # make sure starting vertex is a house

    if cycle.vertexId in agents:
        cycle = cycle.anyNext()

    startingHouse = cycle

    currentVertex = startingHouse.anyNext()

    theAgents = set()

    while currentVertex not in theAgents:
        theAgents.add(currentVertex)

        currentVertex = currentVertex.anyNext()

        currentVertex = currentVertex.anyNext()

    return theAgents


# anyCycle: graph -> vertex

# find any vertex involved in a cycle

def anyCycle(G):
    visited = set()

    v = G.anyVertex()

    while v not in visited:
        visited.add(v)

        v = v.anyNext()

    return v


# find a core matching of agents to pieces

# agents and pieces are unique identifiers for the agents and pieces involved

# agentPreferences is a dictionary with keys being agents and values being

# lists that are permutations of the list of all pieces.

# initiailOwnerships is a dict {pieces:agents}

def topTradingCycles(agents, pieces, agentPreferences, initialOwnership):
    # ensure agent ids and pieces ids are unique
    agents = {'a_' + a for a in agents}
    pieces = {'h_' + h for h in pieces}
    initialOwnership = {'h_' + h: 'a_' + initialOwnership[h] for h in initialOwnership}
    agentPreferences = {'a_' + a: ['h_' + h for h in agentPreferences[a]]
                        for a in agentPreferences}

    # form the initial graph
    agents = set(agents)

    vertexSet = set(agents) | set(pieces)

    G = Graph(vertexSet)

    # maps agent to an index of the list agentPreferences[agent]

    currentPreferenceIndex = dict((a, 0) for a in agents)

    preferredPiece = lambda a: agentPreferences[a][currentPreferenceIndex[a]]

    for a in agents:
        G.addEdge(a, preferredPiece(a))

    for h in pieces:
        G.addEdge(h, initialOwnership[h])

    # iteratively remove top trading cycles

    allocation = dict()

    while len(G.vertices) > 0:

        cycle = anyCycle(G)

        cycleAgents = getAgents(G, cycle, agents)

        # assign agents in the cycle their piece

        for a in cycleAgents:
            h = a.anyNext().vertexId

            allocation[a.vertexId.replace('a_', '')] = h.replace('h_', '')

            G.delete(a)

            G.delete(h)

        for a in agents:

            if a in G.vertices and G[a].outdegree() == 0:

                while preferredPiece(a) not in G.vertices:
                    currentPreferenceIndex[a] += 1

                G.addEdge(a, preferredPiece(a))

    return allocation

#!python3

"""
/**
 * Implementation of the highest-bidder cake-cutting algorithm on a 2-dimensional cake.
 *
 * @author Itay Shtechman
 * @since 2020-04
 */
"""

from functools import lru_cache
from math import log, ceil, floor

from utils.simulation.agent import Agent
from utils.simulation.allocated_piece import AllocatedPiece
from utils.cutters.simple_cutter import SimpleCutter
from utils.simulation.cc_types import CutPattern, AggregationType, AlgType


class AlgorithmHighestBidder:
    """ This algorithm uses a simple method to assign each cell to the highest bidder.
        It is an unfair algorithm and is used to compute the price of fairness """

    def run(self, agents):

        cake_rows = agents[0].value_map_row_count
        cake_cols = agents[0].value_map_col_count
        agents_gain = {agent.get_map_file_number(): 0 for agent in agents}
        for i in range(cake_rows):
            for j in range(cake_cols):
                highest_bidder = max(agents, key=lambda agent: agent.evaluation_cell_bid(i, j))
                agents_gain[highest_bidder.get_map_file_number()] += highest_bidder.evaluation_cell_bid(i, j)

        return agents_gain


    def get_algorithm_type(self):
        return "{}_{}".format("HighestBidder", "NoPattern")


if __name__ == "__main__":
    import doctest

    doctest.testmod()

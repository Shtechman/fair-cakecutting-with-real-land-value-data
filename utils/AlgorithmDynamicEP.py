#!python3

"""
/**
 * Division of a 2-D cake using an Even Paz inspired Dynamic Programming Algorithm
 *
 * @author Itay Shtechman
 * @since 2020-01
 */
"""
from itertools import product

from utils.AllocatedPiece import AllocatedPiece
from utils.Cutter import EPCutter
from utils.Types import CutPattern, CutDirection
from utils.Measurements import Measurements as Measure

VAL_IDX = 0
PAR_IDX = 1

class AlgorithmDynamicEP:

    class CutHistory:
        def __init__(self, cut_direction, partition, evaluation, measure, first_half_cut_history=None, second_half_cut_history=None):
            self.cut_direction = cut_direction
            self.partition = partition
            self.num_of_agents = len(partition)
            self.evaluation = evaluation
            self.measure = measure
            self.first_half_cut_history = first_half_cut_history
            self.second_half_cut_history = second_half_cut_history

    def __init__(self):
        self.measure_func = {
            "EV": Measure.get_egalitarian_gain,
            "UV": Measure.get_utilitarian_gain,
            "LE": Measure.get_largest_envy,
            "AFR": Measure.get_average_face_ratio,
            "SFR": Measure.get_smallest_face_ratio
        }
        self.measure_merge_func = {
            "EV": Measure.merge_egalitarian_gain,
            "UV": Measure.merge_utilitarian_gain,
            "LE": Measure.merge_largest_envy,
            "AFR": Measure.merge_average_face_ratio,
            "SFR": Measure.merge_smallest_face_ratio
        }
        self.comparator = lambda measure, x, y: x < y if measure in "LE" else x > y

        self.measure_list = self.measure_func.keys()
        self.results_by_measure = {}
        self.reset_inner_data()

    def reset_inner_data(self):
        self.results_by_measure = {measure: (-1, []) for measure in self.measure_list}

    def run(self, agents, cut_pattern, measure_to_test = None):
        """
        Calculate a proportional cake-division using the dynamic programming.
        @param agents - a list of n Agents, each with a value-function on the same cake.
        @return a list of n AllocatedPiece-s, each of which contains an Agent and an allocated part of the cake.
        todo: write examples for 2d case
        """
        self.reset_inner_data()

        measures = [measure_to_test] if measure_to_test else self.measure_list

        return self.aggregate_cutters(agents, measures)

    def aggregate_cutters(self, agents, measures):
        results = []
        def _print_inner_cuts(cut_history):
            try:
                zipped_list = [j for i in zip(_print_inner_cuts(cut_history.second_half_cut_history),
                                              _print_inner_cuts(cut_history.first_half_cut_history)) for j in i]
                return [cut_history.cut_direction.name[0]] + zipped_list
            except:
                return [cut_history.cut_direction.name[0]]
        for measure in measures:
            # initially, allocate the entire cake to all agents:
            initial_allocations = list(map(AllocatedPiece, agents))

            # now, recursively divide the cake among the agents using a given cutter:
            final_allocation = self._runRecursive(initial_allocations, measure)
            cut_series = "_".join(_print_inner_cuts(final_allocation))

            results.append((measure+"@"+cut_series, final_allocation))

        # for r in results:
        #     print(r)
        if len(results) > 1:
            return {str(result[0]): result[1].partition for result in results}
        else:
            return results[0][1].partition

    @staticmethod
    def getAlgorithmType():
        return "DynamicEP"

    def choose_partition(self, hor_partition, ver_partition, tested_measure):
        valuation_func = self.measure_func[tested_measure]

        # def __init__(self, cut_direction, partition, evaluation, measure, cut_history=None):
        hor_par_valuation = valuation_func(hor_partition)
        ver_par_valuation = valuation_func(ver_partition)

        if self.comparator(tested_measure, hor_par_valuation, ver_par_valuation):
            return self.CutHistory(CutDirection.Horizontal, hor_partition, hor_par_valuation, tested_measure)
        else:
            return self.CutHistory(CutDirection.Vertical, ver_partition, ver_par_valuation, tested_measure)

    def merge_cut_history(self, first_c_h, second_c_h, cut_direction):
        if first_c_h.measure not in second_c_h.measure:
            raise ValueError("Can't merge evaluation of two different measures.", first_c_h.measure,
                             second_c_h.measure)

        measure = first_c_h.measure
        combined_partition = first_c_h.partition+second_c_h.partition

        valuation = self.measure_merge_func[measure](
            first_c_h.evaluation, first_c_h.num_of_agents,
            second_c_h.evaluation, second_c_h.num_of_agents, combined_partition
        )

        return self.CutHistory(cut_direction,combined_partition,valuation,measure,first_c_h,second_c_h)

    def _runRecursive(self, allocations, tested_measure):
        num_of_agents = len(allocations)

        if num_of_agents < 2:
            raise ValueError("Can not compute division for %s agents" % num_of_agents)

        horizontal_cutter = EPCutter(CutPattern.Hor)
        vertical_cutter = EPCutter(CutPattern.Ver)
        top_hor_allocation, bottom_hor_allocation = horizontal_cutter.allocate_cuts(allocations, num_of_agents)
        left_ver_allocation, right_ver_allocation = vertical_cutter.allocate_cuts(allocations, num_of_agents)

        if num_of_agents == 2:

            hor_partition = top_hor_allocation + bottom_hor_allocation
            ver_partition = left_ver_allocation + right_ver_allocation

            initial_cut = self.choose_partition(hor_partition, ver_partition, tested_measure)

            return initial_cut

        hor_cut = self.merge_cut_history(self._runRecursive(top_hor_allocation, tested_measure),
                                         self._runRecursive(bottom_hor_allocation, tested_measure),
                                         CutDirection.Horizontal)

        ver_cut = self.merge_cut_history(self._runRecursive(left_ver_allocation, tested_measure),
                                         self._runRecursive(right_ver_allocation, tested_measure),
                                         CutDirection.Vertical)

        if self.comparator(tested_measure, hor_cut.evaluation, ver_cut.evaluation):
            return hor_cut
        else:
            return ver_cut


if __name__ == '__main__':
    # from utils.ValueFunction1D import ValueFunction1D
    # from utils.Agent import Agent
    #
    #
    # import doctest
    # # doctest.testmod()
    #
    # # demo test
    # alg = AlgorithmEvenPaz()
    # Alice = Agent(name="Alice", valueFunction=ValueFunction1D([1, 2, 3, 4]))
    # Bob = Agent(name="Bob", valueFunction=ValueFunction1D([40, 30, 20, 10]))
    # Carl = Agent(name="Carl", valueFunction=ValueFunction1D([100, 100, 100, 100]))
    # print("when Alice is the only agent -", alg.run([Alice]))
    # print("when Alice and Bob are the only agents -", alg.run([Alice, Bob]))
    # print("when Carl and Bob are the only agents -", alg.run([Carl, Bob]))
    # print("when Alice, Bob and Carl are the agents -", alg.run([Alice, Bob, Carl]))
    pass

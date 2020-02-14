#!python3

"""
/**
 * Division of a 2-D cake using FOCS, a Dynamic Programming inspired Algorithm
 *
 * @author Itay Shtechman
 * @since 2020-01
 */
"""

from utils.allocated_piece import AllocatedPiece
from utils.cutter import EPCutter
from utils.measurements import Measurements as Measure
from utils.types import CutPattern, CutDirection


class AlgorithmFOCS:
    class _CutHistory:
        def __init__(
            self,
            cut_direction,
            partition,
            evaluation,
            measure,
            first_half_cut_history=None,
            second_half_cut_history=None,
        ):
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
            "SFR": Measure.get_smallest_face_ratio,
        }
        self.measure_merge_func = {
            "EV": Measure.merge_egalitarian_gain,
            "UV": Measure.merge_utilitarian_gain,
            "LE": Measure.merge_largest_envy,
            "AFR": Measure.merge_average_face_ratio,
            "SFR": Measure.merge_smallest_face_ratio,
        }
        self.comparator = (
            lambda measure, x, y: x < y if measure in "LE" else x > y
        )

        self.measure_list = self.measure_func.keys()
        self.results_by_measure = {}
        self.reset_inner_data()

    def reset_inner_data(self):
        self.results_by_measure = {
            measure: (-1, []) for measure in self.measure_list
        }

    def run(self, agents, cut_pattern, measure_to_test=None):

        self.reset_inner_data()
        measures = [measure_to_test] if measure_to_test else self.measure_list

        return self.aggregate_cutters(agents, measures)

    def aggregate_cutters(self, agents, measures):
        results = []

        def _print_inner_cuts(cut_history):
            try:
                zipped_list = [
                    j
                    for i in zip(
                        _print_inner_cuts(cut_history.second_half_cut_history),
                        _print_inner_cuts(cut_history.first_half_cut_history),
                    )
                    for j in i
                ]
                return [cut_history.cut_direction.name[0]] + zipped_list
            except:
                return [cut_history.cut_direction.name[0]]

        for measure in measures:
            # initially, allocate the entire cake to all agents:
            initial_allocations = list(map(AllocatedPiece, agents))

            # now, recursively divide the cake among the agents using a given cutter:
            final_allocation = self._run_recursive(
                initial_allocations, measure
            )
            cut_series = "_".join(_print_inner_cuts(final_allocation))

            results.append((measure + "@" + cut_series, final_allocation))

        if len(results) > 1:
            return {str(result[0]): result[1].partition for result in results}
        else:
            return results[0][1].partition

    @staticmethod
    def get_algorithm_type():
        return "FOCS"

    def choose_better_partition_for_measure(
        self, hor_partition, ver_partition, tested_measure
    ):
        valuation_func = self.measure_func[tested_measure]

        hor_par_valuation = valuation_func(hor_partition)
        ver_par_valuation = valuation_func(ver_partition)

        if self.comparator(
            tested_measure, hor_par_valuation, ver_par_valuation
        ):
            return self._CutHistory(
                CutDirection.Horizontal,
                hor_partition,
                hor_par_valuation,
                tested_measure,
            )
        else:
            return self._CutHistory(
                CutDirection.Vertical,
                ver_partition,
                ver_par_valuation,
                tested_measure,
            )

    def merge_cut_history(self, first_c_h, second_c_h, cut_direction):
        if first_c_h.measure not in second_c_h.measure:
            raise ValueError(
                "Can't merge evaluation of two different measures.",
                first_c_h.measure,
                second_c_h.measure,
            )

        measure = first_c_h.measure
        combined_partition = first_c_h.partition + second_c_h.partition

        valuation = self.measure_merge_func[measure](
            first_c_h.evaluation,
            first_c_h.num_of_agents,
            second_c_h.evaluation,
            second_c_h.num_of_agents,
            combined_partition,
        )

        return self._CutHistory(
            cut_direction,
            combined_partition,
            valuation,
            measure,
            first_c_h,
            second_c_h,
        )

    def _run_recursive(self, allocations, tested_measure):
        num_of_agents = len(allocations)

        if num_of_agents < 2:
            raise ValueError(
                "Can not compute division for %s agents" % num_of_agents
            )

        horizontal_cutter = EPCutter(CutPattern.Hor)
        vertical_cutter = EPCutter(CutPattern.Ver)
        (
            top_hor_allocation,
            bottom_hor_allocation,
        ) = horizontal_cutter.allocate_cuts(allocations, num_of_agents)
        (
            left_ver_allocation,
            right_ver_allocation,
        ) = vertical_cutter.allocate_cuts(allocations, num_of_agents)

        if num_of_agents == 2:
            hor_partition = top_hor_allocation + bottom_hor_allocation
            ver_partition = left_ver_allocation + right_ver_allocation

            initial_cut = self.choose_better_partition_for_measure(
                hor_partition, ver_partition, tested_measure
            )

            return initial_cut

        hor_cut = self.merge_cut_history(
            self._run_recursive(top_hor_allocation, tested_measure),
            self._run_recursive(bottom_hor_allocation, tested_measure),
            CutDirection.Horizontal,
        )

        ver_cut = self.merge_cut_history(
            self._run_recursive(left_ver_allocation, tested_measure),
            self._run_recursive(right_ver_allocation, tested_measure),
            CutDirection.Vertical,
        )

        if self.comparator(
            tested_measure, hor_cut.evaluation, ver_cut.evaluation
        ):
            return hor_cut
        else:
            return ver_cut


if __name__ == "__main__":
    pass

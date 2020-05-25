import numpy as np

from utils.simulation.cc_types import CutPattern
from utils.cutters.basecutter import BaseCutter, SMALLEST_NUMBER


class SimpleCutter(BaseCutter):
    def __init__(
        self,
        cut_pattern,
        cut_query=None,
        cut_direction=None,
        original_cut_pattern=None,
    ):
        super(SimpleCutter, self).__init__(
            cut_pattern, cut_query, cut_direction, original_cut_pattern
        )

        if self.freeplay_mode:
            if len(self.cut_pattern) < 1:
                self.cut_pattern = self.original_cut_pattern
            self.first_part_cut_pattern = self.cut_pattern[2::2]
            self.second_part_cut_pattern = self.cut_pattern[1::2]
        elif cut_pattern not in [CutPattern.Hor, CutPattern.Ver]:
            raise ValueError(
                "NoPattern Cutter can either cut horizontally or vertically %s is not supported."
                % cut_pattern
            )

    def __copy__(self):
        return SimpleCutter(
            self.cut_pattern,
            self.cut_query,
            self.cut_direction,
            self.original_cut_pattern,
        )

    def get_first_part_cutter(self):
        if self.freeplay_mode:
            return SimpleCutter(
                self.first_part_cut_pattern,
                original_cut_pattern=self.original_cut_pattern,
            )
        else:
            return self.__copy__()

    def get_second_part_cutter(self):
        if self.freeplay_mode:
            return SimpleCutter(
                self.second_part_cut_pattern,
                original_cut_pattern=self.original_cut_pattern,
            )
        else:
            return self.__copy__()

    def get_number_of_agents_in_first_partition(self, number_of_agents):
        """ For NoPattern Cutter, first partition holds half of the agents """
        return int(np.ceil(number_of_agents / 2))

    def _calculate_optional_cut_location(self, margin_iFrom, margin_iTo):
        """ For NoPattern cutter, cutting is made at the middle of the margin """
        return (margin_iFrom + margin_iTo) / 2.0

    def _calculate_best_cutmark(
        self, allocation, query_direction, honest_cut_marks
    ):
        if len(honest_cut_marks) % 2 > 0:
            middle_idx = int(np.floor(len(honest_cut_marks) / 2))
        else:
            raise ValueError(
                "Odd number of agents yet to be supported"
            )  # todo: add support for odd num of agents

        honest_cut_marks.sort()
        query_halfcut = honest_cut_marks[middle_idx]

        low_half = allocation.subcut(
            allocation.get_directional_i_from(query_direction),
            query_halfcut,
            query_direction,
        )

        high_half = allocation.subcut(
            query_halfcut,
            allocation.get_directional_i_to(query_direction),
            query_direction,
        )

        if low_half.get_value() > high_half.get_value():
            return query_halfcut - SMALLEST_NUMBER
        else:
            return query_halfcut + SMALLEST_NUMBER
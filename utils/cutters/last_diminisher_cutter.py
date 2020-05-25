import random

from utils.cutters.basecutter import BaseCutter


class LDCutter(BaseCutter):
    def __init__(
        self,
        cut_pattern,
        cut_query=None,
        cut_direction=None,
        original_cut_pattern=None,
    ):
        super(LDCutter, self).__init__(
            cut_pattern, cut_query, cut_direction, original_cut_pattern
        )

        if self.freeplay_mode:
            if len(self.cut_pattern) < 1:
                self.cut_pattern = self.original_cut_pattern
            self.first_part_cut_pattern = self.cut_pattern[0]
            self.second_part_cut_pattern = self.cut_pattern[1:]

    def __copy__(self):
        return LDCutter(
            self.cut_pattern,
            self.cut_query,
            self.cut_direction,
            self.original_cut_pattern,
        )

    def get_first_part_cutter(self):
        if self.freeplay_mode:
            return LDCutter(
                self.first_part_cut_pattern,
                original_cut_pattern=self.original_cut_pattern,
            )
        else:
            return self.__copy__()

    def get_second_part_cutter(self):
        if self.freeplay_mode:
            return LDCutter(
                self.second_part_cut_pattern,
                original_cut_pattern=self.original_cut_pattern,
            )
        else:
            return self.__copy__()

    def _divider_chooser_allocation(self, allocations):
        divider_index = random.randint(0, 1)
        chooser_index = 1 - divider_index
        chooser_agent = allocations[chooser_index].get_agent()
        divider_piece = allocations[divider_index]
        divider_allocations = [
            divider_piece.__copy__(),
            divider_piece.__copy__(),
        ]
        first_piece, second_piece = super()._allocate_cuts(
            divider_allocations, 2, 1
        )
        chooser_allocations = [
            first_piece[0].get_allocated_piece(chooser_agent),
            second_piece[0].get_allocated_piece(chooser_agent),
        ]
        if (
            chooser_allocations[0].get_relative_value()
            > chooser_allocations[1].get_relative_value()
        ):
            return [chooser_allocations[0]], second_piece
        else:
            return first_piece, [chooser_allocations[1]]

    def _allocate_cuts(
        self,
        allocations,
        number_of_agents,
        number_of_agents_in_first_partition,
    ):
        if number_of_agents > 2:
            (
                first_part_allocations,
                second_part_allocations,
            ) = super()._allocate_cuts(
                allocations,
                number_of_agents,
                number_of_agents_in_first_partition,
            )
        elif number_of_agents == 2:
            (
                first_part_allocations,
                second_part_allocations,
            ) = self._divider_chooser_allocation(allocations)
        else:
            raise ValueError("Division is only possible from 2 agents and up.")

        return first_part_allocations, second_part_allocations

    def get_number_of_agents_in_first_partition(self, num_of_agents):
        """ For Last Diminisher, first partition holds only one agent (with the smallest mark) """
        return 1

    def _calculate_optional_cut_location(self, margin_i_from, margin_i_to):
        """ For Last Diminisher, cutting is made at the smallest proposed cut """
        return margin_i_from

    def _calculate_best_cutmark(
        self, allocation, query_direction, honest_cut_marks
    ):
        honest_cut_marks.sort()
        query_cutmark = honest_cut_marks[0]
        raise NotImplementedError(
            "Strategic play for LastDiminisher is not yet implemented."
        )
        # todo: implement a strategic play - note that if query_cutmark results in a piece better then 1/n of the
        #  cake than the player can claim (query_cutmark - SMALLEST_NUMBER) and if not, (query_cutmark +
        #  SMALLEST_NUMBER).
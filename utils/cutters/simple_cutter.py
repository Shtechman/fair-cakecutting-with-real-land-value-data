import numpy as np

from utils.simulation.cc_types import CutPattern, CutDirection
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

        def _manipulate_cut_direction(
                self, current_cut_dir, dishonest_idx, allocations
        ):
            """ for both possible cut direction check which one is more profitable for the dishonest agent,
                if the current cut direction is not the most lucrative, adjust the dishonest agent marks to
                influence the cut direction.
                """

            hor_prospect = self._dishonest_directional_prospect(
                CutDirection.Horizontal,
                allocations,
                dishonest_idx
            )

            ver_prospect = self._dishonest_directional_prospect(
                CutDirection.Vertical,
                allocations,
                dishonest_idx
            )

            if ((current_cut_dir == CutDirection.Horizontal and ver_prospect > hor_prospect) or
                    (current_cut_dir == CutDirection.Vertical and hor_prospect > ver_prospect)):
                self._change_mark_to_influence_cut_direction(current_cut_dir, dishonest_idx,
                                                             allocations, len(allocations))

        def _change_mark_to_influence_cut_direction(
                self, current_cut_dir, dishonest_idx, allocations, num_of_agents, query_jumps=10.0,
                original_cut_mark=None):
            original_cut_mark = allocations[dishonest_idx].cut_marks[current_cut_dir] \
                if original_cut_mark is None else original_cut_mark

            min_cut_mark_possible = allocations[dishonest_idx].get_directional_i_from(current_cut_dir)
            max_cut_mark_possible = allocations[dishonest_idx].get_directional_i_to(current_cut_dir)
            query_jump_size = (max_cut_mark_possible - min_cut_mark_possible) / query_jumps

            allocations[dishonest_idx].cut_marks[current_cut_dir] = original_cut_mark + SMALLEST_NUMBER
            proposed_cut_direction = self._get_proposed_cutting_direction(allocations, num_of_agents)
            if proposed_cut_direction != current_cut_dir:
                print("### +++ Success for cut pattern %s" % self.cut_pattern)
                return

            allocations[dishonest_idx].cut_marks[current_cut_dir] = original_cut_mark - SMALLEST_NUMBER
            proposed_cut_direction = self._get_proposed_cutting_direction(allocations, num_of_agents)
            if proposed_cut_direction != current_cut_dir:
                print("### +++ Success for cut pattern %s" % self.cut_pattern)
                return

            allocations[dishonest_idx].cut_marks[current_cut_dir] = min_cut_mark_possible
            proposed_cut_direction = self._get_proposed_cutting_direction(allocations, num_of_agents)
            if proposed_cut_direction != current_cut_dir:
                print("### +++ Success for cut pattern %s" % self.cut_pattern)
                return

            allocations[dishonest_idx].cut_marks[current_cut_dir] = max_cut_mark_possible
            proposed_cut_direction = self._get_proposed_cutting_direction(allocations, num_of_agents)

            while (proposed_cut_direction == current_cut_dir
                   and allocations[dishonest_idx].cut_marks[current_cut_dir] >= min_cut_mark_possible):
                allocations[dishonest_idx].cut_marks[current_cut_dir] -= query_jump_size
                proposed_cut_direction = self._get_proposed_cutting_direction(allocations, num_of_agents)

            " if the cut direction did not change, try smaller jumps (until limit of 10000 jumps)"
            if proposed_cut_direction == current_cut_dir:
                if query_jumps >= 10000:
                    allocations[dishonest_idx].cut_marks[current_cut_dir] = original_cut_mark
                    print("### --- Failed for cut pattern %s" % self.cut_pattern)
                else:
                    self._change_mark_to_influence_cut_direction(
                        current_cut_dir,
                        dishonest_idx,
                        allocations,
                        num_of_agents,
                        query_jumps * 10,
                        original_cut_mark
                    )
            print("### +++ Success for cut pattern %s" % self.cut_pattern)

        def _dishonest_directional_prospect(self, cut_direction, allocations, dishonest_idx):
            def _get_directional_cut_marks(alloc):
                return alloc.cut_marks[cut_direction]

            dishonest_allocation = allocations[dishonest_idx]

            try:
                sorted_cuts = list(
                    map(
                        _get_directional_cut_marks,
                        sorted(allocations, key=_get_directional_cut_marks),
                    )
                )
            except:
                """ If not both ver and hor cut lists are set, raise error """
                raise NotImplemented(" Something went wrong,"
                                     "this method should only be called when cut pattern supports"
                                     "both cutting directions")

            top_margin_index = self.get_number_of_agents_in_first_partition(
                len(allocations)
            )
            cut_option = self._calculate_cut_option(sorted_cuts, top_margin_index)
            dis_agent_cut = _get_directional_cut_marks(allocations[dishonest_idx])

            if dis_agent_cut < cut_option:
                return dishonest_allocation.subcut(
                    dishonest_allocation.get_directional_i_from(cut_direction),
                    cut_option,
                    cut_direction,
                ).get_value()
            else:
                return dishonest_allocation.subcut(
                    cut_option,
                    dishonest_allocation.get_directional_i_to(cut_direction),
                    cut_direction,
                ).get_value()

    def _manipulate_cut_direction(
            self, current_cut_dir, dishonest_idx, allocations
    ):
        """ for both possible cut direction check which one is more profitable for the dishonest agent,
            if the current cut direction is not the most lucrative, adjust the dishonest agent marks to
            influence the cut direction.
            """

        hor_prospect = self._dishonest_directional_prospect(
            CutDirection.Horizontal,
            allocations,
            dishonest_idx
        )

        ver_prospect = self._dishonest_directional_prospect(
            CutDirection.Vertical,
            allocations,
            dishonest_idx
        )

        if ((current_cut_dir == CutDirection.Horizontal and ver_prospect > hor_prospect) or
                (current_cut_dir == CutDirection.Vertical and hor_prospect > ver_prospect)):
            self._change_mark_to_influence_cut_direction(current_cut_dir, dishonest_idx,
                                                         allocations, len(allocations))

    def _change_mark_to_influence_cut_direction(
            self, current_cut_dir, dishonest_idx, allocations, num_of_agents, query_jumps=10.0, original_cut_mark=None):
        original_cut_mark = allocations[dishonest_idx].cut_marks[current_cut_dir]\
            if original_cut_mark is None else original_cut_mark

        min_cut_mark_possible = allocations[dishonest_idx].get_directional_i_from(current_cut_dir)
        max_cut_mark_possible = allocations[dishonest_idx].get_directional_i_to(current_cut_dir)
        query_jump_size = (max_cut_mark_possible - min_cut_mark_possible) / query_jumps

        allocations[dishonest_idx].cut_marks[current_cut_dir] = original_cut_mark + SMALLEST_NUMBER
        proposed_cut_direction = self._get_proposed_cutting_direction(allocations, num_of_agents)
        if proposed_cut_direction != current_cut_dir:
            print("### +++ Success for cut pattern %s" % self.cut_pattern)
            return

        allocations[dishonest_idx].cut_marks[current_cut_dir] = original_cut_mark - SMALLEST_NUMBER
        proposed_cut_direction = self._get_proposed_cutting_direction(allocations, num_of_agents)
        if proposed_cut_direction != current_cut_dir:
            print("### +++ Success for cut pattern %s" % self.cut_pattern)
            return

        allocations[dishonest_idx].cut_marks[current_cut_dir] = min_cut_mark_possible
        proposed_cut_direction = self._get_proposed_cutting_direction(allocations, num_of_agents)
        if proposed_cut_direction != current_cut_dir:
            print("### +++ Success for cut pattern %s" % self.cut_pattern)
            return

        allocations[dishonest_idx].cut_marks[current_cut_dir] = max_cut_mark_possible
        proposed_cut_direction = self._get_proposed_cutting_direction(allocations, num_of_agents)

        while (proposed_cut_direction == current_cut_dir
               and allocations[dishonest_idx].cut_marks[current_cut_dir] >= min_cut_mark_possible):
            allocations[dishonest_idx].cut_marks[current_cut_dir] -= query_jump_size
            proposed_cut_direction = self._get_proposed_cutting_direction(allocations, num_of_agents)

        " if the cut direction did not change, try smaller jumps (until limit of 10000 jumps)"
        if proposed_cut_direction == current_cut_dir:
            if query_jumps >= 10000:
                allocations[dishonest_idx].cut_marks[current_cut_dir] = original_cut_mark
                print("### --- Failed for cut pattern %s" % self.cut_pattern)
            else:
                self._change_mark_to_influence_cut_direction(
                    current_cut_dir,
                    dishonest_idx,
                    allocations,
                    num_of_agents,
                    query_jumps*10,
                    original_cut_mark
                )
        print("### +++ Success for cut pattern %s" % self.cut_pattern)

    def _dishonest_directional_prospect(self, cut_direction, allocations, dishonest_idx):
        def _get_directional_cut_marks(alloc):
            return alloc.cut_marks[cut_direction]

        dishonest_allocation = allocations[dishonest_idx]

        try:
            sorted_cuts = list(
                map(
                    _get_directional_cut_marks,
                    sorted(allocations, key=_get_directional_cut_marks),
                )
            )
        except:
            """ If not both ver and hor cut lists are set, raise error """
            raise NotImplemented(" Something went wrong,"
                                 "this method should only be called when cut pattern supports"
                                 "both cutting directions")

        top_margin_index = self.get_number_of_agents_in_first_partition(
            len(allocations)
        )
        cut_option = self._calculate_cut_option(sorted_cuts, top_margin_index)
        dis_agent_cut = _get_directional_cut_marks(allocations[dishonest_idx])

        if dis_agent_cut < cut_option:
            return dishonest_allocation.subcut(
                    dishonest_allocation.get_directional_i_from(cut_direction),
                    cut_option,
                    cut_direction,
                ).get_value()
        else:
            return dishonest_allocation.subcut(
                    cut_option,
                    dishonest_allocation.get_directional_i_to(cut_direction),
                    cut_direction,
                ).get_value()

import numpy as np

from utils.simulation.cc_types import CutPattern, CutDirection

SMALLEST_NUMBER = 0.0000000000001


class BaseCutter:
    def __init__(
        self,
        cut_pattern,
        cut_query=None,
        cut_direction=None,
        original_cut_pattern=None,
    ):
        self.cut_pattern = cut_pattern
        self.original_cut_pattern = (
            cut_pattern
            if original_cut_pattern is None
            else original_cut_pattern
        )
        self.cut_query = cut_query
        self.cut_direction = cut_direction
        self.freeplay_mode = type(cut_pattern) is list

    def __repr__(self):
        if self.freeplay_mode:
            str_list = [
                s.replace("CutDirection.", "")
                for s in map(str, self.cut_pattern)
            ]
            return "_".join(str_list)
        return "%s" % self.cut_pattern

    def to_string(self):
        return self.__repr__()

    def __copy__(self):
        return BaseCutter(
            self.cut_pattern,
            self.cut_query,
            self.cut_direction,
            self.original_cut_pattern,
        )

    def get_first_part_cutter(self):
        if self.freeplay_mode:
            return BaseCutter(
                self.first_part_cut_pattern,
                original_cut_pattern=self.original_cut_pattern,
            )
        else:
            return self.__copy__()

    def get_second_part_cutter(self):
        if self.freeplay_mode:
            return BaseCutter(
                self.second_part_cut_pattern,
                original_cut_pattern=self.original_cut_pattern,
            )
        else:
            return self.__copy__()

    def allocate_cuts(self, allocations, num_of_agents):

        number_of_agents_in_first_partition = self.get_number_of_agents_in_first_partition(
            num_of_agents
        )

        proportion_of_first_partition = (
            number_of_agents_in_first_partition / float(num_of_agents)
        )

        self._set_next_query_direction()

        self._set_all_agents_cut_marks(
            allocations, proportion_of_first_partition
        )

        self._set_next_cutting_direction(allocations, num_of_agents)

        allocations.sort(key=lambda alloc: alloc.cut_marks[self.cut_direction])

        first_part_allocations, second_part_allocations = self._allocate_cuts(
            allocations, num_of_agents, number_of_agents_in_first_partition
        )

        return first_part_allocations, second_part_allocations

    def _set_all_agents_cut_marks(
        self, allocations, proportion_of_first_partition
    ):
        dishonest_idx = -1
        for idx, agent_allocation in enumerate(allocations):
            if (
                dishonest_idx < 0
                and agent_allocation.get_agent().is_dishonest()
            ):
                dishonest_idx = idx
            else:
                self._set_relevant_cut_marks(
                    proportion_of_first_partition, agent_allocation
                )

        self._set_dishonest_cut_marks(allocations, dishonest_idx)

    def _set_dishonest_cut_marks(self, allocations, dishonest_idx):
        if dishonest_idx > -1:
            dis_agent_allocation = allocations[dishonest_idx]
            honest_allocations = (
                allocations[:dishonest_idx] + allocations[dishonest_idx + 1 :]
            )

            self._set_best_cut_marks_for_dishonest(
                dis_agent_allocation, honest_allocations
            )

    def _set_relevant_cut_marks(
        self, proportion_of_first_partition, agent_allocation
    ):
        value = proportion_of_first_partition * agent_allocation.get_value()
        agent_allocation.cut_marks = {}
        if (self.cut_query is CutDirection.Horizontal) or (
            self.cut_query is CutDirection.Both
        ):
            agent_allocation.cut_marks[
                CutDirection.Horizontal
            ] = agent_allocation.mark_query_for_given_value(
                value, CutDirection.Horizontal
            )
        if (self.cut_query is CutDirection.Vertical) or (
            self.cut_query is CutDirection.Both
        ):
            agent_allocation.cut_marks[
                CutDirection.Vertical
            ] = agent_allocation.mark_query_for_given_value(
                value, CutDirection.Vertical
            )

    def _set_best_cut_marks_for_dishonest(
        self, allocation, honest_allocations
    ):
        allocation.cut_marks = {}
        if (self.cut_query is CutDirection.Horizontal) or (
            self.cut_query is CutDirection.Both
        ):
            horizontal_hosnest_halfcuts = [
                allocation.cut_marks[CutDirection.Horizontal]
                for allocation in honest_allocations
            ]
            allocation.cut_marks[
                CutDirection.Horizontal
            ] = self._calculate_best_cutmark(
                allocation,
                CutDirection.Horizontal,
                horizontal_hosnest_halfcuts,
            )

        if (self.cut_query is CutDirection.Vertical) or (
            self.cut_query is CutDirection.Both
        ):
            vertical_hosnest_halfcuts = [
                allocation.cut_marks[CutDirection.Vertical]
                for allocation in honest_allocations
            ]
            allocation.cut_marks[
                CutDirection.Vertical
            ] = self._calculate_best_cutmark(
                allocation, CutDirection.Vertical, vertical_hosnest_halfcuts
            )

    def _divide_agents_to_partitions(
        self,
        sorted_allocations,
        cut_location,
        number_of_agents,
        number_of_agents_in_first_partition,
    ):
        first_part_allocations = []
        second_part_allocations = []
        for i in range(0, number_of_agents_in_first_partition):
            i_from = sorted_allocations[i].get_directional_i_from(
                self.cut_direction
            )
            i_to = cut_location
            first_part_allocations.append(
                sorted_allocations[i].subcut(i_from, i_to, self.cut_direction)
            )

        for i in range(number_of_agents_in_first_partition, number_of_agents):
            i_from = cut_location
            i_to = sorted_allocations[i].get_directional_i_to(
                self.cut_direction
            )
            second_part_allocations.append(
                sorted_allocations[i].subcut(i_from, i_to, self.cut_direction)
            )

        return first_part_allocations, second_part_allocations

    def _initial_cut_direction(self):

        if self.freeplay_mode:
            return self._get_next_freeplay_query()
        else:
            switcher = {
                CutPattern.Hor: CutDirection.Horizontal,
                CutPattern.Ver: CutDirection.Vertical,
                CutPattern.HorVer: CutDirection.Horizontal,
                CutPattern.VerHor: CutDirection.Vertical,
            }

            return switcher.get(self.cut_pattern, CutDirection.Both)

    def _cut_pattern_same_query_direction(self):
        return self.cut_query

    def _cut_pattern_opp_query_direction(self):
        switcher = {
            CutDirection.Horizontal: CutDirection.Vertical,
            CutDirection.Vertical: CutDirection.Horizontal,
        }

        return switcher.get(self.cut_direction, CutDirection.Both)

    def _get_next_freeplay_query(self):
        return (
            self.cut_pattern[0]
            if len(self.cut_pattern) > 0
            else CutDirection.Horizontal
        )

    def _get_query_direction_iterator_func(self):
        if self.freeplay_mode:
            return self._get_next_freeplay_query
        else:
            switcher = {
                CutPattern.HorVer: self._cut_pattern_opp_query_direction,
                CutPattern.VerHor: self._cut_pattern_opp_query_direction,
            }

            return switcher.get(
                self.cut_pattern, self._cut_pattern_same_query_direction
            )

    def _set_next_query_direction(self):
        if self.cut_query is None:
            self.cut_query = self._initial_cut_direction()
        else:
            dir_iter_func = self._get_query_direction_iterator_func()
            self.cut_query = dir_iter_func()

    def get_number_of_agents_in_first_partition(self, number_of_agents):
        raise NotImplementedError("This method should be overridden.")

    def _calculate_optional_cut_location(self, margin_iFrom, margin_iTo):
        raise NotImplementedError("This method should be overridden.")

    def _calculate_best_cutmark(
        self, allocation, query_direction, honest_cut_marks
    ):
        raise NotImplementedError("This method should be overridden.")

    def _allocate_cuts(
        self,
        sorted_allocations,
        number_of_agents,
        number_of_agents_in_first_partition,
    ):
        end_of_first_part = sorted_allocations[
            number_of_agents_in_first_partition - 1
        ].cut_marks[self.cut_direction]
        start_of_second_part = sorted_allocations[
            number_of_agents_in_first_partition
        ].cut_marks[self.cut_direction]

        cut_location = self._calculate_optional_cut_location(
            end_of_first_part, start_of_second_part
        )

        (
            first_part_allocations,
            second_part_allocations,
        ) = self._divide_agents_to_partitions(
            sorted_allocations,
            cut_location,
            number_of_agents,
            number_of_agents_in_first_partition,
        )

        return first_part_allocations, second_part_allocations

    def _set_next_cutting_direction(self, allocations, number_of_agents):
        """ Default next cut direction value is the cut query """
        self.cut_direction = self.cut_query

        def _get_horizontal_cut_marks(alloc):
            return alloc.cut_marks[CutDirection.Horizontal]

        def _get_vertical_cut_marks(alloc):
            return alloc.cut_marks[CutDirection.Vertical]

        if self.cut_pattern in [
            CutPattern.SmallestHalfCut,
            CutPattern.SmallestPiece,
        ]:
            first_hor_cut = min(allocations, key=_get_horizontal_cut_marks)
            first_ver_cut = min(allocations, key=_get_vertical_cut_marks)
            first_hor_cut_halfcut = _get_horizontal_cut_marks(first_hor_cut)
            first_ver_cut_halfcut = _get_vertical_cut_marks(first_ver_cut)

            if self.cut_pattern is CutPattern.SmallestHalfCut:
                self.cut_direction = (
                    CutDirection.Horizontal
                    if first_hor_cut_halfcut < first_ver_cut_halfcut
                    else CutDirection.Vertical
                )

            if self.cut_pattern is CutPattern.SmallestPiece:
                start_index_hor = first_hor_cut.get_directional_i_from(
                    CutDirection.Horizontal
                )
                start_index_ver = first_ver_cut.get_directional_i_from(
                    CutDirection.Vertical
                )
                hor_cut_size = (
                    first_hor_cut_halfcut - start_index_hor
                ) * first_ver_cut.get_dimensions()[CutDirection.Vertical]
                ver_cut_size = (
                    first_ver_cut_halfcut - start_index_ver
                ) * first_hor_cut.get_dimensions()[CutDirection.Horizontal]

                self.cut_direction = (
                    CutDirection.Horizontal
                    if hor_cut_size < ver_cut_size
                    else CutDirection.Vertical
                )
            return

        if self.cut_pattern in [CutPattern.LongestDim, CutPattern.ShortestDim]:
            allocation = allocations[0]
            allocation_dimensions = allocation.get_dimensions()

            if (
                allocation_dimensions[CutDirection.Vertical]
                < allocation_dimensions[CutDirection.Horizontal]
            ):
                self.cut_direction = (
                    CutDirection.Horizontal
                    if self.cut_pattern is CutPattern.LongestDim
                    else CutDirection.Vertical
                )
            else:
                self.cut_direction = (
                    CutDirection.Vertical
                    if self.cut_pattern is CutPattern.LongestDim
                    else CutDirection.Horizontal
                )
            return

        try:
            sorted_horizontal_cuts = list(
                map(
                    _get_horizontal_cut_marks,
                    sorted(allocations, key=_get_horizontal_cut_marks),
                )
            )
            sorted_vertical_cuts = list(
                map(
                    _get_vertical_cut_marks,
                    sorted(allocations, key=_get_vertical_cut_marks),
                )
            )
        except:
            """ If not both ver and hor cut lists are set, use default value """
            return

        top_margin_index = self.get_number_of_agents_in_first_partition(
            number_of_agents
        )
        horizontal_margin = (
            sorted_horizontal_cuts[top_margin_index]
            - sorted_horizontal_cuts[top_margin_index - 1]
        )
        vertical_margin = (
            sorted_vertical_cuts[top_margin_index]
            - sorted_vertical_cuts[top_margin_index - 1]
        )

        if self.cut_pattern is CutPattern.LargestMargin:
            self.cut_direction = (
                CutDirection.Horizontal
                if vertical_margin < horizontal_margin
                else CutDirection.Vertical
            )
            return

        if self.cut_pattern is CutPattern.LargestMarginArea:
            allocation_dimensions = allocations[0].get_dimensions()
            horizontal_margin_area = (
                horizontal_margin
                * allocation_dimensions[CutDirection.Vertical]
            )
            vertical_margin_area = (
                vertical_margin
                * allocation_dimensions[CutDirection.Horizontal]
            )
            self.cut_direction = (
                CutDirection.Horizontal
                if vertical_margin_area < horizontal_margin_area
                else CutDirection.Vertical
            )
            return

        hor_margin_i_from = sorted_horizontal_cuts[top_margin_index - 1]
        hor_margin_i_to = sorted_horizontal_cuts[top_margin_index]
        hor_cut_option = self._calculate_optional_cut_location(
            hor_margin_i_from, hor_margin_i_to
        )
        ver_margin_i_from = sorted_vertical_cuts[top_margin_index - 1]
        ver_margin_i_to = sorted_vertical_cuts[top_margin_index]
        ver_cut_option = self._calculate_optional_cut_location(
            ver_margin_i_from, ver_margin_i_to
        )

        if self.cut_pattern is CutPattern.LargestAvgMargin:
            hor_avg_margin = np.average(
                [
                    np.abs(cutmark - hor_cut_option)
                    for cutmark in sorted_horizontal_cuts
                ]
            )
            ver_avg_margin = np.average(
                [
                    np.abs(cutmark - ver_cut_option)
                    for cutmark in sorted_vertical_cuts
                ]
            )
            self.cut_direction = (
                CutDirection.Horizontal
                if ver_avg_margin < hor_avg_margin
                else CutDirection.Vertical
            )
            return

        if self.cut_pattern is CutPattern.MostValuableMargin:
            """
            * We average the value of the entire piece made by cutting in the suggested location in the eyes of each
            * agent.
            * This should be the same as measuring the actual margin because all agents have the same relative
            * eval of the sub-piece from start to their idea of where the margin begins.
            """
            hor_margin_avg_value = np.average(
                list(
                    map(
                        lambda alloc: alloc.get_directional_value(
                            hor_cut_option, CutDirection.Horizontal
                        ),
                        allocations,
                    )
                )
            )
            ver_margin_avg_value = np.average(
                list(
                    map(
                        lambda alloc: alloc.get_directional_value(
                            ver_cut_option, CutDirection.Vertical
                        ),
                        allocations,
                    )
                )
            )
            self.cut_direction = (
                CutDirection.Horizontal
                if ver_margin_avg_value < hor_margin_avg_value
                else CutDirection.Vertical
            )
            return

        if self.cut_pattern is CutPattern.SquarePiece:
            hor_face_ratio_avg_value = np.average(
                list(
                    map(
                        lambda alloc: alloc.get_directional_face_ratio(
                            hor_cut_option, CutDirection.Horizontal
                        ),
                        allocations,
                    )
                )
            )
            ver_face_ratio_value = np.average(
                list(
                    map(
                        lambda alloc: alloc.get_directional_face_ratio(
                            ver_cut_option, CutDirection.Vertical
                        ),
                        allocations,
                    )
                )
            )

            self.cut_direction = (
                CutDirection.Horizontal
                if ver_face_ratio_value < hor_face_ratio_avg_value
                else CutDirection.Vertical
            )
            return

        if self.cut_pattern is CutPattern.HighestScatter:
            neighbor_horizontal_cuts = list(
                zip(
                    sorted_horizontal_cuts, np.roll(sorted_horizontal_cuts, -1)
                )
            )[:-1]
            neighbor_vertical_cuts = list(
                zip(sorted_vertical_cuts, np.roll(sorted_vertical_cuts, -1))
            )[:-1]
            hor_scatter_avg_value = np.average(
                [b - a for (a, b) in neighbor_horizontal_cuts]
            )
            ver_scatter_avg_value = np.average(
                [b - a for (a, b) in neighbor_vertical_cuts]
            )
            self.cut_direction = (
                CutDirection.Horizontal
                if ver_scatter_avg_value < hor_scatter_avg_value
                else CutDirection.Vertical
            )
            return



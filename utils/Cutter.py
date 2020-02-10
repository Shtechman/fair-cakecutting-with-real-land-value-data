import random

import numpy as np

from utils.Types import CutPattern, CutDirection

SMALLEST_NUMBER = 0.0000000000001


class Cutter:

    def __init__(self, cut_pattern, cut_query=None, cut_direction=None, original_cut_pattern=None):
        self.cut_pattern = cut_pattern
        self.original_cut_pattern = cut_pattern if original_cut_pattern is None else original_cut_pattern
        self.cut_query = cut_query
        self.cut_direction = cut_direction
        self.freeplay_mode = type(cut_pattern) is list

    def __repr__(self):
        if self.freeplay_mode:
            str_list = [s.replace("CutDirection.", "") for s in map(str, self.cut_pattern)]
            return '_'.join(str_list)
        return "%s" % self.cut_pattern

    def toString(self):
        return self.__repr__()

    def __copy__(self):
        return Cutter(self.cut_pattern, self.cut_query, self.cut_direction, self.original_cut_pattern)

    def get_firt_part_cutter(self):
        if self.freeplay_mode:
            return Cutter(self.first_part_cut_pattern, original_cut_pattern=self.original_cut_pattern)
        else:
            return self.__copy__()

    def get_second_part_cutter(self):
        if self.freeplay_mode:
            return Cutter(self.second_part_cut_pattern, original_cut_pattern=self.original_cut_pattern)
        else:
            return self.__copy__()

    def allocate_cuts(self, allocations, number_of_agents):

        number_of_agents_in_first_partition = self.get_number_of_agents_in_first_partition(number_of_agents)

        proportion_of_first_partition = number_of_agents_in_first_partition / float(number_of_agents)

        self._set_next_query_direction()

        self._set_all_agents_cutmarks(allocations, proportion_of_first_partition)

        self._set_next_cutting_direction(allocations, number_of_agents)

        allocations.sort(key=lambda alloc: alloc.cutmarks[self.cut_direction])

        first_part_allocations, second_part_allocations = self._allocate_cuts(allocations, number_of_agents,
                                                                              number_of_agents_in_first_partition)

        return first_part_allocations, second_part_allocations

    def _set_all_agents_cutmarks(self, allocations, proportion_of_first_partition):
        dishonestIdx = -1
        for idx, agent_allocation in enumerate(allocations):
            if dishonestIdx < 0 and agent_allocation.getAgent().isDishonest():
                dishonestIdx = idx
            else:
                self._set_relevant_cutmarks(proportion_of_first_partition, agent_allocation)
        self._set_dishonest_cutmarks(allocations, dishonestIdx)

    def _set_dishonest_cutmarks(self, allocations, dishonestIdx):
        if dishonestIdx > -1:
            dis_agent_allocation = allocations[dishonestIdx]
            honest_allocations = allocations[:dishonestIdx] + allocations[dishonestIdx + 1:]
            self._set_best_cutmarks_for_dishonest(dis_agent_allocation, honest_allocations)

    def _set_relevant_cutmarks(self, proportion_of_first_partition, agent_allocation):
        value = proportion_of_first_partition * agent_allocation.getValue()
        agent_allocation.cutmarks = {}
        if (self.cut_query is CutDirection.Horizontal) or (self.cut_query is CutDirection.Both):
            agent_allocation.cutmarks[CutDirection.Horizontal] = agent_allocation.markQuery(value,
                                                                                            CutDirection.Horizontal)
        if (self.cut_query is CutDirection.Vertical) or (self.cut_query is CutDirection.Both):
            agent_allocation.cutmarks[CutDirection.Vertical] = agent_allocation.markQuery(value, CutDirection.Vertical)

    def _set_best_cutmarks_for_dishonest(self, allocation, honestAllocations):
        allocation.cutmarks = {}
        if (self.cut_query is CutDirection.Horizontal) or (self.cut_query is CutDirection.Both):
            horizontalHosnestHalfcuts = [allocation.cutmarks[CutDirection.Horizontal] for allocation in
                                         honestAllocations]
            allocation.cutmarks[CutDirection.Horizontal] = self._calculate_best_cutmark(allocation,
                                                                                        CutDirection.Horizontal,
                                                                                        horizontalHosnestHalfcuts)
        if (self.cut_query is CutDirection.Vertical) or (self.cut_query is CutDirection.Both):
            verticalHosnestHalfcuts = [allocation.cutmarks[CutDirection.Vertical] for allocation in
                                       honestAllocations]
            allocation.cutmarks[CutDirection.Vertical] = self._calculate_best_cutmark(allocation, CutDirection.Vertical,
                                                                                      verticalHosnestHalfcuts)

    def _divide_agents_to_partitions(self, sorted_allocations, cut_location, number_of_agents,
                                     number_of_agents_in_first_partition):
        first_part_allocations = []
        second_part_allocations = []
        for i in range(0, number_of_agents_in_first_partition):
            iFrom = sorted_allocations[i].getDirectionaliFrom(self.cut_direction)
            iTo = cut_location
            first_part_allocations.append(sorted_allocations[i].subCut(iFrom, iTo, self.cut_direction))
        for i in range(number_of_agents_in_first_partition, number_of_agents):
            iFrom = cut_location
            iTo = sorted_allocations[i].getDirectionaliTo(self.cut_direction)
            second_part_allocations.append(sorted_allocations[i].subCut(iFrom, iTo, self.cut_direction))
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
        return self.cut_pattern[0] if len(self.cut_pattern) > 0 else CutDirection.Horizontal

    def _get_query_direction_iterator_func(self):
        if self.freeplay_mode:
            return self._get_next_freeplay_query
        else:
            switcher = {
                CutPattern.HorVer: self._cut_pattern_opp_query_direction,
                CutPattern.VerHor: self._cut_pattern_opp_query_direction,
            }

            return switcher.get(self.cut_pattern, self._cut_pattern_same_query_direction)

    def _set_next_query_direction(self):
        if self.cut_query is None:
            self.cut_query = self._initial_cut_direction()
        else:
            dir_iter_func = self._get_query_direction_iterator_func()
            self.cut_query = dir_iter_func()

    def get_number_of_agents_in_first_partition(self, number_of_agents):
        raise NotImplementedError('This method should be overridden.')

    def _calculate_optional_cut_location(self, margin_iFrom, margin_iTo):
        raise NotImplementedError('This method should be overridden.')

    def _calculate_best_cutmark(self, allocation, query_direction, honest_cutmarks):
        raise NotImplementedError('This method should be overridden.')

    def _allocate_cuts(self, sorted_allocations, number_of_agents, number_of_agents_in_first_partition):
        end_of_first_part = sorted_allocations[number_of_agents_in_first_partition - 1].cutmarks[self.cut_direction]
        start_of_second_part = sorted_allocations[number_of_agents_in_first_partition].cutmarks[self.cut_direction]
        cut_location = self._calculate_optional_cut_location(end_of_first_part, start_of_second_part)
        first_part_allocations, second_part_allocations = self._divide_agents_to_partitions(sorted_allocations,
                                                                                            cut_location,
                                                                                            number_of_agents,
                                                                                            number_of_agents_in_first_partition)
        return first_part_allocations, second_part_allocations

    def _set_next_cutting_direction(self, allocations, number_of_agents):

        self.cut_direction = self.cut_query  # default next cut direction value is the cut query

        def _get_horizontal_cutmarks(alloc):
            return alloc.cutmarks[CutDirection.Horizontal]

        def _get_vertical_cutmarks(alloc):
            return alloc.cutmarks[CutDirection.Vertical]

        if self.cut_pattern in [CutPattern.SmallestHalfCut, CutPattern.SmallestPiece]:
            first_hor_cut = min(allocations, key=_get_horizontal_cutmarks)
            first_ver_cut = min(allocations, key=_get_vertical_cutmarks)
            first_hor_cut_halfcut = _get_horizontal_cutmarks(first_hor_cut)
            first_ver_cut_halfcut = _get_vertical_cutmarks(first_ver_cut)

            if self.cut_pattern is CutPattern.SmallestHalfCut:
                self.cut_direction = CutDirection.Horizontal if first_hor_cut_halfcut < first_ver_cut_halfcut else CutDirection.Vertical

            if self.cut_pattern is CutPattern.SmallestPiece:
                start_index_hor = first_hor_cut.getDirectionaliFrom(CutDirection.Horizontal)
                start_index_ver = first_ver_cut.getDirectionaliFrom(CutDirection.Vertical)
                hor_cut_size = (first_hor_cut_halfcut - start_index_hor) * first_ver_cut.getDimensions()[
                    CutDirection.Vertical]
                ver_cut_size = (first_ver_cut_halfcut - start_index_ver) * first_hor_cut.getDimensions()[
                    CutDirection.Horizontal]

                self.cut_direction = CutDirection.Horizontal if hor_cut_size < ver_cut_size else CutDirection.Vertical
            return

        if self.cut_pattern in [CutPattern.LongestDim, CutPattern.ShortestDim]:
            allocation = allocations[0]
            allocation_dimensions = allocation.getDimensions()

            if allocation_dimensions[CutDirection.Vertical] < allocation_dimensions[CutDirection.Horizontal]:
                self.cut_direction = CutDirection.Horizontal if self.cut_pattern is CutPattern.LongestDim else CutDirection.Vertical
            else:
                self.cut_direction = CutDirection.Vertical if self.cut_pattern is CutPattern.LongestDim else CutDirection.Horizontal
            return

        try:
            sorted_horizontal_cuts = list(
                map(_get_horizontal_cutmarks, sorted(allocations, key=_get_horizontal_cutmarks)))
            sorted_vertical_cuts = list(map(_get_vertical_cutmarks, sorted(allocations, key=_get_vertical_cutmarks)))
        except:  # if not both ver and hor cut lists are set, use default value
            return

        top_margin_index = self.get_number_of_agents_in_first_partition(number_of_agents)
        horizontal_margin = sorted_horizontal_cuts[top_margin_index] - sorted_horizontal_cuts[top_margin_index - 1]
        vertical_margin = sorted_vertical_cuts[top_margin_index] - sorted_vertical_cuts[top_margin_index - 1]

        if self.cut_pattern is CutPattern.LargestMargin:
            self.cut_direction = CutDirection.Horizontal if vertical_margin < horizontal_margin else CutDirection.Vertical
            return

        if self.cut_pattern is CutPattern.LargestMarginArea:
            allocation_dimensions = allocations[0].getDimensions()
            horizontal_margin_area = horizontal_margin * allocation_dimensions[CutDirection.Vertical]
            vertical_margin_area = vertical_margin * allocation_dimensions[CutDirection.Horizontal]
            self.cut_direction = CutDirection.Horizontal if vertical_margin_area < horizontal_margin_area else CutDirection.Vertical
            return

        hor_margin_iFrom = sorted_horizontal_cuts[top_margin_index - 1]
        hor_margin_iTo = sorted_horizontal_cuts[top_margin_index]
        hor_cut_option = self._calculate_optional_cut_location(hor_margin_iFrom, hor_margin_iTo)
        ver_margin_iFrom = sorted_vertical_cuts[top_margin_index - 1]
        ver_margin_iTo = sorted_vertical_cuts[top_margin_index]
        ver_cut_option = self._calculate_optional_cut_location(ver_margin_iFrom, ver_margin_iTo)

        if self.cut_pattern is CutPattern.LargestAvgMargin:
            hor_avg_margin = np.average([np.abs(cutmark - hor_cut_option) for cutmark in sorted_horizontal_cuts])
            ver_avg_margin = np.average([np.abs(cutmark - ver_cut_option) for cutmark in sorted_vertical_cuts])
            self.cut_direction = CutDirection.Horizontal if ver_avg_margin < hor_avg_margin else CutDirection.Vertical
            return

        if self.cut_pattern is CutPattern.MostValuableMargin:
            # we average the value of the entire piece made by cutting in the suggested location in the eyes of each
            # agent. this should be the same as measuring the actual margin because all agents have the same relative
            # eval of the sub-piece from start to their idea of where the margin begins.
            hor_margin_avg_value = np.average(list(map(lambda alloc:
                                                       alloc.getDirectionalValue(hor_cut_option,
                                                                                 CutDirection.Horizontal),
                                                       allocations)))
            ver_margin_avg_value = np.average(list(map(lambda alloc:
                                                       alloc.getDirectionalValue(ver_cut_option, CutDirection.Vertical),
                                                       allocations)))
            self.cut_direction = CutDirection.Horizontal if ver_margin_avg_value < hor_margin_avg_value else CutDirection.Vertical
            return

        if self.cut_pattern is CutPattern.SquarePiece:
            hor_face_ratio_avg_value = np.average(list(map(lambda alloc:
                                                           alloc.getDirectionalFaceRatio(hor_cut_option,
                                                                                         CutDirection.Horizontal),
                                                           allocations)))
            ver_face_ratio_value = np.average(list(map(lambda alloc:
                                                       alloc.getDirectionalFaceRatio(ver_cut_option,
                                                                                     CutDirection.Vertical),
                                                       allocations)))

            self.cut_direction = CutDirection.Horizontal if ver_face_ratio_value < hor_face_ratio_avg_value else CutDirection.Vertical
            return

        if self.cut_pattern is CutPattern.HighestScatter:
            neighbor_horizontal_cuts = list(zip(sorted_horizontal_cuts, np.roll(sorted_horizontal_cuts, -1)))[:-1]
            neighbor_vertical_cuts = list(zip(sorted_vertical_cuts, np.roll(sorted_vertical_cuts, -1)))[:-1]
            hor_scatter_avg_value = np.average([b - a for (a, b) in neighbor_horizontal_cuts])
            ver_scatter_avg_value = np.average([b - a for (a, b) in neighbor_vertical_cuts])
            self.cut_direction = CutDirection.Horizontal if ver_scatter_avg_value < hor_scatter_avg_value else CutDirection.Vertical
            return


class EPCutter(Cutter):
    def __init__(self, cut_pattern, cut_query=None, cut_direction=None, original_cut_pattern=None):
        super(EPCutter, self).__init__(cut_pattern, cut_query, cut_direction, original_cut_pattern)

        if self.freeplay_mode:
            if len(self.cut_pattern) < 1:
                self.cut_pattern = self.original_cut_pattern
            self.first_part_cut_pattern = self.cut_pattern[2::2]
            self.second_part_cut_pattern = self.cut_pattern[1::2]

    def __copy__(self):
        return EPCutter(self.cut_pattern, self.cut_query, self.cut_direction, self.original_cut_pattern)

    def get_firt_part_cutter(self):
        if self.freeplay_mode:
            return EPCutter(self.first_part_cut_pattern, original_cut_pattern=self.original_cut_pattern)
        else:
            return self.__copy__()

    def get_second_part_cutter(self):
        if self.freeplay_mode:
            return EPCutter(self.second_part_cut_pattern, original_cut_pattern=self.original_cut_pattern)
        else:
            return self.__copy__()

    def get_number_of_agents_in_first_partition(self, number_of_agents):
        # for Even Paz, first partition holds half of the agents
        return int(np.ceil(number_of_agents / 2))

    def _calculate_optional_cut_location(self, margin_iFrom, margin_iTo):
        # for Even Paz, cutting is made at the middle of the margin
        return (margin_iFrom + margin_iTo) / 2.0

    def _calculate_best_cutmark(self, allocation, query_direction, honest_cutmarks):
        if len(honest_cutmarks) % 2 > 0:
            middleIdx = int(np.floor(len(honest_cutmarks) / 2))
        else:
            raise ValueError("Odd number of agents yet to be supported")  # todo: add support for odd num of agents

        honest_cutmarks.sort()
        query_halfcut = honest_cutmarks[middleIdx]
        low_half = allocation.subCut(allocation.getDirectionaliFrom(query_direction), query_halfcut, query_direction)
        high_half = allocation.subCut(query_halfcut, allocation.getDirectionaliTo(query_direction), query_direction)

        if low_half.getValue() > high_half.getValue():
            return query_halfcut - SMALLEST_NUMBER
        else:
            return query_halfcut + SMALLEST_NUMBER


class LDCutter(Cutter):
    def __init__(self, cut_pattern, cut_query=None, cut_direction=None, original_cut_pattern=None):
        super(LDCutter, self).__init__(cut_pattern, cut_query, cut_direction, original_cut_pattern)

        if self.freeplay_mode:
            if len(self.cut_pattern) < 1:
                self.cut_pattern = self.original_cut_pattern
            self.first_part_cut_pattern = self.cut_pattern[0]
            self.second_part_cut_pattern = self.cut_pattern[1:]

    def __copy__(self):
        return LDCutter(self.cut_pattern, self.cut_query, self.cut_direction, self.original_cut_pattern)

    def get_firt_part_cutter(self):
        if self.freeplay_mode:
            return LDCutter(self.first_part_cut_pattern, original_cut_pattern=self.original_cut_pattern)
        else:
            return self.__copy__()

    def get_second_part_cutter(self):
        if self.freeplay_mode:
            return LDCutter(self.second_part_cut_pattern, original_cut_pattern=self.original_cut_pattern)
        else:
            return self.__copy__()

    def _divider_chooser_allocation(self, allocations):
        divider_index = random.randint(0, 1)
        chooser_index = 1 - divider_index
        chooser_agent = allocations[chooser_index].getAgent()
        divider_piece = allocations[divider_index]
        divider_allocations = [divider_piece.__copy__(), divider_piece.__copy__()]
        first_piece, second_piece = super()._allocate_cuts(divider_allocations, 2, 1)
        chooser_allocations = [first_piece[0].getAllocatedPiece(chooser_agent),
                               second_piece[0].getAllocatedPiece(chooser_agent)]
        if chooser_allocations[0].getRelativeValue() > chooser_allocations[1].getRelativeValue():
            return [chooser_allocations[0]], second_piece
        else:
            return first_piece, [chooser_allocations[1]]

    def _allocate_cuts(self, allocations, number_of_agents, number_of_agents_in_first_partition):
        if number_of_agents > 2:
            first_part_allocations, second_part_allocations = \
                super()._allocate_cuts(allocations, number_of_agents, number_of_agents_in_first_partition)
        elif number_of_agents == 2:
            first_part_allocations, second_part_allocations = self._divider_chooser_allocation(allocations)
        else:
            # should not happen
            raise ValueError("Division is only possible from 2 agents and up.")

        return first_part_allocations, second_part_allocations

    def get_number_of_agents_in_first_partition(self, number_of_agents):
        # for Last Diminisher, first partition holds only one agent (with the smallest mark)
        return 1

    def _calculate_optional_cut_location(self, margin_iFrom, margin_iTo):
        return margin_iFrom

    def _calculate_best_cutmark(self, allocation, query_direction, honest_cutmarks):
        honest_cutmarks.sort()
        query_cutmark = honest_cutmarks[0]
        raise NotImplementedError("Strategic play for LastDiminisher is not yet implemented.")
        # todo: implement a strategic play - note that if query_cutmark results in a piece better then 1/n of the
        #  cake than the player can claim (query_cutmark - SMALLEST_NUMBER) and if not, (query_cutmark +
        #  SMALLEST_NUMBER).
        #
        # low_half = allocation.subCut(allocation.getDirectionaliFrom(query_direction), query_cutmark, query_direction)
        # high_half = allocation.subCut(query_cutmark, allocation.getDirectionaliTo(query_direction), query_direction)
        #
        # if low_half.getValue() > high_half.getValue():
        #     return query_cutmark - SMALLEST_NUMBER
        # else:
        #     return query_cutmark + SMALLEST_NUMBER


class SimpleCutter(Cutter):
    def __init__(self, cut_pattern, cut_query=None, cut_direction=None, original_cut_pattern=None):
        super(SimpleCutter, self).__init__(cut_pattern, cut_query, cut_direction, original_cut_pattern)

        if self.freeplay_mode:
            if len(self.cut_pattern) < 1:
                self.cut_pattern = self.original_cut_pattern
            self.first_part_cut_pattern = self.cut_pattern[2::2]
            self.second_part_cut_pattern = self.cut_pattern[1::2]
        elif cut_pattern not in [CutPattern.Hor, CutPattern.Ver]:
            raise ValueError(
                'NoPattern Cutter can either cut horizontally or vertically %s is not supported.' % cut_pattern)

    def __copy__(self):
        return SimpleCutter(self.cut_pattern, self.cut_query, self.cut_direction, self.original_cut_pattern)

    def get_firt_part_cutter(self):
        if self.freeplay_mode:
            return SimpleCutter(self.first_part_cut_pattern, original_cut_pattern=self.original_cut_pattern)
        else:
            return self.__copy__()

    def get_second_part_cutter(self):
        if self.freeplay_mode:
            return SimpleCutter(self.second_part_cut_pattern, original_cut_pattern=self.original_cut_pattern)
        else:
            return self.__copy__()

    def get_number_of_agents_in_first_partition(self, number_of_agents):
        # for NoPattern Cutter, first partition holds half of the agents
        return int(np.ceil(number_of_agents / 2))

    def _calculate_optional_cut_location(self, margin_iFrom, margin_iTo):
        # for NoPattern cutter, cutting is made at the middle of the margin
        return (margin_iFrom + margin_iTo) / 2.0

    def _calculate_best_cutmark(self, allocation, query_direction, honest_cutmarks):
        if len(honest_cutmarks) % 2 > 0:
            middleIdx = int(np.floor(len(honest_cutmarks) / 2))
        else:
            raise ValueError("Odd number of agents yet to be supported")  # todo: add support for odd num of agents

        honest_cutmarks.sort()
        query_halfcut = honest_cutmarks[middleIdx]
        low_half = allocation.subCut(allocation.getDirectionaliFrom(query_direction), query_halfcut, query_direction)
        high_half = allocation.subCut(query_halfcut, allocation.getDirectionaliTo(query_direction), query_direction)

        if low_half.getValue() > high_half.getValue():
            return query_halfcut - SMALLEST_NUMBER
        else:
            return query_halfcut + SMALLEST_NUMBER

import ast
import csv
import os
import pickle
from math import log, ceil, floor

from matplotlib import pyplot as plt, patches
from matplotlib.cbook import deprecated

from utils.maps.map_handler import get_original_map_from_index
from utils.simulation.agent import ShadowAgent
from utils.simulation.simulation_log import SimulationLog as slog

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split('utils')[0]

@deprecated
def plot_partition_from_path(partition_path):
    partition_data = {}
    with open(partition_path) as partition_file:
        reader = csv.reader(partition_file)
        for row in reader:
            if row[0] == " ":
                continue
            if row[1][0] == "." or row[1][0].isalpha():
                partition_data[row[0]] = row[1]
            else:
                partition_data[row[0]] = ast.literal_eval(row[1])

    input_path = os.path.join(
        os.path.dirname(partition_data["Agent Files"][0]).replace("./", ""),
        "orig.txt",
    )
    with open(input_path, "rb") as mapfile:
        baseline_map = pickle.load(mapfile)

    plot_partition(baseline_map, partition_data["Partition"])


def plot_partition_from_log_path(log_path: str, show=False):
    sim_log: slog = slog.load_log_file(log_path)

    plot_partition_from_log_object(sim_log, show)


def plot_partition_from_log_object(sim_log: slog, show=False):
    log_path = sim_log.output_log_file_path

    index_path = os.path.join(ROOT_DIR,
        os.path.dirname(sim_log.agent_map_files_list[0]).replace("./", ""),
        "index.txt",
    )

    basemap_path = os.path.join(ROOT_DIR, get_original_map_from_index(index_path))
    with open(basemap_path, "rb") as mapfile:
        baseline_map = pickle.load(mapfile)

    plot_partition(baseline_map, _extract_partition_plots(sim_log.printable_partition), log_path+'.png', show)


def plot_all_plots_from_log_path(log_path: str, show=False):
    sim_log: slog = slog.load_log_file(log_path)

    plot_all_plots_from_log_object(sim_log, show)


def plot_all_plots_from_log_object(sim_log: slog, show=False):
    log_path = sim_log.output_log_file_path
    partition = _extract_partition_plots(sim_log.printable_partition)
    partition_plots = {}
    for agent_map in sim_log.agent_map_files_list:
        agent_map_path = os.path.join(ROOT_DIR, agent_map.replace("./", ""))
        cur_agent_number = ShadowAgent(agent_map_path).get_map_file_number()

        with open(agent_map_path, "rb") as mapfile:
            agent_map = pickle.load(mapfile)
            partition_plots[cur_agent_number] = (partition[cur_agent_number],agent_map)

    plot_separate_partition(partition_plots, log_path+'.png', show)


def plot_separate_partition(partition_plots, out_path='', show=False):
    log_num_agents = log(float(len(partition_plots)), 2)
    cols = pow(2, ceil(log_num_agents / 2.0))
    rows = pow(2, floor(log_num_agents / 2.0))
    ax = []
    fig = plt.figure()
    # plt.axis([0, len(partition_map[0]), len(partition_map), 0])
    for i,agent in enumerate(partition_plots):
        img = partition_plots[agent][1]
        part = partition_plots[agent][0]
        xy = (part[1], part[0])
        height = part[2] - part[0]
        width = part[3] - part[1]
        centerx = xy[0] + 0.5*width - 10
        centery = xy[1] + 0.5*height
        # Create a Rectangle patch
        rect = patches.Rectangle(
            xy, width, height, linewidth=0.5, edgecolor="g", facecolor="g", alpha=0.15
        )
        rect_b = patches.Rectangle(
            xy, width, height, linewidth=1, edgecolor="g", facecolor="none"
        )
        ax.append(fig.add_subplot(rows,cols,i+1))
        ax[-1].set_title("Agent"+agent)
        # Add the patch to the Axes
        ax[-1].add_patch(rect)
        ax[-1].add_patch(rect_b)
        plt.imshow(img, cmap="hot")
    if out_path:
        plt.savefig(out_path)
    if show:
        plt.show()


def plot_partition(partition_map, partition_plots, out_path='', show=False):
    fig, ax = plt.subplots(1)
    ax.imshow(partition_map, cmap="hot")
    plt.axis([0, len(partition_map[0]), len(partition_map), 0])
    for agent in partition_plots:
        part = partition_plots[agent]
        xy = (part[1], part[0])
        height = part[2] - part[0]
        width = part[3] - part[1]
        centerx = xy[0] + 0.5*width - 10
        centery = xy[1] + 0.5*height
        # Create a Rectangle patch
        rect = patches.Rectangle(
            xy, width, height, linewidth=0.5, edgecolor="g", facecolor="none"
        )
        plt.text(centerx, centery, "A" + agent, color="g")
        # Add the patch to the Axes
        ax.add_patch(rect)
    if out_path:
        plt.savefig(out_path)
    if show:
        plt.show()


def _extract_partition_plots(partition_data):
    partition_plots = {}
    for part in partition_data:
        agent = part.split("(")[1].split(")")[0]
        partition_plots[agent] = ast.literal_eval(
            part.split("receives ")[1].split(" - ")[0]
        )
    return partition_plots


def plot_map(path: str):
    with open(path, "rb") as mapfile:
        partition_map = pickle.load(mapfile)
    fig, ax = plt.subplots(1)
    ax.imshow(partition_map, cmap="hot")
    if "_HS" in path:
        hotspot_center = path.split("_HS")[-1].replace(".txt","").split("_")
        hotspot = patches.Circle((float(hotspot_center[0]),float(hotspot_center[1])), 10, edgecolor="g", facecolor="g")
        ax.add_patch(hotspot)
    plt.axis([0, len(partition_map[0]), len(partition_map), 0])
    plt.show()


if __name__ == "__main__":
    plot_map("D:/MSc/221_valueMap_noise0.6_HS100_200.txt")
    plot_partition_from_log_path(
        '../../results/2020-02-14T18-33-34/IsraelMaps02_2020-02-14T18-33-37_NoiseProportion_0.6_2_exp/logs/82_Honest_FOCS_NoPatternUV.log')
    plot_all_plots_from_log_path(
        '../../results/2020-02-14T18-33-34/IsraelMaps02_2020-02-14T18-33-37_NoiseProportion_0.6_2_exp/logs/82_Honest_FOCS_NoPatternUV.log')
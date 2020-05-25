import json
import os
import pickle
import random
import time
from math import exp


def hotspot_noise_function(
    original_map, noise_proportion, normalized_sum, max_value
):
    rows = len(original_map)
    cols = len(original_map[0])
    hotspot_center = (random.randint(0, rows), random.randint(0, cols))
    print(hotspot_center)

    def hotspot_noise(xj, yj):
        """ hotspot_noise = noise*exp(-((xj-xc)^2+(yj-yc)^2)^0.1) """
        dx = pow((hotspot_center[1] - xj), 2)
        dy = pow((hotspot_center[0] - yj), 2)

        noise_addition = noise_proportion * exp(-pow(dx + dy, 0.1))
        return noise_addition

    new_map = [
        [original_map[r][c] * (1 + hotspot_noise(c, r)) for c in range(cols)]
        for r in range(rows)
    ]

    new_map = normalize_map(cols, new_map, normalized_sum, rows)
    return new_map, "_HS%s_%s" % hotspot_center


def nonzero_uniform_noise_function(
    original_map, noise_proportion, normalized_sum, max_value=1000000
):
    rows = len(original_map)
    cols = len(original_map[0])
    new_map = [
        [
            random.uniform(0, max_value) if original_map[r][c] > 0 else 0
            for c in range(cols)
        ]
        for r in range(rows)
    ]
    new_map = normalize_map(cols, new_map, normalized_sum, rows)
    return new_map,""


def uniform_noise_function(
    original_map, noise_proportion, normalized_sum, max_value
):
    rows = len(original_map)
    cols = len(original_map[0])

    """ Done to ensure noisy outcome value is not negative """
    neg_noise_proportion = max(-1, -noise_proportion)

    new_map = [
        [
            original_map[r][c]
            * (1 + random.uniform(neg_noise_proportion, noise_proportion))
            for c in range(cols)
        ]
        for r in range(rows)
    ]

    new_map = normalize_map(cols, new_map, normalized_sum, rows)
    return new_map,""


def normalize_map(cols, new_map, normalized_sum, rows):
    if normalized_sum is not None and normalized_sum > 0:
        aggregated_sum = sum([sum(new_map[r]) for r in range(rows)])
        if aggregated_sum > 0:
            normalization_factor = normalized_sum / aggregated_sum
            new_map = [
                new_map[r][c] * normalization_factor
                for r in rows
                for c in cols
            ]
    return new_map


def random_values(rows, cols, max_value, normalized_sum):
    new_map = [
        [(random.uniform(0, max_value)) for _ in range(cols)]
        for _ in range(rows)
    ]
    if normalized_sum is not None and normalized_sum > 0:
        aggregated_sum = sum([sum(new_map[r]) for r in range(rows)])
        if aggregated_sum > 0:
            normalization_factor = normalized_sum / aggregated_sum
            new_map = [
                new_map[r][c] * normalization_factor
                for r in rows
                for c in cols
            ]
    return new_map


def generate_value_maps_to_file(
    original_map_file,
    folder,
    dataset_name,
    noise,
    num_of_maps,
    normalized_sum,
    noise_function,
    rows=1490,
    cols=1020,
):
    folder = folder + dataset_name

    if not os.path.exists(folder):
        os.mkdir(folder)

    random_maps = False
    if original_map_file is None:
        random_maps = True
        if noise is None:
            noise = 1000000
        print(
            "Creating %s random value maps to folder %s with max value %s"
            % (num_of_maps, folder, noise)
        )

    else:
        with open(original_map_file, "rb") as data_file:
            original_map_data = pickle.load(data_file)
            max_value = max([max(row) for row in original_map_data])
        print(
            "Creating %s value maps to folder %s with noise proportion %s"
            % (num_of_maps, folder, noise)
        )

    index_output_path = "%s/index.txt" % folder
    start_all = time.time()
    paths = []
    end = time.time()
    for i in range(num_of_maps):
        start = end
        output_path = "%s/%s_valueMap_noise%s.txt" % (folder, i, noise)

        print("\tstart saving value maps to file #%s" % i)
        if random_maps:
            new_map = random_values(rows, cols, noise, normalized_sum)
            if i == 0:
                original_map_file = output_path
        else:
            new_map, label = noise_function(
                original_map_data, noise, normalized_sum, max_value
            )
            output_path = output_path.replace(".txt", label+".txt")
        with open(output_path, "wb") as object_file:
            pickle.dump(new_map, object_file)
        paths.append(output_path)
        end = time.time()
        print("\t\tmap %s creation time was %s seconds" % (i, end - start))

    paths = [p.replace("..", ".") for p in paths]
    index = {
        "datasetName": dataset_name,
        "numOfMaps": num_of_maps,
        "folder": folder.replace("..", "."),
        "originalMapFile": original_map_file.replace("..", "."),
        "noise": noise,
        "mapsPaths": paths,
    }

    with open(index_output_path, "w") as index_file:
        json.dump(index, index_file)
    print("The whole creation process time was %s seconds" % (end - start_all))

    return index_output_path
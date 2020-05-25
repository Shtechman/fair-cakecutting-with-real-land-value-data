#!python3
from utils.maps.map_noise_functions import hotspot_noise_function, nonzero_uniform_noise_function, uniform_noise_function, \
    generate_value_maps_to_file

""" Possible base maps (saved to file using Pickle) """
og_maps = {
    "israel": "./data/originalMaps/IsraelMap.txt",
    "newzealand": "./data/originalMaps/newzealand_forests_2D_low_res.txt",
    "random": "./data/originalMaps/RandomNoiseMap.txt",
}

""" Possible noise assignment functions """
noise_methods = {
    "uniform": uniform_noise_function,
    "uniformNZ": nonzero_uniform_noise_function,
    "hotspot": hotspot_noise_function,
}

""" Base folder to save the agents to (a sub-folder will be generate for each set) """
base_folder = "./data/"

if __name__ == "__main__":

    """ Noise paramater r to add to base map. """
    noise = 0.6

    """ Number of different agents to generate.
        NOTE! since simulation picks some agents from the folder at random,
        you should create more then actually needed """
    num_of_agents = 1024

    """ datasets dictionary - from each dataset entry, a set of agents will be generated """
    datasets = [
        # <datasetName> <input_file> <noise> <num_of_agents> <noise_method>"
        {
            "datasetName": "newZealandLowRes06HS",
            "input_file": og_maps["newzealand"],
            "noise": noise,
            "num_of_agents": num_of_agents,
            "noise_method": noise_methods["hotspot"],
        },
        {"datasetName": "Israel06HS",
         "input_file": og_maps["israel"],
         "noise": noise,
         "num_of_agents": num_of_agents,
         "noise_method": noise_methods["hotspot"]},
    ]

    for dataset in datasets:
        indexFile = generate_value_maps_to_file(
            dataset["input_file"],
            base_folder,
            dataset["datasetName"],
            dataset["noise"],
            dataset["num_of_agents"],
            None,
            dataset["noise_method"],
        )
        print(
            "Save index file path to use in the simulation (main.py):\n\t\t",
            indexFile,
        )

print("all done")

#!python3
from utils.MapFileHandler import uniform_noise_function, nonzero_uniform_noise_function, hotspot_noise_function, \
    generate_valueMaps_to_file

""" Possible base maps (saved to file using Pickle) """
og_maps = {"israel": '../data/originalMaps/IsraelMap.txt',
           "newzealand": '../data/originalMaps/newzealand_forests_2D_low_res.txt',
           "random": '../data/originalMaps/RandomNoiseMap.txt'}

""" Possible noise assignment functions """
noise_methods = {"uniform": uniform_noise_function,
                 "uniformNZ": nonzero_uniform_noise_function,
                 "hotspot": hotspot_noise_function}

""" Base folder to save the agents to (a sub-folder will be generate for each set) """
base_folder = "../data/"

if __name__ == '__main__':

    """ Noise paramater r to add to base map. """
    noise = 0.6

    """ Number of different agents to generate.
        NOTE! since simulation picks some agents from the folder at random,
        you should create more then actually needed """
    numOfAgents = 512

    """ datasets dictionary - from each dataset entry, a set of agents will be generated """
    datasets = [
        # <datasetName> <input_file> <noise> <numOfAgents> <noise_method>"
        {"datasetName": "newZealand_nonZuniform",
         "input_file": og_maps["newzealand"],
         "noise": noise,
         "numOfAgents": numOfAgents,
         "noise_method": noise_methods["uniformNZ"]},

        # {"datasetName": "newZealandLowResAgents06HS",
        #  "input_file": og_maps["israel"],
        #  "noise": noise,
        #  "numOfAgents": numOfAgents,
        #  "noise_method": noise_methods["hotspot"]},
    ]

    for dataset in datasets:
        indexFile = generate_valueMaps_to_file(dataset["input_file"], base_folder, dataset["datasetName"],
                                               dataset["noise"], dataset["numOfAgents"], None, dataset["noise_method"])
        print('Save index file path to use in the simulation (main.py):\n\t\t', indexFile)

print("all done")

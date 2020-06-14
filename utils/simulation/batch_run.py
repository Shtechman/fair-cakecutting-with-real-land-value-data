import json
import os
from datetime import datetime


def build_batch_file_lines(json_path, prefix):
    return ["#!/bin/bash",
            "#SBATCH -p work",
            "#SBATCH -c 4",
            "#SBATCH -n 1",
            "#SBATCH --mem 50000",
            "#SBATCH --job-name=CCExp",
            "#SBATCH --output=out_logs/%x-%j.out",
            "#SBATCH --error=err_logs/%x-%j.err",
            "echo \"running experiments with json params.\"",
            "python jsonexprunner.py \"%s\" \"%s\"" % (json_path, prefix),
            "exit"]


def schedule_batch_run(prefix, params):
    prefix = prefix.replace("/", "")
    batch_folder = './batch_files'
    if not os.path.exists(batch_folder):
        os.makedirs(batch_folder)
    json_file_name = batch_folder\
                     + '/'\
                     + prefix \
                     + "-" \
                     + datetime.now().isoformat(timespec="microseconds").replace(":", "-").replace(".", "-")\
                     + ".json"
    batch_file_name = json_file_name.replace('.json', '')

    with open(json_file_name, "w") as json_file:
        json.dump(params, json_file)

    with open(batch_file_name, "w") as batch_file:
        [batch_file.write(line + '\n') for line in build_batch_file_lines(json_file_name, prefix)]

    os.system('sbatch %s' % batch_file_name)

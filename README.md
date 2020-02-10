Fair-Division
=============

Algorithms and experiments related to fair division of land.

The first version was in Node.js. It is working and kept in the folder "oldNodeJsCode".

The current version is in Python 3. 


Prerequisites
-------------

    sudo pip3 install matplotlib numpy pandas pyparsing pytz scipy six xlrd


Running
-------

For manual tests, run:

    python3 freePlay.py

For agent maps dataset creation, edit inner script params in mapSetGenerator.py, then run:

    python3 mapSetGenerator.py
    
For full simulation, edit inner config params at the top of main.py, then run: 

    python3 main.py [<num_of_repetitions> [<num_of_parallel_task> [<log_min_num_of_agents> <log_max_num_of_agents>]]]
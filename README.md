fair-division
=============

Algorithms and experiments related to fair division of land.

The first version was in Node.js. It is working and kept in the folder "oldNodeJsCode".

The second version was in Python 3. It is not working very well.

Prerequisites
-------------
To run the experiments you need [Node.js](http://nodejs.org/).

To show the plots you need [gnuplut](http://www.gnuplot.info/). 
The results are produced in plain text format so you can probably use other plotting software. 

Installation
------------
Clone the repository, then run:

		npm install
		
Experiments
-----------	

Edit the main.js file and change the experiment details, i.e. the numbers of agents or the noise proportion.

Go to the oldNodeJsCode folder and run: 

		node main.js

This will create some data files in the "results/" folder. 

On the console, you will see some gnuplot commands that you can use to plot the results, such as:

		gnuplot --persist  -e "filename='results/evenpaz-noise-0.2.dat'; xcolumn=1" main.gnuplot
 
Plots
-----	
The plots show the performance of the fair division algorithm ("fair"), vs. the performance of the assessor division ("shamay").

Currently there are three performance measures: 

* Egalitarian social welfare
* Utilitarian social welfare
* Maximum envy felt by a single person


Folders
-------

* data/ - contains land-value maps and files for processing these maps. 
* lib/  - contains the main code implementing the fair division algorithms and data structures.
* node_modules/ - contains third-party modules automatically installed by npm install.
* results/ - holds the result files created by the main program.


/**
 * @author Erel Segal-Halevi
 * @since 2016-11
 */

var _ = require("underscore")
  , fs = require("fs")
  , ValueFunction1D = require("./lib/ValueFunction1D")
  , AllocatedPiece1D = require("./lib/AllocatedPiece1D")
  , inputvalues = require("./lib/inputvalues")
  , evenpaz1d = require("./lib/evenpaz1d")
  , cakepartitions = require("./lib/cakepartitions")
  , rungnuplot = require('./lib/rungnuplot')
  ;

//var NOISE_PROPORTIONS = [0,0.25,0.5,1];
//var NOISE_PROPORTIONS = _.range(0.05, 1, 0.05);
var NOISE_PROPORTIONS = [0.5];

var AGENT_NUMS = [2,4,8,16,32,64,128];
//var AGENT_NUMS = [2,4,16,256,1024];
//var AGENT_NUMS = [128,512,2048];
//var AGENT_NUMS = [1024];

var AUTOMATICALLY_RUN_GNUPLOT = true;

var EXPERIMENTS_PER_CELL = 10;

var LAND_SIZE = 1000;
var VALUE_PER_CELL = 100;
var FILENAME = "../data/newzealand_forests_npv_4q.1d.json";

var meanValues = inputvalues.valuesFromFile(FILENAME);
console.log("cells in land: "+meanValues.length);

var AGGREGATE_BY_AGENT_NUM = false;


if (!Math.log2) {
	LOG2 = Math.log(2);
	Math.log2 = function(x) {
		return Math.log(x)/LOG2;
	}
}


if (AGGREGATE_BY_AGENT_NUM) {
	for (var numOfAgents in AGENT_NUMS) {
		var numOfAgents = AGENT_NUMS[iAgentNum];
		console.log(numOfAgents+" agents");
		var resultsFileName = "results/evenpaz-agents-"+numOfAgents+".dat";
		var resultsFile = fs.createWriteStream(resultsFileName);
		var identicalValueFunctions = inputvalues.noisyValuesArray(meanValues, 0, null, numOfAgents).map(ValueFunction1D.fromValues);
		var identicalPartition = evenpaz1d(identicalValueFunctions);
		for (var iNoise in NOISE_PROPORTIONS) {
			var noiseProportion = NOISE_PROPORTIONS[iNoise];
			console.log("\t"+noiseProportion+" noise");
			calculateSingleDatapoint(numOfAgents,noiseProportion,resultsFile);
		}
		resultsFile.end();
		rungnuplot("utilitarian.gnuplot", "filename='"+resultsFileName+"'; xcolumn=3; xlabel='amplitude of deviation in utilities'", /*dry-run=*/!AUTOMATICALLY_RUN_GNUPLOT);
		rungnuplot("egalitarian.gnuplot", "filename='"+resultsFileName+"'; xcolumn=3; xlabel='amplitude of deviation in utilities'", /*dry-run=*/!AUTOMATICALLY_RUN_GNUPLOT);
	}
} else {  // aggregate by noise
	for (var iNoise in NOISE_PROPORTIONS) {
		var noiseProportion = NOISE_PROPORTIONS[iNoise];
		console.log(noiseProportion+" noise");
		var resultsFileName = "results/evenpaz-noise-"+noiseProportion+".dat";
		var resultsFile = fs.createWriteStream(resultsFileName);
		for (var iAgentNum in AGENT_NUMS) {
			var numOfAgents = AGENT_NUMS[iAgentNum];
			console.log("\t"+numOfAgents+" agents");
			var identicalValueFunctions = inputvalues.noisyValuesArray(meanValues, 0, null, numOfAgents).map(ValueFunction1D.fromValues);
			var identicalPartition = evenpaz1d(identicalValueFunctions);
			calculateSingleDatapoint(numOfAgents,noiseProportion,resultsFile);
		}
		resultsFile.end();
    rungnuplot("utilitarian.gnuplot", "filename='"+resultsFileName+"'; xcolumn=2; xlabel='log num of people'", /*dry-run=*/!AUTOMATICALLY_RUN_GNUPLOT);
    rungnuplot("egalitarian.gnuplot", "filename='"+resultsFileName+"'; xcolumn=2; xlabel='log num of people'", /*dry-run=*/!AUTOMATICALLY_RUN_GNUPLOT);
	}
}

function calculateSingleDatapoint(numOfAgents,noiseProportion,resultsFile) {
	for (var iExperiment=0; iExperiment<EXPERIMENTS_PER_CELL; ++iExperiment) {
		var valueFunctions = inputvalues.noisyValuesArray(meanValues, noiseProportion, null, numOfAgents).map(ValueFunction1D.fromValues);
		var partition = evenpaz1d(valueFunctions);
		var egalitarianGain = cakepartitions.normalizedEgalitarianValue(partition)-1;
		if (egalitarianGain<-0.001) throw new Error("In proportional division, normalized egalitarian gain must be at least 0; got "+egalitarianGain);
		var utilitarianGain = cakepartitions.utilitarianValue(partition)-1;
		if (utilitarianGain<-0.001) throw new Error("In proportional division, utilitarian gain must be at least 0; got "+utilitarianGain);
		var envy = cakepartitions.largestEnvy(partition);


		var identicalPartitionWithDifferentAgents = _.zip(valueFunctions,identicalPartition).map(function(pair) {
			return new AllocatedPiece1D(pair[0], pair[1].iFrom, pair[1].iTo);
		});
		var egalitarianGainIPWDA = cakepartitions.normalizedEgalitarianValue(identicalPartitionWithDifferentAgents)-1;
		var utilitarianGainIPWDA = cakepartitions.utilitarianValue(identicalPartitionWithDifferentAgents)-1;
		var envyIPWDA = cakepartitions.largestEnvy(identicalPartitionWithDifferentAgents);

		var data = numOfAgents+"\t"+Math.log2(numOfAgents)+"\t"+noiseProportion+"\t"+
			egalitarianGain+"\t"+utilitarianGain+"\t"+envy+"\t"+
			egalitarianGainIPWDA+"\t"+utilitarianGainIPWDA+"\t"+envyIPWDA;
		resultsFile.write(data+"\n");
	}
}

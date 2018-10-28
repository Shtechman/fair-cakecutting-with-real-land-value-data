// A utility to convert a given 2-dimensional data file to a 1-dimensional JSON containing the sums of the rows.

var input = "./newzealand_forests_npv_4q.asc";
var output = "./newzealand_forests_npv_4q.1d.json";

var fs = require('fs')
var liner = require('./liner')
var source = fs.createReadStream(input)
var sums = [];
source.pipe(liner)
liner.on('readable', function () {
     var line
     while (line = liner.read()) {
    	 if (/^[a-z]/i.test(line))
        	  continue;
         var values = line.split(" ").
         	map(function(x){return parseInt(x)}).
         	filter(function(x) { return !isNaN(x)});
         //console.dir(values)
         var sum = values.reduce( function(sum,x){
        	  return x>0? sum+x: sum} ,0 )
          sums.push(sum)
     }
})
liner.on('end', function() {
	fs.writeFile(output, JSON.stringify(sums));
})

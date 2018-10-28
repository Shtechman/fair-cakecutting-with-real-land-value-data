/**
 * A simple utility to run gnuplot on a given results file.
 * 
 * @author Erel Segal-Halevi
 * @since 2014-08
 * 
 */

var exec = require("child_process").exec;

module.exports = function rungnuplot(gnuplotFilename, params, dryRun) {
	var command = "gnuplot --persist "+(params? " -e \""+params+"\" ": "")+gnuplotFilename;
	if (dryRun) {
		console.log(command);
	} else {
		console.log("Running "+command);
		exec(command, function (error, stdout, stderr) {
				if (stdout) console.log('stdout: ' + stdout);
				if (stderr) console.log('stderr: ' + stderr);
				if (error !== null) {
					console.log('exec error: ' + error);
				}
			}
		);
	}
}

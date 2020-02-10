#!python3

# From http://stackoverflow.com/a/1057534/827927

import glob
from os.path import dirname, basename, isfile

modules = glob.glob(dirname(__file__) + "/*.py")
__all__ = [basename(f)[:-3] for f in modules if isfile(f)]
